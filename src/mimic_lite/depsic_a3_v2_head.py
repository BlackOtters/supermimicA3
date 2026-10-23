"""Versioned SMPL-root-head to A3 external head-target projector."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from mimic_lite.depsic_a3_contract import A3_HEAD_DIM
from mimic_lite.tasks.depsic_a3 import _A3ReferencePool, _load_a3_reference_pool
from mimic_lite.tasks.depsic_paired_smpl import (
    PairedSmplReferencePool,
    load_paired_smpl_reference_pool,
)


# v5 consumes column-major root-local head 6D from paired-cache v5.  Its
# coefficients are representation-specific and must be refit, not reused.
PROJECTOR_VERSION = 5
A3_HEAD_LIMITS = ((-1.0472, 1.0472), (-0.436332, 0.261799))


@dataclass(frozen=True)
class A3HeadProjector:
    weights: torch.Tensor
    bias: torch.Tensor
    manifest_hash: str
    train_rmse: float
    heldout_rmse: float
    heldout_max_error: float
    cache_path: Path

    def __call__(self, head_rot6_root: torch.Tensor) -> torch.Tensor:
        if head_rot6_root.shape[-1] != 6:
            raise ValueError(f"A3 head projector expects [..., 6], got {tuple(head_rot6_root.shape)}")
        result = head_rot6_root @ self.weights.to(head_rot6_root) + self.bias.to(head_rot6_root)
        low = result.new_tensor((A3_HEAD_LIMITS[0][0], A3_HEAD_LIMITS[1][0]))
        high = result.new_tensor((A3_HEAD_LIMITS[0][1], A3_HEAD_LIMITS[1][1]))
        return result.clamp(low, high)


def _fingerprint(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_root(value: str | Path | None) -> Path:
    configured = value or os.getenv("DEPSIC_REFERENCE_CACHE_ROOT")
    root = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[3] / ".cache" / "depsic"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _fit_ridge(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(features.shape[0], 1, dtype=torch.float64)
    x = torch.cat((features.to(torch.float64), ones), dim=1)
    y = targets.to(torch.float64)
    regularizer = torch.eye(x.shape[1], dtype=x.dtype) * float(ridge)
    regularizer[-1, -1] = 0.0
    solution = torch.linalg.solve(x.T @ x + regularizer, x.T @ y)
    return solution[:-1].float(), solution[-1].float()


def load_or_fit_a3_head_projector(
    *,
    a3_motion_file: str | Path,
    smpl_motion_file: str | Path,
    motion_pool_size: int | None = None,
    ridge: float = 1.0e-4,
    heldout_fraction: float = 0.1,
    max_heldout_rmse: float = 0.20,
    max_heldout_error: float = 0.60,
    cache_root: str | Path | None = None,
    paired_smpl_pool: PairedSmplReferencePool | None = None,
    a3_pool: _A3ReferencePool | None = None,
) -> A3HeadProjector:
    """Fit/reuse a deterministic affine ridge mapping with held-out gates."""
    if not 0.0 < heldout_fraction < 0.5:
        raise ValueError("heldout_fraction must be in (0, 0.5)")
    if ridge < 0.0 or max_heldout_rmse <= 0.0 or max_heldout_error <= 0.0:
        raise ValueError("ridge and projector error thresholds must be positive")
    a3_pool = a3_pool or _load_a3_reference_pool(
        a3_motion_file, motion_pool_size=motion_pool_size
    )
    smpl_pool = paired_smpl_pool or load_paired_smpl_reference_pool(
        a3_pool.paths, a3_pool.lengths, a3_pool.fps,
        smpl_motion_file=smpl_motion_file, cache_root=cache_root,
    )
    if not torch.equal(smpl_pool.lengths, a3_pool.lengths) or not torch.allclose(smpl_pool.fps, a3_pool.fps):
        raise ValueError("A3 head projector paired-SMPL pool is not aligned to the A3 reference pool")
    root = _cache_root(cache_root)
    inputs = {
        "version": PROJECTOR_VERSION,
        "a3": [_fingerprint(Path(path)) for path in a3_pool.paths],
        "smpl_manifest_hash": smpl_pool.manifest_hash,
        "ridge": float(ridge),
        "heldout_fraction": float(heldout_fraction),
    }
    manifest_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_path = root / f"a3-head-projector-{manifest_hash[:24]}.pt"
    lock_path = cache_path.with_suffix(".lock")

    import fcntl
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        payload = None
        if cache_path.is_file():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, ValueError):
                payload = None
        if not isinstance(payload, dict) or payload.get("manifest_hash") != manifest_hash:
            # This property delegates to the same explicit column-major
            # serializer used by the runtime command.  Do not replace it with
            # ``matrix[..., :2].reshape(6)``: that is row-major.
            features = smpl_pool.head_rot6_root
            targets = a3_pool.head_q
            # Hold out complete clip tails deterministically, avoiding adjacent
            # frames from the same pose in both split partitions.
            train_mask = torch.ones(features.shape[0], dtype=torch.bool)
            offset = 0
            for length in a3_pool.lengths.tolist():
                heldout = max(1, round(int(length) * heldout_fraction))
                train_mask[offset + int(length) - heldout:offset + int(length)] = False
                offset += int(length)
            weights, bias = _fit_ridge(features[train_mask], targets[train_mask], ridge)
            prediction = features @ weights + bias
            error = prediction - targets
            train_rmse = float(error[train_mask].square().mean().sqrt())
            heldout_rmse = float(error[~train_mask].square().mean().sqrt())
            heldout_max_error = float(error[~train_mask].abs().max())
            payload = {
                "manifest_hash": manifest_hash,
                "weights": weights,
                "bias": bias,
                "train_rmse": train_rmse,
                "heldout_rmse": heldout_rmse,
                "heldout_max_error": heldout_max_error,
            }
            with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, cache_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    weights, bias = payload.get("weights"), payload.get("bias")
    if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != (6, A3_HEAD_DIM):
        raise ValueError(f"Invalid A3 head projector weights in {cache_path}")
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (A3_HEAD_DIM,):
        raise ValueError(f"Invalid A3 head projector bias in {cache_path}")
    projector = A3HeadProjector(
        weights=weights.float(), bias=bias.float(), manifest_hash=manifest_hash,
        train_rmse=float(payload["train_rmse"]), heldout_rmse=float(payload["heldout_rmse"]),
        heldout_max_error=float(payload["heldout_max_error"]), cache_path=cache_path,
    )
    if projector.heldout_rmse > max_heldout_rmse or projector.heldout_max_error > max_heldout_error:
        raise RuntimeError(
            "A3 SMPL-to-head projector rejected: "
            f"heldout_rmse={projector.heldout_rmse:.5f} (limit {max_heldout_rmse:.5f}), "
            f"heldout_max_error={projector.heldout_max_error:.5f} (limit {max_heldout_error:.5f})."
        )
    return projector


__all__ = ["A3HeadProjector", "A3_HEAD_LIMITS", "load_or_fit_a3_head_projector"]
