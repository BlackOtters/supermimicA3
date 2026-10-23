"""Cached paired-SMPL references used by the deployable DEPSIC V2 inputs.

The robot PKLs used by DEPSIC V1 contain the teacher labels.  V2 additionally
needs the matching SMPL clip, identified by its basename.  This module keeps
that pairing explicit, validates time alignment, and materializes the world
signals needed by the runtime sampler:

* left shoulder/elbow/wrist and right shoulder/elbow/wrist positions,
* left/right wrist orientations, and
* the current three-point/head fields retained only for A3's head projector.

The runtime deliberately interpolates those world signals before expressing
them in the root frame.  This matches DEPSIC HEAD at fractional source frames:
interpolating an already root-local wrist/arm trajectory is not equivalent
when the root rotates between the two frames.

The cache is deliberately independent of Hydra's run directory so DDP ranks
and later runs reuse the expensive PKL decoding safely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mimic_lite.tasks.depsic import _discover_clips


SMPL24_PARENTS = np.asarray(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21),
    dtype=np.int64,
)
SMPL_ARM_JOINTS = (16, 18, 20, 17, 19, 21)
SMPL_WRIST_JOINTS = (20, 21)
SMPL_HEAD_JOINT = 15
# v5 adds the current G1 full-SMPL V2 action-path fields: the six arm points
# and two wrist orientations.  Values are always interpolated in world space
# and root-localized only after interpolation.  Bump this whenever cached
# signals or their meaning changes so stale caches can never be reused.
CACHE_VERSION = 5


@dataclass(frozen=True)
class PairedSmplReferencePool:
    """Flattened world-space SMPL tracker signals aligned to robot clips."""

    paths: tuple[str, ...]
    lengths: torch.Tensor
    fps: torch.Tensor
    tracker_pos_w: torch.Tensor
    tracker_quat_w: torch.Tensor
    arm_pos_w: torch.Tensor
    wrist_quat_w: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    manifest_hash: str
    cache_path: Path

    @property
    def head_rot6_root(self) -> torch.Tensor:
        """Discrete-frame root-local head 6D used to fit A3's projector.

        Runtime tracker samples must use :func:`interpolate_smpl_world_trackers`
        instead.  This property is only for a projector fit at source frames,
        where it is exactly the same calculation.
        """
        return root_local_head_rot6(self.root_quat_w, self.tracker_quat_w[:, 2])


def _raw_payload(path: Path) -> Mapping[str, object]:
    try:
        decompressed = zlib.decompress(path.read_bytes())
    except (OSError, zlib.error) as error:
        raise ValueError(f"Could not read compressed SMPL PKL {path}") from error
    try:
        import joblib
    except ModuleNotFoundError:
        try:
            payload = pickle.loads(decompressed)
        except (pickle.UnpicklingError, EOFError, ValueError) as error:
            raise ModuleNotFoundError("DEPSIC V2 paired-SMPL loading requires joblib.") from error
    else:
        payload = joblib.load(io.BytesIO(decompressed))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a mapping payload")
    return payload


def _smpl_payload(path: Path) -> Mapping[str, object]:
    payload = _raw_payload(path)
    if "smpl_joints" in payload:
        return payload
    candidates = [value for value in payload.values() if isinstance(value, Mapping) and "smpl_joints" in value]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"{path}: contains multiple SMPL clips; select a concrete file")
    raise ValueError(f"{path}: missing smpl_joints")


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (lw * rw - lx * rx - ly * ry - lz * rz,
         lw * rx + lx * rw + ly * rz - lz * ry,
         lw * ry - lx * rz + ly * rw + lz * rx,
         lw * rz + lx * ry - ly * rx + lz * rw),
        axis=-1,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(1.0e-8)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def lerp_unit_quaternion(
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate wxyz quaternions with GR00T's hemisphere rule."""
    if lower.shape != upper.shape or lower.shape[-1] != 4:
        raise ValueError(
            "Quaternion interpolation needs matching [..., 4] inputs, got "
            f"{tuple(lower.shape)} and {tuple(upper.shape)}"
        )
    # q and -q represent the same rotation.  Without this sign alignment an
    # otherwise stationary tracker can interpolate through a zero quaternion.
    upper = torch.where((lower * upper).sum(dim=-1, keepdim=True) < 0.0, -upper, upper)
    result = torch.lerp(lower, upper, alpha)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _torch_quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _torch_quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _torch_quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    """Inverse of a (normally unit) wxyz quaternion, robust to bad input."""
    return _torch_quat_conjugate(quat) / quat.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _torch_quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply a wxyz quaternion with broadcast-compatible shapes."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    vector_part = quat[..., 1:]
    twice_cross = 2.0 * torch.cross(vector_part, vector, dim=-1)
    return vector + quat[..., :1] * twice_cross + torch.cross(vector_part, twice_cross, dim=-1)


def _torch_quat_to_rot6(quat: torch.Tensor) -> torch.Tensor:
    """DEPSIC cont6d: concatenate rotation matrix columns 0 and 1."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    w, x, y, z = quat.unbind(dim=-1)
    column_0 = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        dim=-1,
    )
    column_1 = torch.stack(
        (2.0 * (x * y - z * w), 1.0 - 2.0 * (x.square() + z.square()), 2.0 * (y * z + x * w)),
        dim=-1,
    )
    return torch.cat((column_0, column_1), dim=-1)


def root_local_head_rot6(root_quat_w: torch.Tensor, head_quat_w: torch.Tensor) -> torch.Tensor:
    """Convert world head orientation to DEPSIC root-local rotation-6D."""
    # Delegate to the public serializer also used by A3's runtime target.
    # This makes fit and inference share one column-major 6D definition.
    return quat_to_column_rot6(_torch_quat_mul(_torch_quat_inverse(root_quat_w), head_quat_w))


def quat_to_column_rot6(quat: torch.Tensor) -> torch.Tensor:
    """Return rotation matrix columns 0 and 1 in DEPSIC's 6-D layout.

    ``matrix[..., :2].reshape(...)`` is *not* equivalent: PyTorch flattens
    that slice row-major.  The explicit helper is shared by the A3 projector
    fit and runtime path so the learned affine map sees identical features.
    """
    return _torch_quat_to_rot6(quat)


def interpolate_smpl_world_trackers(
    *,
    tracker_pos_w: torch.Tensor,
    tracker_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    starts: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate tracker signals, then convert them to SMPL-root-local.

    ``starts/lower/upper`` have shape ``[N, H]`` and ``alpha`` is ``[N, H]``.
    This is intentionally the same ordering as GR00T's
    ``DepsicSmplVrReferenceCommand._refresh``.  The interpolated root world
    position is returned alongside its orientation so consumers that need a
    world-space tracker quantity (currently the V2 head-height calibration)
    do not re-index the source cache with subtly different fractional-frame
    logic.
    """
    point_alpha = alpha[..., None, None]
    quat_alpha = alpha[..., None]
    tracker_world = torch.lerp(
        tracker_pos_w[starts + lower], tracker_pos_w[starts + upper], point_alpha
    )
    tracker_quat_world = lerp_unit_quaternion(
        tracker_quat_w[starts + lower], tracker_quat_w[starts + upper], point_alpha
    )
    root_pos = torch.lerp(root_pos_w[starts + lower], root_pos_w[starts + upper], quat_alpha)
    root_quat = lerp_unit_quaternion(
        root_quat_w[starts + lower], root_quat_w[starts + upper], quat_alpha
    )
    root_inv = _torch_quat_inverse(root_quat)
    tracker_root = _torch_quat_apply(
        root_inv[..., None, :], tracker_world - root_pos[..., None, :]
    )
    tracker_quat_root = _torch_quat_mul(root_inv[..., None, :], tracker_quat_world)
    return tracker_root, tracker_quat_root, root_pos, root_quat


def interpolate_smpl_world_signals(
    *,
    arm_pos_w: torch.Tensor,
    wrist_quat_w: torch.Tensor,
    head_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    starts: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate full-SMPL arm/wrist signals, then convert to root-local.

    This is the deployable G1 V2 action-path ordering.  Interpolation must
    precede root localization: localizing the two endpoint frames first would
    produce a different trajectory whenever the SMPL root rotates.
    """
    arm_alpha = alpha[..., None, None]
    quat_alpha = alpha[..., None]
    arm_world = torch.lerp(arm_pos_w[starts + lower], arm_pos_w[starts + upper], arm_alpha)
    wrist_world = lerp_unit_quaternion(
        wrist_quat_w[starts + lower], wrist_quat_w[starts + upper], arm_alpha
    )
    head_world = lerp_unit_quaternion(
        head_quat_w[starts + lower], head_quat_w[starts + upper], quat_alpha
    )
    root_pos = torch.lerp(root_pos_w[starts + lower], root_pos_w[starts + upper], quat_alpha)
    root_quat = lerp_unit_quaternion(
        root_quat_w[starts + lower], root_quat_w[starts + upper], quat_alpha
    )
    root_inv = _torch_quat_inverse(root_quat)
    arm_root = _torch_quat_apply(root_inv[..., None, :], arm_world - root_pos[..., None, :])
    wrist_root = _torch_quat_mul(root_inv[..., None, :], wrist_world)
    return arm_root, wrist_root, root_local_head_rot6(root_quat, head_world)


def _load_tracker_world_signals(
    path: Path, *, expected_frames: int, expected_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load exactly the three world-space tracker signals used by V2."""
    clip = _smpl_payload(path)
    _, wrist, head, root_pos, root_quat = _load_world_signals(
        path, expected_frames=expected_frames, expected_fps=expected_fps
    )
    joints = np.asarray(clip["smpl_joints"], dtype=np.float32)
    tracker_pos = np.ascontiguousarray(
        joints[:, (SMPL_WRIST_JOINTS[0], SMPL_WRIST_JOINTS[1], SMPL_HEAD_JOINT)],
        dtype=np.float32,
    )
    tracker_quat = np.concatenate((wrist, head[:, None]), axis=1)
    return (
        np.ascontiguousarray(tracker_pos, dtype=np.float32),
        np.ascontiguousarray(tracker_quat, dtype=np.float32),
        root_pos,
        root_quat,
    )


def _global_quat_from_pose_aa(pose_aa: object, *, num_joints: int, source: Path) -> np.ndarray:
    local = np.asarray(pose_aa, dtype=np.float32)
    if local.ndim == 2 and local.shape[1] % 3 == 0:
        local = local.reshape(local.shape[0], -1, 3)
    if local.ndim != 3 or local.shape[0] < 2 or local.shape[-1] != 3:
        raise ValueError(f"{source}: expected pose_aa [T, 72] or [T, 24, 3], got {local.shape}")
    if local.shape[1] < 22 or num_joints > len(SMPL24_PARENTS):
        raise ValueError(f"{source}: paired-SMPL V2 needs standard SMPL-24 wrist/head joints")
    local = local[:, :num_joints]
    if not np.isfinite(local).all():
        raise ValueError(f"{source}: pose_aa contains non-finite values")
    angle = np.linalg.norm(local, axis=-1, keepdims=True)
    half = angle * 0.5
    quat = np.empty(local.shape[:-1] + (4,), dtype=np.float32)
    quat[..., 0] = np.cos(half[..., 0])
    quat[..., 1:] = local * np.divide(
        np.sin(half), angle, out=np.full_like(angle, 0.5), where=angle > 1.0e-8
    )
    global_quat = np.empty_like(quat)
    for joint, parent in enumerate(SMPL24_PARENTS[:num_joints]):
        global_quat[:, joint] = quat[:, joint] if parent < 0 else _quat_mul(global_quat[:, parent], quat[:, joint])
    return global_quat


def _load_world_signals(
    path: Path, *, expected_frames: int, expected_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clip = _smpl_payload(path)
    joints = np.asarray(clip.get("smpl_joints"), dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] != expected_frames or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(
            f"{path}: expected paired smpl_joints [{expected_frames}, >=22, 3], got {joints.shape}"
        )
    if not np.isfinite(joints).all() or np.allclose(joints, 0.0):
        raise ValueError(f"{path}: smpl_joints is non-finite or all zero")
    fps = float(clip.get("fps", expected_fps))
    if not math.isfinite(fps) or not math.isclose(fps, expected_fps, rel_tol=1.0e-5):
        raise ValueError(f"{path}: SMPL fps={fps} does not match robot fps={expected_fps}")
    if "smpl_joint_quat" in clip:
        quats = np.asarray(clip["smpl_joint_quat"], dtype=np.float32)
        if quats.shape != joints.shape[:-1] + (4,):
            raise ValueError(f"{path}: smpl_joint_quat has shape {quats.shape}, expected {joints.shape[:-1] + (4,)}")
    elif "pose_aa" in clip:
        quats = _global_quat_from_pose_aa(clip["pose_aa"], num_joints=joints.shape[1], source=path)
    else:
        raise ValueError(f"{path}: missing smpl_joint_quat and pose_aa")
    if not np.isfinite(quats).all() or np.any(np.linalg.norm(quats, axis=-1) < 1.0e-6):
        raise ValueError(f"{path}: SMPL quaternions are non-finite or zero-norm")
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    # Match GR00T's V2 loader exactly: paired corpora may provide an explicit
    # SMPL model-root orientation that is not identical to joint 0. ``root_rot``
    # is the legacy xyzw spelling, whereas all other paths are wxyz.
    if "smpl_root_quat" in clip:
        root_quat = np.asarray(clip["smpl_root_quat"], dtype=np.float32)
    elif "root_rot" in clip:
        root_rot_xyzw = np.asarray(clip["root_rot"], dtype=np.float32)
        if root_rot_xyzw.shape != (expected_frames, 4):
            raise ValueError(
                f"{path}: expected root_rot [{expected_frames}, 4], got {root_rot_xyzw.shape}"
            )
        root_quat = root_rot_xyzw[:, [3, 0, 1, 2]]
    else:
        root_quat = quats[:, 0]
    if root_quat.shape != (expected_frames, 4) or not np.isfinite(root_quat).all():
        raise ValueError(f"{path}: expected finite root quaternion [{expected_frames}, 4]")
    root_norm = np.linalg.norm(root_quat, axis=-1, keepdims=True)
    if np.any(root_norm < 1.0e-6):
        raise ValueError(f"{path}: root quaternion contains a zero-norm value")
    root_quat = root_quat / root_norm
    root_pos = joints[:, 0]
    return (
        np.ascontiguousarray(joints[:, SMPL_ARM_JOINTS], dtype=np.float32),
        np.ascontiguousarray(quats[:, SMPL_WRIST_JOINTS], dtype=np.float32),
        np.ascontiguousarray(quats[:, SMPL_HEAD_JOINT], dtype=np.float32),
        np.ascontiguousarray(root_pos, dtype=np.float32),
        np.ascontiguousarray(root_quat, dtype=np.float32),
    )


def _load_root_local_signals(
    path: Path, *, expected_frames: int, expected_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper for offline fixtures and projector tests.

    The runtime no longer uses this pre-localized representation; it is kept
    as a small discrete-frame adapter for callers that only need to inspect a
    single source frame.
    """
    arm, wrist, head, root_pos, root_quat = _load_world_signals(
        path, expected_frames=expected_frames, expected_fps=expected_fps
    )
    root_inv = _quat_conjugate(root_quat)
    root_inv_matrix = _quat_to_matrix(root_inv)
    arm_root = np.einsum(
        "tij,tkj->tki", root_inv_matrix, arm - root_pos[:, None, :], optimize=True
    )
    wrist_root = _quat_mul(root_inv[:, None, :], wrist)
    head_root = _quat_mul(root_inv, head)
    head_matrix = _quat_to_matrix(head_root)
    head_rot6 = np.concatenate((head_matrix[..., 0], head_matrix[..., 1]), axis=-1).reshape(
        expected_frames, 6
    )
    return arm_root.astype(np.float32), wrist_root.astype(np.float32), head_rot6.astype(np.float32)


def _fingerprint(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_root(value: str | Path | None) -> Path:
    if value is None:
        value = os.getenv("DEPSIC_REFERENCE_CACHE_ROOT")
    root = Path(value).expanduser() if value else Path(__file__).resolve().parents[4] / ".cache" / "depsic"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _cache_key(inputs: Mapping[str, object]) -> str:
    encoded = json.dumps({"version": CACHE_VERSION, **inputs}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _paired_paths(robot_paths: Sequence[str], smpl_motion_file: str | Path) -> tuple[Path, ...]:
    smpl_candidates = _discover_clips(smpl_motion_file)
    by_name: dict[str, Path] = {}
    for path in smpl_candidates:
        if path.name in by_name:
            raise ValueError(f"Paired SMPL basename is ambiguous: {path.name}")
        by_name[path.name] = path
    result = []
    for robot_path in robot_paths:
        name = Path(robot_path).name
        if name not in by_name:
            raise FileNotFoundError(f"No paired SMPL clip named {name} under {smpl_motion_file}")
        result.append(by_name[name].resolve())
    return tuple(result)


def load_paired_smpl_reference_pool(
    robot_paths: Sequence[str],
    lengths: torch.Tensor,
    fps: torch.Tensor,
    *,
    smpl_motion_file: str | Path,
    cache_root: str | Path | None = None,
) -> PairedSmplReferencePool:
    """Load or atomically materialize world-space V2 signals for robot clips."""
    if len(robot_paths) != int(lengths.numel()) or len(robot_paths) != int(fps.numel()):
        raise ValueError("robot_paths, lengths, and fps must have one entry per paired clip")
    resolved_robot = tuple(Path(path).expanduser().resolve() for path in robot_paths)
    smpl_paths = _paired_paths(tuple(str(path) for path in resolved_robot), smpl_motion_file)
    lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
    fps_cpu = fps.detach().to(device="cpu", dtype=torch.float32)
    inputs = {
        "robot": [_fingerprint(path) for path in resolved_robot],
        "smpl": [_fingerprint(path) for path in smpl_paths],
        "lengths": lengths_cpu.tolist(),
        "fps": [float(value) for value in fps_cpu.tolist()],
        "signals": {
            "arm_joints": SMPL_ARM_JOINTS,
            "wrist_joints": SMPL_WRIST_JOINTS,
            "head_joint": SMPL_HEAD_JOINT,
        },
    }
    manifest_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    key = _cache_key(inputs)
    root = _cache_root(cache_root)
    cache_path = root / f"paired-smpl-{key}.pt"
    lock_path = root / f"paired-smpl-{key}.lock"

    import fcntl
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        payload: dict[str, object] | None = None
        if cache_path.is_file():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, ValueError, pickle.UnpicklingError):
                payload = None
        if not isinstance(payload, dict) or payload.get("manifest_hash") != manifest_hash:
            tracker_pos_parts: list[torch.Tensor] = []
            tracker_quat_parts: list[torch.Tensor] = []
            arm_pos_parts: list[torch.Tensor] = []
            wrist_quat_parts: list[torch.Tensor] = []
            root_pos_parts: list[torch.Tensor] = []
            root_quat_parts: list[torch.Tensor] = []
            for path, clip_len, clip_fps in zip(smpl_paths, lengths_cpu.tolist(), fps_cpu.tolist(), strict=True):
                arm_pos, wrist_quat, _, _, _ = _load_world_signals(
                    path, expected_frames=int(clip_len), expected_fps=float(clip_fps)
                )
                tracker_pos, tracker_quat, root_pos, root_quat = _load_tracker_world_signals(
                    path, expected_frames=int(clip_len), expected_fps=float(clip_fps)
                )
                tracker_pos_parts.append(torch.from_numpy(tracker_pos))
                tracker_quat_parts.append(torch.from_numpy(tracker_quat))
                arm_pos_parts.append(torch.from_numpy(arm_pos))
                wrist_quat_parts.append(torch.from_numpy(wrist_quat))
                root_pos_parts.append(torch.from_numpy(root_pos))
                root_quat_parts.append(torch.from_numpy(root_quat))
            payload = {
                "manifest_hash": manifest_hash,
                "paths": tuple(str(path) for path in smpl_paths),
                "lengths": lengths_cpu,
                "fps": fps_cpu,
                "tracker_pos_w": torch.cat(tracker_pos_parts, dim=0).contiguous(),
                "tracker_quat_w": torch.cat(tracker_quat_parts, dim=0).contiguous(),
                "arm_pos_w": torch.cat(arm_pos_parts, dim=0).contiguous(),
                "wrist_quat_w": torch.cat(wrist_quat_parts, dim=0).contiguous(),
                "root_pos_w": torch.cat(root_pos_parts, dim=0).contiguous(),
                "root_quat_w": torch.cat(root_quat_parts, dim=0).contiguous(),
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

    required_shapes = {
        "tracker_pos_w": (int(lengths_cpu.sum().item()), 3, 3),
        "tracker_quat_w": (int(lengths_cpu.sum().item()), 3, 4),
        "arm_pos_w": (int(lengths_cpu.sum().item()), len(SMPL_ARM_JOINTS), 3),
        "wrist_quat_w": (int(lengths_cpu.sum().item()), len(SMPL_WRIST_JOINTS), 4),
        "root_pos_w": (int(lengths_cpu.sum().item()), 3),
        "root_quat_w": (int(lengths_cpu.sum().item()), 4),
    }
    for name, shape in required_shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid paired-SMPL cache {cache_path}: {name} must have finite shape {shape}")
    return PairedSmplReferencePool(
        paths=tuple(str(path) for path in smpl_paths),
        lengths=payload["lengths"].to(dtype=torch.long),
        fps=payload["fps"].to(dtype=torch.float32),
        tracker_pos_w=payload["tracker_pos_w"].to(dtype=torch.float32),
        tracker_quat_w=payload["tracker_quat_w"].to(dtype=torch.float32),
        arm_pos_w=payload["arm_pos_w"].to(dtype=torch.float32),
        wrist_quat_w=payload["wrist_quat_w"].to(dtype=torch.float32),
        root_pos_w=payload["root_pos_w"].to(dtype=torch.float32),
        root_quat_w=payload["root_quat_w"].to(dtype=torch.float32),
        manifest_hash=manifest_hash,
        cache_path=cache_path,
    )


__all__ = [
    "PairedSmplReferencePool",
    "SMPL_ARM_JOINTS",
    "SMPL_HEAD_JOINT",
    "SMPL_WRIST_JOINTS",
    "interpolate_smpl_world_signals",
    "interpolate_smpl_world_trackers",
    "lerp_unit_quaternion",
    "load_paired_smpl_reference_pool",
    "quat_to_column_rot6",
    "root_local_head_rot6",
]
