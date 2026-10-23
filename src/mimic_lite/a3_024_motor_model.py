"""Runtime helpers for the published A3 024 closed-chain model.

This module intentionally implements only the conventions and numbers that
ship in ``a3_024_sim2real_materials``.  In particular, the ankle and waist
tables use

``q_joint_dot = J(q) q_motor_dot`` and ``tau_motor = J(q).T tau_joint``.

The tables stay in their published discrete form and are bilinearly
interpolated on the device; no constant transmission-ratio approximation is
introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "a3_024"
RUNTIME_TABLE_DIR = ASSET_DIR / "runtime_tables"

REWARD_024_CURVES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "PFP41": ((0.0, 20.943951, 26.179939), (5.8, 5.8, 0.0)),
    "PFP59": ((0.0, 13.613568, 17.0), (34.0, 34.0, 0.0)),
    "PFP78": ((0.0, 16.755161, 23.561945), (58.0, 58.0, 0.0)),
    "PFP93": ((0.0, 10.471976, 18.5), (215.0, 215.0, 0.0)),
    "PFP110": ((0.0, 9.424778, 17.3), (300.0, 300.0, 0.0)),
}

PARALLEL_PAIRS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("left_ankle", ("left_ankle_pitch_joint", "left_ankle_roll_joint")),
    ("right_ankle", ("right_ankle_pitch_joint", "right_ankle_roll_joint")),
    ("waist", ("waist_pitch_joint", "waist_roll_joint")),
)

@dataclass(frozen=True)
class A3PenaltyTerms:
    """Unweighted reward_024 soft/hard penalty components."""

    soft: torch.Tensor
    hard: torch.Tensor
    combined: torch.Tensor


def motor_family_for_serial_joint(joint_name: str) -> str | None:
    """Return the published reward_024 motor family for a direct joint."""

    if "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
        return None
    if joint_name in {"waist_pitch_joint", "waist_roll_joint"}:
        return None
    if joint_name.endswith("_knee_joint"):
        return "PFP110"
    if (
        "_hip_pitch_joint" in joint_name
        or "_hip_roll_joint" in joint_name
        or "_hip_yaw_joint" in joint_name
        or joint_name == "waist_yaw_joint"
    ):
        return "PFP93"
    if "_shoulder_pitch_joint" in joint_name or "_shoulder_roll_joint" in joint_name:
        return "PFP78"
    if "_shoulder_yaw_joint" in joint_name or "_elbow_joint" in joint_name or "_wrist_roll_joint" in joint_name:
        return "PFP59"
    if "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
        return "PFP41"
    return None


def reward_024_curve(
    family: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact three-vertex 024 T--N envelope for ``family``."""

    try:
        speed, torque = REWARD_024_CURVES[family]
    except KeyError as error:
        raise KeyError(f"unknown A3 reward_024 motor family {family!r}") from error
    return (
        torch.tensor(speed, device=device, dtype=dtype),
        torch.tensor(torque, device=device, dtype=dtype),
    )


def interpolate_tn_limit(speed_rad_s: torch.Tensor, speed_grid: torch.Tensor, torque_grid: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate the symmetric torque envelope at ``abs(speed)``."""

    if speed_grid.shape != (3,) or torque_grid.shape != (3,):
        raise ValueError("A3 024 curves must contain exactly plateau/zero three vertices")
    speed = speed_rad_s.abs()
    upper = torch.searchsorted(speed_grid, speed.contiguous(), right=False).clamp_(1, 2)
    lower = upper - 1
    x0, x1 = speed_grid[lower], speed_grid[upper]
    y0, y1 = torque_grid[lower], torque_grid[upper]
    alpha = ((speed - x0) / (x1 - x0).clamp_min(torch.finfo(speed.dtype).eps)).clamp_(0.0, 1.0)
    result = y0 + alpha * (y1 - y0)
    result = torch.where(speed <= speed_grid[0], torque_grid[0], result)
    result = torch.where(speed >= speed_grid[-1], torque_grid[-1], result)
    return result.clamp_min(0.0)


def peak_mechanical_power(speed_grid: torch.Tensor, torque_grid: torch.Tensor) -> torch.Tensor:
    """Return the maximum of the exact two-segment torque-speed envelope."""

    slope = (torque_grid[2] - torque_grid[1]) / (speed_grid[2] - speed_grid[1]).clamp_min(
        torch.finfo(speed_grid.dtype).eps
    )
    intercept = torque_grid[1] - slope * speed_grid[1]
    decline_peak = (-intercept / (2.0 * slope).clamp_max(-torch.finfo(speed_grid.dtype).eps)).clamp(
        speed_grid[1], speed_grid[2]
    )
    candidates = torch.stack(
        (
            speed_grid[0] * torque_grid[0],
            speed_grid[1] * torque_grid[1],
            speed_grid[2] * torque_grid[2],
            decline_peak * (slope * decline_peak + intercept),
        )
    )
    return candidates.max()


@dataclass(frozen=True)
class ParallelJacobianTable:
    """One published A3 024 ``J(q)`` and motor-coordinate table."""

    name: str
    pitch_axis: torch.Tensor
    roll_axis: torch.Tensor
    jacobian: torch.Tensor
    motor_position: torch.Tensor
    reachable: torch.Tensor
    near_singular: torch.Tensor

    _CACHE: ClassVar[dict[tuple[str, str, torch.dtype, str], "ParallelJacobianTable"]] = {}

    @classmethod
    def load(
        cls,
        name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        table_dir: Path | str = RUNTIME_TABLE_DIR,
    ) -> "ParallelJacobianTable":
        table_dir = Path(table_dir).expanduser().resolve()
        key = (name, str(torch.device(device or "cpu")), dtype, str(table_dir))
        cached = cls._CACHE.get(key)
        if cached is not None:
            return cached
        path = table_dir / f"{name}_jacobi.npz"
        if not path.is_file():
            raise FileNotFoundError(f"A3 024 Jacobian table not found: {path}")
        with np.load(path) as raw:
            required = {"pitch_axis", "roll_axis", "J", "m", "reachable", "near_singular"}
            missing = required.difference(raw.files)
            if missing:
                raise ValueError(f"{path} misses required A3 024 fields: {sorted(missing)}")
            table = cls(
                name=name,
                pitch_axis=torch.as_tensor(np.ascontiguousarray(raw["pitch_axis"]), device=device, dtype=dtype),
                roll_axis=torch.as_tensor(np.ascontiguousarray(raw["roll_axis"]), device=device, dtype=dtype),
                jacobian=torch.as_tensor(np.ascontiguousarray(raw["J"]), device=device, dtype=dtype),
                motor_position=torch.as_tensor(np.ascontiguousarray(raw["m"]), device=device, dtype=dtype),
                reachable=torch.as_tensor(np.ascontiguousarray(raw["reachable"]), device=device, dtype=torch.bool),
                near_singular=torch.as_tensor(np.ascontiguousarray(raw["near_singular"]), device=device, dtype=torch.bool),
            )
        if table.jacobian.shape != (*table.motor_position.shape[:2], 2, 2):
            raise ValueError(f"invalid J/m table shape for {path}")
        if table.motor_position.shape[-1] != 2:
            raise ValueError(f"invalid motor-coordinate width for {path}")
        if not bool(table.reachable.all()) or bool(table.near_singular.any()):
            raise ValueError(f"A3 024 {name} table contains unsupported unreachable/singular cells")
        cls._CACHE[key] = table
        return table

    def _corners(self, position_pr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_pr.shape[-1] != 2:
            raise ValueError(f"expected position [..., 2], got {tuple(position_pr.shape)}")
        pitch = position_pr[..., 0].clamp(self.pitch_axis[0], self.pitch_axis[-1])
        roll = position_pr[..., 1].clamp(self.roll_axis[0], self.roll_axis[-1])
        p1 = torch.searchsorted(self.pitch_axis, pitch.contiguous(), right=False).clamp_(1, self.pitch_axis.numel() - 1)
        r1 = torch.searchsorted(self.roll_axis, roll.contiguous(), right=False).clamp_(1, self.roll_axis.numel() - 1)
        p0, r0 = p1 - 1, r1 - 1
        pw = ((pitch - self.pitch_axis[p0]) / (self.pitch_axis[p1] - self.pitch_axis[p0]).clamp_min(torch.finfo(pitch.dtype).eps))[..., None]
        rw = ((roll - self.roll_axis[r0]) / (self.roll_axis[r1] - self.roll_axis[r0]).clamp_min(torch.finfo(roll.dtype).eps))[..., None]
        return p0, p1, r0, r1, pw.clamp_(0.0, 1.0), rw.clamp_(0.0, 1.0)

    def _interpolate(self, values: torch.Tensor, position_pr: torch.Tensor) -> torch.Tensor:
        p0, p1, r0, r1, pw, rw = self._corners(position_pr)
        # ``values[p0, r0]`` has the batch dimensions of ``position_pr`` plus
        # its trailing payload dimensions (two for motor coordinates, four
        # for a 2x2 Jacobian).  Broadcast the scalar bilinear weights over
        # exactly those payload dimensions.
        payload_ndim = values.ndim - 2
        while pw.ndim < position_pr.ndim - 1 + payload_ndim:
            pw = pw.unsqueeze(-1)
            rw = rw.unsqueeze(-1)
        c00 = values[p0, r0]
        c01 = values[p0, r1]
        c10 = values[p1, r0]
        c11 = values[p1, r1]
        low = c00 * (1.0 - rw) + c01 * rw
        high = c10 * (1.0 - rw) + c11 * rw
        return low * (1.0 - pw) + high * pw

    def interpolate_jacobian(self, position_pr: torch.Tensor) -> torch.Tensor:
        return self._interpolate(self.jacobian, position_pr)

    def interpolate_motor_position(self, position_pr: torch.Tensor) -> torch.Tensor:
        return self._interpolate(self.motor_position, position_pr)


def motor_velocity_from_joint_velocity(jacobian: torch.Tensor, joint_velocity: torch.Tensor) -> torch.Tensor:
    """Solve the published ``J qdot_motor = qdot_joint`` relation."""

    if jacobian.shape[-2:] != (2, 2) or joint_velocity.shape[-1] != 2:
        raise ValueError("A3 024 parallel velocity map requires [..., 2, 2] and [..., 2]")
    a, b = jacobian[..., 0, 0], jacobian[..., 0, 1]
    c, d = jacobian[..., 1, 0], jacobian[..., 1, 1]
    det = a * d - b * c
    eps = torch.finfo(jacobian.dtype).eps
    if torch.any(det.abs() <= eps):
        raise RuntimeError("A3 024 Jacobian interpolation produced a singular matrix")
    return torch.stack(((d * joint_velocity[..., 0] - b * joint_velocity[..., 1]) / det,
                        (a * joint_velocity[..., 1] - c * joint_velocity[..., 0]) / det), dim=-1)


def motor_torque_from_joint_torque(jacobian: torch.Tensor, joint_torque: torch.Tensor) -> torch.Tensor:
    """Map virtual joint torque using the published ``tau_motor = J.T tau_joint``."""

    if jacobian.shape[-2:] != (2, 2) or joint_torque.shape[-1] != 2:
        raise ValueError("A3 024 parallel torque map requires [..., 2, 2] and [..., 2]")
    return torch.einsum("...ji,...j->...i", jacobian, joint_torque)


def motor_damping_from_joint_damping(
    jacobian: torch.Tensor, joint_damping: torch.Tensor,
) -> torch.Tensor:
    """Return the released loop controller's per-motor damping gains.

    The A3 loop solver's ``RlConvertKd`` does not emit a scalar transmission
    ratio.  At the current loop configuration it returns the diagonal of

    ``J.T @ diag(joint_damping) @ J``.

    This is the motor-diagonal damping used by the published MuJoCo sim2sim
    controller.  Keeping only that diagonal (rather than applying the full
    coupled matrix to motor velocity) is intentional: it reproduces the
    solver's two independent motor damping commands while the physical loop
    itself remains in the MJCF.
    """

    if jacobian.shape[-2:] != (2, 2) or joint_damping.shape[-1] != 2:
        raise ValueError("A3 024 parallel damping requires [..., 2, 2] and [..., 2]")
    return torch.einsum("...ji,...j,...ji->...i", jacobian, joint_damping, jacobian)


def motor_penalty(
    utilization: torch.Tensor,
    *,
    lambda_soft: float,
    lambda_violation: float,
    soft_exponent: float,
    violation_exponent: float,
    utilization_cap: float,
) -> A3PenaltyTerms:
    """Exact 024 dense T--N shaping from the released reward snapshot."""

    if lambda_soft < 0.0 or lambda_violation < 0.0:
        raise ValueError("A3 024 penalty weights must be non-negative")
    if soft_exponent <= 0.0 or violation_exponent <= 0.0 or utilization_cap < 1.0:
        raise ValueError("invalid A3 024 penalty shaping parameters")
    capped = utilization.clamp(min=0.0, max=float(utilization_cap))
    soft = torch.minimum(capped, torch.ones_like(capped)).pow(float(soft_exponent))
    hard = (capped - 1.0).clamp_min(0.0).pow(float(violation_exponent))
    return A3PenaltyTerms(
        soft=soft,
        hard=hard,
        combined=float(lambda_soft) * soft + float(lambda_violation) * hard,
    )


def threshold_penalty(
    utilization: torch.Tensor,
    *,
    threshold_ratio: float,
    lambda_soft: float,
    lambda_violation: float,
    soft_exponent: float,
    violation_exponent: float,
    utilization_cap: float,
) -> A3PenaltyTerms:
    """024 threshold penalty used by ankle speed/power/fixed-torque terms."""

    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in [0, 1]")
    if lambda_soft < 0.0 or lambda_violation < 0.0:
        raise ValueError("A3 024 penalty weights must be non-negative")
    if soft_exponent <= 0.0 or violation_exponent <= 0.0 or utilization_cap < 1.0:
        raise ValueError("invalid A3 024 penalty shaping parameters")
    capped = utilization.clamp(min=0.0, max=float(utilization_cap))
    soft = (capped - float(threshold_ratio)).clamp_min(0.0).pow(float(soft_exponent))
    hard = (capped - 1.0).clamp_min(0.0).pow(float(violation_exponent))
    return A3PenaltyTerms(
        soft=soft,
        hard=hard,
        combined=float(lambda_soft) * soft + float(lambda_violation) * hard,
    )


def cross_penalty(normalized_pair_load: torch.Tensor, *, threshold_ratio: float = 0.75) -> torch.Tensor:
    """Released 024 pitch/roll simultaneous-load shaping."""

    if normalized_pair_load.shape[-1] != 2:
        raise ValueError("A3 024 cross penalty expects [..., 2]")
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in [0, 1]")
    excess = (normalized_pair_load - float(threshold_ratio)).clamp_min(0.0)
    return excess[..., 0] * excess[..., 1]


def position_threshold_penalty(
    joint_position: torch.Tensor,
    hard_joint_limits: torch.Tensor,
    *,
    threshold_ratio: float = 0.85,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Released 024 hard-range ankle position penalty, per environment."""

    if hard_joint_limits.shape[-1] != 2:
        raise ValueError("hard_joint_limits must have trailing [lower, upper]")
    if not 0.0 <= threshold_ratio <= 1.0 or exponent <= 0.0:
        raise ValueError("invalid A3 024 position-penalty parameters")
    lower, upper = hard_joint_limits[..., 0], hard_joint_limits[..., 1]
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    valid = half_range > torch.finfo(joint_position.dtype).eps
    ratio = torch.where(
        valid,
        (joint_position - center).abs() / half_range.clamp_min(torch.finfo(joint_position.dtype).eps),
        torch.zeros_like(joint_position),
    )
    return torch.where(valid, (ratio - float(threshold_ratio)).clamp_min(0.0).pow(float(exponent)), 0.0)


__all__ = [
    "ASSET_DIR",
    "A3PenaltyTerms",
    "PARALLEL_PAIRS",
    "ParallelJacobianTable",
    "REWARD_024_CURVES",
    "RUNTIME_TABLE_DIR",
    "cross_penalty",
    "interpolate_tn_limit",
    "motor_penalty",
    "motor_family_for_serial_joint",
    "motor_torque_from_joint_torque",
    "motor_damping_from_joint_damping",
    "motor_velocity_from_joint_velocity",
    "peak_mechanical_power",
    "position_threshold_penalty",
    "reward_024_curve",
    "threshold_penalty",
]
