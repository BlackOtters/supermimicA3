"""mjlab implementation of the DEPSIC A3 T3D0 V1 robot-only task.

This module deliberately does not reuse the G1 command directly.  A3 V1 has
31 physical DoFs but a strict 29-DoF policy contract: the PKL-provided head
yaw/pitch targets are externally replayed and must never enter policy action,
actor history, critic proprioception, or policy-only penalties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

from mimic_lite.depsic_a3_contract import (
    A3_ACTOR_OBS_DIM,
    A3_ARM_DIM,
    A3_ARM_JOINT_NAMES,
    A3_CRITIC_DIM,
    A3_HEAD_DIM,
    A3_HEAD_JOINT_NAMES,
    A3_LBC_DIM,
    A3_POLICY_DIM,
    A3_POLICY_JOINT_NAMES,
    A3_REFERENCE_FRAMES,
    A3_TOKENIZER_DIM,
    A3_T3D0_MJCF_JOINT_NAMES,
    A3_31DOF_JOINT_NAMES,
)
from mimic_lite.depsic_a3_amp_contract import A3_AMP_SQUAT_HEIGHT_THRESHOLD
from mimic_lite.tasks.depsic import (
    LBC_METRIC_NAMES,
    LBC_WANDB_KEYS,
    SOFT_JOINT_POS_LIMIT_FACTOR,
    _G1ArmKinematics,
    _lbc_rpy,
    _load_compressed_pkl,
    _discover_clips,
)


A3_AXIS_ALIGNED_ROUTE_NAMES = (
    "forward",
    "backward",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "stationary_height",
    "squat",
    "idle",
)
_A3_AMP_ROUTE_SPEED_THRESHOLD = 0.10
_A3_AMP_YAW_RATE_SCALE = 0.55 / 1.57


@dataclass(frozen=True)
class _A3ReferencePool:
    paths: tuple[str, ...]
    lengths: torch.Tensor
    fps: torch.Tensor
    arm_q: torch.Tensor
    head_q: torch.Tensor


def _load_a3_reference_pool(
    path: str | Path,
    *,
    motion_pool_size: int | None,
) -> _A3ReferencePool:
    """Load only the A3 V1 PKL contract and canonicalize it by joint name."""
    paths = _discover_clips(path)
    if motion_pool_size is not None:
        if motion_pool_size < 1:
            raise ValueError("motion_pool_size must be positive or null")
        if len(paths) > motion_pool_size:
            stride = max(1, len(paths) // motion_pool_size)
            paths = paths[::stride][:motion_pool_size]

    arm_parts: list[torch.Tensor] = []
    head_parts: list[torch.Tensor] = []
    lengths: list[int] = []
    fps_values: list[float] = []
    valid_paths: list[str] = []
    invalid: list[tuple[Path, str]] = []
    for source in paths:
        try:
            clip = _load_compressed_pkl(source)
            dof = np.asarray(clip["dof"], dtype=np.float32)
            if "joint_names" in clip:
                joint_names = tuple(str(name) for name in clip["joint_names"])
            else:
                # The redirector's no-name output is the one accepted legacy
                # format.  Treat every other nameless layout as a hard error.
                joint_names = A3_T3D0_MJCF_JOINT_NAMES
            if dof.ndim != 2 or dof.shape != (dof.shape[0], len(joint_names)):
                raise ValueError(
                    f"expected dof [T, {len(joint_names)}] for declared A3 joint order, got {dof.shape}"
                )
            if dof.shape[1] != len(A3_31DOF_JOINT_NAMES):
                raise ValueError(f"expected exactly 31 A3 DoFs, got {dof.shape[1]}")
            if len(set(joint_names)) != len(joint_names):
                raise ValueError("joint_names contains duplicates")
            missing = [name for name in A3_31DOF_JOINT_NAMES if name not in joint_names]
            if missing:
                raise ValueError(f"missing required A3 joints: {missing}")
            if dof.shape[0] < 2 or not np.isfinite(dof).all():
                raise ValueError("clip needs at least two finite frames")
            # A3 intentionally rejects silent 50 Hz fallback: a wrong frame
            # rate corrupts both arm velocity labels and head replay timing.
            if "fps" not in clip:
                raise ValueError("A3 PKL is missing required fps")
            fps = float(clip["fps"])
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"invalid fps={fps}")
            indices = {name: index for index, name in enumerate(joint_names)}
            arm_q = np.ascontiguousarray(dof[:, [indices[name] for name in A3_ARM_JOINT_NAMES]])
            head_q = np.ascontiguousarray(dof[:, [indices[name] for name in A3_HEAD_JOINT_NAMES]])
        except (KeyError, TypeError, ValueError, ModuleNotFoundError) as error:
            invalid.append((source, str(error)))
            continue
        arm_parts.append(torch.from_numpy(arm_q))
        head_parts.append(torch.from_numpy(head_q))
        lengths.append(int(dof.shape[0]))
        fps_values.append(fps)
        valid_paths.append(str(source.resolve()))

    if not arm_parts:
        detail = invalid[0][1] if invalid else "no clips"
        raise ValueError(f"No valid DEPSIC A3 AMASS clips were found: {detail}")
    if invalid:
        first_path, first_error = invalid[0]
        print(f"[depsic-a3] skipped {len(invalid)} invalid PKLs; first: {first_path} ({first_error})")
    arm_q = torch.cat(arm_parts, dim=0).contiguous()
    head_q = torch.cat(head_parts, dim=0).contiguous()
    print(f"[depsic-a3] loaded {len(valid_paths)} A3 clips / {arm_q.shape[0]} frames")
    return _A3ReferencePool(
        paths=tuple(valid_paths),
        lengths=torch.tensor(lengths, dtype=torch.long),
        fps=torch.tensor(fps_values, dtype=torch.float32),
        arm_q=arm_q,
        head_q=head_q,
    )


class DepsicA3V1RobotOnly(Command, namespace="mimic_lite"):
    """A3 V1 lower-body command plus 14-DoF arm / external-head replay."""

    def __init__(
        self,
        env,
        motion_file: str,
        arm_urdf_path: str,
        runtime_head_joint_names: Sequence[str] | None = None,
        motion_pool_size: int | None = None,
        default_fps: float = 50.0,
        future_frames: int = A3_REFERENCE_FRAMES,
        future_frame_dt: float = 0.1,
        resampling_time_s: float = 10.0,
        standing_probability: float = 0.4,
        lbc_sampling_mode: str = "independent",
        axis_aligned_route_weights: Mapping[str, float] | None = None,
        nominal_motion_height: float = 1.075,
        stationary_height_range: Sequence[float] = (0.63, 1.08),
        moving_rpy_scale: float = 0.5,
        stationary_rpy_scale: float = 1.0,
        min_linear_speed: float = 0.12,
        min_turn_rate: float = 0.30,
        high_height_probability: float | None = None,
        reset_to_reference: bool = True,
        lbc_ranges: Mapping[str, Sequence[float]] | None = None,
        curriculum: Mapping[str, Mapping[str, Sequence[float] | int | float]] | None = None,
        curriculum_steps_divisor: float = 1.0,
        amp_forward_command_override_until_iter: int | None = None,
        amp_forward_command_override_height: float | None = None,
        nominal_reset: bool = False,
    ) -> None:
        super().__init__(env)
        del default_fps  # A3 PKLs carry a mandatory fps; see loader above.
        if future_frames != A3_REFERENCE_FRAMES:
            raise ValueError(f"A3 V1 requires future_frames={A3_REFERENCE_FRAMES}")
        if future_frame_dt <= 0.0 or resampling_time_s <= 0.0:
            raise ValueError("future_frame_dt and resampling_time_s must be positive")
        if curriculum_steps_divisor <= 0.0:
            raise ValueError("curriculum_steps_divisor must be positive")
        self.future_frames = int(future_frames)
        self.future_frame_dt = float(future_frame_dt)
        self.resample_steps = max(1, round(float(resampling_time_s) / env.step_dt))
        self.standing_probability = float(standing_probability)
        self.lbc_sampling_mode = str(lbc_sampling_mode)
        if self.lbc_sampling_mode not in {"independent", "axis_aligned"}:
            raise ValueError(
                "lbc_sampling_mode must be either 'independent' or 'axis_aligned'"
            )
        if self.lbc_sampling_mode == "axis_aligned" and self.standing_probability != 0.0:
            raise ValueError(
                "axis_aligned sampling encodes idle through stationary_height; "
                "standing_probability must be 0"
            )
        configured_weights = dict(axis_aligned_route_weights or {})
        unknown_routes = set(configured_weights).difference(A3_AXIS_ALIGNED_ROUTE_NAMES)
        if unknown_routes:
            raise ValueError(
                f"Unknown axis-aligned LBC routes: {sorted(unknown_routes)}"
            )
        default_weights = dict.fromkeys(A3_AXIS_ALIGNED_ROUTE_NAMES, 1.0)
        default_weights.update(stationary_height=0.8, squat=0.0, idle=0.0)
        default_weights.update(configured_weights)
        route_weights = tuple(float(default_weights[name]) for name in A3_AXIS_ALIGNED_ROUTE_NAMES)
        if any(not math.isfinite(value) or value < 0.0 for value in route_weights):
            raise ValueError("axis_aligned_route_weights must be finite and non-negative")
        if not any(value > 0.0 for value in route_weights):
            raise ValueError("axis_aligned_route_weights must enable at least one route")
        self.axis_aligned_route_weights = torch.tensor(
            route_weights, dtype=torch.float32, device=self.device
        )
        self.nominal_motion_height = float(nominal_motion_height)
        if not math.isfinite(self.nominal_motion_height):
            raise ValueError("nominal_motion_height must be finite")
        if len(stationary_height_range) != 2:
            raise ValueError("stationary_height_range must contain [low, high]")
        self.stationary_height_range = tuple(float(value) for value in stationary_height_range)
        if (
            not all(math.isfinite(value) for value in self.stationary_height_range)
            or self.stationary_height_range[0] >= self.stationary_height_range[1]
        ):
            raise ValueError("stationary_height_range must be finite with low < high")
        self.moving_rpy_scale = float(moving_rpy_scale)
        self.stationary_rpy_scale = float(stationary_rpy_scale)
        if (
            not math.isfinite(self.moving_rpy_scale)
            or not math.isfinite(self.stationary_rpy_scale)
            or self.moving_rpy_scale < 0.0
            or self.stationary_rpy_scale < 0.0
        ):
            raise ValueError("RPY range scales must be finite and non-negative")
        self.min_linear_speed = float(min_linear_speed)
        self.min_turn_rate = float(min_turn_rate)
        if (
            not math.isfinite(self.min_linear_speed)
            or self.min_linear_speed < _A3_AMP_ROUTE_SPEED_THRESHOLD
        ):
            raise ValueError(
                "min_linear_speed must be finite and remain outside the AMP stationary band"
            )
        if (
            not math.isfinite(self.min_turn_rate)
            or self.min_turn_rate * _A3_AMP_YAW_RATE_SCALE
            < _A3_AMP_ROUTE_SPEED_THRESHOLD
        ):
            raise ValueError(
                "min_turn_rate must be finite and remain outside the AMP stationary band "
                "after yaw scaling"
            )
        self.high_height_probability = (
            None if high_height_probability is None else float(high_height_probability)
        )
        if self.high_height_probability is not None and not 0.0 <= self.high_height_probability <= 1.0:
            raise ValueError("high_height_probability must be in [0, 1]")
        self.reset_to_reference = bool(reset_to_reference)
        # Table-II evaluation uses the same nominal root/joint reset as the
        # IsaacLab evaluator. Keep the training default randomized.
        self.nominal_reset = bool(nominal_reset)
        self.curriculum_steps_divisor = float(curriculum_steps_divisor)
        self.amp_forward_command_override_until_iter = (
            None
            if amp_forward_command_override_until_iter is None
            else int(amp_forward_command_override_until_iter)
        )
        if (
            self.amp_forward_command_override_until_iter is not None
            and self.amp_forward_command_override_until_iter < 0
        ):
            raise ValueError("AMP forward command override iteration must be non-negative")
        self.amp_forward_command_override_height = (
            None
            if amp_forward_command_override_height is None
            else float(amp_forward_command_override_height)
        )
        if (
            self.amp_forward_command_override_height is not None
            and not math.isfinite(self.amp_forward_command_override_height)
        ):
            raise ValueError("AMP forward command override height must be finite")

        self.runtime_head_joint_names = tuple(
            A3_HEAD_JOINT_NAMES if runtime_head_joint_names is None else runtime_head_joint_names
        )
        if len(self.runtime_head_joint_names) != A3_HEAD_DIM:
            raise ValueError(
                f"DEPSIC A3 requires exactly {A3_HEAD_DIM} runtime head joints, "
                f"got {self.runtime_head_joint_names}"
            )
        self.policy_joint_ids = self._named_joint_ids(A3_POLICY_JOINT_NAMES, "policy")
        self.arm_joint_ids = self._named_joint_ids(A3_ARM_JOINT_NAMES, "arm")
        self.head_joint_ids = self._named_joint_ids(self.runtime_head_joint_names, "head")
        self.torso_id = self._one_body_id("torso_Link")
        self.pelvis_id = self._one_body_id("pelvis_link")

        self.pool = _load_a3_reference_pool(motion_file, motion_pool_size=motion_pool_size)
        self.lengths = self.pool.lengths.to(device=self.device)
        self.fps = self.pool.fps.to(device=self.device)
        self.clip_starts = torch.cat(
            (torch.zeros(1, device=self.device, dtype=torch.long), self.lengths.cumsum(0)[:-1])
        )
        self.arm_q_data = self.pool.arm_q.to(device=self.device)
        self.head_q_data = self.pool.head_q.to(device=self.device)
        self.kinematics = _G1ArmKinematics(
            arm_urdf_path,
            torso_link="torso_Link",
            wrist_links=("left_wrist_yaw_Link", "right_wrist_yaw_Link"),
            arm_joint_names=A3_ARM_JOINT_NAMES,
            robot_label="A3",
        )

        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_ids = torch.zeros_like(self.clip_ids)
        self.frame_fraction = torch.zeros(self.num_envs, device=self.device)
        self.lbc = torch.zeros(self.num_envs, A3_LBC_DIM, device=self.device)
        self.arm_q = torch.zeros(self.num_envs, A3_ARM_DIM, device=self.device)
        self.arm_dq = torch.zeros_like(self.arm_q)
        self.head_q = torch.zeros(self.num_envs, A3_HEAD_DIM, device=self.device)
        self.wrist_pos_torso = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.wrist_rot_torso = torch.eye(3, device=self.device).expand(self.num_envs, 2, 3, 3).clone()
        self.lbc_elapsed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.velocity_ema_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.velocity_ema_wz = torch.zeros(self.num_envs, device=self.device)
        self.velocity_ema_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lbc_metric_steps = torch.zeros(self.num_envs, device=self.device)
        self.lbc_metric_means = torch.zeros(self.num_envs, len(LBC_METRIC_NAMES), device=self.device)
        self._future_arm_q: torch.Tensor | None = None
        self._future_arm_dq: torch.Tensor | None = None

        ranges = lbc_ranges or {}
        defaults = (
            (-0.55, 0.55), (-0.55, 0.55), (-1.57, 1.57),
            (-0.35, 0.35), (-0.489, 1.27), (-1.27, 1.27), (0.63, 1.08),
        )
        self.lbc_range_keys = (
            "lin_vel_x", "lin_vel_y", "ang_vel_z", "torso_roll", "torso_pitch", "torso_yaw", "base_height"
        )
        self.lbc_ranges = torch.tensor(
            [ranges.get(key, default) for key, default in zip(self.lbc_range_keys, defaults, strict=True)],
            dtype=torch.float32,
            device=self.device,
        )
        self.curriculum = dict(curriculum or {})

    @staticmethod
    def _uniform_rows(
        count: int,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return low + torch.rand(count, low.numel(), device=device) * (high - low)

    def _sample_axis_aligned_lbc(
        self,
        env_ids: torch.Tensor,
        ranges: torch.Tensor,
    ) -> None:
        """Sample one deployable cardinal route per environment.

        Translation and yaw rate are mutually exclusive. Moving/turning
        routes keep the torso-height target fixed, while the stationary route
        splits squat below 1.0 m and AMP-silent idle at or above 1.0 m.
        The legacy stationary_height route still samples the combined range.
        """
        count = len(env_ids)
        routes = torch.multinomial(
            self.axis_aligned_route_weights,
            count,
            replacement=True,
        )
        sampled = torch.zeros(count, A3_LBC_DIM, device=self.device)
        low, high = ranges[:, 0], ranges[:, 1]

        rpy_low, rpy_high = low[3:6], high[3:6]
        if bool(((rpy_low > 0.0) | (rpy_high < 0.0)).any()):
            raise ValueError("axis-aligned RPY ranges must contain zero")
        stationary = routes >= A3_AXIS_ALIGNED_ROUTE_NAMES.index("stationary_height")
        moving = ~stationary
        moving_count = int(moving.sum().item())
        stationary_count = count - moving_count
        if moving_count:
            sampled[moving, 3:6] = self._uniform_rows(
                moving_count,
                rpy_low * self.moving_rpy_scale,
                rpy_high * self.moving_rpy_scale,
                device=self.device,
            )
            sampled[moving, 6] = self.nominal_motion_height
        if stationary_count:
            sampled[stationary, 3:6] = self._uniform_rows(
                stationary_count,
                rpy_low * self.stationary_rpy_scale,
                rpy_high * self.stationary_rpy_scale,
                device=self.device,
            )
            height_low = max(float(low[6].item()), self.stationary_height_range[0])
            height_high = min(float(high[6].item()), self.stationary_height_range[1])
            if height_low >= height_high:
                raise ValueError(
                    "stationary height curriculum does not overlap "
                    f"stationary_height_range: [{height_low}, {height_high}]"
                )
            for route_name in ("stationary_height", "squat", "idle"):
                mask = routes.eq(A3_AXIS_ALIGNED_ROUTE_NAMES.index(route_name))
                route_count = int(mask.sum().item())
                if not route_count:
                    continue
                route_low, route_high = height_low, height_high
                if route_name == "idle":
                    route_low = max(route_low, A3_AMP_SQUAT_HEIGHT_THRESHOLD)
                elif route_name == "squat":
                    route_high = min(route_high, A3_AMP_SQUAT_HEIGHT_THRESHOLD)
                if route_low >= route_high:
                    raise ValueError(f"{route_name} has no height range in the current curriculum")
                values = route_low + torch.rand(
                    route_count, device=self.device
                ) * (route_high - route_low)
                if route_name == "squat":
                    # Rounding must never turn a squat sample into idle.
                    upper = values.new_tensor(route_high)
                    values.clamp_max_(torch.nextafter(upper, values.new_tensor(-torch.inf)))
                sampled[mask, 6] = values

        route_specs = (
            ("forward", 0, 1.0, self.min_linear_speed, float(high[0].item())),
            ("backward", 0, -1.0, self.min_linear_speed, float(-low[0].item())),
            ("left", 1, 1.0, self.min_linear_speed, float(high[1].item())),
            ("right", 1, -1.0, self.min_linear_speed, float(-low[1].item())),
            ("turn_left", 2, 1.0, self.min_turn_rate, float(high[2].item())),
            ("turn_right", 2, -1.0, self.min_turn_rate, float(-low[2].item())),
        )
        for route_name, dimension, sign, minimum, maximum in route_specs:
            mask = routes.eq(A3_AXIS_ALIGNED_ROUTE_NAMES.index(route_name))
            route_count = int(mask.sum().item())
            if not route_count:
                continue
            if maximum < minimum:
                raise ValueError(
                    f"{route_name} range cannot provide magnitude "
                    f"[{minimum}, {maximum}]"
                )
            magnitude = minimum + torch.rand(route_count, device=self.device) * (
                maximum - minimum
            )
            sampled[mask, dimension] = sign * magnitude

        self.lbc[env_ids] = sampled

    def _named_joint_ids(self, expected: Sequence[str], label: str) -> torch.Tensor:
        ids, names = find_joints(self.asset, list(expected))
        if tuple(names) != tuple(expected):
            raise ValueError(f"DEPSIC A3 {label} order mismatch: expected {list(expected)}, got {names}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _one_body_id(self, name: str) -> int:
        ids, names = find_bodies(self.asset, [name])
        if names != [name]:
            raise ValueError(f"DEPSIC A3 requires body {name}")
        return int(ids[0])

    @property
    def command(self) -> torch.Tensor:
        return self.lbc

    @property
    def lbc_future(self) -> torch.Tensor:
        return self.lbc[:, None].expand(-1, self.future_frames, -1)

    @property
    def arm_future(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._future_arm_q is None or self._future_arm_dq is None:
            offsets = torch.arange(self.future_frames, device=self.device, dtype=self.frame_fraction.dtype)
            source = (
                self.frame_ids[:, None].to(self.frame_fraction.dtype)
                + self.frame_fraction[:, None]
                + offsets[None] * self.future_frame_dt * self.fps[self.clip_ids, None]
            )
            self._future_arm_q, self._future_arm_dq = self._interpolate_arm(self.clip_ids, source)
        return self._future_arm_q, self._future_arm_dq

    def _effective_ranges(self) -> torch.Tensor:
        result = self.lbc_ranges.clone()
        iteration = float(self.env.current_iter)
        for index, key in enumerate(self.lbc_range_keys):
            spec = self.curriculum.get(key)
            if not spec:
                continue
            start = float(spec["start_iter"]) / self.curriculum_steps_divisor
            end = float(spec["end_iter"]) / self.curriculum_steps_divisor
            start_range = torch.as_tensor(spec["start_range"], device=self.device, dtype=result.dtype)
            end_range = torch.as_tensor(spec["end_range"], device=self.device, dtype=result.dtype)
            if iteration <= start:
                result[index] = start_range
            elif iteration >= end:
                result[index] = end_range
            else:
                result[index] = torch.lerp(start_range, end_range, (iteration - start) / max(end - start, 1.0))
        return result

    def _sample_lbc(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        ranges = self._effective_ranges()
        if getattr(self, "lbc_sampling_mode", "independent") == "axis_aligned":
            self._sample_axis_aligned_lbc(env_ids, ranges)
            self.lbc_elapsed[env_ids] = 0
            return
        low, high = ranges[:, 0], ranges[:, 1]
        # Commands remain uniformly sampled from their possibly
        # curriculum-adjusted ranges unless an explicitly configured
        # experiment enables the optional two-band height mixture below.
        self.lbc[env_ids] = low + torch.rand(len(env_ids), A3_LBC_DIM, device=self.device) * (high - low)
        if self.high_height_probability is not None:
            # Optional legacy experiment: its height mixture is independent
            # of the normal curriculum.  The default/AMP V1 path above keeps
            # the original uniform V1 height sampler.
            height_low = low[6]
            height_high = high[6]
            if height_low >= 0.95 or height_high < 0.95:
                raise ValueError(
                    "A3 height mixture requires the effective base_height range to straddle 0.95 m, "
                    f"got [{height_low.item():.3f}, {height_high.item():.3f}]"
                )
            high_height = torch.rand(len(env_ids), device=self.device) < self.high_height_probability
            sampled_height = torch.empty(len(env_ids), device=self.device)
            high_count = int(high_height.sum().item())
            low_count = len(env_ids) - high_count
            if high_count:
                sampled_height[high_height] = 0.95 + torch.rand(high_count, device=self.device) * (height_high - 0.95)
            if low_count:
                # ``low`` is curriculum-controlled (0.75 -> 0.60), while
                # the upper endpoint is strict: low-height commands are < .95.
                sampled_height[~high_height] = height_low + torch.rand(low_count, device=self.device) * (0.95 - height_low)
            self.lbc[env_ids, 6] = sampled_height
        standing = torch.rand(len(env_ids), device=self.device) < self.standing_probability
        self.lbc[env_ids[standing], :3] = 0.0
        # Forward AMP bootstrap: neutralize torso RPY on forward-routed
        # commands. Height stays on the sampled/curriculum value so walking
        # crouch is still a training target (024 probe: vx>0 + h=0.75 was
        # executed at 1.07 whenever this wrote height to 1.075).
        forward_override_until = getattr(
            self, "amp_forward_command_override_until_iter", None
        )
        if (
            forward_override_until is not None
            and int(self.env.current_iter) < forward_override_until
        ):
            selected = self.lbc[env_ids]
            vx, vy, wz = selected[:, 0], selected[:, 1], selected[:, 2]
            scaled_wz = wz / 1.57 * 0.55
            speed = torch.sqrt(vx.square() + vy.square() + scaled_wz.square())
            forward = (
                (speed >= 0.10)
                & (vx >= 0.0)
                & (vx.abs() >= vy.abs())
                & (vx.abs() >= scaled_wz.abs())
            )
            forward_env_ids = env_ids[forward]
            self.lbc[forward_env_ids, 3:6] = 0.0
            override_height = getattr(self, "amp_forward_command_override_height", None)
            if override_height is not None:
                self.lbc[forward_env_ids, 6] = override_height
        # ``wz`` is always a sampled, held yaw-rate command.  It is never
        # replaced with a heading-feedback value in A3 V1.
        self.lbc_elapsed[env_ids] = 0

    def _interpolate_arm(
        self, clip_ids: torch.Tensor, source_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last = self.lengths[clip_ids] - 1
        if source_frames.ndim == 2:
            last = last[:, None]
            starts = self.clip_starts[clip_ids][:, None]
            fps = self.fps[clip_ids][:, None, None]
        else:
            starts = self.clip_starts[clip_ids]
            fps = self.fps[clip_ids][:, None]
        source_frames = torch.minimum(source_frames.clamp_min(0.0), last.to(source_frames.dtype))
        lower = torch.floor(source_frames).long()
        upper = torch.minimum(lower + 1, last)
        alpha = (source_frames - lower).unsqueeze(-1)
        lower_q = self.arm_q_data[starts + lower]
        upper_q = self.arm_q_data[starts + upper]
        q = torch.lerp(lower_q, upper_q, alpha)
        dq = (upper_q - lower_q) * fps
        dq = torch.where((upper == lower).unsqueeze(-1), torch.zeros_like(dq), dq)
        return q, dq

    def _interpolate_head(self, clip_ids: torch.Tensor, source_frames: torch.Tensor) -> torch.Tensor:
        last = self.lengths[clip_ids] - 1
        source_frames = torch.minimum(source_frames.clamp_min(0.0), last.to(source_frames.dtype))
        lower = torch.floor(source_frames).long()
        upper = torch.minimum(lower + 1, last)
        alpha = (source_frames - lower).unsqueeze(-1)
        starts = self.clip_starts[clip_ids]
        return torch.lerp(self.head_q_data[starts + lower], self.head_q_data[starts + upper], alpha)

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        source = self.frame_ids[env_ids].to(self.frame_fraction.dtype) + self.frame_fraction[env_ids]
        arm_q, arm_dq = self._interpolate_arm(self.clip_ids[env_ids], source)
        self.arm_q[env_ids] = arm_q
        self.arm_dq[env_ids] = arm_dq
        self.head_q[env_ids] = self._interpolate_head(self.clip_ids[env_ids], source)
        wrist_pos, wrist_rot = self.kinematics.forward(arm_q)
        self.wrist_pos_torso[env_ids] = wrist_pos
        self.wrist_rot_torso[env_ids] = wrist_rot
        # The head remains an external controller target even in a headless
        # rollout.  This direct selected-joint write is intentionally separate
        # from the policy action's selected 29-joint write.
        # mjlab's convenience method uses two advanced tensor indices as a
        # paired index.  Use the explicit outer index so a partial reset writes
        # all two head targets for every selected environment.
        self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = self.head_q[env_ids]

    def _set_episode_actuator_delay(self, env_ids: torch.Tensor) -> None:
        """Sample one 0..2 physics-step body delay per episode/environment."""
        if getattr(self.env, "_depsic_episode_delay_owner", None) is not None:
            # The motor DR reset callback owns a different delay distribution
            # and covers custom elastic drives as well as fused builtin drives.
            return
        body_actuator = next(
            (
                actuator
                for actuator in self.asset.actuators
                if getattr(actuator.cfg, "delay_max_lag", 0) == 2
                and set(getattr(actuator, "target_names", ())).intersection(A3_POLICY_JOINT_NAMES)
            ),
            None,
        )
        if body_actuator is None:
            raise RuntimeError("A3 V1 requires a delayed mjlab body position actuator")
        lags = torch.randint(0, 3, (len(env_ids),), device=self.device)
        # Built-in groups may use one fused buffer, while the 024 closed-loop
        # actuator owns one buffer itself.  Both expose ``set_lags`` and must
        # receive this once-per-episode fixed 0--2 physics-step realization.
        body_actuator.set_lags(lags, env_ids)

    def _root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        root = self.init_root_state[env_ids].clone()
        root[:, :3] += self.env.scene.get_spawn_origins(env_ids)
        if self.nominal_reset:
            root[:, 7:13] = 0.0
            return root
        root[:, :2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.5, 0.5)
        rpy = torch.zeros(len(env_ids), 3, device=self.device)
        rpy[:, 2] = torch.empty(len(env_ids), device=self.device).uniform_(-3.14, 3.14)
        root[:, 3:7] = quat_from_euler_xyz(rpy)
        root[:, 7:13] = torch.empty(len(env_ids), 6, device=self.device).uniform_(-0.5, 0.5)
        return root

    def sample_init(self, env_ids: torch.Tensor, reset_td=None) -> torch.Tensor:
        del reset_td
        return self._root_state(env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._sample_lbc(env_ids)
        self.clip_ids[env_ids] = torch.randint(len(self.lengths), (len(env_ids),), device=self.device)
        episode_frames = torch.ceil(
            self.env.max_episode_length[env_ids, 0].to(self.fps.dtype)
            * self.env.step_dt
            * self.fps[self.clip_ids[env_ids]]
        ).long()
        latest_start = (self.lengths[self.clip_ids[env_ids]] - episode_frames).clamp_min(0)
        self.frame_ids[env_ids] = torch.floor(
            torch.rand(len(env_ids), device=self.device) * (latest_start + 1)
        ).long()
        self.frame_fraction[env_ids] = 0.0
        self.velocity_ema_xy[env_ids] = 0.0
        self.velocity_ema_wz[env_ids] = 0.0
        self.velocity_ema_initialized[env_ids] = False
        self.lbc_metric_steps[env_ids] = 0.0
        self.lbc_metric_means[env_ids] = 0.0
        self._future_arm_q = None
        self._future_arm_dq = None
        self._set_episode_actuator_delay(env_ids)
        self._refresh_reference(env_ids)
        if self.reset_to_reference:
            joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
            if not self.nominal_reset:
                joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
            joint_pos[:, self.arm_joint_ids] = self.arm_q[env_ids]
            joint_pos[:, self.head_joint_ids] = self.head_q[env_ids]
            limits = self.asset.data.joint_pos_limits[env_ids]
            finite_limits = torch.isfinite(limits).all(dim=-1)
            center = limits.mean(dim=-1)
            half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * SOFT_JOINT_POS_LIMIT_FACTOR
            clamped = joint_pos.clamp(center - half_range, center + half_range)
            # MuJoCo represents ball/unlimited joint ranges as +/-inf.  Do
            # not turn their scalar projection into NaN during a 024 reset.
            joint_pos = torch.where(finite_limits, clamped, joint_pos)
            joint_vel = torch.zeros_like(joint_pos)
            joint_vel[:, self.arm_joint_ids] = self.arm_dq[env_ids]
            self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = self.head_q[env_ids]

    def _update_velocity_ema(self) -> None:
        pelvis_quat = self.asset.data.body_link_quat_w[:, self.pelvis_id]
        velocity_xy = quat_rotate_inverse(yaw_quat(pelvis_quat), self.asset.data.root_link_lin_vel_w)[:, :2]
        yaw_velocity = self.asset.data.root_link_ang_vel_w[:, 2]
        initialized = self.velocity_ema_initialized
        self.velocity_ema_xy = torch.where(
            initialized[:, None], 0.9 * self.velocity_ema_xy + 0.1 * velocity_xy, velocity_xy
        )
        self.velocity_ema_wz = torch.where(
            initialized, 0.9 * self.velocity_ema_wz + 0.1 * yaw_velocity, yaw_velocity
        )
        self.velocity_ema_initialized.fill_(True)

    def _update_metrics(self) -> None:
        pelvis_quat = self.asset.data.body_link_quat_w[:, self.pelvis_id]
        velocity_b = quat_rotate_inverse(yaw_quat(pelvis_quat), self.asset.data.root_link_lin_vel_w)
        rpy = _lbc_rpy(self.asset, self.torso_id, self.pelvis_id)
        errors = torch.stack(
            (
                (self.lbc[:, :2] - velocity_b[:, :2]).norm(dim=-1),
                (self.lbc[:, 2] - self.asset.data.root_link_ang_vel_w[:, 2]).abs(),
                (self.lbc[:, 6] - self.asset.data.body_link_pos_w[:, self.torso_id, 2]).abs(),
                wrap_to_pi(rpy[:, 2] - self.lbc[:, 5]).abs(),
                wrap_to_pi(rpy[:, 1] - self.lbc[:, 4]).abs(),
                wrap_to_pi(rpy[:, 0] - self.lbc[:, 3]).abs(),
            ),
            dim=-1,
        )
        self.lbc_metric_steps.add_(1.0)
        self.lbc_metric_means.add_((errors - self.lbc_metric_means) / self.lbc_metric_steps[:, None])
        for index, (metric_name, wandb_key) in enumerate(zip(LBC_METRIC_NAMES, LBC_WANDB_KEYS, strict=True)):
            value = self.lbc_metric_means[:, index].mean()
            self.env.extra[f"Env/Metrics/lbc/{metric_name}"] = value
            self.env.extra[f"metric/lbc/{metric_name}"] = value
            self.env.extra[wandb_key] = value
            self.env.extra[f"train/{metric_name}"] = value
        clip_frame = self.frame_ids.float().mean()
        self.env.extra["Env/Metrics/arm_ref/clip_frame"] = clip_frame
        self.env.extra["train/arm_ref_clip_frame"] = clip_frame

    def update(self) -> None:
        self.frame_fraction += self.env.step_dt * self.fps[self.clip_ids]
        advance = torch.floor(self.frame_fraction).long()
        self.frame_fraction -= advance
        self.frame_ids = torch.minimum(self.frame_ids + advance, self.lengths[self.clip_ids] - 1)
        self._future_arm_q = None
        self._future_arm_dq = None
        self._refresh_reference(torch.arange(self.num_envs, device=self.device))
        self.lbc_elapsed += 1
        self._sample_lbc((self.lbc_elapsed >= self.resample_steps).nonzero(as_tuple=False).flatten())
        self._update_velocity_ema()
        self._update_metrics()


class DepsicA3UltraV1RobotOnly(DepsicA3V1RobotOnly, namespace="mimic_lite"):
    """A3 Ultra T3D0 V1 using the ordinary 29-DoF A3 policy contract.

    The source replay remains the 31-column A3 PKL.  Only the runtime head
    pitch name changes; Ultra finger joints are intentionally not included in
    the actor or replay path and stay at their asset initial state.
    """

    def __init__(self, env, **kwargs):
        kwargs["runtime_head_joint_names"] = ("head_yaw_joint", "ultra_head_pitch_joint")
        super().__init__(env, **kwargs)


class _A3Observation(Observation[DepsicA3V1RobotOnly]):
    pass


class depsic_a3_actor_history(_A3Observation, namespace="mimic_lite"):
    """10 x [gravity, omega, q29, dq29, action29] = 930."""

    def __init__(
        self,
        env,
        gravity_noise: float = 0.05,
        ang_vel_noise: float = 0.2,
        joint_pos_noise: float = 0.01,
        joint_vel_noise: float = 1.5,
        ankle_pitch_joint_vel_noise: float | None = None,
        ankle_roll_joint_vel_noise: float | None = None,
        joint_pos_source_names: dict[str, str] | None = None,
        ankle_encoder_state: bool = False,
    ):
        super().__init__(env)
        self.gravity_noise = float(gravity_noise)
        self.ang_vel_noise = float(ang_vel_noise)
        self.joint_pos_noise = float(joint_pos_noise)
        self.joint_vel_noise = float(joint_vel_noise)
        if not math.isfinite(self.joint_vel_noise) or self.joint_vel_noise < 0.0:
            raise ValueError("joint_vel_noise must be finite and non-negative")
        # Per-joint TOTAL uniform half-widths in rad/s, not extra noise.
        # None inherits joint_vel_noise; the legacy path keeps its exact RNG.
        self.joint_vel_noise_amplitudes = None
        overrides = {
            "pitch": ankle_pitch_joint_vel_noise,
            "roll": ankle_roll_joint_vel_noise,
        }
        for axis, amplitude in overrides.items():
            if amplitude is None:
                continue
            amplitude = float(amplitude)
            if not math.isfinite(amplitude) or amplitude < 0.0:
                raise ValueError(f"ankle_{axis}_joint_vel_noise must be finite and non-negative")
            if self.joint_vel_noise_amplitudes is None:
                self.joint_vel_noise_amplitudes = torch.full(
                    (A3_POLICY_DIM,), self.joint_vel_noise, device=self.device
                )
            ids = [i for i, name in enumerate(A3_POLICY_JOINT_NAMES) if f"_ankle_{axis}_" in name]
            self.joint_vel_noise_amplitudes[ids] = amplitude
        self.action_manager = env.input_managers["action"]
        self.policy_joint_ids = self.command_manager.policy_joint_ids
        self.position_joint_ids = self.policy_joint_ids
        self.ankle_encoder_state = getattr(env, "_four_ankle_state", None) if ankle_encoder_state else None
        if ankle_encoder_state and self.ankle_encoder_state is None:
            raise ValueError("ankle_encoder_state requires the four-axis dynamics reset hook")
        if ankle_encoder_state and joint_pos_source_names:
            raise ValueError("Use either mixed encoder state or static joint source mapping")
        if joint_pos_source_names:
            names = list(self.env.scene.articulations["robot"].joint_names)
            if set(joint_pos_source_names) - set(A3_POLICY_JOINT_NAMES):
                raise ValueError("Position source overrides must name policy joints")
            self.position_joint_ids = torch.tensor([
                names.index(joint_pos_source_names.get(name, name)) for name in A3_POLICY_JOINT_NAMES
            ], device=self.device)
        self.buffer = torch.zeros(self.num_envs, A3_REFERENCE_FRAMES, 93, device=self.device)
        self.head = 0

    def _value(self) -> torch.Tensor:
        asset = self.env.scene.articulations["robot"]
        gravity = asset.data.projected_gravity_b + torch.empty_like(asset.data.projected_gravity_b).uniform_(
            -self.gravity_noise, self.gravity_noise
        )
        omega = asset.data.root_com_ang_vel_b + torch.empty_like(asset.data.root_com_ang_vel_b).uniform_(
            -self.ang_vel_noise, self.ang_vel_noise
        )
        q = asset.data.joint_pos[:, self.position_joint_ids] - asset.data.default_joint_pos[:, self.policy_joint_ids]
        dq = asset.data.joint_vel[:, self.policy_joint_ids]
        if self.ankle_encoder_state is not None:
            state = self.ankle_encoder_state
            enc_q, enc_dq = state.encoder_state()
            q[:, state.roll_cols] = enc_q - asset.data.default_joint_pos[:, state.output_ids]
            dq[:, state.roll_cols] = enc_dq
        q = q + torch.empty_like(q).uniform_(-self.joint_pos_noise, self.joint_pos_noise)
        # Exactly one noise sample per joint/new observation. Never modify
        # simulator state, driver PD, rewards or critic inputs.
        if self.joint_vel_noise_amplitudes is None:
            dq_noise = torch.empty_like(dq).uniform_(-self.joint_vel_noise, self.joint_vel_noise)
        else:
            dq_noise = torch.empty_like(dq).uniform_(-1.0, 1.0) * self.joint_vel_noise_amplitudes
        dq = dq + dq_noise
        return torch.cat((gravity, omega, q, dq, self.action_manager.applied_action), dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = self._value()[env_ids, None]

    def update(self) -> None:
        self.head = (self.head - 1) % self.buffer.shape[1]
        self.buffer[:, self.head] = self._value()

    def compute(self) -> torch.Tensor:
        indices = (torch.arange(A3_REFERENCE_FRAMES, device=self.device) + self.head + 1) % A3_REFERENCE_FRAMES
        result = self.buffer[:, indices].reshape(self.num_envs, -1)
        if result.shape[-1] != A3_ACTOR_OBS_DIM:
            raise RuntimeError(f"A3 actor observation contract broken: expected {A3_ACTOR_OBS_DIM}, got {result.shape[-1]}")
        return result


class depsic_a3_reference(_A3Observation, namespace="mimic_lite"):
    """A3 tokenizer input: 10 x [arm q14, arm dq14, LBC7] = 350."""

    def compute(self) -> torch.Tensor:
        q, dq = self.command_manager.arm_future
        result = torch.cat((q, dq, self.command_manager.lbc_future), dim=-1).reshape(self.num_envs, -1)
        if result.shape[-1] != A3_TOKENIZER_DIM:
            raise RuntimeError(f"A3 tokenizer contract broken: expected {A3_TOKENIZER_DIM}, got {result.shape[-1]}")
        return result


class depsic_a3_privileged(_A3Observation, namespace="mimic_lite"):
    """A3 critic state with q/dq restricted to the 29 policy DoFs."""

    def __init__(self, env):
        super().__init__(env)
        self.action_manager = env.input_managers["action"]
        self.policy_joint_ids = self.command_manager.policy_joint_ids

    def compute(self) -> torch.Tensor:
        asset = self.env.scene.articulations["robot"]
        arm_q, arm_dq = self.command_manager.arm_future
        q = asset.data.joint_pos[:, self.policy_joint_ids] - asset.data.default_joint_pos[:, self.policy_joint_ids]
        result = torch.cat(
            (
                self.command_manager.lbc_future.reshape(self.num_envs, -1),
                arm_q.reshape(self.num_envs, -1),
                arm_dq.reshape(self.num_envs, -1),
                asset.data.projected_gravity_b,
                asset.data.root_com_lin_vel_b,
                asset.data.root_com_ang_vel_b,
                q,
                asset.data.joint_vel[:, self.policy_joint_ids],
                self.action_manager.applied_action,
            ),
            dim=-1,
        )
        if result.shape[-1] != A3_CRITIC_DIM:
            raise RuntimeError(f"A3 critic contract broken: expected {A3_CRITIC_DIM}, got {result.shape[-1]}")
        return result


__all__ = ["DepsicA3V1RobotOnly", "DepsicA3UltraV1RobotOnly"]
