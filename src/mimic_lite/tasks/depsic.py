"""MjLab implementation of the DEPSIC G1 robot-only task.

The task keeps DEPSIC's split command contract: a sampled, held lower-body
command (LBC) and an independently sampled AMASS arm reference.  It is kept
separate from MimicLite's full-body motion tracker because the reference has
no root, leg, or contact labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.resources
from pathlib import Path
import io
import math
import pickle
import re
import xml.etree.ElementTree as et
import zlib

import numpy as np
import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.randomizations.base import Randomization
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.envs.mdp.terminations.base import Termination
from active_adaptation.envs.utils import find_bodies, find_joints, find_sensor_bodies
from active_adaptation.utils.math import (
    euler_from_quat,
    matrix_from_quat,
    quat_angle_magnitude,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ARM_JOINT_NAMES = G1_JOINT_NAMES[15:]
LBC_DIM = 7
ARM_DIM = len(ARM_JOINT_NAMES)
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
LBC_METRIC_NAMES = ("E_v", "E_omega", "E_h", "E_y", "E_p", "E_r")
LBC_WANDB_KEYS = (
    "train/E_v_mps",
    "train/E_omega_radps",
    "train/E_h_m",
    "train/E_y_rad",
    "train/E_p_rad",
    "train/E_r_rad",
)


def _find_clip(payload: object, source: Path) -> Mapping[str, object]:
    if isinstance(payload, Mapping) and "dof" in payload:
        return payload
    if isinstance(payload, Mapping):
        clips = [value for value in payload.values() if isinstance(value, Mapping) and "dof" in value]
        if len(clips) == 1:
            return clips[0]
    raise ValueError(f"{source}: expected a mapping containing a 'dof' array")


def _load_compressed_pkl(source: Path) -> Mapping[str, object]:
    """Load DEPSIC's zlib-compressed joblib PKL with a clear dependency error."""
    try:
        raw = zlib.decompress(source.read_bytes())
    except (OSError, zlib.error) as error:
        raise ValueError(f"Could not read compressed DEPSIC clip {source}") from error

    try:
        import joblib
    except ModuleNotFoundError:
        # Standard pickle is useful for small hand-authored fixtures, while
        # production AMASS PKLs are normally written by joblib.
        try:
            payload = pickle.loads(raw)
        except (pickle.UnpicklingError, EOFError, ValueError) as error:
            raise ModuleNotFoundError(
                "DEPSIC AMASS loading requires joblib. Install the MimicLite mjlab environment "
                "after syncing its updated pyproject.toml."
            ) from error
    else:
        payload = joblib.load(io.BytesIO(raw))

    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: expected a mapping payload")
    return _find_clip(payload, source)


def _discover_clips(path: str | Path) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        if root.suffix != ".pkl":
            raise ValueError(f"Expected a .pkl AMASS clip, got {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"AMASS path does not exist: {root}")
    clips = sorted(root.rglob("*.pkl"))
    if not clips:
        raise FileNotFoundError(f"No .pkl AMASS clips found below {root}")
    return clips


@dataclass(frozen=True)
class _ArmPool:
    paths: tuple[str, ...]
    lengths: torch.Tensor
    fps: torch.Tensor
    q: torch.Tensor


def _load_arm_pool(
    path: str | Path,
    *,
    motion_pool_size: int | None,
    default_fps: float,
) -> _ArmPool:
    paths = _discover_clips(path)
    if motion_pool_size is not None:
        if motion_pool_size < 1:
            raise ValueError("motion_pool_size must be positive or null")
        if len(paths) > motion_pool_size:
            stride = max(1, len(paths) // motion_pool_size)
            paths = paths[::stride][:motion_pool_size]

    q_parts: list[torch.Tensor] = []
    lengths: list[int] = []
    fps_values: list[float] = []
    valid_paths: list[str] = []
    invalid: list[tuple[Path, str]] = []
    for source in paths:
        try:
            clip = _load_compressed_pkl(source)
            dof = np.asarray(clip["dof"], dtype=np.float32)
            if dof.ndim != 2 or dof.shape[1] != len(G1_JOINT_NAMES):
                raise ValueError(f"expected dof [T, 29], got {dof.shape}")
            if dof.shape[0] < 2 or not np.isfinite(dof).all():
                raise ValueError("clip needs at least two finite frames")
            fps = float(clip.get("fps", default_fps))
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"invalid fps={fps}")
        except (KeyError, TypeError, ValueError, ModuleNotFoundError) as error:
            invalid.append((source, str(error)))
            continue
        q_parts.append(torch.from_numpy(np.ascontiguousarray(dof[:, 15:29])))
        lengths.append(int(dof.shape[0]))
        fps_values.append(fps)
        valid_paths.append(str(source.resolve()))

    if not q_parts:
        detail = invalid[0][1] if invalid else "no clips"
        raise ValueError(f"No valid DEPSIC G1 AMASS clips were found: {detail}")
    if invalid:
        first_path, first_error = invalid[0]
        print(
            f"[depsic] skipped {len(invalid)} invalid AMASS clips; first: {first_path} ({first_error})"
        )
    q = torch.cat(q_parts, dim=0).contiguous()
    print(f"[depsic] loaded {len(valid_paths)} arm clips / {q.shape[0]} frames")
    return _ArmPool(
        paths=tuple(valid_paths),
        lengths=torch.tensor(lengths, dtype=torch.long),
        fps=torch.tensor(fps_values, dtype=torch.float32),
        q=q,
    )


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def _vector(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    values = tuple(float(item) for item in value.split())
    if len(values) != 3:
        raise ValueError(f"Expected a three-vector, got {value!r}")
    return values


def _rpy_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zeros, ones = torch.zeros_like(roll), torch.ones_like(roll)
    rx = torch.stack((ones, zeros, zeros, zeros, cr, -sr, zeros, sr, cr), dim=-1).reshape(*roll.shape, 3, 3)
    ry = torch.stack((cp, zeros, sp, zeros, ones, zeros, -sp, zeros, cp), dim=-1).reshape(*roll.shape, 3, 3)
    rz = torch.stack((cy, -sy, zeros, sy, cy, zeros, zeros, zeros, ones), dim=-1).reshape(*roll.shape, 3, 3)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z = axis.unbind(dim=-1)
    c, s = torch.cos(angle), torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack(
        (
            c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s,
            y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s,
            z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c,
        ),
        dim=-1,
    ).reshape(*angle.shape, 3, 3)


class _G1ArmKinematics:
    """Torch FK for the reference arm pose, parsed once from the G1 URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        torso_link: str = "torso_link",
        wrist_links: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        arm_joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
        robot_label: str = "G1",
    ):
        source_ref = str(urdf_path)
        if source_ref.startswith("package://"):
            package, _, resource = source_ref.removeprefix("package://").partition("/")
            if not package or not resource:
                raise ValueError(f"Invalid package asset reference: {source_ref}")
            source = Path(importlib.resources.files(package).joinpath(resource))
        else:
            source = Path(source_ref).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                f"DEPSIC {robot_label} wrist rewards need the matching URDF, not found: {source}"
            )
        joints: dict[str, _UrdfJoint] = {}
        for node in et.parse(source).getroot().findall("joint"):
            parent, child = node.find("parent"), node.find("child")
            if parent is None or child is None:
                continue
            origin, axis = node.find("origin"), node.find("axis")
            joints[node.attrib["name"]] = _UrdfJoint(
                name=node.attrib["name"],
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                joint_type=node.attrib.get("type", "fixed"),
                xyz=_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_vector(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            )
        by_child = {joint.child: joint for joint in joints.values()}
        self.chains = tuple(
            self._chain(by_child, target, torso_link)
            for target in wrist_links
        )
        chain_joint_names = {
            joint.name for chain in self.chains for joint in chain if joint.joint_type in {"revolute", "continuous"}
        }
        if chain_joint_names != set(arm_joint_names):
            raise ValueError(
                f"The configured URDF arm chains do not match the DEPSIC {robot_label} arm contract: "
                f"{sorted(chain_joint_names)}"
            )
        self.arm_joint_names = arm_joint_names
        self.arm_indices = {name: index for index, name in enumerate(arm_joint_names)}

    @staticmethod
    def _chain(by_child: Mapping[str, _UrdfJoint], target: str, torso: str) -> list[_UrdfJoint]:
        chain: list[_UrdfJoint] = []
        link = target
        while link != torso:
            if link not in by_child:
                raise ValueError(f"No URDF chain from {torso} to {target}")
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    @staticmethod
    def _transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        transform = torch.zeros(*rotation.shape[:-2], 4, 4, dtype=rotation.dtype, device=rotation.device)
        transform[..., :3, :3] = rotation
        transform[..., :3, 3] = translation
        transform[..., 3, 3] = 1.0
        return transform

    def forward(self, arm_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if arm_q.shape[-1] != len(self.arm_joint_names):
            raise ValueError(
                f"Expected [..., {len(self.arm_joint_names)}] arm q, got {tuple(arm_q.shape)}"
            )
        transforms = []
        for chain in self.chains:
            transform = torch.eye(4, dtype=arm_q.dtype, device=arm_q.device).expand(*arm_q.shape[:-1], 4, 4).clone()
            for joint in chain:
                xyz = torch.tensor(joint.xyz, dtype=arm_q.dtype, device=arm_q.device).expand(*arm_q.shape[:-1], 3)
                rpy = torch.tensor(joint.rpy, dtype=arm_q.dtype, device=arm_q.device).expand(*arm_q.shape[:-1], 3)
                transform = transform @ self._transform(_rpy_matrix(rpy), xyz)
                if joint.joint_type in {"revolute", "continuous"}:
                    axis = torch.tensor(joint.axis, dtype=arm_q.dtype, device=arm_q.device).expand(*arm_q.shape[:-1], 3)
                    rotation = _axis_angle_matrix(axis, arm_q[..., self.arm_indices[joint.name]])
                    transform = transform @ self._transform(rotation, torch.zeros_like(xyz))
            transforms.append(transform)
        pose = torch.stack(transforms, dim=-3)
        return pose[..., :3, 3], pose[..., :3, :3]


class DepsicG1RobotOnly(Command, namespace="mimic_lite"):
    """Synthetic 7-D LBC plus AMASS 14-D arm replay for the G1 DEPSIC task."""

    def __init__(
        self,
        env,
        motion_file: str,
        arm_urdf_path: str,
        motion_pool_size: int | None = None,
        default_fps: float = 50.0,
        future_frames: int = 10,
        future_frame_dt: float = 0.1,
        resampling_time_s: float = 10.0,
        standing_probability: float = 0.4,
        reset_to_reference: bool = True,
        lbc_ranges: Mapping[str, Sequence[float]] | None = None,
        curriculum: Mapping[str, Mapping[str, Sequence[float] | int | float]] | None = None,
        nominal_reset: bool = False,
    ) -> None:
        super().__init__(env)
        if future_frames < 1 or future_frame_dt <= 0.0 or resampling_time_s <= 0.0:
            raise ValueError("future_frames, future_frame_dt, and resampling_time_s must be positive")
        self.future_frames = int(future_frames)
        self.future_frame_dt = float(future_frame_dt)
        self.resample_steps = max(1, round(float(resampling_time_s) / env.step_dt))
        self.standing_probability = float(standing_probability)
        self.reset_to_reference = bool(reset_to_reference)
        # Table-II evaluation uses the same nominal root/joint reset as the
        # IsaacLab evaluator. Keep the training default randomized.
        self.nominal_reset = bool(nominal_reset)
        arm_joint_ids, arm_joint_names = find_joints(self.asset, list(ARM_JOINT_NAMES))
        if set(arm_joint_names) != set(ARM_JOINT_NAMES) or len(arm_joint_names) != ARM_DIM:
            raise ValueError(f"DEPSIC arm joint set mismatch: {arm_joint_names}")
        # mjlab stores G1 joints in its simulator order (left/right interleaved),
        # while DEPSIC PKLs keep all seven left-arm entries before the right arm.
        ids_by_name = dict(zip(arm_joint_names, arm_joint_ids, strict=True))
        self.arm_joint_ids = torch.tensor(
            [ids_by_name[name] for name in ARM_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.torso_id, torso_names = find_bodies(self.asset, ["torso_link"])
        self.pelvis_id, pelvis_names = find_bodies(self.asset, ["pelvis"])
        if len(torso_names) != 1 or len(pelvis_names) != 1:
            raise ValueError("DEPSIC requires torso_link and pelvis bodies")
        self.torso_id, self.pelvis_id = self.torso_id[0], self.pelvis_id[0]
        self.pool = _load_arm_pool(
            motion_file,
            motion_pool_size=motion_pool_size,
            default_fps=float(default_fps),
        )
        self.lengths = self.pool.lengths.to(device=self.device)
        self.fps = self.pool.fps.to(device=self.device)
        self.clip_starts = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), self.lengths.cumsum(0)[:-1]))
        self.arm_q_data = self.pool.q.to(device=self.device)
        self.kinematics = _G1ArmKinematics(arm_urdf_path)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_ids = torch.zeros_like(self.clip_ids)
        self.frame_fraction = torch.zeros(self.num_envs, device=self.device)
        self.lbc = torch.zeros(self.num_envs, LBC_DIM, device=self.device)
        self.arm_q = torch.zeros(self.num_envs, ARM_DIM, device=self.device)
        self.arm_dq = torch.zeros_like(self.arm_q)
        self.wrist_pos_torso = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.wrist_rot_torso = torch.eye(3, device=self.device).expand(self.num_envs, 2, 3, 3).clone()
        self.lbc_elapsed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.velocity_ema_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.velocity_ema_wz = torch.zeros(self.num_envs, device=self.device)
        self.velocity_ema_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Original DEPSIC logs an episode mean for each HTD Table-II LBC error.
        self.lbc_metric_steps = torch.zeros(self.num_envs, device=self.device)
        self.lbc_metric_means = torch.zeros(self.num_envs, len(LBC_METRIC_NAMES), device=self.device)
        self._future_arm_q: torch.Tensor | None = None
        self._future_arm_dq: torch.Tensor | None = None
        ranges = lbc_ranges or {}
        default_ranges = (
            (-0.55, 0.55), (-0.55, 0.55), (-1.57, 1.57),
            (-0.5, 0.5), (-0.52, 1.22), (-1.27, 1.27), (0.35, 0.8),
        )
        range_keys = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "torso_roll", "torso_pitch", "torso_yaw", "base_height")
        self.lbc_range_keys = range_keys
        self.lbc_ranges = torch.tensor(
            [ranges.get(key, default) for key, default in zip(range_keys, default_ranges, strict=True)],
            dtype=torch.float32,
            device=self.device,
        )
        self.curriculum = dict(curriculum or {})

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
            source_frames = (
                self.frame_ids[:, None].to(self.frame_fraction.dtype)
                + self.frame_fraction[:, None]
                + offsets[None] * self.future_frame_dt * self.fps[self.clip_ids, None]
            )
            self._future_arm_q, self._future_arm_dq = self._interpolate(self.clip_ids, source_frames)
        return self._future_arm_q, self._future_arm_dq

    def _effective_ranges(self) -> torch.Tensor:
        result = self.lbc_ranges.clone()
        iteration = float(self.env.current_iter)
        for index, key in enumerate(self.lbc_range_keys):
            spec = self.curriculum.get(key)
            if not spec:
                continue
            start = float(spec["start_iter"])
            end = float(spec["end_iter"])
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
        ranges = self._effective_ranges()
        low, high = ranges[:, 0], ranges[:, 1]
        self.lbc[env_ids] = low + torch.rand(len(env_ids), LBC_DIM, device=self.device) * (high - low)
        standing = torch.rand(len(env_ids), device=self.device) < self.standing_probability
        self.lbc[env_ids[standing], :3] = 0.0
        self.lbc_elapsed[env_ids] = 0

    def _interpolate(
        self,
        clip_ids: torch.Tensor,
        source_frames: torch.Tensor,
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

    def _refresh_arm_reference(self, env_ids: torch.Tensor) -> None:
        source = self.frame_ids[env_ids].to(self.frame_fraction.dtype) + self.frame_fraction[env_ids]
        q, dq = self._interpolate(self.clip_ids[env_ids], source)
        self.arm_q[env_ids] = q
        self.arm_dq[env_ids] = dq
        wrist_pos, wrist_rot = self.kinematics.forward(q)
        self.wrist_pos_torso[env_ids] = wrist_pos
        self.wrist_rot_torso[env_ids] = wrist_rot

    def _root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        root = self.init_root_state[env_ids].clone()
        root[:, :3] += self.env.scene.get_spawn_origins(env_ids)
        if self.nominal_reset:
            root[:, 7:13] = 0.0
            return root
        root[:, :2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.5, 0.5)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        rpy = torch.zeros(len(env_ids), 3, device=self.device)
        rpy[:, 2] = yaw
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
        self._refresh_arm_reference(env_ids)
        if self.reset_to_reference:
            joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
            if not self.nominal_reset:
                joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
            joint_pos[:, self.arm_joint_ids] = self.arm_q[env_ids]
            limits = self.asset.data.joint_pos_limits[env_ids]
            # IsaacLab's reset_joints_by_scale clamps to the asset's soft limits.
            limit_center = limits.mean(dim=-1)
            soft_half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * SOFT_JOINT_POS_LIMIT_FACTOR
            joint_pos = joint_pos.clamp(limit_center - soft_half_range, limit_center + soft_half_range)
            joint_vel = torch.zeros_like(joint_pos)
            # GR00T's reset event writes the sampled AMASS velocity for the
            # arm joints as well as position. Keep lower-body reset velocity
            # at zero, matching its reset_joints_by_scale contract.
            joint_vel[:, self.arm_joint_ids] = self.arm_dq[env_ids]
            self.asset.write_joint_state_to_sim(
                joint_pos,
                joint_vel,
                env_ids=env_ids,
            )

    def _update_velocity_ema(self) -> None:
        pelvis_quat = self.asset.data.body_link_quat_w[:, self.pelvis_id]
        root_lin_vel_w = self.asset.data.root_link_lin_vel_w
        root_ang_vel_w = self.asset.data.root_link_ang_vel_w
        velocity_xy = quat_rotate_inverse(yaw_quat(pelvis_quat), root_lin_vel_w)[:, :2]
        yaw_velocity = root_ang_vel_w[:, 2]
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
        lbc = self.lbc
        errors = torch.stack(
            (
                (lbc[:, :2] - velocity_b[:, :2]).norm(dim=-1),
                (lbc[:, 2] - self.asset.data.root_link_ang_vel_w[:, 2]).abs(),
                (lbc[:, 6] - self.asset.data.body_link_pos_w[:, self.torso_id, 2]).abs(),
                wrap_to_pi(rpy[:, 2] - lbc[:, 5]).abs(),
                wrap_to_pi(rpy[:, 1] - lbc[:, 4]).abs(),
                wrap_to_pi(rpy[:, 0] - lbc[:, 3]).abs(),
            ),
            dim=-1,
        )
        self.lbc_metric_steps.add_(1.0)
        self.lbc_metric_means.add_(
            (errors - self.lbc_metric_means) / self.lbc_metric_steps[:, None]
        )

        # Preserve original DEPSIC names and provide its W&B train/* aliases.
        for index, (metric_name, wandb_key) in enumerate(
            zip(LBC_METRIC_NAMES, LBC_WANDB_KEYS, strict=True)
        ):
            value = self.lbc_metric_means[:, index].mean()
            self.env.extra[f"Env/Metrics/lbc/{metric_name}"] = value
            self.env.extra[f"metric/lbc/{metric_name}"] = value
            self.env.extra[wandb_key] = value
            # Preserve unit-bearing DEPSIC keys above and expose the compact
            # aliases used by older dashboard templates.
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
        self._refresh_arm_reference(torch.arange(self.num_envs, device=self.device))
        self.lbc_elapsed += 1
        self._sample_lbc((self.lbc_elapsed >= self.resample_steps).nonzero(as_tuple=False).flatten())
        self._update_velocity_ema()
        self._update_metrics()


class depsic_joint_default_offset(Randomization, namespace="mimic_lite"):
    """Apply DEPSIC's startup encoder-offset randomization to action targets."""

    def __init__(
        self,
        env,
        low: float = -0.01,
        high: float = 0.01,
        joint_names: str | list[str] | None = None,
    ) -> None:
        super().__init__(env)
        if high < low:
            raise ValueError("DEPSIC joint default offset requires low <= high")
        self.low = float(low)
        self.high = float(high)
        self.asset = env.scene.articulations["robot"]
        self.action_manager = env.action_manager
        self.nominal_joint_pos = self.asset.data.default_joint_pos.clone()
        self.joint_ids = None
        if joint_names is not None:
            ids, _ = find_joints(self.asset, joint_names)
            self.joint_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

    def startup(self) -> None:
        offset = torch.zeros_like(self.asset.data.default_joint_pos)
        if self.joint_ids is None:
            offset.uniform_(self.low, self.high)
        else:
            offset[:, self.joint_ids] = torch.empty(
                self.num_envs, len(self.joint_ids), device=self.device
            ).uniform_(self.low, self.high)
        # Isaac Lab changes both the reset reference and the position-action
        # offset. Keep the two mjlab representations synchronized.
        self.asset.data.default_joint_pos.copy_(self.nominal_joint_pos + offset)
        self.action_manager.offset.copy_(offset)


class depsic_torso_mass_offset(Randomization, namespace="mimic_lite"):
    """Add the committed DEPSIC torso mass offset in mjlab's batched model."""

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        low: float = -5.0,
        high: float = 5.0,
        torso_body_name: str = "torso_link",
    ) -> None:
        super().__init__(env)
        if high < low:
            raise ValueError("DEPSIC torso mass offset requires low <= high")
        torso_ids, torso_names = find_bodies(env.scene.articulations["robot"], [torso_body_name])
        if len(torso_names) != 1:
            raise ValueError(f"DEPSIC torso mass randomization requires {torso_body_name}")
        self.low = float(low)
        self.high = float(high)
        self.asset = env.scene.articulations["robot"]
        self.global_body_id = self.asset.indexing.body_ids[torso_ids[0]]
        env.sim.expand_model_fields(("body_mass", "body_inertia"))
        from mjlab.managers.event_manager import RecomputeLevel

        self.recompute_level = RecomputeLevel.set_const
        self.default_mass = env.sim.model.body_mass[:, self.global_body_id].clone()
        self.default_inertia = env.sim.model.body_inertia[:, self.global_body_id].clone()

    def startup(self) -> None:
        offset = torch.empty(self.num_envs, device=self.device).uniform_(self.low, self.high)
        mass = (self.default_mass + offset).clamp_min(1.0e-3)
        ratio = mass / self.default_mass.clamp_min(1.0e-3)
        self.env.sim.model.body_mass[:, self.global_body_id] = mass
        self.env.sim.model.body_inertia[:, self.global_body_id] = self.default_inertia * ratio[:, None]
        self.env.sim.recompute_constants(self.recompute_level)


class _DepsicObservation(Observation[DepsicG1RobotOnly]):
    pass


class depsic_actor_history(_DepsicObservation, namespace="mimic_lite"):
    """Ten DEPSIC proprioception frames in IsaacLab's oldest-to-newest order."""

    def __init__(self, env, gravity_noise: float = 0.05, ang_vel_noise: float = 0.2, joint_pos_noise: float = 0.01, joint_vel_noise: float = 1.5):
        super().__init__(env)
        self.gravity_noise = float(gravity_noise)
        self.ang_vel_noise = float(ang_vel_noise)
        self.joint_pos_noise = float(joint_pos_noise)
        self.joint_vel_noise = float(joint_vel_noise)
        self.action_manager = env.input_managers["action"]
        self.buffer = torch.zeros(self.num_envs, 10, 93, device=self.device)
        self.head = 0

    def _value(self) -> torch.Tensor:
        asset = self.env.scene.articulations["robot"]
        gravity = asset.data.projected_gravity_b + torch.empty_like(asset.data.projected_gravity_b).uniform_(-self.gravity_noise, self.gravity_noise)
        # IsaacLab's legacy ``root_ang_vel_b`` alias used by GR00T resolves
        # to root-COM angular velocity for articulations.
        omega = asset.data.root_com_ang_vel_b + torch.empty_like(asset.data.root_com_ang_vel_b).uniform_(-self.ang_vel_noise, self.ang_vel_noise)
        q = asset.data.joint_pos - asset.data.default_joint_pos
        q = q + torch.empty_like(q).uniform_(-self.joint_pos_noise, self.joint_pos_noise)
        dq = asset.data.joint_vel + torch.empty_like(asset.data.joint_vel).uniform_(-self.joint_vel_noise, self.joint_vel_noise)
        return torch.cat((gravity, omega, q, dq, self.action_manager.applied_action), dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = self._value()[env_ids, None]

    def update(self) -> None:
        self.head = (self.head - 1) % self.buffer.shape[1]
        self.buffer[:, self.head] = self._value()

    def compute(self) -> torch.Tensor:
        # IsaacLab flattens observation histories oldest-to-newest.  ``head``
        # marks the newest ring-buffer entry, so start one slot after it.
        indices = (torch.arange(10, device=self.device) + self.head + 1) % 10
        return self.buffer[:, indices].reshape(self.num_envs, -1)


class depsic_reference(_DepsicObservation, namespace="mimic_lite"):
    """Flattened 10 x [arm q, arm dq, LBC] DEPSIC tokenizer input."""

    def compute(self) -> torch.Tensor:
        q, dq = self.command_manager.arm_future
        return torch.cat((q, dq, self.command_manager.lbc_future), dim=-1).reshape(self.num_envs, -1)


class depsic_privileged(_DepsicObservation, namespace="mimic_lite"):
    """DEPSIC critic state in the committed GR00T field order."""

    def __init__(self, env):
        super().__init__(env)
        self.action_manager = env.input_managers["action"]

    def compute(self) -> torch.Tensor:
        asset = self.env.scene.articulations["robot"]
        arm_q, arm_dq = self.command_manager.arm_future
        lbc_future = self.command_manager.lbc_future
        q = asset.data.joint_pos - asset.data.default_joint_pos
        return torch.cat(
            (
                # GR00T's critic config declares LBC before the two AMASS
                # fields; preserve that ordering independently of tokenizer.
                lbc_future.reshape(self.num_envs, -1),
                arm_q.reshape(self.num_envs, -1),
                arm_dq.reshape(self.num_envs, -1),
                asset.data.projected_gravity_b,
                # IsaacLab's legacy root velocity aliases resolve to these
                # root-COM fields for articulations.
                asset.data.root_com_lin_vel_b,
                asset.data.root_com_ang_vel_b,
                q,
                asset.data.joint_vel,
                self.action_manager.applied_action,
            ),
            dim=-1,
        )


def _lbc_rpy(asset, torso_id: int, pelvis_id: int) -> torch.Tensor:
    torso_quat = asset.data.body_link_quat_w[:, torso_id]
    pelvis_quat = asset.data.body_link_quat_w[:, pelvis_id]
    return wrap_to_pi(euler_from_quat(quat_mul(quat_conjugate(yaw_quat(pelvis_quat)), torso_quat)))


def _contact_force_history(sensor, sensor_ids: torch.Tensor) -> torch.Tensor:
    data = sensor.data
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        return force_history[:, sensor_ids]
    net_forces = getattr(data, "net_forces_w", None)
    if net_forces is not None:
        return net_forces[:, sensor_ids].unsqueeze(2)
    raise RuntimeError("The mjlab contact sensor does not expose force history")


def _contact_mask(sensor, sensor_ids: torch.Tensor, threshold: float) -> torch.Tensor:
    forces = _contact_force_history(sensor, sensor_ids)
    return forces.norm(dim=-1).amax(dim=-1) > threshold


def _wrist_anchor_height_mask(
    asset,
    command: DepsicG1RobotOnly,
    torso_id: int,
    pelvis_id: int,
    wrist_ids: torch.Tensor,
    threshold: float,
    down_threshold: float,
    height_threshold: float,
) -> torch.Tensor:
    torso_pos = asset.data.body_link_pos_w[:, torso_id]
    torso_quat = asset.data.body_link_quat_w[:, torso_id, None]
    pelvis_z = asset.data.body_link_pos_w[:, pelvis_id, 2]
    wrists = asset.data.body_link_pos_w[:, wrist_ids]
    reference = torso_pos[:, None] + _quat_rotate(torso_quat, command.wrist_pos_torso)
    error = ((wrists[..., 2] - pelvis_z[:, None]) - (reference[..., 2] - pelvis_z[:, None])).abs()
    allowed = torch.where(
        command.lbc[:, 6, None] < height_threshold,
        torch.full_like(error, down_threshold),
        torch.full_like(error, threshold),
    )
    return (error > allowed).any(dim=-1, keepdim=True)


class _DepsicReward(Reward[DepsicG1RobotOnly]):
    def __init__(self, env, weight: float, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = env.scene.articulations["robot"]


class depsic_track_lin_vel(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        command = self.command_manager.lbc[:, :2]
        pelvis_quat = self.asset.data.body_link_quat_w[:, self.command_manager.pelvis_id]
        velocity = quat_rotate_inverse(yaw_quat(pelvis_quat), self.asset.data.root_link_lin_vel_w)[:, :2]
        broad = torch.exp(-(command - velocity).square().sum(dim=-1, keepdim=True) / 0.65**2)
        sharp = torch.exp(-(command - self.command_manager.velocity_ema_xy).square().sum(dim=-1, keepdim=True) / 0.18**2)
        return 1.5 * (0.35 * broad + 0.65 * sharp)


class depsic_track_ang_vel(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        command = self.command_manager.lbc[:, 2]
        velocity = self.asset.data.root_link_ang_vel_w[:, 2]
        broad = torch.exp(-((command - velocity) / 0.80).square())
        sharp = torch.exp(-((command - self.command_manager.velocity_ema_wz) / 0.20).square())
        return (2.0 * (0.40 * broad + 0.60 * sharp)).unsqueeze(-1)


class depsic_yaw_rate_high_pass(_DepsicReward, namespace="mimic_lite"):
    """Suppress only high-frequency yaw correction, not commanded turning.

    ``velocity_ema_wz`` is the same low-pass state already used by the V1
    yaw-rate tracker.  Penalising the residual lets natural turning retain its
    mean yaw velocity while removing the rapid left/right corrections that can
    otherwise improve instantaneous command reward in simulation.
    """

    def __init__(self, env, weight: float, std: float = 0.35, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        if std <= 0.0:
            raise ValueError("DEPSIC yaw high-pass std must be positive")
        self.std = float(std)

    def _compute(self) -> torch.Tensor:
        raw_wz = self.asset.data.root_link_ang_vel_w[:, 2]
        residual = raw_wz - self.command_manager.velocity_ema_wz
        return (residual / self.std).square().unsqueeze(-1)


class depsic_track_heading(_DepsicReward, namespace="mimic_lite"):
    """Bounded heading-error reward for dynamic turning.

    A3 AMP uses this in place of instantaneous yaw-rate tracking.  The
    command's yaw slot is interpreted as a short-horizon heading target by
    integrating the requested rate at the task control period.
    """

    def __init__(self, env, weight: float, std: float = 0.35, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.std = float(std)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        env._depsic_heading_reward = self
        # Reward reset callbacks normally run before mjlab's reset-state
        # forward pass. Heading is read from a derived body quaternion, so
        # schedule the same reset once more in the post-forward observation
        # phase; the second write is the authoritative one.
        env._observation_reset_callbacks.append(self.reset)

    def reset(self, env_ids: torch.Tensor) -> None:
        pelvis = self.asset.data.body_link_quat_w[env_ids, self.command_manager.pelvis_id]
        # Project the wxyz quaternion onto the horizontal plane.  The
        # denominator is 1-2(y^2+z^2); using w^2 here (an earlier prototype
        # did so) makes the initial target wrong even for a level pelvis.
        self.heading_target[env_ids] = torch.atan2(
            2.0 * (pelvis[:, 0] * pelvis[:, 3] + pelvis[:, 1] * pelvis[:, 2]),
            1.0 - 2.0 * (pelvis[:, 2].square() + pelvis[:, 3].square()),
        )

    def _compute(self) -> torch.Tensor:
        self.heading_target = wrap_to_pi(
            self.heading_target + self.command_manager.lbc[:, 2] * self.env.step_dt
        )
        pelvis = self.asset.data.body_link_quat_w[:, self.command_manager.pelvis_id]
        heading = torch.atan2(
            2.0 * (pelvis[:, 0] * pelvis[:, 3] + pelvis[:, 1] * pelvis[:, 2]),
            1.0 - 2.0 * (pelvis[:, 2].square() + pelvis[:, 3].square()),
        )
        error = wrap_to_pi(heading - self.heading_target)
        return torch.exp(-((error / self.std).square())).unsqueeze(-1)


class depsic_track_lin_vel_direction(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        command = self.command_manager.lbc[:, :2]
        velocity = self.command_manager.velocity_ema_xy
        command_norm = command.norm(dim=-1)
        velocity_norm = velocity.norm(dim=-1)
        cosine = (command * velocity).sum(dim=-1) / (command_norm * velocity_norm).clamp_min(1.0e-6)
        valid = (command_norm >= 0.05) & (velocity_norm >= 0.05)
        return torch.where(valid, cosine.clamp_min(0.0), torch.zeros_like(cosine)).unsqueeze(-1)


class depsic_track_height(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        height = self.asset.data.body_link_pos_w[:, self.command_manager.torso_id, 2]
        return torch.exp(-((height - self.command_manager.lbc[:, 6]) / 0.1).square()).unsqueeze(-1)


class depsic_track_torso_orientation(_DepsicReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, axis: int, std: float = 0.1,
                 broad_std: float | None = None, broad_fraction: float = 0.5, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be roll=0, pitch=1, or yaw=2")
        self.axis = axis
        self.std = float(std)
        self.broad_std = None if broad_std is None else float(broad_std)
        self.broad_fraction = float(broad_fraction)
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Orientation std must be finite and positive")
        if self.broad_std is not None and (not math.isfinite(self.broad_std) or self.broad_std < self.std):
            raise ValueError("Broad orientation std must be finite and >= fine std")
        if not 0 <= self.broad_fraction <= 1:
            raise ValueError("Broad orientation fraction must lie in [0, 1]")

    def _compute(self) -> torch.Tensor:
        rpy = _lbc_rpy(self.asset, self.command_manager.torso_id, self.command_manager.pelvis_id)
        error = wrap_to_pi(rpy[:, self.axis] - self.command_manager.lbc[:, self.axis + 3])
        fine = torch.exp(-(error / self.std).square())
        if self.broad_std is None:
            return fine.unsqueeze(-1)
        broad = torch.exp(-(error / self.broad_std).square())
        return ((1.0 - self.broad_fraction) * fine + self.broad_fraction * broad).unsqueeze(-1)


class depsic_arm_joint_pos(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        error = self.asset.data.joint_pos[:, self.command_manager.arm_joint_ids] - self.command_manager.arm_q
        return torch.exp(-error.square().mean(dim=-1, keepdim=True) / 0.25**2)


class depsic_arm_joint_vel(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        error = self.asset.data.joint_vel[:, self.command_manager.arm_joint_ids] - self.command_manager.arm_dq
        return torch.exp(-error.square().mean(dim=-1, keepdim=True) / 1.0**2)


class depsic_arm_wrist_pos(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        std: float = 0.1,
        torso_body_name: str = "torso_link",
        wrist_body_names: list[str] | tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        narrow_std: float | None = None,
        narrow_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.std = float(std)
        self.narrow_std = None if narrow_std is None else float(narrow_std)
        self.narrow_fraction = float(narrow_fraction)
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Wrist position std must be finite and positive")
        if self.narrow_std is not None and not 0 < self.narrow_std < self.std:
            raise ValueError("Wrist narrow_std must be positive and smaller than std")
        if not 0 <= self.narrow_fraction <= 1:
            raise ValueError("Wrist narrow_fraction must be in [0, 1]")
        self.torso_id, _ = find_bodies(self.asset, [torso_body_name])
        self.wrist_ids, _ = find_bodies(self.asset, list(wrist_body_names))

    def _compute(self) -> torch.Tensor:
        torso_pos = self.asset.data.body_link_pos_w[:, self.torso_id[0]]
        torso_quat = self.asset.data.body_link_quat_w[:, self.torso_id[0], None]
        wrists = self.asset.data.body_link_pos_w[:, self.wrist_ids]
        actual = quat_rotate_inverse(torso_quat, wrists - torso_pos[:, None])
        error_sq = (actual - self.command_manager.wrist_pos_torso).square().sum(dim=-1).mean(dim=-1)
        broad = torch.exp(-error_sq[:, None] / self.std**2)
        if self.narrow_std is None:
            return broad
        narrow = torch.exp(-error_sq[:, None] / self.narrow_std**2)
        return (1 - self.narrow_fraction) * broad + self.narrow_fraction * narrow


class depsic_arm_wrist_ori(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        std: float = 0.4,
        torso_body_name: str = "torso_link",
        wrist_body_names: list[str] | tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        narrow_std: float | None = None,
        narrow_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.std = float(std)
        self.narrow_std = None if narrow_std is None else float(narrow_std)
        self.narrow_fraction = float(narrow_fraction)
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Wrist orientation std must be finite and positive")
        if self.narrow_std is not None and not 0 < self.narrow_std < self.std:
            raise ValueError("Wrist narrow_std must be positive and smaller than std")
        if not 0 <= self.narrow_fraction <= 1:
            raise ValueError("Wrist narrow_fraction must be in [0, 1]")
        self.torso_id, _ = find_bodies(self.asset, [torso_body_name])
        self.wrist_ids, _ = find_bodies(self.asset, list(wrist_body_names))

    def _compute(self) -> torch.Tensor:
        torso_quat = self.asset.data.body_link_quat_w[:, self.torso_id[0], None]
        wrists = self.asset.data.body_link_quat_w[:, self.wrist_ids]
        actual = quat_mul(quat_conjugate(torso_quat), wrists)
        actual_matrix = matrix_from_quat(actual)
        relative = self.command_manager.wrist_rot_torso.transpose(-1, -2) @ actual_matrix
        angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0))
        error_sq = angle.square().mean(dim=-1, keepdim=True)
        broad = torch.exp(-error_sq / self.std**2)
        if self.narrow_std is None:
            return broad
        narrow = torch.exp(-error_sq / self.narrow_std**2)
        return (1 - self.narrow_fraction) * broad + self.narrow_fraction * narrow


class depsic_energy_l2(_DepsicReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, joint_names: str | list[str] | None = None, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.joint_ids = None
        self.ctrl_local_ids = None
        if joint_names is not None:
            joint_ids, resolved = find_joints(self.asset, joint_names)
            self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
            actuator_names = list(self.asset.actuator_names)
            missing = [name for name in resolved if name not in actuator_names]
            if missing:
                raise RuntimeError(f"DEPSIC energy reward has no actuator for joints: {missing}")
            self.ctrl_local_ids = torch.tensor(
                [actuator_names.index(name) for name in resolved], dtype=torch.long, device=self.device
            )

    def _compute(self) -> torch.Tensor:
        torque = getattr(self.asset.data, "actuator_force", None)
        if torque is None:
            torque = self.asset.data.applied_torque
        if self.joint_ids is not None:
            torque = torque[:, self.ctrl_local_ids]
            velocity = self.asset.data.joint_vel[:, self.joint_ids]
        else:
            velocity = self.asset.data.joint_vel
        power = (torque * velocity).abs()
        return power.norm(dim=-1, keepdim=True)


class depsic_action_rate_l2(_DepsicReward, namespace="mimic_lite"):
    """Action delta squared; optionally select joints and scale to target radians.

    Target deltas refer to commanded targets before actuator delay, not
    measured joint velocity. Defaults preserve the original whole-body cost.
    """

    def __init__(self, env, weight: float, joint_names: str | list[str] | None = None,
                 use_target_delta: bool = False, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.action_manager = env.input_managers["action"]
        self.action_ids = slice(None)
        self.target_scale = None
        if joint_names is not None:
            patterns = [joint_names] if isinstance(joint_names, str) else list(joint_names)
            names = self.action_manager.joint_names
            matches = [[i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
                       for pattern in patterns]
            if not patterns or any(not ids for ids in matches):
                raise ValueError(f"Action-rate patterns must match policy joints: {patterns}")
            # Select in POLICY action order, never full-asset joint order.
            self.action_ids = torch.tensor(sorted({i for ids in matches for i in ids}),
                                           dtype=torch.long, device=self.device)
        if use_target_delta:
            scale = torch.as_tensor(self.action_manager.action_scaling, device=self.device)
            if scale.ndim != 1 or scale.numel() != self.action_manager.action_buf.shape[-1]:
                raise ValueError("Target action-rate requires one scale per policy action")
            self.target_scale = scale[self.action_ids].detach().clone()
            if not bool((torch.isfinite(self.target_scale) & (self.target_scale > 0)).all()):
                raise ValueError("Target action-rate scales must be finite and positive")

    def _compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        delta = (action_buf[:, 0] - action_buf[:, 1])[:, self.action_ids]
        if self.target_scale is not None:
            delta = delta * self.target_scale
        return delta.square().sum(dim=-1, keepdim=True)


class depsic_joint_acc_l2(_DepsicReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, joint_names: str | list[str], **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        ids, _ = find_joints(self.asset, joint_names)
        self.joint_ids = torch.tensor(ids, device=self.device)

    def _compute(self) -> torch.Tensor:
        return self.asset.data.joint_acc[:, self.joint_ids].square().sum(dim=-1, keepdim=True)


class depsic_vertical_velocity(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        # GR00T's legacy root_lin_vel_b alias resolves to root-COM velocity.
        return self.asset.data.root_com_lin_vel_b[:, 2:3].square()


class depsic_root_ang_vel_xy(_DepsicReward, namespace="mimic_lite"):
    """A3 V1 root roll/pitch angular-velocity cost; yaw remains commanded."""

    def _compute(self) -> torch.Tensor:
        return self.asset.data.root_com_ang_vel_b[:, :2].square().sum(dim=-1, keepdim=True)


class depsic_anti_shake_body_ang_vel(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        threshold: float = 1.5,
        body_names: list[str] | tuple[str, ...] = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        ids, _ = find_bodies(self.asset, list(body_names))
        self.body_ids = torch.tensor(ids, device=self.device)
        self.threshold = float(threshold)

    def _compute(self) -> torch.Tensor:
        magnitude = self.asset.data.body_com_ang_vel_w[:, self.body_ids].norm(dim=-1)
        # IsaacLab's DEPSIC term averages the selected torso/wrist links.
        return (magnitude - self.threshold).clamp_min(0.0).square().mean(dim=-1, keepdim=True)


class depsic_undesired_contacts(_DepsicReward, namespace="mimic_lite"):
    """Count non-support terrain contacts over the sensor history window."""

    def __init__(
        self,
        env,
        weight: float,
        threshold: float = 1.0,
        body_names: str | list[str] | None = None,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.contact_sensor = env.scene.sensors["contact_forces"]
        # This is the same exclusion set as DEPSIC's source profile: feet
        # support normally, while the listed arm links may intentionally touch.
        names = body_names or [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
            r"(?!left_elbow_link$)(?!right_elbow_link$).+$"
        ]
        ids, _ = find_sensor_bodies(self.asset, self.contact_sensor, names)
        self.sensor_ids = torch.tensor(ids, device=self.device)
        self.threshold = float(threshold)

    def _compute(self) -> torch.Tensor:
        return _contact_mask(self.contact_sensor, self.sensor_ids, self.threshold).sum(
            dim=-1, keepdim=True
        ).float()


class _DepsicFootReward(_DepsicReward):
    def __init__(
        self,
        env,
        weight: float,
        threshold: float = 1.0,
        foot_body_names: list[str] | tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.contact_sensor = env.scene.sensors["contact_forces"]
        self.sensor_ids, foot_names = find_sensor_bodies(self.asset, self.contact_sensor, list(foot_body_names))
        self.sensor_ids = torch.tensor(self.sensor_ids, device=self.device)
        self.body_ids, _ = find_bodies(self.asset, foot_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.threshold = float(threshold)

    def _forces(self) -> torch.Tensor:
        return _contact_force_history(self.contact_sensor, self.sensor_ids)

    def _in_contact(self) -> torch.Tensor:
        return _contact_mask(self.contact_sensor, self.sensor_ids, self.threshold)


class depsic_foot_slide(_DepsicFootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        velocity = self.asset.data.body_link_lin_vel_w[:, self.body_ids, :2].norm(dim=-1)
        return (velocity * self._in_contact()).sum(dim=-1, keepdim=True)


class depsic_flying(_DepsicFootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        return (~self._in_contact().any(dim=-1, keepdim=True)).float()


class depsic_feet_force(_DepsicFootReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, force_threshold: float = 500.0, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.force_threshold = float(force_threshold)

    def _compute(self) -> torch.Tensor:
        # Both simulators keep the newest contact sample at history index 0.
        vertical_force = self._forces()[:, :, 0, 2]
        force_norm = vertical_force.norm(dim=-1, keepdim=True)
        return (force_norm - self.force_threshold).clamp_min(0.0).clamp_max(400.0)


class depsic_feet_air_time(_DepsicFootReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, max_air_time: float = 0.4, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.max_air_time = float(max_air_time)

    def _compute(self) -> torch.Tensor:
        data = self.contact_sensor.data
        contact_time = getattr(data, "current_contact_time", None)
        air_time = getattr(data, "current_air_time", None)
        if contact_time is None or air_time is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        contact_time = contact_time[:, self.sensor_ids]
        air_time = air_time[:, self.sensor_ids]
        contact = contact_time > 0.0
        single_stance = contact.sum(dim=-1) == 1
        mode_time = torch.where(contact, contact_time, air_time)
        reward = torch.where(single_stance[:, None], mode_time, torch.zeros_like(mode_time)).amin(dim=-1)
        moving = self.command_manager.lbc[:, :2].norm(dim=-1) + self.command_manager.lbc[:, 2].abs() > 0.1
        return (reward.clamp_max(self.max_air_time) * moving).unsqueeze(-1)


class depsic_feet_stumble(_DepsicFootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        force = self._forces()[:, :, 0]
        horizontal = force[..., :2].norm(dim=-1)
        return (horizontal > 5.0 * force[..., 2].abs()).any(dim=-1, keepdim=True).float()


class depsic_feet_contact_standing_stable(_DepsicFootReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, min_contact_time: float = 0.5, command_threshold: float = 0.1, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.min_contact_time = float(min_contact_time)
        self.command_threshold = float(command_threshold)

    def _compute(self) -> torch.Tensor:
        contact_time = getattr(self.contact_sensor.data, "current_contact_time", None)
        if contact_time is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        contact_time = contact_time[:, self.sensor_ids]
        standing = (
            self.command_manager.lbc[:, :2].norm(dim=-1)
            + self.command_manager.lbc[:, 2].abs()
            <= self.command_threshold
        )
        stable = (contact_time > self.min_contact_time).sum(dim=-1) == 2
        reward = torch.tanh(contact_time.amin(dim=-1) / self.min_contact_time)
        return torch.where(standing & stable, reward, torch.zeros_like(reward)).unsqueeze(-1)


class depsic_joint_limits(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        joint_names: str | list[str] | None = None,
        factor_overrides: Mapping[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        limits = self.asset.data.joint_pos_limits
        center = limits.mean(dim=-1)
        # Per-joint soft factor; ``factor_overrides`` maps joint-name patterns
        # to a narrower factor (SONIC-A3 035 uses 0.8 on the waist).
        factor = torch.full_like(center, SOFT_JOINT_POS_LIMIT_FACTOR)
        for pattern, value in (factor_overrides or {}).items():
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"soft joint limit factor must be in (0, 1], got {value}")
            ids, _ = find_joints(self.asset, pattern)
            if not ids:
                raise ValueError(f"depsic_joint_limits factor override matched no joint: {pattern!r}")
            factor[:, ids] = float(value)
        half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * factor
        self.soft_lower = center - half_range
        self.soft_upper = center + half_range
        self.joint_ids = None
        if joint_names is not None:
            ids, _ = find_joints(self.asset, joint_names)
            self.joint_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

    def _compute(self) -> torch.Tensor:
        # Match IsaacLab's joint_pos_limits term used by GR00T 321f91b.
        position = self.asset.data.joint_pos
        lower, upper = self.soft_lower, self.soft_upper
        if self.joint_ids is not None:
            position = position[:, self.joint_ids]
            lower, upper = lower[:, self.joint_ids], upper[:, self.joint_ids]
        below = (lower - position).clamp_min(0.0)
        above = (position - upper).clamp_min(0.0)
        return (below + above).sum(dim=-1, keepdim=True)


class depsic_flat_orientation(_DepsicReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        return self.asset.data.projected_gravity_b[:, :2].square().sum(dim=-1, keepdim=True)


class depsic_feet_distance(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        threshold: float = 0.2,
        foot_body_names: list[str] | tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        ids, _ = find_bodies(self.asset, list(foot_body_names))
        self.body_ids = torch.tensor(ids, device=self.device)
        self.threshold = float(threshold)

    def _compute(self) -> torch.Tensor:
        feet = self.asset.data.body_link_pos_w[:, self.body_ids]
        return (self.threshold - (feet[:, 0] - feet[:, 1]).norm(dim=-1)).clamp_min(0.0).unsqueeze(-1)


class depsic_knee_stance_width(_DepsicReward, namespace="mimic_lite"):
    """Reward an expert-like knee stance, optionally conditioned on height.

    ``ideal_width`` is retained as the configuration name for compatibility,
    and ``lower_bound_only`` optionally treats it as a minimum acceptable knee
    separation.  The latter lets the AMP squat prior use the wider stance
    present in its experts without changing the ordinary V1 baseline.

    When ``height_conditioned`` is enabled, a nearly stationary command uses
    a target interpolating between a narrow standing stance and a wider squat
    stance. Moving commands retain the ordinary lower-bound semantics, so
    locomotion and turns keep their anti-collapse constraint without being
    pushed toward a squat-width stance.

    ``lock_standing_width`` (default True) is the ordinary AMP standing lock:
    a still command at standing height is pulled to ``standing_width``. Set
    it False to keep that standing command on the lower-bound path and only
    apply the interpolated wide target when the commanded height is a squat.
    """
    def __init__(
        self,
        env,
        weight: float,
        ideal_width: float = 0.27,
        std: float = 0.1,
        lower_bound_only: bool = False,
        height_conditioned: bool = False,
        standing_height: float = 1.05,
        squat_height: float = 0.85,
        standing_width: float = 0.35,
        squat_width: float = 0.53,
        speed_threshold: float = 0.10,
        lock_standing_width: bool = True,
        posture_threshold: float | None = None,
        moving_std: float | None = None,
        knee_body_names: list[str] | tuple[str, str] = ("left_knee_link", "right_knee_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        ids, _ = find_bodies(self.asset, list(knee_body_names))
        self.body_ids = torch.tensor(ids, device=self.device)
        self.ideal_width = float(ideal_width)
        self.std = float(std)
        self.lower_bound_only = bool(lower_bound_only)
        self.height_conditioned = bool(height_conditioned)
        self.standing_height = float(standing_height)
        self.squat_height = float(squat_height)
        self.standing_width = float(standing_width)
        self.squat_width = float(squat_width)
        self.speed_threshold = float(speed_threshold)
        self.lock_standing_width = bool(lock_standing_width)
        self.posture_threshold = (
            None if posture_threshold is None else float(posture_threshold)
        )
        self.moving_std = self.std if moving_std is None else float(moving_std)
        if self.std <= 0.0:
            raise ValueError("DEPSIC knee stance std must be positive")
        if self.moving_std <= 0.0:
            raise ValueError("DEPSIC moving knee stance std must be positive")
        if self.standing_height <= self.squat_height:
            raise ValueError("standing_height must be greater than squat_height")
        if self.speed_threshold < 0.0:
            raise ValueError("speed_threshold must be non-negative")
        if self.posture_threshold is not None and self.posture_threshold < 0.0:
            raise ValueError("posture_threshold must be non-negative")

    def _compute(self) -> torch.Tensor:
        knees = self.asset.data.body_link_pos_w[:, self.body_ids, :2]
        separation = (knees[:, 0] - knees[:, 1]).norm(dim=-1)
        if self.height_conditioned:
            command = self.command_manager.lbc
            # Use the requested height, rather than the current torso height:
            # otherwise the reward would ask for a wide stance only after the
            # robot had already crouched, providing no gradient that initiates
            # the crouch in the first place.
            height = command[:, 6]
            speed = torch.sqrt(
                command[:, 0].square()
                + command[:, 1].square()
                + (command[:, 2] / 1.57 * 0.55).square()
            )
            # The measured squat experts widen their knees as torso height
            # falls.  Clamp the interpolation so out-of-range commands cannot
            # request an unbounded stance.
            fraction = ((height - self.squat_height) / (self.standing_height - self.squat_height)).clamp(0.0, 1.0)
            target = self.squat_width + fraction * (self.standing_width - self.squat_width)
            error_sq = (separation - target).square()
            reward = torch.exp(-error_sq / self.std**2)
            use_height_target = speed < self.speed_threshold
            if not self.lock_standing_width:
                # Standing is anti-collapse only. The interpolated wide
                # target is the squat prior that initiates a crouch.
                use_height_target = use_height_target & (height < self.standing_height)
            posture_threshold = getattr(self, "posture_threshold", None)
            if posture_threshold is not None:
                # jikw1gbo: still + commanded torso yaw is not a standing-width
                # sample. Default yaml leaves this unset.
                posture = command[:, 3:6].abs().amax(dim=-1)
                use_height_target = use_height_target & (
                    posture < posture_threshold
                )
            # The historical AMP/V1 semantics remain in force while moving:
            # all widths at/above ``ideal_width`` receive the same maximum
            # reward, and only an inward-collapsed stance is penalized.
            moving_error_sq = (self.ideal_width - separation).clamp_min(0.0).square()
            moving_reward = torch.exp(-moving_error_sq / self.moving_std**2)
            return torch.where(use_height_target, reward, moving_reward).unsqueeze(-1)
        if self.lower_bound_only:
            # One-sided lower-bound penalty.  Widths at/above the threshold
            # all receive the same maximum reward; only a collapsed stance is
            # penalized.  This avoids opposing a wide AMP squat.
            error_sq = (self.ideal_width - separation).clamp_min(0.0).square()
        else:
            error_sq = (separation - self.ideal_width).square()
        return torch.exp(-error_sq[:, None] / self.std**2)


class depsic_joint_torque_limits(_DepsicReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, joint_names: str | list[str], soft_torque_limit: float, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        _, resolved_joint_names = find_joints(self.asset, joint_names)
        actuator_names = list(self.asset.actuator_names)
        ctrl_ids = torch.as_tensor(
            self.asset.indexing.ctrl_ids,
            dtype=torch.long,
            device=self.device,
        )
        if len(actuator_names) != int(ctrl_ids.numel()):
            raise RuntimeError(
                "DEPSIC torque-limit reward expected one actuator name per control id, "
                f"but got {len(actuator_names)} names and {int(ctrl_ids.numel())} ids"
            )

        name_to_local_ctrl = {name: index for index, name in enumerate(actuator_names)}
        missing = [name for name in resolved_joint_names if name not in name_to_local_ctrl]
        if missing:
            raise RuntimeError(
                "DEPSIC torque-limit reward has no mjlab actuator for joints: "
                f"{missing}"
            )

        self.ctrl_local_ids = torch.tensor(
            [name_to_local_ctrl[name] for name in resolved_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        selected_ctrl_ids = ctrl_ids[self.ctrl_local_ids]
        force_ranges = self.env.sim.get_default_field("actuator_forcerange").to(self.device)
        self.limits = force_ranges[selected_ctrl_ids].abs().amax(dim=-1).clamp_min(1.0e-6)
        self.soft_torque_limit = float(soft_torque_limit)

    def _compute(self) -> torch.Tensor:
        # actuator_force is entity-local and follows asset.indexing.ctrl_ids.
        torque = self.asset.data.actuator_force[:, self.ctrl_local_ids]
        normalized = torque.abs() / self.limits
        return (normalized - self.soft_torque_limit).clamp_min(0.0).sum(dim=-1, keepdim=True)


class depsic_center_of_gravity_tracking(_DepsicReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.2,
        foot_body_names: list[str] | tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        foot_ids, _ = find_bodies(self.asset, list(foot_body_names))
        self.foot_ids = torch.tensor(foot_ids, device=self.device)
        self.global_body_ids = self.asset.indexing.body_ids.to(device=self.device)
        self.sigma = float(sigma)

    def _compute(self) -> torch.Tensor:
        masses = self.env.sim.model.body_mass[:, self.global_body_ids]
        body_com = self.asset.data.body_com_pos_w
        cog_xy = (body_com * masses[..., None]).sum(dim=1)[:, :2]
        cog_xy = cog_xy / masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        ankles_xy = self.asset.data.body_link_pos_w[:, self.foot_ids, :2].mean(dim=1)
        error_sq = (cog_xy - ankles_xy).square().sum(dim=-1)
        return torch.exp(-error_sq[:, None] / self.sigma**2)


class depsic_joint_deviation(_DepsicReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, joint_names: str | list[str], **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        ids, _ = find_joints(self.asset, joint_names)
        self.joint_ids = torch.tensor(ids, device=self.device)

    def _compute(self) -> torch.Tensor:
        error = self.asset.data.joint_pos[:, self.joint_ids] - self.asset.data.default_joint_pos[:, self.joint_ids]
        return error.abs().sum(dim=-1, keepdim=True)


class _DepsicTermination(Termination[DepsicG1RobotOnly]):
    pass


class depsic_torso_contact(_DepsicTermination, namespace="mimic_lite"):
    def __init__(self, env, threshold: float = 1.0, torso_body_name: str = "torso_link", **kwargs):
        super().__init__(env, **kwargs)
        sensor = env.scene.sensors["contact_forces"]
        self.sensor = sensor
        ids, _ = find_sensor_bodies(env.scene.articulations["robot"], sensor, [torso_body_name])
        self.sensor_ids = torch.tensor(ids, device=self.device)
        self.threshold = float(threshold)

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        del termination
        return _contact_mask(self.sensor, self.sensor_ids, self.threshold).any(dim=-1, keepdim=True)


class depsic_wrist_anchor_height(_DepsicTermination, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        threshold: float = 0.15,
        down_threshold: float = 0.75,
        height_threshold: float = 0.5,
        torso_body_name: str = "torso_link",
        pelvis_body_name: str = "pelvis",
        wrist_body_names: list[str] | tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        asset = env.scene.articulations["robot"]
        self.asset = asset
        self.torso_id, _ = find_bodies(asset, [torso_body_name])
        self.pelvis_id, _ = find_bodies(asset, [pelvis_body_name])
        self.wrist_ids, _ = find_bodies(asset, list(wrist_body_names))
        self.threshold = float(threshold)
        self.down_threshold = float(down_threshold)
        self.height_threshold = float(height_threshold)

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        del termination
        return _wrist_anchor_height_mask(
            self.asset,
            self.command_manager,
            self.torso_id[0],
            self.pelvis_id[0],
            self.wrist_ids,
            self.threshold,
            self.down_threshold,
            self.height_threshold,
        )


class depsic_termination_penalty(_DepsicReward, namespace="mimic_lite"):
    """Mirror DEPSIC's -200 terminal term before MimicLite computes done flags."""

    def __init__(
        self,
        env,
        weight: float,
        torso_contact_threshold: float = 1.0,
        wrist_threshold: float = 0.15,
        wrist_down_threshold: float = 0.75,
        wrist_height_threshold: float = 0.5,
        torso_body_name: str = "torso_link",
        pelvis_body_name: str = "pelvis",
        wrist_body_names: list[str] | tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        sensor = env.scene.sensors["contact_forces"]
        torso_ids, _ = find_sensor_bodies(self.asset, sensor, [torso_body_name])
        self.sensor = sensor
        self.torso_sensor_ids = torch.tensor(torso_ids, device=self.device)
        torso_ids, _ = find_bodies(self.asset, [torso_body_name])
        pelvis_ids, _ = find_bodies(self.asset, [pelvis_body_name])
        wrist_ids, _ = find_bodies(self.asset, list(wrist_body_names))
        self.torso_id = torso_ids[0]
        self.pelvis_id = pelvis_ids[0]
        self.wrist_ids = torch.tensor(wrist_ids, device=self.device)
        self.torso_contact_threshold = float(torso_contact_threshold)
        self.wrist_threshold = float(wrist_threshold)
        self.wrist_down_threshold = float(wrist_down_threshold)
        self.wrist_height_threshold = float(wrist_height_threshold)

    def _compute(self) -> torch.Tensor:
        torso_contact = _contact_mask(
            self.sensor, self.torso_sensor_ids, self.torso_contact_threshold
        ).any(dim=-1, keepdim=True)
        wrist_error = _wrist_anchor_height_mask(
            self.asset,
            self.command_manager,
            self.torso_id,
            self.pelvis_id,
            self.wrist_ids,
            self.wrist_threshold,
            self.wrist_down_threshold,
            self.wrist_height_threshold,
        )
        return (torso_contact | wrist_error).float()


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    t = xyz.cross(vector, dim=-1) * 2.0
    return vector + quat[..., :1] * t + xyz.cross(t, dim=-1)


__all__ = [
    "DepsicG1RobotOnly",
]
