"""DEPSIC V2 full-SMPL command and observations.

This module follows GR00T's current G1 IsaacLab V2 route exactly.  The
deployable student sees an independently sampled LBC plan together with a
synchronized 10 x 20-ms SMPL two-arm horizon.  The former 97-D current
three-point/anchor route is intentionally absent: it is an incompatible,
superseded checkpoint contract.
"""

from __future__ import annotations

import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.utils.math import (
    euler_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)

from mimic_lite.depsic_v2_contract import (
    LBC_DIM,
    MIXED_REFERENCE_DIM,
    REFERENCE_FRAMES,
    SMPL_ARM_POS_DIM,
    SMPL_WRIST_QUAT_DIM,
    V1_REFERENCE_DIM,
)
from mimic_lite.tasks.depsic import DepsicG1RobotOnly
from mimic_lite.tasks.depsic_paired_smpl import (
    PairedSmplReferencePool,
    interpolate_smpl_world_signals,
    interpolate_smpl_world_trackers,
    load_paired_smpl_reference_pool,
    quat_to_column_rot6,
)


class _DepsicV2TrackerMixin:
    """Shared full-SMPL V2 reference state for G1 and A3.

    The existing V1 arm PKL remains the single owner of clip selection and
    continuous time.  Its source coordinates drive both teacher labels and
    the paired SMPL fields, so a partial reset cannot mix clips or phases.
    """

    _vr_nominal_height: float
    _vr_rpy_min: tuple[float, float, float]
    _vr_rpy_max: tuple[float, float, float]
    _vr_height_range: tuple[float, float]

    def _init_v2_tracker(
        self,
        *,
        smpl_motion_file: str,
        smpl_cache_root: str | None,
        nominal_height: float,
        rpy_min: tuple[float, float, float],
        rpy_max: tuple[float, float, float],
        height_range: tuple[float, float],
        mixed_future_frame_dt: float = 0.02,
    ) -> None:
        if mixed_future_frame_dt <= 0.0:
            raise ValueError("DEPSIC V2 mixed_future_frame_dt must be positive.")
        self.smpl_pool: PairedSmplReferencePool = load_paired_smpl_reference_pool(
            self.pool.paths,
            self.pool.lengths,
            self.pool.fps,
            smpl_motion_file=smpl_motion_file,
            cache_root=smpl_cache_root,
        )
        self.smpl_tracker_pos_data = self.smpl_pool.tracker_pos_w.to(self.device)
        self.smpl_tracker_quat_data = self.smpl_pool.tracker_quat_w.to(self.device)
        self.smpl_arm_pos_data = self.smpl_pool.arm_pos_w.to(self.device)
        self.smpl_wrist_quat_data = self.smpl_pool.wrist_quat_w.to(self.device)
        self.smpl_root_pos_data = self.smpl_pool.root_pos_w.to(self.device)
        self.smpl_root_quat_data = self.smpl_pool.root_quat_w.to(self.device)
        self.vr_tracker_pos_root = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.vr_tracker_quat_root = torch.zeros(self.num_envs, 3, 4, device=self.device)
        self.vr_tracker_quat_root[..., 0] = 1.0
        self.vr_rpy_height = torch.zeros(self.num_envs, 4, device=self.device)
        self.vr_rpy_height[:, 3] = float(nominal_height)
        self.smpl_arm_future_root = torch.zeros(
            self.num_envs, REFERENCE_FRAMES, SMPL_ARM_POS_DIM, device=self.device
        )
        self.smpl_wrist_quat_future_root = torch.zeros(
            self.num_envs, REFERENCE_FRAMES, SMPL_WRIST_QUAT_DIM, device=self.device
        )
        self.smpl_wrist_quat_future_root[..., 0] = 1.0
        self.smpl_wrist_quat_future_root[..., 4] = 1.0
        self._vr_head_quat_baseline = torch.zeros(self.num_envs, 4, device=self.device)
        self._vr_head_height_baseline = torch.zeros(self.num_envs, device=self.device)
        self._vr_has_calibration = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._vr_nominal_height = float(nominal_height)
        self._vr_rpy_min = tuple(float(value) for value in rpy_min)
        self._vr_rpy_max = tuple(float(value) for value in rpy_max)
        self._vr_height_range = tuple(float(value) for value in height_range)
        self.mixed_future_frame_dt = float(mixed_future_frame_dt)
        self.smpl_cache_path = str(self.smpl_pool.cache_path)
        self.smpl_manifest_hash = self.smpl_pool.manifest_hash

    def _interpolate_v2_trackers(
        self, clip_ids: torch.Tensor, source_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_frames.ndim != 2 or source_frames.shape[0] != clip_ids.numel():
            raise ValueError(
                "DEPSIC V2 tracker interpolation expects clip_ids [N] and source_frames [N,H]."
            )
        last = (self.lengths[clip_ids] - 1)[:, None]
        starts = self.clip_starts[clip_ids][:, None]
        source = torch.minimum(source_frames.clamp_min(0.0), last.to(source_frames.dtype))
        lower = torch.floor(source).long()
        upper = torch.minimum(lower + 1, last)
        return interpolate_smpl_world_trackers(
            tracker_pos_w=self.smpl_tracker_pos_data,
            tracker_quat_w=self.smpl_tracker_quat_data,
            root_pos_w=self.smpl_root_pos_data,
            root_quat_w=self.smpl_root_quat_data,
            starts=starts,
            lower=lower,
            upper=upper,
            alpha=source - lower,
        )

    def _interpolate_v2_full_smpl(
        self, clip_ids: torch.Tensor, source_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_frames.ndim != 2 or source_frames.shape[0] != clip_ids.numel():
            raise ValueError(
                "DEPSIC V2 full-SMPL interpolation expects clip_ids [N] and source_frames [N,H]."
            )
        last = (self.lengths[clip_ids] - 1)[:, None]
        starts = self.clip_starts[clip_ids][:, None]
        source = torch.minimum(source_frames.clamp_min(0.0), last.to(source_frames.dtype))
        lower = torch.floor(source).long()
        upper = torch.minimum(lower + 1, last)
        # ``head_quat_w`` is unused by the full-SMPL branch.  Passing the
        # tracker head preserves the public helper's compact, tested API.
        arm_root, wrist_root, _ = interpolate_smpl_world_signals(
            arm_pos_w=self.smpl_arm_pos_data,
            wrist_quat_w=self.smpl_wrist_quat_data,
            head_quat_w=self.smpl_tracker_quat_data[:, 2],
            root_pos_w=self.smpl_root_pos_data,
            root_quat_w=self.smpl_root_quat_data,
            starts=starts,
            lower=lower,
            upper=upper,
            alpha=source - lower,
        )
        return arm_root, wrist_root, source

    def _refresh_v2_full_smpl(self) -> None:
        offsets = torch.arange(
            REFERENCE_FRAMES, device=self.device, dtype=self.frame_fraction.dtype
        )
        source = (
            self.frame_ids[:, None].to(self.frame_fraction.dtype)
            + self.frame_fraction[:, None]
            + offsets[None] * self.mixed_future_frame_dt * self.fps[self.clip_ids, None]
        )
        arm_root, wrist_root, _ = self._interpolate_v2_full_smpl(self.clip_ids, source)
        self.smpl_arm_future_root.copy_(arm_root.flatten(start_dim=2))
        self.smpl_wrist_quat_future_root.copy_(wrist_root.flatten(start_dim=2))

    def _refresh_v2_tracker(self, env_ids: torch.Tensor, *, calibrate: bool = False) -> None:
        """Refresh only tracker-derived physical posture/head state.

        Trackers never enter either V2 token.  A3 alone also uses the current
        root-local head orientation to drive its separate physical head PD
        target through a fitted projector.
        """
        if len(env_ids) == 0:
            return
        source = (
            self.frame_ids[env_ids].to(self.frame_fraction.dtype)
            + self.frame_fraction[env_ids]
        )[:, None]
        points_root, quat_root, root_pos, root_quat = self._interpolate_v2_trackers(
            self.clip_ids[env_ids], source
        )
        self.vr_tracker_pos_root[env_ids] = points_root[:, 0]
        self.vr_tracker_quat_root[env_ids] = quat_root[:, 0]
        root_for_head = root_quat[:, 0]
        head_world = quat_mul(root_for_head, quat_root[:, 0, 2])
        head_height = (
            root_pos[:, 0] + quat_rotate(root_for_head, points_root[:, 0, 2])
        )[:, 2]
        if calibrate:
            self._vr_head_quat_baseline[env_ids] = head_world
            self._vr_head_height_baseline[env_ids] = head_height
            self._vr_has_calibration[env_ids] = True
        pending = ~self._vr_has_calibration[env_ids]
        if pending.any():
            rows = env_ids[pending]
            self._vr_head_quat_baseline[rows] = head_world[pending]
            self._vr_head_height_baseline[rows] = head_height[pending]
            self._vr_has_calibration[rows] = True
        head_delta = quat_mul(quat_conjugate(self._vr_head_quat_baseline[env_ids]), head_world)
        rpy = euler_from_quat(head_delta)
        rpy = torch.maximum(
            torch.minimum(rpy, rpy.new_tensor(self._vr_rpy_max)), rpy.new_tensor(self._vr_rpy_min)
        )
        height = (
            self._vr_nominal_height + head_height - self._vr_head_height_baseline[env_ids]
        ).clamp(*self._vr_height_range)
        self.vr_rpy_height[env_ids] = torch.cat((rpy, height[:, None]), dim=-1)

    def _v2_mixed_reference(self) -> torch.Tensor:
        """Serialize current HEAD's ``LBC70 + 10 x [arm18 | wrist8]``.

        The outer LBC block is exactly the input to the LBC branch.  The arm
        branch must be frame-interleaved: GR00T first concatenates its named
        ``[N, 10, 18]`` and ``[N, 10, 8]`` tensors on the last axis and only
        then flattens.  Flattening all 180 arm values before all 80 wrist
        values would retain 330 dimensions yet silently permute the actor's
        learned arm features.
        """
        result = torch.cat(
            (
                self.lbc_future.reshape(self.num_envs, -1),
                torch.cat(
                    (self.smpl_arm_future_root, self.smpl_wrist_quat_future_root), dim=-1
                ).reshape(self.num_envs, -1),
            ),
            dim=-1,
        )
        if result.shape[-1] != MIXED_REFERENCE_DIM:
            raise RuntimeError(
                "DEPSIC V2 mixed reference contract broken: "
                f"expected {MIXED_REFERENCE_DIM}, got {result.shape[-1]}"
            )
        return result

    def _v2_head_rot6(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Current root-local head orientation in projector column-major 6D."""
        head = self.vr_tracker_quat_root[:, 2] if env_ids is None else self.vr_tracker_quat_root[env_ids, 2]
        return quat_to_column_rot6(head)


class DepsicG1V2Mixed(_DepsicV2TrackerMixin, DepsicG1RobotOnly, namespace="mimic_lite"):
    """G1 V1 physics/rewards with GR00T HEAD's full-SMPL V2 token route."""

    def __init__(self, env, *, smpl_motion_file: str, smpl_cache_root: str | None = None, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self._init_v2_tracker(
            smpl_motion_file=smpl_motion_file,
            smpl_cache_root=smpl_cache_root,
            nominal_height=0.75,
            rpy_min=(-0.5, -0.52, -1.27),
            rpy_max=(0.5, 1.22, 1.27),
            height_range=(0.35, 0.8),
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self._refresh_v2_tracker(env_ids, calibrate=True)
        self._refresh_v2_full_smpl()

    def update(self) -> None:
        # Keep exact V1 arm/reward timing.  LBC remains independently sampled
        # HTD intent; SMPL never overwrites its velocity or posture channels.
        self.frame_fraction += self.env.step_dt * self.fps[self.clip_ids]
        advance = torch.floor(self.frame_fraction).long()
        self.frame_fraction -= advance
        self.frame_ids = torch.minimum(self.frame_ids + advance, self.lengths[self.clip_ids] - 1)
        self._future_arm_q = None
        self._future_arm_dq = None
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._refresh_arm_reference(env_ids)
        self._refresh_v2_tracker(env_ids)
        self._refresh_v2_full_smpl()
        self.lbc_elapsed += 1
        self._sample_lbc((self.lbc_elapsed >= self.resample_steps).nonzero(as_tuple=False).flatten())
        self._update_velocity_ema()
        self._update_metrics()
        self.env.extra["depsic_v2/smpl_cache"] = self.smpl_cache_path
        self.env.extra["depsic_v2/smpl_manifest"] = self.smpl_manifest_hash


class _V2Observation(Observation[DepsicG1V2Mixed]):
    pass


class depsic_v2_mixed_reference(_V2Observation, namespace="mimic_lite"):
    """V2 action-path input: LBC70 + SMPL arm180 + wrist-quat80 = 330."""

    def compute(self) -> torch.Tensor:
        return self.command_manager._v2_mixed_reference()


class depsic_v2_teacher_reference(_V2Observation, namespace="mimic_lite"):
    """Train-only V1 350-D robot labels for frozen-teacher supervision."""

    def compute(self) -> torch.Tensor:
        q, dq = self.command_manager.arm_future
        result = torch.cat((q, dq, self.command_manager.lbc_future), dim=-1).reshape(self.num_envs, -1)
        if result.shape[-1] != V1_REFERENCE_DIM:
            raise RuntimeError(
                f"DEPSIC V2 teacher contract broken: expected {V1_REFERENCE_DIM}, got {result.shape[-1]}"
            )
        return result
