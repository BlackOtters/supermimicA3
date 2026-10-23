"""A3 DEPSIC V2: A3 V1 mechanics with the common 330-D full-SMPL route."""

from __future__ import annotations

from dataclasses import replace

import torch

from mimic_lite.depsic_a3_v2_head import load_or_fit_a3_head_projector
from mimic_lite.depsic_v2_contract import V1_REFERENCE_DIM
from mimic_lite.tasks.depsic_a3 import (
    DepsicA3UltraV1RobotOnly,
    DepsicA3V1RobotOnly,
)
from mimic_lite.tasks.depsic_v2 import _DepsicV2TrackerMixin
from active_adaptation.envs.mdp.observations.base import Observation


class DepsicA3V2Mixed(_DepsicV2TrackerMixin, DepsicA3V1RobotOnly, namespace="mimic_lite"):
    """A3 V1 physics/rewards with HEAD's 330-D full-SMPL token input.

    The external head target is projected only from the current paired-SMPL
    root-local head rotation.  A3 PKL head columns are used once to fit and
    validate that fixed projector, then discarded from the runtime command.
    """

    def __init__(
        self,
        env,
        *,
        smpl_motion_file: str,
        smpl_cache_root: str | None = None,
        head_projector_ridge: float = 1.0e-4,
        head_projector_heldout_fraction: float = 0.1,
        head_projector_max_rmse: float = 0.2,
        head_projector_max_error: float = 0.6,
        **kwargs,
    ) -> None:
        motion_file = kwargs["motion_file"]
        super().__init__(env, **kwargs)
        self._init_v2_tracker(
            smpl_motion_file=smpl_motion_file,
            smpl_cache_root=smpl_cache_root,
            nominal_height=0.75,
            rpy_min=(-0.40, -0.489, -1.57),
            rpy_max=(0.40, 1.57, 1.57),
            height_range=(0.60, 1.05),
        )
        self.head_projector = load_or_fit_a3_head_projector(
            a3_motion_file=motion_file,
            smpl_motion_file=smpl_motion_file,
            motion_pool_size=kwargs.get("motion_pool_size"),
            ridge=head_projector_ridge,
            heldout_fraction=head_projector_heldout_fraction,
            max_heldout_rmse=head_projector_max_rmse,
            max_heldout_error=head_projector_max_error,
            cache_root=smpl_cache_root,
            paired_smpl_pool=self.smpl_pool,
            a3_pool=self.pool,
        )
        # The physical V2 route has no PKL-head replay path after this point.
        del self.head_q_data
        self.pool = replace(self.pool, head_q=torch.empty(0, dtype=self.pool.head_q.dtype))
        self.head_projector_cache_path = str(self.head_projector.cache_path)

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        """Refresh V1 arm teacher/current target and projected external head."""
        source = self.frame_ids[env_ids].to(self.frame_fraction.dtype) + self.frame_fraction[env_ids]
        arm_q, arm_dq = self._interpolate_arm(self.clip_ids[env_ids], source)
        self.arm_q[env_ids] = arm_q
        self.arm_dq[env_ids] = arm_dq
        wrist_pos, wrist_rot = self.kinematics.forward(arm_q)
        self.wrist_pos_torso[env_ids] = wrist_pos
        self.wrist_rot_torso[env_ids] = wrist_rot
        # ``reset`` first calibrates the tracker, then calls this method.  An
        # early base reset during construction has no tracker state yet.
        if hasattr(self, "vr_tracker_quat_root"):
            self.head_q[env_ids] = self.head_projector(self._v2_head_rot6(env_ids))
            self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = self.head_q[env_ids]

    def reset(self, env_ids: torch.Tensor) -> None:
        # Reproduce V1 reset ordering while ensuring the physical head uses a
        # calibrated current SMPL sample rather than a historical PKL column.
        self._sample_lbc(env_ids)
        self.clip_ids[env_ids] = torch.randint(len(self.lengths), (len(env_ids),), device=self.device)
        episode_frames = torch.ceil(
            self.env.max_episode_length[env_ids, 0].to(self.fps.dtype)
            * self.env.step_dt * self.fps[self.clip_ids[env_ids]]
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
        self._refresh_v2_tracker(env_ids, calibrate=True)
        self._refresh_v2_full_smpl()
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
            half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * 0.9
            clamped = joint_pos.clamp(center - half_range, center + half_range)
            # Unlimited motor coordinates have [-inf, inf] bounds: their
            # midpoint is NaN. Match V1's reset and leave these coordinates alone.
            joint_pos = torch.where(finite_limits, clamped, joint_pos)
            joint_vel = torch.zeros_like(joint_pos)
            joint_vel[:, self.arm_joint_ids] = self.arm_dq[env_ids]
            self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = self.head_q[env_ids]

    def update(self) -> None:
        self.frame_fraction += self.env.step_dt * self.fps[self.clip_ids]
        advance = torch.floor(self.frame_fraction).long()
        self.frame_fraction -= advance
        self.frame_ids = torch.minimum(self.frame_ids + advance, self.lengths[self.clip_ids] - 1)
        self._future_arm_q = None
        self._future_arm_dq = None
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._refresh_v2_tracker(env_ids)
        self._refresh_v2_full_smpl()
        self._refresh_reference(env_ids)
        self.lbc_elapsed += 1
        self._sample_lbc((self.lbc_elapsed >= self.resample_steps).nonzero(as_tuple=False).flatten())
        self._update_velocity_ema()
        self._update_metrics()
        self.env.extra["depsic_v2/smpl_cache"] = self.smpl_cache_path
        self.env.extra["depsic_v2/smpl_manifest"] = self.smpl_manifest_hash
        self.env.extra["depsic_a3_v2/head_projector_cache"] = self.head_projector_cache_path
        self.env.extra["depsic_a3_v2/head_projector_manifest"] = self.head_projector.manifest_hash
        self.env.extra["depsic_a3_v2/head_projector_heldout_rmse"] = self.head_projector.heldout_rmse
        self.env.extra["depsic_a3_v2/head_projector_heldout_max_error"] = self.head_projector.heldout_max_error


class DepsicA3UltraV2Mixed(
    DepsicA3V2Mixed,
    DepsicA3UltraV1RobotOnly,
    namespace="mimic_lite",
):
    """Ultra T3D0 V2 with the unchanged 29-DoF A3 mixed-token contract.

    The V2 state machine, task rewards and 330-D SMPL token are deliberately
    identical to ordinary A3 V2.  The second parent only supplies the Ultra
    runtime head target mapping: source PKLs still contain
    ``head_pitch_joint`` while physics writes that value to
    ``ultra_head_pitch_joint``.
    """

    def __init__(self, env, **kwargs) -> None:
        kwargs["runtime_head_joint_names"] = (
            "head_yaw_joint",
            "ultra_head_pitch_joint",
        )
        super().__init__(env, **kwargs)


class _A3V2Observation(Observation[DepsicA3V2Mixed]):
    pass


class depsic_a3_v2_mixed_reference(_A3V2Observation, namespace="mimic_lite"):
    def compute(self) -> torch.Tensor:
        return self.command_manager._v2_mixed_reference()


class depsic_a3_v2_teacher_reference(_A3V2Observation, namespace="mimic_lite"):
    def compute(self) -> torch.Tensor:
        q, dq = self.command_manager.arm_future
        result = torch.cat((q, dq, self.command_manager.lbc_future), dim=-1).reshape(self.num_envs, -1)
        if result.shape[-1] != V1_REFERENCE_DIM:
            raise RuntimeError(
                f"DEPSIC A3 V2 teacher contract broken: expected {V1_REFERENCE_DIM}, got {result.shape[-1]}"
            )
        return result


__all__ = ["DepsicA3V2Mixed", "DepsicA3UltraV2Mixed"]
