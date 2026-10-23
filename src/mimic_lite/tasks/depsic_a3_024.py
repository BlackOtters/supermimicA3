"""DEPSIC A3 024 tasks.

The closed-chain 024 asset locks its head and therefore does not use the A3
head projector.  The verified serial/passive-foot asset keeps a physical head,
so its V2 route uses the same versioned SMPL-head projector as ordinary A3 V2.
Both V2 variants use the current 330-D full-SMPL token contract.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies

from mimic_lite.a3_024_motor_model import (
    cross_penalty,
    interpolate_tn_limit,
    motor_family_for_serial_joint,
    motor_penalty,
    peak_mechanical_power,
    position_threshold_penalty,
    reward_024_curve,
    threshold_penalty,
)
from mimic_lite.tasks.depsic import (
    SOFT_JOINT_POS_LIMIT_FACTOR,
    _contact_force_history,
    _contact_mask,
)
from mimic_lite.tasks.depsic_a3 import (
    DepsicA3V1RobotOnly,
    depsic_a3_actor_history,
    depsic_a3_privileged,
    depsic_a3_reference,
)
from mimic_lite.tasks.depsic_v2 import _DepsicV2TrackerMixin
from mimic_lite.tasks.depsic_a3_v2 import (
    depsic_a3_v2_mixed_reference,
    depsic_a3_v2_teacher_reference,
)


# In the 024 MJCF, rear-foot collision geometry belongs to ankle_roll while
# forefoot and toe are distinct compliant bodies. The order is left rear,
# forefoot, toe, then right rear, forefoot, toe.
A3_024_SUPPORT_BODY_NAMES = (
    "left_ankle_roll_Link",
    "left_foot_forefoot_Link",
    "left_foot_toe_Link",
    "right_ankle_roll_Link",
    "right_foot_forefoot_Link",
    "right_foot_toe_Link",
)


class DepsicA3V1RobotOnly024(DepsicA3V1RobotOnly, namespace="mimic_lite"):
    """A3 V1 command paired with the 024 closed-chain asset."""

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        # The pool continues to validate the legacy 31-DoF PKL layout, but
        # its two head columns must have no runtime path in the 024 task: the
        # released MJCF hard-locks the physical head at zero.  Do not call
        # the base method here, since it interpolates and writes those head
        # columns before we could overwrite them.
        source = self.frame_ids[env_ids].to(self.frame_fraction.dtype) + self.frame_fraction[env_ids]
        arm_q, arm_dq = self._interpolate_arm(self.clip_ids[env_ids], source)
        self.arm_q[env_ids] = arm_q
        self.arm_dq[env_ids] = arm_dq
        wrist_pos, wrist_rot = self.kinematics.forward(arm_q)
        self.wrist_pos_torso[env_ids] = wrist_pos
        self.wrist_rot_torso[env_ids] = wrist_rot
        self.head_q[env_ids] = 0.0
        self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = 0.0


class DepsicA3V1RobotOnly024Serial(DepsicA3V1RobotOnly024, namespace="mimic_lite"):
    """Explicit command type for the 024 serial asset.

    The command logic is identical to the released 024 V1 contract; the
    robot asset selected by the task config supplies the serial actuator and
    passive-foot topology.  Keeping a distinct registry class prevents a
    serial task from silently falling back to the closed-chain command.
    """

    pass


class DepsicA3V2Mixed024(_DepsicV2TrackerMixin, DepsicA3V1RobotOnly024, namespace="mimic_lite"):
    """A3 024 V2 with the current 330-D full-SMPL reference path."""

    def __init__(self, env, *, smpl_motion_file: str, smpl_cache_root: str | None = None, **kwargs):
        super().__init__(env, **kwargs)
        self._init_v2_tracker(
            smpl_motion_file=smpl_motion_file,
            smpl_cache_root=smpl_cache_root,
            nominal_height=0.75,
            rpy_min=(-0.40, -0.489, -1.57),
            rpy_max=(0.40, 1.57, 1.57),
            height_range=(0.60, 1.05),
        )

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        super()._refresh_reference(env_ids)
        self.head_q[env_ids] = 0.0
        self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = 0.0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the V2 route without ever loading/replaying a head target.

        The V1 PKL remains the owner of clip selection and arm teacher
        labels.  The paired SMPL cache supplies only the 330-D mixed token;
        the 024 head is a physical zero-range joint and is not an auxiliary
        target to fit or replay.
        """
        self._sample_lbc(env_ids)
        self.clip_ids[env_ids] = torch.randint(
            len(self.lengths), (len(env_ids),), device=self.device
        )
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
        self._refresh_v2_tracker(env_ids, calibrate=True)
        self._refresh_v2_full_smpl()
        self._refresh_reference(env_ids)
        if self.reset_to_reference:
            joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
            if not self.nominal_reset:
                joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
            joint_pos[:, self.arm_joint_ids] = self.arm_q[env_ids]
            # This explicit assignment is intentionally zero rather than a
            # projection of a SMPL or PKL signal.
            joint_pos[:, self.head_joint_ids] = 0.0
            limits = self.asset.data.joint_pos_limits[env_ids]
            finite_limits = torch.isfinite(limits).all(dim=-1)
            center = limits.mean(dim=-1)
            half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * SOFT_JOINT_POS_LIMIT_FACTOR
            clamped = joint_pos.clamp(center - half_range, center + half_range)
            joint_pos = torch.where(finite_limits, clamped, joint_pos)
            joint_vel = torch.zeros_like(joint_pos)
            joint_vel[:, self.arm_joint_ids] = self.arm_dq[env_ids]
            self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = 0.0

    def update(self) -> None:
        """Advance V1 timing and refresh the synchronized 330-D SMPL token."""
        self.frame_fraction += self.env.step_dt * self.fps[self.clip_ids]
        advance = torch.floor(self.frame_fraction).long()
        self.frame_fraction -= advance
        self.frame_ids = torch.minimum(
            self.frame_ids + advance, self.lengths[self.clip_ids] - 1
        )
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


class DepsicA3V2Mixed024Serial(
    _DepsicV2TrackerMixin,
    DepsicA3V1RobotOnly024Serial,
    namespace="mimic_lite",
):
    """A3 V2 full-SMPL route on the verified 024 serial/passive-foot model."""

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
        from mimic_lite.depsic_a3_v2_head import load_or_fit_a3_head_projector

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
        del self.head_q_data
        self.pool = replace(self.pool, head_q=torch.empty(0, dtype=self.pool.head_q.dtype))
        self.head_projector_cache_path = str(self.head_projector.cache_path)

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        source = self.frame_ids[env_ids].to(self.frame_fraction.dtype) + self.frame_fraction[env_ids]
        arm_q, arm_dq = self._interpolate_arm(self.clip_ids[env_ids], source)
        self.arm_q[env_ids] = arm_q
        self.arm_dq[env_ids] = arm_dq
        wrist_pos, wrist_rot = self.kinematics.forward(arm_q)
        self.wrist_pos_torso[env_ids] = wrist_pos
        self.wrist_rot_torso[env_ids] = wrist_rot
        if hasattr(self, "vr_tracker_quat_root"):
            self.head_q[env_ids] = self.head_projector(self._v2_head_rot6(env_ids))
            self.asset.data.joint_pos_target[env_ids[:, None], self.head_joint_ids] = self.head_q[env_ids]

    def reset(self, env_ids: torch.Tensor) -> None:
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
            half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * SOFT_JOINT_POS_LIMIT_FACTOR
            joint_pos = torch.where(
                finite_limits,
                joint_pos.clamp(center - half_range, center + half_range),
                joint_pos,
            )
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


class depsic_a3_024_actor_history(depsic_a3_actor_history, namespace="mimic_lite"):
    pass


class depsic_a3_024_reference(depsic_a3_reference, namespace="mimic_lite"):
    pass


class depsic_a3_024_privileged(depsic_a3_privileged, namespace="mimic_lite"):
    pass


class _A3024Reward(Reward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = env.scene.articulations["robot"]
        self.action = env.input_managers["action"]

    @property
    def actuator(self):
        actuator = getattr(self.asset, "_a3_024_actuator", None)
        if actuator is None:
            raise RuntimeError("A3 024 rewards require the A3_024LoopActuator")
        return actuator

    # Motor-space sources.  Subclasses for assets without the 024 actuator
    # (e.g. Ultra) override these with a reward-side estimate.
    def _requested_joint_torque(self) -> torch.Tensor:
        return self.asset.data.a3_024_requested_joint_torque

    def _parallel_motor_torque(self) -> torch.Tensor:
        return self.asset.data.a3_024_parallel_motor_torque

    def _parallel_motor_velocity(self) -> torch.Tensor:
        return self.asset.data.a3_024_parallel_motor_velocity


class depsic_a3_024_undesired_contacts(_A3024Reward, namespace="mimic_lite"):
    """Count non-support contacts while allowing all three 024 foot segments."""

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
        names = body_names or [
            r"^(?!.*head.*)(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$)"
            r"(?!left_foot_forefoot_Link$)(?!right_foot_forefoot_Link$)"
            r"(?!left_foot_toe_Link$)(?!right_foot_toe_Link$)"
            r"(?!left_wrist_yaw_Link$)(?!right_wrist_yaw_Link$)"
            r"(?!left_elbow_Link$)(?!right_elbow_Link$).+$"
        ]
        ids, _ = find_sensor_bodies(self.asset, self.contact_sensor, names)
        self.sensor_ids = torch.tensor(ids, device=self.device)
        self.threshold = float(threshold)

    def _compute(self) -> torch.Tensor:
        return _contact_mask(self.contact_sensor, self.sensor_ids, self.threshold).sum(
            dim=-1, keepdim=True
        ).float()


class _A3024FootReward(_A3024Reward):
    """Aggregate 024 rear/forefoot/toe signals into the two physical feet."""

    def __init__(
        self,
        env,
        weight: float,
        threshold: float = 1.0,
        foot_body_names: list[str] | tuple[str, ...] = A3_024_SUPPORT_BODY_NAMES,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        foot_body_names = tuple(foot_body_names)
        if len(foot_body_names) != 6:
            raise ValueError(
                "A3 024 foot rewards require [left rear, left forefoot, left toe, "
                "right rear, right forefoot, right toe]."
            )
        self.contact_sensor = env.scene.sensors["contact_forces"]
        sensor_ids, resolved_names = find_sensor_bodies(
            self.asset, self.contact_sensor, list(foot_body_names)
        )
        body_ids, _ = find_bodies(self.asset, resolved_names)
        self.sensor_ids = torch.tensor(sensor_ids, device=self.device).reshape(2, 3)
        self.body_ids = torch.tensor(body_ids, device=self.device).reshape(2, 3)
        self.threshold = float(threshold)

    def _segment_forces(self) -> torch.Tensor:
        forces = _contact_force_history(self.contact_sensor, self.sensor_ids.flatten())
        return forces.reshape(self.num_envs, 2, 3, *forces.shape[-2:])

    def _segment_in_contact(self) -> torch.Tensor:
        forces = self._segment_forces()
        return forces.norm(dim=-1).amax(dim=-1) > self.threshold

    def _side_in_contact(self) -> torch.Tensor:
        return self._segment_in_contact().any(dim=-1)

    def _side_forces(self) -> torch.Tensor:
        return self._segment_forces().sum(dim=2)


class depsic_a3_024_foot_slide(_A3024FootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        velocity = self.asset.data.body_link_lin_vel_w[:, self.body_ids, :2].norm(dim=-1)
        # A three-piece contact is one physical foot. Use the largest sliding
        # segment per side instead of counting an ordinary support three times.
        supported_speed = velocity * self._segment_in_contact()
        return supported_speed.amax(dim=-1).sum(dim=-1, keepdim=True)


class depsic_a3_024_flying(_A3024FootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        return (~self._side_in_contact().any(dim=-1, keepdim=True)).float()


class depsic_a3_024_feet_force(_A3024FootReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, force_threshold: float = 500.0, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.force_threshold = float(force_threshold)

    def _compute(self) -> torch.Tensor:
        # Sum per-foot segment force before retaining rigid A3's two-foot norm.
        vertical_force = self._side_forces()[:, :, 0, 2]
        force_norm = vertical_force.norm(dim=-1, keepdim=True)
        return (force_norm - self.force_threshold).clamp_min(0.0).clamp_max(400.0)


class _A3024FootTimingReward(_A3024FootReward):
    """Per-foot clocks continuous through heel-to-toe rolling contact."""

    def __init__(self, env, weight: float, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.current_contact_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.current_air_time = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.current_contact_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0

    def _update_support_time(self) -> torch.Tensor:
        contact = self._side_in_contact()
        step_dt = self.env.step_dt
        self.current_contact_time = torch.where(
            contact,
            self.current_contact_time + step_dt,
            torch.zeros_like(self.current_contact_time),
        )
        self.current_air_time = torch.where(
            contact,
            torch.zeros_like(self.current_air_time),
            self.current_air_time + step_dt,
        )
        return contact


class depsic_a3_024_feet_air_time(_A3024FootTimingReward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, max_air_time: float = 0.4, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.max_air_time = float(max_air_time)

    def _compute(self) -> torch.Tensor:
        contact = self._update_support_time()
        single_stance = contact.sum(dim=-1) == 1
        mode_time = torch.where(contact, self.current_contact_time, self.current_air_time)
        reward = torch.where(
            single_stance,
            mode_time.amin(dim=-1),
            torch.zeros(self.num_envs, device=self.device),
        )
        moving = (
            self.command_manager.lbc[:, :2].norm(dim=-1)
            + self.command_manager.lbc[:, 2].abs()
            > 0.1
        )
        return (reward.clamp_max(self.max_air_time) * moving).unsqueeze(-1)


class depsic_a3_024_feet_stumble(_A3024FootReward, namespace="mimic_lite"):
    def _compute(self) -> torch.Tensor:
        force = self._side_forces()[:, :, 0]
        horizontal = force[..., :2].norm(dim=-1)
        return (horizontal > 5.0 * force[..., 2].abs()).any(dim=-1, keepdim=True).float()


class depsic_a3_024_feet_contact_standing_stable(
    _A3024FootTimingReward, namespace="mimic_lite"
):
    def __init__(
        self,
        env,
        weight: float,
        min_contact_time: float = 0.5,
        command_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.min_contact_time = float(min_contact_time)
        self.command_threshold = float(command_threshold)

    def _compute(self) -> torch.Tensor:
        contact = self._update_support_time()
        standing = (
            self.command_manager.lbc[:, :2].norm(dim=-1)
            + self.command_manager.lbc[:, 2].abs()
            <= self.command_threshold
        )
        stable = contact.all(dim=-1) & (self.current_contact_time > self.min_contact_time).all(dim=-1)
        reward = torch.tanh(self.current_contact_time.amin(dim=-1) / self.min_contact_time)
        return torch.where(standing & stable, reward, torch.zeros_like(reward)).unsqueeze(-1)


class depsic_a3_024_motor_tn(_A3024Reward, namespace="mimic_lite"):
    """Dense T-N utilization penalty for serial or parallel motors."""

    def __init__(self, env, weight: float, group: str = "serial", lambda_soft: float = 0.05,
                 lambda_violation: float = 5.0, soft_exponent: float = 8.0,
                 violation_exponent: float = 2.0, utilization_cap: float = 2.0, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        if group not in {"serial", "parallel"}:
            raise ValueError("A3 024 motor_tn group must be serial or parallel")
        self.group = group
        self.params = (lambda_soft, lambda_violation, soft_exponent, violation_exponent, utilization_cap)
        if group == "serial":
            names = [n for n in self.asset.joint_names if motor_family_for_serial_joint(n) is not None]
            self.ids = torch.tensor([self.asset.joint_names.index(n) for n in names], device=self.device)
            self.names = names
            self.curves = {n: reward_024_curve(motor_family_for_serial_joint(n), device=self.device) for n in names}

    def _compute(self):
        if self.group == "parallel":
            speed, torque = reward_024_curve("PFP78", device=self.device)
            limit = interpolate_tn_limit(self._parallel_motor_velocity(), speed, torque)
            utilization = self._parallel_motor_torque().abs() / limit.clamp_min(1e-6)
        else:
            values = []
            qvel = self.asset.data.joint_vel[:, self.ids]
            requested = self._requested_joint_torque()[:, self.ids]
            for i, name in enumerate(self.names):
                speed, torque = self.curves[name]
                limit = interpolate_tn_limit(qvel[:, i], speed, torque)
                values.append(requested[:, i].abs() / limit.clamp_min(1e-6))
            utilization = torch.stack(values, dim=-1)
        return motor_penalty(
            utilization,
            lambda_soft=self.params[0], lambda_violation=self.params[1],
            soft_exponent=self.params[2], violation_exponent=self.params[3], utilization_cap=self.params[4],
        ).combined.sum(dim=-1, keepdim=True)


class depsic_a3_024_ankle_motor_constraint(_A3024Reward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, constraint: str = "speed", threshold_ratio: float = 0.8,
                 lambda_soft: float = 1.0, lambda_violation: float = 5.0,
                 soft_exponent: float = 2.0, violation_exponent: float = 2.0,
                 utilization_cap: float = 2.0, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        if constraint not in {"speed", "power", "fixed_torque"}:
            raise ValueError("unknown A3 024 ankle constraint")
        self.constraint = constraint
        self.threshold_ratio = threshold_ratio
        self.penalty_args = (lambda_soft, lambda_violation, soft_exponent, violation_exponent, utilization_cap)

    def _compute(self):
        speed = self._parallel_motor_velocity()[:, :4]
        torque = self._parallel_motor_torque()[:, :4]
        curve_speed, curve_torque = reward_024_curve("PFP78", device=self.device)
        if self.constraint == "speed":
            ratio = speed.abs() / curve_speed[-1]
        elif self.constraint == "power":
            ratio = torch.relu(speed * torque) / peak_mechanical_power(curve_speed, curve_torque).clamp_min(1e-6)
        else:
            ratio = torque.abs() / curve_torque[0].clamp_min(1e-6)
        return threshold_penalty(
            ratio, threshold_ratio=self.threshold_ratio,
            lambda_soft=self.penalty_args[0], lambda_violation=self.penalty_args[1],
            soft_exponent=self.penalty_args[2], violation_exponent=self.penalty_args[3], utilization_cap=self.penalty_args[4],
        ).combined.sum(dim=-1, keepdim=True)


class depsic_a3_024_parallel_cross(_A3024Reward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, group: str = "ankle", threshold_ratio: float = 0.75, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        if group not in {"ankle", "waist"}:
            raise ValueError("A3 024 cross group must be ankle or waist")
        self.group = group
        self.threshold_ratio = threshold_ratio
        names = (("left_ankle_pitch_joint", "left_ankle_roll_joint"), ("right_ankle_pitch_joint", "right_ankle_roll_joint")) if group == "ankle" else (("waist_pitch_joint", "waist_roll_joint"),)
        self.ids = torch.tensor([[self.asset.joint_names.index(a), self.asset.joint_names.index(b)] for a, b in names], device=self.device)
        # Published reward_024 normalization: the waist pair is ordered
        # [pitch, roll] and uses 115 / 46 Nm, respectively.
        limits = ((118.2, 54.75), (118.2, 54.75)) if group == "ankle" else ((115.0, 46.0),)
        self.limits = torch.tensor(limits, device=self.device)

    def _compute(self):
        requested = self._requested_joint_torque()[:, self.ids]
        return cross_penalty(requested.abs() / self.limits, threshold_ratio=self.threshold_ratio).sum(dim=-1, keepdim=True)


class depsic_a3_024_ankle_position(_A3024Reward, namespace="mimic_lite"):
    def __init__(self, env, weight: float, threshold_ratio: float = 0.85, exponent: float = 2.0, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        names = ("left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint")
        self.ids = torch.tensor([self.asset.joint_names.index(n) for n in names], device=self.device)
        self.threshold_ratio, self.exponent = threshold_ratio, exponent

    def _compute(self):
        return position_threshold_penalty(
            self.asset.data.joint_pos[:, self.ids], self.asset.data.joint_pos_limits[:, self.ids],
            threshold_ratio=self.threshold_ratio, exponent=self.exponent,
        ).sum(dim=-1, keepdim=True)


__all__ = [
    "DepsicA3V1RobotOnly024", "DepsicA3V1RobotOnly024Serial",
    "DepsicA3V2Mixed024", "DepsicA3V2Mixed024Serial",
    "depsic_a3_024_actor_history", "depsic_a3_024_reference", "depsic_a3_024_privileged",
    "depsic_a3_024_undesired_contacts", "depsic_a3_024_foot_slide",
    "depsic_a3_024_flying", "depsic_a3_024_feet_force", "depsic_a3_024_feet_air_time",
    "depsic_a3_024_feet_stumble", "depsic_a3_024_feet_contact_standing_stable",
    "depsic_a3_024_motor_tn", "depsic_a3_024_ankle_motor_constraint",
    "depsic_a3_024_parallel_cross", "depsic_a3_024_ankle_position",
]
