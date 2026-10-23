"""A3 Ultra sim2real terms ported from the SONIC-for-A3 035 contract.

Ultra shares A3's joints, feet and motors, but its MJCF is serial and uses
mjlab's builtin position actuators.  The 024 motor rewards therefore cannot
read the 024 actuator's cached motor state.  This module reconstructs the same
quantities on the reward side, exactly as the released 035 reward does:

* joint torque is the pre-clip implicit PD request ``kp (q* - q) - kd dq``,
  read from the per-world actuator gains so that gain randomization is seen;
* ankle and waist pitch/roll are mapped to their two PFP78 motors through the
  published 024 Jacobian tables (``tau_m = J.T tau_j``, ``J dq_m = dq_j``).

The motor-state reconstruction used by rewards does not feed back into the
simulated dynamics. The domain-randomization classes later in this module do
intentionally modify per-world model parameters and root velocity.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from active_adaptation.envs.mdp.randomizations.base import Randomization
from active_adaptation.envs.mdp.randomizations.common import (
    _mjlab_ensure_recompute_fields_expanded,
    _mjlab_recompute_constants,
)
from active_adaptation.envs.utils import find_bodies, find_joints, find_sensor_bodies

from mimic_lite.a3_024_motor_model import (
    PARALLEL_PAIRS,
    ParallelJacobianTable,
    motor_torque_from_joint_torque,
    motor_velocity_from_joint_velocity,
)
from mimic_lite.tasks.depsic import _contact_mask
from mimic_lite.tasks.depsic_a3_024 import (
    _A3024Reward,
    depsic_a3_024_ankle_motor_constraint,
    depsic_a3_024_ankle_position,
    depsic_a3_024_motor_tn,
    depsic_a3_024_parallel_cross,
)


class A3UltraMotorState:
    """Per-step reward-side motor estimate shared by all Ultra S2R rewards."""

    def __init__(self, env):
        self.env = env
        self.asset = env.scene.articulations["robot"]
        self.device = env.device
        joint_names = list(self.asset.joint_names)
        actuator_names = list(self.asset.actuator_names)
        ctrl_ids = torch.as_tensor(self.asset.indexing.ctrl_ids, dtype=torch.long, device=self.device)
        # a3_ultra.py names each actuator ``joint_<joint name>``.
        self.joint_ids, self.global_ctrl_ids = [], []
        for local_ctrl, name in enumerate(actuator_names):
            joint_name = name.removeprefix("joint_")
            if joint_name in joint_names:
                self.joint_ids.append(joint_names.index(joint_name))
                self.global_ctrl_ids.append(int(ctrl_ids[local_ctrl]))
        self.joint_ids = torch.tensor(self.joint_ids, dtype=torch.long, device=self.device)
        self.global_ctrl_ids = torch.tensor(self.global_ctrl_ids, dtype=torch.long, device=self.device)
        self.feedback_joint_ids = self.joint_ids.clone()
        self.four_ankle_state = getattr(env, "_four_ankle_state", None)
        self.elastic_pairs = []
        for side in ("left", "right"):
            output, motor = f"{side}_ankle_roll_joint", f"{side}_ankle_roll_motor_joint"
            if motor in joint_names and self.four_ankle_state is None:
                jid, mid = joint_names.index(output), joint_names.index(motor)
                self.feedback_joint_ids[self.joint_ids == jid] = mid
                self.elastic_pairs.append((jid, mid))
        self.pairs = [
            (
                torch.tensor([joint_names.index(p), joint_names.index(r)], dtype=torch.long, device=self.device),
                ParallelJacobianTable.load(name, device=self.device),
            )
            for name, (p, r) in PARALLEL_PAIRS
        ]
        self._timestamp = -1
        self.requested = torch.zeros(env.num_envs, len(joint_names), device=self.device)
        self.parallel_torque = torch.zeros(env.num_envs, 6, device=self.device)
        self.parallel_velocity = torch.zeros(env.num_envs, 6, device=self.device)

    @classmethod
    def get(cls, env) -> "A3UltraMotorState":
        state = getattr(env, "_a3_ultra_motor_state", None)
        if state is None:
            state = cls(env)
            env._a3_ultra_motor_state = state
        return state

    def refresh(self) -> None:
        if self._timestamp == self.env.timestamp:
            return
        self._timestamp = self.env.timestamp
        model, data = self.env.sim.model, self.asset.data
        # MuJoCo position actuator: force = gain * ctrl + b1 * q + b2 * dq,
        # with gain = kp, b1 = -kp and b2 = -kd.  ``ctrl`` is the delayed
        # target actually applied, matching 035's post-delay computed_torque.
        ctrl = data.data.ctrl[:, self.global_ctrl_ids]
        gain = model.actuator_gainprm[:, self.global_ctrl_ids, 0]
        bias_q = model.actuator_biasprm[:, self.global_ctrl_ids, 1]
        bias_dq = model.actuator_biasprm[:, self.global_ctrl_ids, 2]
        q = data.joint_pos[:, self.feedback_joint_ids]
        dq = data.joint_vel[:, self.feedback_joint_ids]
        self.requested[:, self.joint_ids] = gain * ctrl + bias_q * q + bias_dq * dq
        drive_velocity = data.joint_vel
        if self.four_ankle_state is not None:
            state = self.four_ankle_state
            self.requested[:, state.output_ids] = state.motor_effort()
            drive_velocity = drive_velocity.clone()
            drive_velocity[:, state.output_ids] = state.encoder_state()[1]
        if self.elastic_pairs:
            drive_velocity = drive_velocity.clone()
            for output, motor in self.elastic_pairs:
                drive_velocity[:, output] = data.joint_vel[:, motor]
        for slot, (ids, table) in enumerate(self.pairs):
            jac = table.interpolate_jacobian(data.joint_pos[:, ids])
            cols = slice(2 * slot, 2 * slot + 2)
            self.parallel_torque[:, cols] = motor_torque_from_joint_torque(jac, self.requested[:, ids])
            self.parallel_velocity[:, cols] = motor_velocity_from_joint_velocity(jac, drive_velocity[:, ids])


class _UltraMotorMixin(_A3024Reward):
    def __init__(self, env, weight: float, **kwargs):
        super().__init__(env, weight=weight, **kwargs)
        self.motor_state = A3UltraMotorState.get(env)

    def _requested_joint_torque(self) -> torch.Tensor:
        self.motor_state.refresh()
        return self.motor_state.requested

    def _parallel_motor_torque(self) -> torch.Tensor:
        self.motor_state.refresh()
        return self.motor_state.parallel_torque

    def _parallel_motor_velocity(self) -> torch.Tensor:
        self.motor_state.refresh()
        return self.motor_state.parallel_velocity


class depsic_a3_ultra_motor_tn(_UltraMotorMixin, depsic_a3_024_motor_tn, namespace="mimic_lite"):
    pass


class depsic_a3_ultra_ankle_motor_constraint(
    _UltraMotorMixin, depsic_a3_024_ankle_motor_constraint, namespace="mimic_lite"
):
    pass


class depsic_a3_ultra_parallel_cross(_UltraMotorMixin, depsic_a3_024_parallel_cross, namespace="mimic_lite"):
    pass


class depsic_a3_ultra_ankle_position(depsic_a3_024_ankle_position, namespace="mimic_lite"):
    pass


class depsic_a3_ultra_push_if_in_contact(Randomization, namespace="mimic_lite"):
    """035-style contact-gated root velocity push on a per-env timer.

    Each env draws a push interval; when it expires, a root COM velocity delta
    is added only if a support body is loaded.  An airborne env keeps its timer
    at zero and is pushed on its next support step.  Note that the released 035
    config itself uses the ungated ``push_by_setting_velocity`` (see
    ``depsic_a3_ultra_push``); this gated variant is kept for ablations.
    """

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        min_s: float,
        max_s: float,
        foot_body_names: list[str] | tuple[str, ...] = ("left_ankle_roll_Link", "right_ankle_roll_Link"),
        force_threshold: float = 1.0,
        x: tuple[float, float] = (0.0, 0.0),
        y: tuple[float, float] = (0.0, 0.0),
        z: tuple[float, float] = (0.0, 0.0),
        roll: tuple[float, float] = (0.0, 0.0),
        pitch: tuple[float, float] = (0.0, 0.0),
        yaw: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        if not 0.0 <= min_s <= max_s:
            raise ValueError("push interval requires 0 <= min_s <= max_s")
        self.asset = env.scene.articulations["robot"]
        self.contact_sensor = env.scene.sensors["contact_forces"]
        sensor_ids, _ = find_sensor_bodies(self.asset, self.contact_sensor, list(foot_body_names))
        self.sensor_ids = torch.tensor(sensor_ids, device=self.device)
        self.force_threshold = float(force_threshold)
        self.min_s, self.max_s = float(min_s), float(max_s)
        ranges = torch.tensor((x, y, z, roll, pitch, yaw), dtype=torch.float32, device=self.device)
        if torch.any(ranges[:, 1] < ranges[:, 0]):
            raise ValueError("push velocity ranges require low <= high")
        self.low, self.high = ranges[:, 0], ranges[:, 1]
        self.time_left_s = torch.zeros(self.num_envs, device=self.device)

    def _interval(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device) * (self.max_s - self.min_s) + self.min_s

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            self.time_left_s[env_ids] = self._interval(env_ids.numel())

    def update(self):
        self.time_left_s.sub_(self.env.step_dt).clamp_(min=0.0)
        due = self.time_left_s <= 1.0e-6
        in_contact = _contact_mask(self.contact_sensor, self.sensor_ids, self.force_threshold).any(dim=-1)
        ids = torch.nonzero(due & in_contact, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        _push_root_com_velocity(self.asset, ids, self.low, self.high)
        self.time_left_s[ids] = self._interval(ids.numel())


def _push_root_com_velocity(asset, ids: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> None:
    # IsaacLab ``push_by_setting_velocity``: root_vel_w is the root COM
    # velocity, perturbed additively and written back as COM velocity.
    delta = low + (high - low) * torch.rand(ids.numel(), 6, device=low.device)
    asset.write_root_com_velocity_to_sim(asset.data.root_com_vel_w[ids] + delta, env_ids=ids)


def _velocity_ranges(device, **ranges) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [ranges[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")], dtype=torch.float32, device=device
    )
    if values.shape != (6, 2) or torch.any(values[:, 1] < values[:, 0]):
        raise ValueError("push velocity ranges must be [low, high] pairs with low <= high")
    return values[:, 0], values[:, 1]


class depsic_a3_ultra_push(Randomization, namespace="mimic_lite"):
    """035 ``push_by_setting_velocity`` interval event.

    Matches IsaacLab's interval semantics: every env keeps a timer sampled in
    ``[min_s, max_s]``, re-sampled when it fires and on episode reset; the
    root COM velocity receives an additive uniform 6-D delta.
    """

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        min_s: float,
        max_s: float,
        x: tuple[float, float] = (0.0, 0.0),
        y: tuple[float, float] = (0.0, 0.0),
        z: tuple[float, float] = (0.0, 0.0),
        roll: tuple[float, float] = (0.0, 0.0),
        pitch: tuple[float, float] = (0.0, 0.0),
        yaw: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        if not 0.0 <= min_s <= max_s:
            raise ValueError("push interval requires 0 <= min_s <= max_s")
        self.asset = env.scene.articulations["robot"]
        self.min_s, self.max_s = float(min_s), float(max_s)
        self.low, self.high = _velocity_ranges(self.device, x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        self.time_left_s = self._interval(self.num_envs)

    def _interval(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device) * (self.max_s - self.min_s) + self.min_s

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            self.time_left_s[env_ids] = self._interval(env_ids.numel())

    def update(self):
        self.time_left_s.sub_(self.env.step_dt)
        ids = torch.nonzero(self.time_left_s < 1.0e-6, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self.time_left_s[ids] = self._interval(ids.numel())
        _push_root_com_velocity(self.asset, ids, self.low, self.high)


class depsic_a3_ultra_body_com(Randomization, namespace="mimic_lite"):
    """035 ``randomize_rigid_body_com``: per-axis additive startup offset."""

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        body_names: list[str] | tuple[str, ...],
        x: tuple[float, float] = (0.0, 0.0),
        y: tuple[float, float] = (0.0, 0.0),
        z: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        from mjlab.managers.event_manager import RecomputeLevel

        asset = env.scene.articulations["robot"]
        local_ids, names = find_bodies(asset, list(body_names))
        if len(names) != len(body_names):
            raise ValueError(f"CoM bodies not found: {body_names}, matched {names}")
        self.global_ids = asset.indexing.body_ids[torch.as_tensor(local_ids, device=self.device)]
        ranges = torch.tensor((x, y, z), dtype=torch.float32, device=self.device)
        if torch.any(ranges[:, 1] < ranges[:, 0]):
            raise ValueError("CoM ranges require low <= high")
        self.low, self.high = ranges[:, 0], ranges[:, 1]
        env.sim.expand_model_fields(("body_ipos",))
        self.recompute_level = RecomputeLevel.set_const
        _mjlab_ensure_recompute_fields_expanded(env, self.recompute_level)

    def startup(self) -> None:
        model = self.env.sim.model
        ipos = model.body_ipos[:, self.global_ids]
        offset = self.low + (self.high - self.low) * torch.rand_like(ipos)
        model.body_ipos[:, self.global_ids] = ipos + offset
        _mjlab_recompute_constants(self.env, self.recompute_level)


class depsic_a3_ultra_joint_params(Randomization, namespace="mimic_lite"):
    """035 ``randomize_joint_parameters`` (mode=reset, scale, uniform).

    Armature and frictionloss are re-drawn from the nominal values on episode
    reset, but at most once every ``min_step_count_between_reset`` global env
    steps per env (IsaacLab EventManager semantics, including the unconditional
    first trigger).  Each group is ``{joint regex: ...}`` sharing one armature
    and one friction scale range.
    """

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        groups: Mapping[str, Mapping[str, object]],
        min_step_count_between_reset: int = 0,
        randomize_armature: bool = True,
    ):
        super().__init__(env)
        from mjlab.managers.event_manager import RecomputeLevel

        if int(min_step_count_between_reset) < 0:
            raise ValueError("min_step_count_between_reset must be non-negative")
        self.min_steps = int(min_step_count_between_reset)
        self.randomize_armature = bool(randomize_armature)
        asset = env.scene.articulations["robot"]
        sim = env.sim
        sim.expand_model_fields(
            ("dof_armature", "dof_frictionloss", "dof_invweight0", "body_invweight0", "tendon_length0", "tendon_invweight0")
        )
        default_armature = sim.get_default_field("dof_armature").to(self.device)
        default_friction = sim.get_default_field("dof_frictionloss").to(self.device)
        dof_ids, arm_ranges, fric_ranges = [], [], []
        claimed: set[int] = set()
        for group_name, group in groups.items():
            ids, _ = find_joints(asset, list(group["joint_names"]))
            if not ids:
                raise ValueError(f"joint parameter group {group_name!r} matched no joint")
            overlap = claimed.intersection(ids)
            if overlap:
                raise ValueError(f"joint parameter group {group_name!r} repeats joints {sorted(overlap)}")
            claimed.update(ids)
            dofs = asset.indexing.joint_v_adr[torch.as_tensor(ids, device=self.device)].tolist()
            dof_ids += dofs
            arm_ranges += [tuple(group["armature_scale"])] * len(dofs)
            fric_ranges += [tuple(group["friction_scale"])] * len(dofs)
        self.dof_ids = torch.tensor(dof_ids, dtype=torch.long, device=self.device)
        arm = torch.tensor(arm_ranges, dtype=torch.float32, device=self.device)
        fric = torch.tensor(fric_ranges, dtype=torch.float32, device=self.device)
        if torch.any(arm[:, 1] < arm[:, 0]) or torch.any(fric[:, 1] < fric[:, 0]) or torch.any(arm <= 0) or torch.any(fric < 0):
            raise ValueError("joint parameter scale ranges must be non-negative with low <= high")
        self.arm_low, self.arm_high = arm[:, 0], arm[:, 1]
        self.fric_low, self.fric_high = fric[:, 0], fric[:, 1]
        self.default_armature = default_armature[self.dof_ids]
        self.default_friction = default_friction[self.dof_ids]
        self.recompute_level = RecomputeLevel.set_const_0
        self.last_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.triggered_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._global_env_step = 0

    def update(self):
        self._global_env_step += 1

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self.device)
        step = self._global_env_step
        if self.min_steps > 0:
            last = self.last_step[env_ids]
            valid = (step - last) >= self.min_steps
            valid |= (last == 0) & ~self.triggered_once[env_ids]
            env_ids = env_ids[valid]
            if env_ids.numel() == 0:
                return
        self.last_step[env_ids] = step
        self.triggered_once[env_ids] = True
        n = env_ids.numel()
        rows = env_ids[:, None]
        arm_scale = self.arm_low + (self.arm_high - self.arm_low) * torch.rand(n, self.dof_ids.numel(), device=self.device)
        fric_scale = self.fric_low + (self.fric_high - self.fric_low) * torch.rand(n, self.dof_ids.numel(), device=self.device)
        model = self.env.sim.model
        if self.randomize_armature:
            model.dof_armature[rows, self.dof_ids] = self.default_armature * arm_scale
        model.dof_frictionloss[rows, self.dof_ids] = self.default_friction * fric_scale
        # Armature enters the constraint inverse weights.  set_const_0 runs
        # at qpos0 and restores qpos; the reset path forwards afterwards.
        if self.randomize_armature:
            _mjlab_recompute_constants(self.env, self.recompute_level)


class depsic_a3_ultra_contact_friction(Randomization, namespace="mimic_lite"):
    """035 ``randomize_rigid_body_material`` static friction on robot geoms.

    PhysX combines the robot and terrain friction by ``multiply`` with a unit
    terrain, so the effective contact friction equals the sampled robot value.
    MuJoCo instead takes the per-axis maximum of the two geoms, which would
    clamp every sample below the terrain's 1.0. Lowering only the flat
    ground's sliding friction to the sampling lower bound makes that maximum
    equal the robot sample. Keep priorities and solver parameters unchanged:
    raising priority would also change contact stiffness and impedance.
    Samples come from ``num_buckets`` random buckets, one per geom and env.

    MuJoCo has one sliding coefficient, so the dynamic-friction and
    restitution ranges have no direct counterpart and are not applied.
    """

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        static_friction_range: tuple[float, float],
        body_names: str = ".*",
        num_buckets: int = 64,
        plane_only: bool = True,
    ):
        super().__init__(env)
        low, high = map(float, static_friction_range)
        if not 0.0 < low <= high or int(num_buckets) < 1:
            raise ValueError("invalid contact friction range or bucket count")
        asset = env.scene.articulations["robot"]
        local_ids, _ = find_bodies(asset, body_names)
        selected = set(asset.indexing.body_ids[torch.as_tensor(local_ids, device=self.device)].tolist())
        mj_model = env.sim.mj_model
        import mujoco

        robot_geoms = set(asset.indexing.geom_ids.tolist())
        geoms = [
            int(g) for g in asset.indexing.geom_ids.tolist()
            if int(mj_model.geom_bodyid[g]) in selected
            and (int(mj_model.geom_contype[g]) or int(mj_model.geom_conaffinity[g]))
        ]
        if not geoms:
            raise ValueError("contact friction matched no collision geom")
        ground_geoms = [
            g for g in range(mj_model.ngeom)
            if g not in robot_geoms
            and (int(mj_model.geom_contype[g]) or int(mj_model.geom_conaffinity[g]))
        ]
        if not ground_geoms or (plane_only and any(
            mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE for g in ground_geoms
        )):
            raise ValueError("Ultra S2R contact friction requires a plane-only ground")
        if any(mj_model.body_weldid[mj_model.geom_bodyid[g]] != 0 for g in ground_geoms):
            raise ValueError("Ultra S2R ground friction requires static world-welded terrain")
        if any(mj_model.geom_priority[g] != mj_model.geom_priority[r] for g in ground_geoms for r in geoms):
            raise ValueError("Ultra S2R friction requires equal robot/ground geom priorities")
        self.geom_ids = torch.tensor(geoms, dtype=torch.long, device=self.device)
        self.buckets = torch.rand(int(num_buckets), device=self.device) * (high - low) + low
        env.sim.expand_model_fields(("geom_friction",))
        # Leave torsional/rolling friction and all contact solver fields intact.
        ground_ids = torch.tensor(ground_geoms, dtype=torch.long, device=self.device)
        ground_friction = env.sim.model.geom_friction[:, ground_ids, 0]
        env.sim.model.geom_friction[:, ground_ids, 0] = ground_friction.clamp_max(low)
        mj_model.geom_friction[ground_geoms, 0] = mj_model.geom_friction[ground_geoms, 0].clip(max=low)

    def startup(self) -> None:
        idx = torch.randint(self.buckets.numel(), (self.num_envs, self.geom_ids.numel()), device=self.device)
        self.env.sim.model.geom_friction[:, self.geom_ids, 0] = self.buckets[idx]


class depsic_a3_ultra_body_mass_scale(Randomization, namespace="mimic_lite"):
    """035 ``randomize_rigid_body_mass`` (scale, recompute_inertia) at startup."""

    supported_backends = ("mjlab",)

    def __init__(self, env, body_names: str | list[str], low: float, high: float):
        super().__init__(env)
        from mjlab.managers.event_manager import RecomputeLevel

        if not 0.0 < low <= high:
            raise ValueError("mass scale requires 0 < low <= high")
        asset = env.scene.articulations["robot"]
        local_ids, names = find_bodies(asset, body_names)
        if not names:
            raise ValueError(f"mass scale matched no body: {body_names}")
        self.global_ids = asset.indexing.body_ids[torch.as_tensor(local_ids, device=self.device)]
        env.sim.expand_model_fields(("body_mass", "body_inertia"))
        self.recompute_level = RecomputeLevel.set_const
        _mjlab_ensure_recompute_fields_expanded(env, self.recompute_level)
        self.low, self.high = float(low), float(high)

    def startup(self) -> None:
        model = self.env.sim.model
        scale = torch.empty(self.num_envs, self.global_ids.numel(), device=self.device).uniform_(self.low, self.high)
        model.body_mass[:, self.global_ids] = model.body_mass[:, self.global_ids] * scale
        model.body_inertia[:, self.global_ids] = model.body_inertia[:, self.global_ids] * scale[..., None]
        _mjlab_recompute_constants(self.env, self.recompute_level)


class depsic_a3_ultra_body_mass_add(Randomization, namespace="mimic_lite"):
    """Startup additive payload on selected bodies with a minimum-mass clamp."""

    supported_backends = ("mjlab",)

    def __init__(
        self,
        env,
        body_names: list[str] | tuple[str, ...],
        low: float,
        high: float,
        min_mass: float = 0.05,
    ):
        super().__init__(env)
        from mjlab.managers.event_manager import RecomputeLevel

        if high < low:
            raise ValueError("payload mass requires low <= high")
        asset = env.scene.articulations["robot"]
        local_ids, names = find_bodies(asset, list(body_names))
        if len(names) != len(body_names):
            raise ValueError(f"payload bodies not found: {body_names}, matched {names}")
        self.global_ids = asset.indexing.body_ids[torch.as_tensor(local_ids, device=self.device)]
        env.sim.expand_model_fields(("body_mass", "body_inertia"))
        self.recompute_level = RecomputeLevel.set_const
        _mjlab_ensure_recompute_fields_expanded(env, self.recompute_level)
        self.low, self.high, self.min_mass = float(low), float(high), float(min_mass)

    def startup(self) -> None:
        model = self.env.sim.model
        mass = model.body_mass[:, self.global_ids].clone()
        inertia = model.body_inertia[:, self.global_ids].clone()
        offset = torch.empty_like(mass).uniform_(self.low, self.high)
        new_mass = (mass + offset).clamp_min(self.min_mass)
        model.body_mass[:, self.global_ids] = new_mass
        model.body_inertia[:, self.global_ids] = inertia * (new_mass / mass.clamp_min(1.0e-6))[..., None]
        _mjlab_recompute_constants(self.env, self.recompute_level)


__all__ = [
    "A3UltraMotorState",
    "depsic_a3_ultra_motor_tn",
    "depsic_a3_ultra_ankle_motor_constraint",
    "depsic_a3_ultra_parallel_cross",
    "depsic_a3_ultra_ankle_position",
    "depsic_a3_ultra_push",
    "depsic_a3_ultra_push_if_in_contact",
    "depsic_a3_ultra_body_com",
    "depsic_a3_ultra_joint_params",
    "depsic_a3_ultra_contact_friction",
    "depsic_a3_ultra_body_mass_scale",
    "depsic_a3_ultra_body_mass_add",
]
