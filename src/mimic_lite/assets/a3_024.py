"""A3 024 closed-chain MuJoCo asset for the mjlab DEPSIC tasks.

The released 024 MJCF contains the policy joints *and* the physical motor,
rod, ball, and passive-foot joints.  This module keeps that physical model
intact while exposing the same scalar 31-joint view used by the DEPSIC policy.
Ankle and waist targets are converted through the published position-dependent
Jacobian tables before being written to the six XML motor actuators.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from active_adaptation.assets.asset_cfg import ContactSensorCfg, EntityCfg as ActiveEntityCfg
from active_adaptation.registry import Registry
from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd, CommandField
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg, EntityData, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.string import resolve_expr

from mimic_lite.a3_024_motor_model import (
    PARALLEL_PAIRS,
    ParallelJacobianTable,
    motor_damping_from_joint_damping,
    motor_torque_from_joint_torque,
    motor_velocity_from_joint_velocity,
)
from mimic_lite.depsic_a3_contract import A3_HEAD_JOINT_NAMES, A3_POLICY_JOINT_NAMES


registry = Registry.instance()
A3_024_ASSET_DIR = Path(__file__).resolve().parent / "a3_024"
A3_024_MJCF_PATH = A3_024_ASSET_DIR / "mjcf" / "a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
A3_024_URDF_PATH = A3_024_ASSET_DIR / "urdf" / "a3" / "model_collision_optimized_passive_foot_twostage_fit_optimized.urdf"

# The published 024 asset is a closed-chain model.  The sim2real release also
# trains a serial equivalent: it retains the three-piece compliant foot, but
# removes the two ankle and one waist four-bar mechanisms.  Building the
# serial variant from the versioned source MJCF (rather than maintaining a
# second hand-edited XML) makes the exact removed physical topology explicit.
_SERIAL_LOOP_EQUALITIES = (
    "left_ankle_long_loop", "left_ankle_short_loop",
    "right_ankle_long_loop", "right_ankle_short_loop",
    "left_waist_loop", "right_waist_loop",
)
_SERIAL_LOOP_BODIES = (
    "left_ankle_motor_up_Link", "left_ankle_motor_down_Link",
    "right_ankle_motor_up_Link", "right_ankle_motor_down_Link",
    "left_waist_motor_Link", "right_waist_motor_Link",
)
_SERIAL_DIRECT_TRANSMISSIONS = {
    "left_ankle_pitch_joint": "left_ankle_pitch_joint",
    "left_ankle_roll_joint": "left_ankle_roll_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
    "right_ankle_roll_joint": "right_ankle_roll_joint",
    "waist_roll_joint": "waist_roll_joint",
    "waist_pitch_joint": "waist_pitch_joint",
}
_SERIAL_MOTOR_TO_JOINT = {
    "left_ankle_motor_up_joint": "left_ankle_pitch_joint",
    "left_ankle_motor_down_joint": "left_ankle_roll_joint",
    "right_ankle_motor_up_joint": "right_ankle_pitch_joint",
    "right_ankle_motor_down_joint": "right_ankle_roll_joint",
    "left_waist_motor_joint": "waist_roll_joint",
    "right_waist_motor_joint": "waist_pitch_joint",
}

PARALLEL_NAMES = {
    "left_ankle": ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
    "right_ankle": ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    "waist": ("waist_pitch_joint", "waist_roll_joint"),
}
PARALLEL_MOTOR_NAMES = {
    "left_ankle": ("left_ankle_motor_up_joint", "left_ankle_motor_down_joint"),
    "right_ankle": ("right_ankle_motor_up_joint", "right_ankle_motor_down_joint"),
    "waist": ("left_waist_motor_joint", "right_waist_motor_joint"),
}
PARALLEL_TABLE_NAMES = {"left_ankle": "left_ankle", "right_ankle": "right_ankle", "waist": "waist"}
PASSIVE_FOOT_JOINT_NAMES = (
    "left_foot_forefoot_joint", "left_foot_toe_joint",
    "right_foot_forefoot_joint", "right_foot_toe_joint",
)
BALL_JOINT_NAMES = (
    "left_waist_rod_up_joint", "right_waist_rod_up_joint",
    "left_ankle_rod_long_up_joint", "left_ankle_rod_short_up_joint",
    "right_ankle_rod_long_up_joint", "right_ankle_rod_short_up_joint",
)

# Logical position-PD values from the released ``A3_024_CYLINDER_CFG``.  Its
# controller changes are intentionally kept separate from the base A3:
# knee/waist effort is lower and the closed-chain ankles use 60 / 1.2 gains.
# The XML motors on the parallel mechanisms additionally retain their
# physical 60 Nm force range.
KP = {"knee": 250.0, "ankle": 60.0, "waist": 50.0}
KD = {"knee": 8.0, "ankle": 1.2, "waist": 2.0}
EFFORT = {
    "knee": 300.0, "ankle_pitch": 118.2, "ankle_roll": 54.75,
    "waist_yaw": 200.0, "waist_roll": 42.0, "waist_pitch": 105.0,
    "hip": 220.0, "shoulder_pitch_roll": 60.0, "shoulder_yaw_elbow_wrist_roll": 24.0,
    "wrist": 6.0, "head": 6.0,
}

def _joint_gain(name: str) -> tuple[float, float, float]:
    if "ankle_" in name:
        if name.endswith("pitch_joint"):
            return KP["ankle"], KD["ankle"], EFFORT["ankle_pitch"]
        return KP["ankle"], KD["ankle"], EFFORT["ankle_roll"]
    if name.startswith("waist_"):
        if name == "waist_yaw_joint":
            return 85.0, 3.0, EFFORT["waist_yaw"]
        return KP["waist"], KD["waist"], EFFORT["waist_roll" if name.endswith("roll_joint") else "waist_pitch"]
    if name.endswith("knee_joint"):
        return KP["knee"], KD["knee"], EFFORT["knee"]
    if any(x in name for x in ("hip_pitch", "hip_roll", "hip_yaw")):
        return 80.0 if name.endswith(("pitch_joint", "yaw_joint")) else 120.0, 3.0 if not name.endswith("roll_joint") else 4.0, EFFORT["hip"]
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return 40.0, 3.0, EFFORT["shoulder_pitch_roll"]
    if any(x in name for x in ("shoulder_yaw", "elbow", "wrist_roll")):
        return 30.0 if "shoulder_yaw" in name or "elbow" in name else 30.0, 2.0, EFFORT["shoulder_yaw_elbow_wrist_roll"]
    if "wrist_" in name:
        return 20.0, 2.0, EFFORT["wrist"]
    if name in A3_HEAD_JOINT_NAMES:
        return 20.0, 1.0, EFFORT["head"]
    return 0.0, 0.0, 0.0


def _all_body_names() -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(A3_024_MJCF_PATH))
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]


def _make_a3_024_mjlab_spec() -> mujoco.MjSpec:
    """Load the released 024 MJCF, including its calibrated contact floor.

    The fitted passive-foot model was released together with this floor's
    contact parameters.  024 mjlab tasks therefore suppress mjlab's generic
    plane at scene construction rather than deleting this geom or allowing
    two coincident collision planes.
    """
    return mujoco.MjSpec.from_file(str(A3_024_MJCF_PATH))


# ``CollisionCfg`` assigns its defaults for contype/conaffinity/condim to
# every matched geom.  The 024 XML floor deliberately has a different mask
# (contype=0, conaffinity=7): it is the passive-foot model's calibrated world
# contact surface, not a robot self-collider.  Keep it out of the generic
# robot collision edit so attaching the asset to mjlab preserves that released
# contact contract verbatim.
_A3_024_ROBOT_COLLISION_GEOMS = (r"^(?!floor$).*",)


def _make_a3_024_serial_spec() -> mujoco.MjSpec:
    """Return the released A3 024 model with only serial policy joints.

    This deliberately does *not* remove the rear/forefoot/toe subtree.  The
    latter is the fitted passive foot used to make toe loading a physical
    consequence of a policy action, not a separate policy DoF.
    """
    spec = _make_a3_024_mjlab_spec()

    # Retarget first: MjSpec deletes an actuator/sensor when its source joint
    # disappears with a body subtree.
    for actuator_name, joint_name in _SERIAL_DIRECT_TRANSMISSIONS.items():
        spec.actuator(actuator_name).target = joint_name
    for sensor in spec.sensors:
        if sensor.objname in _SERIAL_MOTOR_TO_JOINT:
            sensor.objname = _SERIAL_MOTOR_TO_JOINT[sensor.objname]

    for equality_name in _SERIAL_LOOP_EQUALITIES:
        spec.delete(spec.equality(equality_name))
    for body_name in _SERIAL_LOOP_BODIES:
        spec.delete(spec.body(body_name))
    return spec


def _serial_body_names() -> list[str]:
    model = _make_a3_024_serial_spec().compile()
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]


class A3_024EntityData(EntityData):
    """EntityData projection with scalar logical values for every MJCF joint."""

    def _scalar_qpos(self) -> torch.Tensor:
        return self.data.qpos[:, self._logical_qadr]

    def _scalar_qvel(self) -> torch.Tensor:
        return self.data.qvel[:, self._logical_vadr]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._scalar_qpos()

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        return self.joint_pos + self.encoder_bias

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._scalar_qvel()

    @property
    def joint_acc(self) -> torch.Tensor:
        return self.data.qacc[:, self._logical_vadr]

    @property
    def a3_024_requested_joint_torque(self) -> torch.Tensor:
        actuator = getattr(self._entity, "_a3_024_actuator", None)
        if actuator is None:
            return self.joint_pos_target.new_zeros(self.joint_pos_target.shape)
        return actuator._joint_requested

    @property
    def a3_024_parallel_motor_torque(self) -> torch.Tensor:
        actuator = getattr(self._entity, "_a3_024_actuator", None)
        if actuator is None:
            return self.joint_pos_target.new_zeros((self.data.nworld, 6))
        return actuator._parallel_torque

    @property
    def a3_024_parallel_motor_velocity(self) -> torch.Tensor:
        actuator = getattr(self._entity, "_a3_024_actuator", None)
        if actuator is None:
            return self.joint_pos_target.new_zeros((self.data.nworld, 6))
        return actuator._parallel_velocity

    def write_joint_position(self, position, joint_ids=None, env_ids=None):
        self._entity._write_logical_state(position, None, joint_ids, env_ids)

    def write_joint_velocity(self, velocity, joint_ids=None, env_ids=None):
        self._entity._write_logical_state(None, velocity, joint_ids, env_ids)

    def write_joint_state(self, position, velocity, joint_ids=None, env_ids=None):
        self._entity._write_logical_state(position, velocity, joint_ids, env_ids)


class A3_024LoopActuatorCfg(ActuatorCfg):
    target_names_expr: tuple[str, ...] = tuple(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES)
    transmission_type: TransmissionType = TransmissionType.JOINT

    def build(self, entity, target_ids, target_names):
        return A3_024LoopActuator(self, entity, target_ids, target_names)


class A3_024LoopActuator(Actuator[A3_024LoopActuatorCfg]):
    """Position-PD actuator that writes the XML's serial and loop motors."""

    @property
    def command_field(self) -> CommandField:
        return "position"

    def edit_spec(self, spec, target_names):
        # XML actuator names preserve the logical policy names. Their target
        # joints are physical motor joints for ankles/waist and direct joints
        # elsewhere. Reuse them instead of adding a second set of actuators.
        by_name = {act.name: act for act in spec.actuators}
        for name in target_names:
            try:
                self._mjs_actuators.append(by_name[name])
            except KeyError as exc:
                raise ValueError(f"A3 024 XML actuator {name!r} is missing") from exc

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._name_to_col = {name: i for i, name in enumerate(self.target_names)}
        self._table = {
            name: ParallelJacobianTable.load(name, device=device)
            for name in PARALLEL_NAMES
        }
        self._logical_to_index = {name: i for i, name in enumerate(self.entity.joint_names)}
        self._logical_target_ids = torch.tensor(
            [self._logical_to_index[name] for name in self.target_names],
            dtype=torch.long,
            device=device,
        )
        self._requested = torch.zeros(data.nworld, len(self.entity.joint_names), device=device)
        self._parallel_torque = torch.zeros(data.nworld, 6, device=device)
        self._parallel_velocity = torch.zeros(data.nworld, 6, device=device)
        self.entity._a3_024_actuator = self

    def _pair(self, pair_name, cmd):
        names = PARALLEL_NAMES[pair_name]
        cols = [self._name_to_col[n] for n in names]
        q = cmd.pos[:, cols]
        dq = cmd.vel[:, cols]
        target = cmd.position_target[:, cols]
        kp = q.new_tensor([KP["ankle"] if "ankle" in pair_name else KP["waist"]] * 2)
        kd = q.new_tensor([KD["ankle"] if "ankle" in pair_name else KD["waist"]] * 2)
        table = self._table[pair_name]
        jac = table.interpolate_jacobian(q)
        motor_dq = motor_velocity_from_joint_velocity(jac, dq)
        position_torque = motor_torque_from_joint_torque(jac, kp * (target - q))
        # This is exactly the documented solver sequence used by the 024
        # MuJoCo sim2sim controller:
        #   RlConvertPos(kp, error) -> J.T @ error, then multiply by kp;
        #   RlConvertKd(kd) -> diag(J.T @ diag(kd) @ J).
        # Its motor-space damping is deliberately diagonal, so mapping the
        # full joint PD torque with J.T would introduce off-diagonal damping
        # that the released controller does not command.
        motor_kd = motor_damping_from_joint_damping(jac, kd)
        motor_tau = position_torque - motor_kd * motor_dq
        # Rewards use the training-side pre-clip implicit joint PD torque,
        # matching the released reward's ``computed_torque`` input rather
        # than the actuator's motor-space command.
        joint_tau = kp * (target - q) - kd * dq
        return motor_tau, motor_dq, joint_tau, cols

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        result = torch.zeros_like(cmd.position_target)
        # ``result`` is the command that reaches MuJoCo and can therefore be
        # limited by the logical actuator and by the XML force range.  Keep a
        # second, logical tensor for the 024 reward: the release defines its
        # T--N and cross penalties from implicit-PD torque *before* either
        # actuator limit is applied.  Reconstructing it later from ``result``
        # would silently turn a saturated command into an in-range one.
        logical_requested = torch.zeros(
            cmd.position_target.shape[0], len(self.entity.joint_names), device=cmd.position_target.device,
            dtype=cmd.position_target.dtype,
        )
        for i, name in enumerate(self.target_names):
            if name not in {n for pair in PARALLEL_NAMES.values() for n in pair}:
                kp, kd, effort = _joint_gain(name)
                requested = kp * (cmd.position_target[:, i] - cmd.pos[:, i]) - kd * cmd.vel[:, i]
                logical_requested[:, self._logical_target_ids[i]] = requested
                result[:, i] = requested.clamp(-effort, effort)
        slot = 0
        for pair_name in ("left_ankle", "right_ankle", "waist"):
            motor_tau, motor_dq, joint_tau, cols = self._pair(pair_name, cmd)
            if pair_name == "waist":
                # Table order is [pitch, roll]; XML/policy control order is
                # [waist_roll motor, waist_pitch motor].
                result[:, cols[0]] = motor_tau[:, 1]
                result[:, cols[1]] = motor_tau[:, 0]
            else:
                result[:, cols] = motor_tau
            self._parallel_torque[:, slot : slot + 2] = motor_tau
            self._parallel_velocity[:, slot : slot + 2] = motor_dq
            logical_ids = torch.tensor(
                [self._logical_to_index[name] for name in PARALLEL_NAMES[pair_name]],
                dtype=torch.long,
                device=self._logical_target_ids.device,
            )
            logical_requested[:, logical_ids] = joint_tau
            slot += 2
        self._requested.copy_(logical_requested)
        self._joint_requested = self._requested
        return result


class A3_024SerialActuatorCfg(ActuatorCfg):
    """Direct joint actuator for the serial 024 training model."""

    target_names_expr: tuple[str, ...] = tuple(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES)
    transmission_type: TransmissionType = TransmissionType.JOINT

    def build(self, entity, target_ids, target_names):
        return A3_024SerialActuator(self, entity, target_ids, target_names)


class A3_024SerialActuator(Actuator[A3_024SerialActuatorCfg]):
    """Direct PD control while retaining motor-space reward estimates.

    The six parallel motors do not exist in the serial simulation.  Their
    torque and velocity are nevertheless reconstructed through the published
    Jacobian tables for the motor-envelope rewards.  The reconstruction is a
    reward-only feasibility check; it never feeds forces back into the serial
    dynamics.
    """

    @property
    def command_field(self) -> CommandField:
        return "position"

    def edit_spec(self, spec, target_names):
        by_name = {act.name: act for act in spec.actuators}
        for name in target_names:
            try:
                self._mjs_actuators.append(by_name[name])
            except KeyError as exc:
                raise ValueError(f"A3 024 serial XML actuator {name!r} is missing") from exc

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._name_to_col = {name: i for i, name in enumerate(self.target_names)}
        self._table = {
            name: ParallelJacobianTable.load(name, device=device)
            for name in PARALLEL_NAMES
        }
        self._logical_to_index = {name: i for i, name in enumerate(self.entity.joint_names)}
        self._logical_target_ids = torch.tensor(
            [self._logical_to_index[name] for name in self.target_names],
            dtype=torch.long,
            device=device,
        )
        self._requested = torch.zeros(data.nworld, len(self.entity.joint_names), device=device)
        self._parallel_torque = torch.zeros(data.nworld, 6, device=device)
        self._parallel_velocity = torch.zeros(data.nworld, 6, device=device)
        self.entity._a3_024_actuator = self

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        result = torch.zeros_like(cmd.position_target)
        requested = torch.zeros(
            cmd.position_target.shape[0],
            len(self.entity.joint_names),
            dtype=cmd.position_target.dtype,
            device=cmd.position_target.device,
        )
        for column, name in enumerate(self.target_names):
            kp, kd, effort = _joint_gain(name)
            joint_torque = kp * (cmd.position_target[:, column] - cmd.pos[:, column]) - kd * cmd.vel[:, column]
            requested[:, self._logical_target_ids[column]] = joint_torque
            result[:, column] = joint_torque.clamp(-effort, effort)

        slot = 0
        for pair_name in ("left_ankle", "right_ankle", "waist"):
            names = PARALLEL_NAMES[pair_name]
            columns = [self._name_to_col[name] for name in names]
            logical_ids = [self._logical_to_index[name] for name in names]
            jacobian = self._table[pair_name].interpolate_jacobian(cmd.pos[:, columns])
            self._parallel_torque[:, slot : slot + 2] = motor_torque_from_joint_torque(
                jacobian, requested[:, logical_ids]
            )
            self._parallel_velocity[:, slot : slot + 2] = motor_velocity_from_joint_velocity(
                jacobian, cmd.vel[:, columns]
            )
            slot += 2

        self._requested.copy_(requested)
        self._joint_requested = self._requested
        return result


class A3_024EntityCfg(ActiveEntityCfg):
    def build(self):
        return A3_024Entity(self)


class A3_024SerialEntityCfg(ActiveEntityCfg):
    def build(self):
        return A3_024SerialEntity(self)


class A3_024Entity(Entity):
    def _add_initial_state_keyframe(self):
        values = resolve_expr(self.cfg.init_state.joint_pos or {}, self.joint_names, 0.0)
        by_name = dict(zip(self.joint_names, values, strict=True))
        qpos: list[float] = [*self.cfg.init_state.pos, *self.cfg.init_state.rot]
        for joint in self._non_free_joints:
            if joint.type == mujoco.mjtJoint.mjJNT_BALL:
                qpos.extend((1.0, 0.0, 0.0, 0.0))
            elif joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            else:
                qpos.append(float(by_name.get(joint.name.split("/")[-1], 0.0)))
        self._spec.add_key(name="init_state", qpos=qpos)

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        logical_qadr = [int(mj_model.jnt_qposadr[j.id]) for j in self._non_free_joints]
        logical_vadr = [int(mj_model.jnt_dofadr[j.id]) for j in self._non_free_joints]
        projected = A3_024EntityData(**vars(self._data))
        projected._entity = self
        projected._logical_qadr = torch.tensor(logical_qadr, dtype=torch.long, device=device)
        projected._logical_vadr = torch.tensor(logical_vadr, dtype=torch.long, device=device)
        self._data = projected
        self._logical_names = tuple(self.joint_names)
        self._logical_index = {name: i for i, name in enumerate(self._logical_names)}
        self._physical_index = {name: i for i, name in enumerate(self._logical_names)}
        self._qadr_by_name = {name: int(mj_model.jnt_qposadr[j.id]) for j, name in zip(self._non_free_joints, self._logical_names, strict=True)}
        self._vadr_by_name = {name: int(mj_model.jnt_dofadr[j.id]) for j, name in zip(self._non_free_joints, self._logical_names, strict=True)}
        self._tables = {name: ParallelJacobianTable.load(name, device=device) for name in PARALLEL_NAMES}

    def _write_logical_state(self, position, velocity, joint_ids=None, env_ids=None):
        env = slice(None) if env_ids is None else env_ids
        n = self.data.qpos.shape[0] if isinstance(env, slice) else len(env)
        if position is None:
            position = self._data.joint_pos[env].clone()
        if velocity is None:
            velocity = self._data.joint_vel[env].clone()
        if joint_ids is not None:
            ids = joint_ids if isinstance(joint_ids, torch.Tensor) else torch.tensor(joint_ids, device=self._data.device)
            full_p = self._data.joint_pos[env].clone()
            full_v = self._data.joint_vel[env].clone()
            full_p[:, ids] = position
            full_v[:, ids] = velocity
            position, velocity = full_p, full_v
        del n
        env_index = env[:, None] if isinstance(env, torch.Tensor) else env
        self._data.data.qpos[env_index, self._data._logical_qadr] = position
        self._data.data.qvel[env_index, self._data._logical_vadr] = velocity
        # Ball joints have quaternion qpos and three qvel coordinates. They
        # are closed-chain coordinates, not policy states.
        for name in BALL_JOINT_NAMES:
            qadr, vadr = self._qadr_by_name[name], self._vadr_by_name[name]
            self._data.data.qpos[env_index, qadr : qadr + 4] = self._data.data.qpos.new_tensor([1.0, 0.0, 0.0, 0.0])
            self._data.data.qvel[env_index, vadr : vadr + 3] = 0.0
        for pair_name, logical in PARALLEL_NAMES.items():
            q = position[:, [self._logical_index[x] for x in logical]]
            dq = velocity[:, [self._logical_index[x] for x in logical]]
            motor_q = self._tables[pair_name].interpolate_motor_position(q)
            jac = self._tables[pair_name].interpolate_jacobian(q)
            motor_dq = motor_velocity_from_joint_velocity(jac, dq)
            for value, name in zip(motor_q.unbind(-1), PARALLEL_MOTOR_NAMES[pair_name], strict=True):
                self._data.data.qpos[env_index, self._qadr_by_name[name]] = value[:, None]
            for value, name in zip(motor_dq.unbind(-1), PARALLEL_MOTOR_NAMES[pair_name], strict=True):
                self._data.data.qvel[env_index, self._vadr_by_name[name]] = value[:, None]


class A3_024SerialEntity(A3_024Entity):
    """Serial 024 entity: reset only direct and passive physical joints.

    ``A3_024Entity`` projects a logical reset into motor and ball-joint loop
    coordinates.  Those coordinates do not exist in the serial variant, so
    normal scalar joint writes are exactly the desired reset behavior here.
    """

    def _write_logical_state(self, position, velocity, joint_ids=None, env_ids=None):
        env = slice(None) if env_ids is None else env_ids
        if position is None:
            position = self._data.joint_pos[env].clone()
        if velocity is None:
            velocity = self._data.joint_vel[env].clone()
        if joint_ids is not None:
            ids = joint_ids if isinstance(joint_ids, torch.Tensor) else torch.tensor(
                joint_ids, dtype=torch.long, device=self._data.device
            )
            full_position = self._data.joint_pos[env].clone()
            full_velocity = self._data.joint_vel[env].clone()
            full_position[:, ids] = position
            full_velocity[:, ids] = velocity
            position, velocity = full_position, full_velocity
        env_index = env[:, None] if isinstance(env, torch.Tensor) else env
        self._data.data.qpos[env_index, self._data._logical_qadr] = position
        self._data.data.qvel[env_index, self._data._logical_vadr] = velocity


def _initial_state():
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09),
        joint_pos={
            "left_hip_pitch_joint": -0.1311, "right_hip_pitch_joint": -0.1311,
            "left_hip_roll_joint": 0.0056, "right_hip_roll_joint": -0.0056,
            "left_hip_yaw_joint": -0.0348, "right_hip_yaw_joint": 0.0348,
            "left_knee_joint": 0.2468, "right_knee_joint": 0.2468,
            "left_ankle_pitch_joint": -0.1204, "right_ankle_pitch_joint": -0.1204,
            "left_ankle_roll_joint": -0.0078, "right_ankle_roll_joint": 0.0078,
            "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12,
            "left_elbow_joint": 0.8, "right_elbow_joint": 0.8,
            "head_yaw_joint": 0.0, "head_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )


def make_a3_024_cfg(backend: str):
    if backend != "mjlab":
        raise ValueError("a3-depsic-024 is implemented for mjlab only")
    cfg = A3_024EntityCfg(
        init_state=_initial_state(),
        spec_fn=_make_a3_024_mjlab_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(A3_024LoopActuatorCfg(
                target_names_expr=tuple(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES),
                delay_min_lag=0,
                delay_max_lag=2,
                delay_hold_prob=1.0,
                delay_update_period=1,
                delay_per_env_phase=False,
            ),),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=_A3_024_ROBOT_COLLISION_GEOMS,
                disable_other_geoms=False,
            ),
        ),
        joint_names_simulation=list(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES),
        body_names_simulation=_all_body_names(),
    )
    sensor = ContactSensorCfg(
        name="contact_forces", primary_contact_match_mode="body", primary_contact_match_pattern=r".*",
        primary_contact_match_entity="robot", track_air_time=True, history_length=4, reduce="netforce",
    ).mjlab()
    return cfg, (sensor,)


def make_a3_024_serial_cfg(backend: str):
    """Construct the 1 ms serial-plus-passive-foot A3 024 training asset."""
    if backend != "mjlab":
        raise ValueError("a3-depsic-024-serial is implemented for mjlab only")
    cfg = A3_024SerialEntityCfg(
        init_state=_initial_state(),
        spec_fn=_make_a3_024_serial_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(A3_024SerialActuatorCfg(
                target_names_expr=tuple(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES),
                delay_min_lag=0,
                delay_max_lag=2,
                delay_hold_prob=1.0,
                delay_update_period=1,
                delay_per_env_phase=False,
            ),),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=_A3_024_ROBOT_COLLISION_GEOMS,
                disable_other_geoms=False,
            ),
        ),
        joint_names_simulation=list(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES),
        body_names_simulation=_serial_body_names(),
    )
    sensor = ContactSensorCfg(
        name="contact_forces", primary_contact_match_mode="body", primary_contact_match_pattern=r".*",
        primary_contact_match_entity="robot", track_air_time=True, history_length=4, reduce="netforce",
    ).mjlab()
    return cfg, (sensor,)


registry.register("asset", "a3-depsic-024", make_a3_024_cfg)
registry.register("asset", "a3-depsic-024-serial", make_a3_024_serial_cfg)

__all__ = [
    "A3_024Entity", "A3_024EntityCfg", "A3_024_MJCF_PATH", "make_a3_024_cfg",
    "A3_024SerialEntity", "A3_024SerialActuator", "A3_024SerialActuatorCfg",
    "make_a3_024_serial_cfg", "_make_a3_024_serial_spec",
]
