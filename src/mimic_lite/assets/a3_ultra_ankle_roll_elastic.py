"""Rigid-foot Ultra with two native serial-elastic roll transmissions."""

from dataclasses import dataclass, asdict, replace
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator.actuator import Actuator
from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg
from mjlab.utils.spec import create_position_actuator

from active_adaptation.registry import Registry
from mimic_lite.assets.a3_ultra import make_a3_ultra_depsic_v1_cfg, ultra_mjcf_path
from mimic_lite.assets.a3_ultra_s2r_plus import hand_internal_body_pairs

ROLL_JOINTS = ("left_ankle_roll_joint", "right_ankle_roll_joint")
MOTOR_JOINTS = tuple(name.replace("_joint", "_motor_joint") for name in ROLL_JOINTS)
ROTOR_INERTIA = 0.020137175616
HALF_PITCH_ARMATURE = 0.032220302655
HALF_ROLL_INERTIA = ROTOR_INERTIA * 0.5


def elastic_spec(*, rotor_inertia=ROTOR_INERTIA, compliant_foot=False):
    source = ultra_mjcf_path()
    root = ET.parse(source).getroot()
    for element in list(root):
        if element.tag in {"actuator", "keyframe"} or (
            element.tag == "include" and element.get("file") == "../../terrain/mujoco/flat.xml"
        ):
            root.remove(element)
    # Reuse only the hand filter, NOT the passive-foot/contact package. Warp
    # produced a NaN witness for L_index_pip/L_middle_pip in the captured run.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {frozenset((e.get("body1"), e.get("body2"))) for e in contact.findall("exclude")}
    for body1, body2 in hand_internal_body_pairs(root):
        if frozenset((body1, body2)) not in existing:
            ET.SubElement(contact, "exclude", body1=body1, body2=body2)
    tendons = root.find("tendon")
    if tendons is None:
        tendons = ET.SubElement(root, "tendon")
    for side, output, motor in zip(("left", "right"), ROLL_JOINTS, MOTOR_JOINTS):
        joint = root.find(f".//joint[@name='{output}']")
        joint.set("armature", "0")
        joint.set("damping", "2.0")
        joint.set("frictionloss", "0.25")
        # Abstract upstream rotor, fixed to world as in the validated replay.
        # No collision geoms or translational DOFs; its tiny mass is not payload.
        body = ET.SubElement(root.find("worldbody"), "body", name=f"{side}_virtual_ankle_rotor", pos="0 0 3")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.000001",
                      diaginertia=" ".join([str(rotor_inertia)] * 3))
        ET.SubElement(body, "joint", name=motor, type="hinge", axis="0 0 1",
                      limited="false", damping="0", frictionloss="0", armature="0")
        tendon = ET.SubElement(tendons, "fixed", name=f"{side}_ankle_roll_transmission",
                               stiffness="77.5", damping="1.0", springlength="0")
        ET.SubElement(tendon, "joint", joint=motor, coef="1")
        ET.SubElement(tendon, "joint", joint=output, coef="-1")
        if compliant_foot:
            foot = root.find(f".//body[@name='{side}_ankle_roll_Link']")
            for geom in foot.findall("geom"):
                if geom.get("type") == "sphere":
                    # Released 035 MuJoCo sole contact, not dimensional PhysX
                    # stiffness/damping. Keep the original rigid shoe geometry.
                    geom.set("solref", "0.018 1")
                    geom.set("solimp", "0.80 0.95 0.0129123 0.5 2")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelfiledir = str(source.parent)
    return spec


@dataclass(kw_only=True)
class RollMotorPositionCfg(BuiltinPositionActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return RollMotorPosition(self, entity, target_ids, target_names)


class RollMotorPosition(Actuator[RollMotorPositionCfg]):
    """Keep output-joint command/delay indexing; drive only its upstream rotor."""

    @property
    def command_field(self):
        return "position"

    def compute(self, cmd):
        return cmd.position_target

    def edit_spec(self, spec, target_names):
        for output in target_names:
            motor = output.replace("_joint", "_motor_joint")
            actuator = create_position_actuator(
                spec, motor, stiffness=self.cfg.stiffness, damping=self.cfg.damping,
                effort_limit=self.cfg.effort_limit,
                armature=0., frictionloss=0., viscous_damping=0.,
            )
            actuator.name = output
            self._mjs_actuators.append(actuator)


def make_elastic_cfg(backend):
    cfg, sensors = make_a3_ultra_depsic_v1_cfg(backend)
    cfg.spec_fn = elastic_spec
    cfg.articulation.actuators = tuple(
        RollMotorPositionCfg(**{**asdict(a), "stiffness": 50., "damping": 2.,
                                "effort_limit": 54.75, "armature": 0., "frictionloss": 0.})
        if tuple(a.target_names_expr) in [(name,) for name in ROLL_JOINTS] else a
        for a in cfg.articulation.actuators
    )
    spec = elastic_spec()
    cfg.joint_names_simulation = [j.name for j in spec.joints if j.name != "free_joint"]
    cfg.body_names_simulation = [b.name for b in spec.bodies if b.name != "world"]
    return cfg, sensors


Registry.instance().register("asset", "a3-ultra-ankle-roll-elastic", make_elastic_cfg)


def elastic_dr_spec():
    return elastic_spec(rotor_inertia=HALF_ROLL_INERTIA, compliant_foot=True)


def make_elastic_dr_cfg(backend):
    """New asset identity keeps old checkpoint physics reproducible."""
    from mimic_lite.depsic_a3_contract import A3_POLICY_JOINT_NAMES

    cfg, sensors = make_elastic_cfg(backend)
    cfg.spec_fn = elastic_dr_spec
    actuators = []
    for actuator in cfg.articulation.actuators:
        name, = actuator.target_names_expr
        changes = {}
        if name in A3_POLICY_JOINT_NAMES:
            # Capacity for 10 ms at the task's 0.5 ms physical timestep.
            # A reset event sets discrete lags and shares them across the group.
            changes.update(delay_max_lag=20, delay_hold_prob=1.0)
        if "ankle" in name:
            changes.update(stiffness=60., damping=1.2)
            if "pitch" in name:
                changes["armature"] = HALF_PITCH_ARMATURE
        actuators.append(replace(actuator, **changes))
    cfg.articulation.actuators = tuple(actuators)
    return cfg, sensors


Registry.instance().register("asset", "a3-ultra-ankle-roll-elastic-dr", make_elastic_dr_cfg)
