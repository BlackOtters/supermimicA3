"""Four native serial transmissions and a topology-free rigid control asset."""

from dataclasses import dataclass, asdict, replace
import xml.etree.ElementTree as ET

import mujoco
from mjlab.utils.spec import create_position_actuator

from active_adaptation.registry import Registry
from mimic_lite.assets.a3_ultra import make_a3_ultra_depsic_v1_cfg, ultra_mjcf_path
from mimic_lite.assets.a3_ultra_ankle_roll_elastic import (
    elastic_spec, RollMotorPosition, RollMotorPositionCfg,
    HALF_PITCH_ARMATURE, HALF_ROLL_INERTIA,
)
from mimic_lite.depsic_a3_contract import A3_POLICY_JOINT_NAMES

ANKLES = tuple(f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll"))
ENCODERS = tuple(n.replace("_joint", "_motor_joint") for n in ANKLES)
ROTORS = tuple(n.replace("_joint", "_rotor") for n in ANKLES)
TRANSMISSIONS = tuple(n.replace("_joint", "_transmission") for n in ANKLES)
INERTIAS = (HALF_PITCH_ARMATURE, HALF_ROLL_INERTIA) * 2


def four_ankle_spec(*, elastic=True):
    root = ET.fromstring(elastic_spec().to_xml())
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if "virtual_ankle_rotor" in body.get("name", ""):
            world.remove(body)
    tendon = root.find("tendon")
    for item in list(tendon):
        if "ankle_roll_transmission" in item.get("name", ""):
            tendon.remove(item)
    for output, motor, rotor, transmission, inertia in zip(ANKLES, ENCODERS, ROTORS, TRANSMISSIONS, INERTIAS):
        joint = root.find(f".//joint[@name='{output}']")
        joint.set("armature", str(0. if elastic else inertia))
        joint.set("frictionloss", "0.25")
        joint.set("damping", "2.0")
        if elastic:
            body = ET.SubElement(world, "body", name=rotor, pos="0 0 3")
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.000001", diaginertia=f"{inertia} {inertia} {inertia}")
            ET.SubElement(body, "joint", name=motor, type="hinge", axis="0 0 1", limited="false",
                          damping="0", frictionloss="0", armature="0")
            spring = ET.SubElement(tendon, "fixed", name=transmission, stiffness="0", damping="0", springlength="0")
            ET.SubElement(spring, "joint", joint=motor, coef="1")
            ET.SubElement(spring, "joint", joint=output, coef="-1")
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_ankle_roll_Link']")
        for geom in foot.findall("geom"):
            if geom.get("type") == "sphere":
                geom.set("solref", "0.018 1")
                geom.set("solimp", "0.80 0.95 0.0129123 0.5 2")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelfiledir = str(ultra_mjcf_path().parent)
    return spec


def rigid_ankle_spec():
    return four_ankle_spec(elastic=False)


@dataclass(kw_only=True)
class SwitchableAnkleCfg(RollMotorPositionCfg):
    def build(self, entity, target_ids, target_names):
        return SwitchableAnkle(self, entity, target_ids, target_names)


class SwitchableAnkle(RollMotorPosition):
    """One policy target/delay buffer, two mutually exclusive native PD drives."""

    def compute(self, cmd):
        return cmd.position_target.repeat_interleave(2, dim=-1)

    def edit_spec(self, spec, target_names):
        for output in target_names:
            for joint, name in ((output, output), (output.replace("_joint", "_motor_joint"), output + "_encoder_drive")):
                act = create_position_actuator(spec, joint, stiffness=self.cfg.stiffness,
                    damping=self.cfg.damping, effort_limit=self.cfg.effort_limit,
                    armature=0., frictionloss=.25 if joint == output else 0.,
                    viscous_damping=2. if joint == output else 0.)
                act.name = name
                self._mjs_actuators.append(act)


def make_four_ankle_cfg(backend, *, elastic=True):
    cfg, sensors = make_a3_ultra_depsic_v1_cfg(backend)
    cfg.spec_fn = four_ankle_spec if elastic else rigid_ankle_spec
    actuators = []
    for actuator in cfg.articulation.actuators:
        name, = actuator.target_names_expr
        changes = {}
        if name in A3_POLICY_JOINT_NAMES:
            changes.update(delay_max_lag=20, delay_hold_prob=1.)
        if name in ANKLES:
            changes.update(stiffness=60., damping=1.2,
                           armature=0. if elastic else INERTIAS[ANKLES.index(name)])
        actuator = replace(actuator, **changes)
        if elastic and name in ANKLES:
            actuator = SwitchableAnkleCfg(**asdict(actuator))
        actuators.append(actuator)
    cfg.articulation.actuators = tuple(actuators)
    spec = cfg.spec_fn()
    cfg.joint_names_simulation = [j.name for j in spec.joints if j.name != "free_joint"]
    cfg.body_names_simulation = [b.name for b in spec.bodies if b.name != "world"]
    return cfg, sensors


def make_rigid_ankle_cfg(backend):
    return make_four_ankle_cfg(backend, elastic=False)


Registry.instance().register("asset", "a3-ultra-four-ankle-mixed", make_four_ankle_cfg)
Registry.instance().register("asset", "a3-ultra-ankle-rigid-dr", make_rigid_ankle_cfg)
