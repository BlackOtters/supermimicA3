"""Native 29-axis rigid A3: fixed head/hands, unchanged masses and collision geometry."""
from dataclasses import replace
import xml.etree.ElementTree as ET
import mujoco
from active_adaptation.registry import Registry
from mimic_lite.assets.a3_ultra import ultra_mjcf_path
from mimic_lite.assets.a3_ultra_four_ankle import rigid_ankle_spec, make_rigid_ankle_cfg
from a3_supermimic.joint_names import POLICY_JOINTS


def compact_rigid_spec():
    root = ET.fromstring(rigid_ankle_spec().to_xml())
    removed = set()
    for body in root.iter('body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name')
            if joint.get('type') != 'free' and name not in POLICY_JOINTS:
                # The omitted head/hand hinges have neutral qpos=0. Their links,
                # inertials and geoms stay attached at that exact configuration.
                if float(joint.get('ref', '0')) != 0:
                    raise ValueError(f'Cannot weld nonzero reference joint {name}')
                removed.add(name)
                body.remove(joint)
    for section in ('sensor', 'equality', 'tendon'):
        node = root.find(section)
        if node is not None:
            for item in list(node):
                if any(value in removed for elem in item.iter() for value in elem.attrib.values()):
                    node.remove(item)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    spec.modelfiledir = str(ultra_mjcf_path().parent)
    return spec


def make_compact_rigid_cfg(backend):
    cfg, sensors = make_rigid_ankle_cfg(backend)
    cfg.spec_fn = compact_rigid_spec
    cfg.articulation.actuators = tuple(
        replace(a, delay_min_lag=0, delay_max_lag=0, delay_hold_prob=0.)
        for a in cfg.articulation.actuators if a.target_names_expr[0] in POLICY_JOINTS)
    cfg.init_state.joint_pos = {k:v for k,v in cfg.init_state.joint_pos.items() if k == '.*' or k in POLICY_JOINTS}
    spec = cfg.spec_fn()
    cfg.joint_names_simulation = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    cfg.body_names_simulation = [b.name for b in spec.bodies if b.name != 'world']
    return cfg, sensors


Registry.instance().register('asset', 'a3-supermimic-compact-rigid', make_compact_rigid_cfg)
