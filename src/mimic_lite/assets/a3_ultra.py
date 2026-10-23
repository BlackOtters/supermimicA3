"""A3 Ultra T3D0 asset adapter.

The Ultra model is intentionally kept outside this repository.  The adapter
loads the checked-out model from ``DEPSIC_A3_ULTRA_MODEL_ROOT`` (or the
developer's standard model checkout), removes the XML actuator block, and
lets mjlab create the same delayed position actuators used by the ordinary
A3 task.  The hand joints remain physical, but are passive at training time:
they are initialized to zero and are not part of the policy/action contract.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as et
from pathlib import Path

from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry
import mujoco

from mimic_lite.assets.a3 import (
    _ARMATURE,
    _DAMPING,
    _EFFORT,
    _STIFFNESS,
    _VELOCITY,
)
from mimic_lite.depsic_a3_contract import A3_HEAD_JOINT_NAMES, A3_POLICY_JOINT_NAMES


registry = Registry.instance()
import a3_supermimic

DEFAULT_ULTRA_MODEL_ROOT = Path(a3_supermimic.__file__).resolve().parent / "assets/a3_ultra_t3d0"


def ultra_model_root() -> Path:
    return Path(os.environ.get("DEPSIC_A3_ULTRA_MODEL_ROOT", str(DEFAULT_ULTRA_MODEL_ROOT))).expanduser()


def ultra_mjcf_path() -> Path:
    return ultra_model_root() / "mjcf" / "a3_ultra_t3d0.xml"


def ultra_urdf_path() -> Path:
    return ultra_model_root() / "urdf" / "model.urdf"


def _xml_names() -> tuple[list[str], list[str]]:
    source = ultra_mjcf_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"A3 Ultra model not found: {source}. Set DEPSIC_A3_ULTRA_MODEL_ROOT."
        )
    root = et.parse(source).getroot()
    joints = [node.attrib["name"] for node in root.iter("joint") if node.attrib.get("name")]
    bodies = [node.attrib["name"] for node in root.iter("body") if node.attrib.get("name")]
    # The free joint is not an articulation joint in mjlab's local tensor.
    if joints and joints[0] == "free_joint":
        joints = joints[1:]
    return joints, bodies


def ultra_passive_joint_positions() -> dict[str, float]:
    """Return fixed zero positions for Ultra-only finger joints."""
    joints, _ = _xml_names()
    known = set(A3_POLICY_JOINT_NAMES) | {"head_yaw_joint", "ultra_head_pitch_joint"}
    return {name: 0.0 for name in joints if name not in known}


def _spec_fn() -> mujoco.MjSpec:
    source = ultra_mjcf_path()
    text = source.read_text(encoding="utf-8")
    # The public Ultra XML includes a repository-local terrain include which is
    # not present in the MimicLite checkout.  The task supplies the plane.
    text = re.sub(r"\s*<include\s+file=[\"']\.\./\.\./terrain/mujoco/flat\.xml[\"']\s*/>", "", text)
    # XML position actuators would expose all 51 joints to the action manager.
    # Remove them (the sensor declarations are retained) and add only the
    # 29 policy + 2 replay-head actuators below.
    text = re.sub(r"\s*<actuator>.*?</actuator>", "", text, flags=re.DOTALL)
    spec = mujoco.MjSpec.from_string(text)
    spec.modelfiledir = str(source.parent)
    return spec


def _actuator(name: str) -> ActuatorCfg:
    if name in _STIFFNESS:
        stiffness, damping, effort = _STIFFNESS[name], _DAMPING[name], _EFFORT[name]
        friction = 0.25 if "ankle" in name else 0.15 if any(x in name for x in ("hip", "knee", "waist")) else 0.1
        armature = _ARMATURE[name]
    else:
        stiffness, damping, effort, friction, armature = 20.0, 1.0, 6.0, 0.1, 0.0
    return ActuatorCfg(
        joint_names_expr=name,
        effort_limit=effort,
        velocity_limit=_VELOCITY.get(name, 6.0),
        stiffness=stiffness,
        damping=damping,
        friction=friction,
        armature=armature,
        delay_min_lag=0 if name not in A3_POLICY_JOINT_NAMES else 0,
        delay_max_lag=2 if name in A3_POLICY_JOINT_NAMES else 0,
        delay_hold_prob=1.0 if name in A3_POLICY_JOINT_NAMES else 0.0,
        delay_update_period=1 if name in A3_POLICY_JOINT_NAMES else 0,
        delay_per_env_phase=False,
    )


def make_a3_ultra_depsic_v1_cfg(backend: str):
    if backend != "mjlab":
        raise ValueError("a3-ultra-depsic-v1 is currently implemented for mjlab only")
    joints, bodies = _xml_names()
    expected = list(A3_POLICY_JOINT_NAMES) + ["head_yaw_joint", "ultra_head_pitch_joint"]
    missing = [name for name in expected if name not in joints]
    if missing:
        raise ValueError(f"A3 Ultra model is missing required joints: {missing}")
    # Preserve the exact model traversal order for tensors while all task code
    # resolves named subsets through this list.  Hand joints remain in the
    # simulation state, but have no task actuator and stay at their passive
    # initial positions.
    init_joint_pos = {
        ".*": 0.0,
        "left_hip_pitch_joint": -0.1311, "right_hip_pitch_joint": -0.1311,
        "left_hip_roll_joint": 0.0056, "right_hip_roll_joint": -0.0056,
        "left_hip_yaw_joint": -0.0348, "right_hip_yaw_joint": 0.0348,
        "left_knee_joint": 0.2468, "right_knee_joint": 0.2468,
        "left_ankle_pitch_joint": -0.1204, "right_ankle_pitch_joint": -0.1204,
        "left_ankle_roll_joint": -0.0078, "right_ankle_roll_joint": 0.0078,
        "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12,
        "left_elbow_joint": 0.8, "right_elbow_joint": 0.8,
    }
    asset = AssetCfg(
        mjcf_path=str(ultra_mjcf_path()),
        usd_path=str(ultra_urdf_path()),
        init_state=InitialStateCfg(
            pos=(0.0, 0.0, 1.07), joint_pos=init_joint_pos, joint_vel={".*": 0.0}
        ),
        actuators={f"joint_{name}": _actuator(name) for name in expected},
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                primary_contact_match_mode="body",
                primary_contact_match_pattern=r".*",
                primary_contact_match_entity="robot",
                track_air_time=True,
                history_length=4,
                reduce="netforce",
            )
        ],
        mjlab_collisions=[
            MjlabCollisionCfg(geom_names_expr=(r".*",), disable_other_geoms=False)
        ],
        joint_names_simulation=joints,
        body_names_simulation=bodies,
        mjlab_spec_fn=_spec_fn,
    )
    return asset.mjlab(), tuple(sensor.mjlab() for sensor in asset.sensors_mjlab)


registry.register("asset", "a3-ultra-depsic-v1", make_a3_ultra_depsic_v1_cfg)

__all__ = [
    "DEFAULT_ULTRA_MODEL_ROOT", "ultra_model_root", "ultra_mjcf_path",
    "ultra_urdf_path", "ultra_passive_joint_positions",
    "make_a3_ultra_depsic_v1_cfg",
]
