"""Bundled A3 T3D0 asset used by the DEPSIC A3 V1 mjlab task."""

from __future__ import annotations

from pathlib import Path

from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry

from mimic_lite.depsic_a3_contract import A3_HEAD_JOINT_NAMES, A3_POLICY_JOINT_NAMES


registry = Registry.instance()
A3_ASSET_DIR = Path(__file__).resolve().parent / "a3_t3d0"
A3_MJCF_PATH = A3_ASSET_DIR / "mjcf" / "a3_t3d0_robot_only.xml"
A3_URDF_PATH = A3_ASSET_DIR / "urdf" / "model.urdf"

_STIFFNESS = {
    "left_hip_pitch_joint": 80.0, "left_hip_roll_joint": 120.0, "left_hip_yaw_joint": 80.0,
    "left_knee_joint": 250.0, "left_ankle_pitch_joint": 50.0, "left_ankle_roll_joint": 50.0,
    "right_hip_pitch_joint": 80.0, "right_hip_roll_joint": 120.0, "right_hip_yaw_joint": 80.0,
    "right_knee_joint": 250.0, "right_ankle_pitch_joint": 50.0, "right_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 85.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
    "left_shoulder_pitch_joint": 40.0, "left_shoulder_roll_joint": 40.0, "left_shoulder_yaw_joint": 30.0,
    "left_elbow_joint": 30.0, "left_wrist_roll_joint": 30.0, "left_wrist_pitch_joint": 20.0,
    "left_wrist_yaw_joint": 20.0,
    "right_shoulder_pitch_joint": 40.0, "right_shoulder_roll_joint": 40.0, "right_shoulder_yaw_joint": 30.0,
    "right_elbow_joint": 30.0, "right_wrist_roll_joint": 30.0, "right_wrist_pitch_joint": 20.0,
    "right_wrist_yaw_joint": 20.0,
}
_DAMPING = {
    "left_hip_pitch_joint": 3.0, "left_hip_roll_joint": 4.0, "left_hip_yaw_joint": 3.0,
    "left_knee_joint": 8.0, "left_ankle_pitch_joint": 2.0, "left_ankle_roll_joint": 2.0,
    "right_hip_pitch_joint": 3.0, "right_hip_roll_joint": 4.0, "right_hip_yaw_joint": 3.0,
    "right_knee_joint": 8.0, "right_ankle_pitch_joint": 2.0, "right_ankle_roll_joint": 2.0,
    "waist_yaw_joint": 3.0, "waist_roll_joint": 2.0, "waist_pitch_joint": 2.0,
    "left_shoulder_pitch_joint": 3.0, "left_shoulder_roll_joint": 3.0, "left_shoulder_yaw_joint": 2.0,
    "left_elbow_joint": 2.0, "left_wrist_roll_joint": 2.0, "left_wrist_pitch_joint": 2.0,
    "left_wrist_yaw_joint": 2.0,
    "right_shoulder_pitch_joint": 3.0, "right_shoulder_roll_joint": 3.0, "right_shoulder_yaw_joint": 2.0,
    "right_elbow_joint": 2.0, "right_wrist_roll_joint": 2.0, "right_wrist_pitch_joint": 2.0,
    "right_wrist_yaw_joint": 2.0,
}
_EFFORT = {
    "left_hip_pitch_joint": 220.0, "left_hip_roll_joint": 220.0, "left_hip_yaw_joint": 220.0,
    "left_knee_joint": 320.0, "left_ankle_pitch_joint": 118.2, "left_ankle_roll_joint": 54.75,
    "right_hip_pitch_joint": 220.0, "right_hip_roll_joint": 220.0, "right_hip_yaw_joint": 220.0,
    "right_knee_joint": 320.0, "right_ankle_pitch_joint": 118.2, "right_ankle_roll_joint": 54.75,
    "waist_yaw_joint": 220.0, "waist_roll_joint": 46.0, "waist_pitch_joint": 115.0,
    "left_shoulder_pitch_joint": 60.0, "left_shoulder_roll_joint": 60.0, "left_shoulder_yaw_joint": 24.0,
    "left_elbow_joint": 24.0, "left_wrist_roll_joint": 24.0, "left_wrist_pitch_joint": 6.0,
    "left_wrist_yaw_joint": 6.0,
    "right_shoulder_pitch_joint": 60.0, "right_shoulder_roll_joint": 60.0, "right_shoulder_yaw_joint": 24.0,
    "right_elbow_joint": 24.0, "right_wrist_roll_joint": 24.0, "right_wrist_pitch_joint": 6.0,
    "right_wrist_yaw_joint": 6.0,
}
_VELOCITY = {
    "left_hip_pitch_joint": 12.04, "left_hip_roll_joint": 12.04, "left_hip_yaw_joint": 12.04,
    "left_knee_joint": 14.66, "left_ankle_pitch_joint": 10.80, "left_ankle_roll_joint": 19.37,
    "right_hip_pitch_joint": 12.04, "right_hip_roll_joint": 12.04, "right_hip_yaw_joint": 12.04,
    "right_knee_joint": 14.66, "right_ankle_pitch_joint": 10.80, "right_ankle_roll_joint": 19.37,
    "waist_yaw_joint": 12.04, "waist_roll_joint": 22.70, "waist_pitch_joint": 9.25,
    "left_shoulder_pitch_joint": 13.61, "left_shoulder_roll_joint": 13.61, "left_shoulder_yaw_joint": 15.71,
    "left_elbow_joint": 15.71, "left_wrist_roll_joint": 15.71, "left_wrist_pitch_joint": 12.78,
    "left_wrist_yaw_joint": 12.78,
    "right_shoulder_pitch_joint": 13.61, "right_shoulder_roll_joint": 13.61, "right_shoulder_yaw_joint": 15.71,
    "right_elbow_joint": 15.71, "right_wrist_roll_joint": 15.71, "right_wrist_pitch_joint": 12.78,
    "right_wrist_yaw_joint": 12.78,
}
_ARMATURE = {
    "left_hip_pitch_joint": 0.066465698909, "left_hip_roll_joint": 0.066465698909,
    "left_hip_yaw_joint": 0.066465698909, "left_knee_joint": 0.120340286840,
    "left_ankle_pitch_joint": 0.064440605310, "left_ankle_roll_joint": 0.020137175616,
    "right_hip_pitch_joint": 0.066465698909, "right_hip_roll_joint": 0.066465698909,
    "right_hip_yaw_joint": 0.066465698909, "right_knee_joint": 0.120340286840,
    "right_ankle_pitch_joint": 0.064440605310, "right_ankle_roll_joint": 0.020137175616,
    "waist_yaw_joint": 0.066465698909, "waist_roll_joint": 0.014620876135,
    "waist_pitch_joint": 0.088208591555,
    "left_shoulder_pitch_joint": 0.012083368706, "left_shoulder_roll_joint": 0.012083368706,
    "left_shoulder_yaw_joint": 0.004967351303, "left_elbow_joint": 0.004967351303,
    "left_wrist_roll_joint": 0.004967351303, "left_wrist_pitch_joint": 0.000810089334,
    "left_wrist_yaw_joint": 0.000810089334,
    "right_shoulder_pitch_joint": 0.012083368706, "right_shoulder_roll_joint": 0.012083368706,
    "right_shoulder_yaw_joint": 0.004967351303, "right_elbow_joint": 0.004967351303,
    "right_wrist_roll_joint": 0.004967351303, "right_wrist_pitch_joint": 0.000810089334,
    "right_wrist_yaw_joint": 0.000810089334,
}

A3_ACTION_SCALE = {name: 0.25 * _EFFORT[name] / _STIFFNESS[name] for name in A3_POLICY_JOINT_NAMES}

# ``find_bodies`` deliberately uses this simulation-order list rather than
# mjlab's spec traversal order.  It contains every named body in the bundled
# robot-only MJCF, with source capitalization intact.
A3_BODY_NAMES = (
    "pelvis_link",
    "waist_yaw_Link",
    "waist_roll_Link",
    "torso_Link",
    "head_yaw_Link",
    "head_pitch_Link",
    "left_shoulder_pitch_Link",
    "left_shoulder_roll_Link",
    "left_shoulder_yaw_Link",
    "left_elbow_Link",
    "left_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_pitch_Link",
    "right_shoulder_roll_Link",
    "right_shoulder_yaw_Link",
    "right_elbow_Link",
    "right_wrist_roll_Link",
    "right_wrist_pitch_Link",
    "right_wrist_yaw_Link",
    "left_hip_pitch_Link",
    "left_hip_roll_Link",
    "left_hip_yaw_Link",
    "left_knee_Link",
    "left_ankle_pitch_Link",
    "left_ankle_roll_Link",
    "right_hip_pitch_Link",
    "right_hip_roll_Link",
    "right_hip_yaw_Link",
    "right_knee_Link",
    "right_ankle_pitch_Link",
    "right_ankle_roll_Link",
)


def _joint_actuators() -> dict[str, ActuatorCfg]:
    # AssetCfg.mjlab intentionally accepts scalar actuator parameters.  One
    # actuator config per joint preserves the source V1 values exactly.
    result = {
        f"body_{name}": ActuatorCfg(
            joint_names_expr=name,
            effort_limit=_EFFORT[name],
            velocity_limit=_VELOCITY[name],
            stiffness=_STIFFNESS[name],
            damping=_DAMPING[name],
            friction=0.25 if "ankle" in name else 0.15 if any(x in name for x in ("hip", "knee", "waist")) else 0.1,
            armature=_ARMATURE[name],
            # mjlab would refresh this delay every physics step by default.
            # A3's command reset hook fixes one sampled lag for the episode.
            delay_min_lag=0,
            delay_max_lag=2,
            delay_hold_prob=1.0,
            delay_update_period=1,
            delay_per_env_phase=False,
        )
        for name in A3_POLICY_JOINT_NAMES
    }
    result["head"] = ActuatorCfg(
        joint_names_expr=list(A3_HEAD_JOINT_NAMES),
        effort_limit=6.0,
        velocity_limit=6.0,
        stiffness=20.0,
        damping=1.0,
        friction=0.1,
        armature=0.0,
    )
    return result


def make_depsic_a3_v1_cfg(backend: str):
    asset = AssetCfg(
        mjcf_path=str(A3_MJCF_PATH),
        usd_path=str(A3_URDF_PATH),
        init_state=InitialStateCfg(
            pos=(0.0, 0.0, 1.07),
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
            },
            joint_vel={".*": 0.0},
        ),
        actuators=_joint_actuators(),
        self_collisions=False,
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                # DEPSIC needs a per-link tensor for support, torso/wrist
                # terminations, and undesired-contact rewards.
                primary_contact_match_mode="body",
                primary_contact_match_pattern=r".*",
                primary_contact_match_entity="robot",
                track_air_time=True,
                history_length=4,
                reduce="netforce",
            ),
        ],
        mjlab_collisions=[
            MjlabCollisionCfg(geom_names_expr=(".*",), disable_other_geoms=False),
        ],
        joint_names_simulation=list(A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES),
        body_names_simulation=list(A3_BODY_NAMES),
    )
    if backend != "mjlab":
        raise ValueError("a3-depsic-v1 is currently implemented for mjlab only")
    return asset.mjlab(), tuple(sensor.mjlab() for sensor in asset.sensors_mjlab)


registry.register("asset", "a3-depsic-v1", make_depsic_a3_v1_cfg)
