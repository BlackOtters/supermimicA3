"""Static tensor and initialization contracts for DEPSIC V2.

The authoritative source is the current G1 ``g1_robot_mixed`` IsaacLab
profile in GR00T.  Its deployable mixed input is a synchronized 10 x 20-ms
full-SMPL horizon, not the superseded single-frame three-point tracker route::

    [ LBC(10 x 7) | 10 x (two-arm position(18) | wrist quat(8)) ]

All SMPL geometry is interpolated in world coordinates and converted to the
SMPL-root frame *after* interpolation.  The V1 robot AMASS reference remains
the train-only frozen-teacher/reconstruction label.
"""

from __future__ import annotations

from typing import Final


REFERENCE_FRAMES: Final[int] = 10
V1_REFERENCE_DIM: Final[int] = REFERENCE_FRAMES * (14 + 14 + 7)
LBC_DIM: Final[int] = 7
SMPL_ARM_JOINT_COUNT: Final[int] = 6
SMPL_ARM_POS_DIM: Final[int] = SMPL_ARM_JOINT_COUNT * 3
SMPL_WRIST_COUNT: Final[int] = 2
SMPL_WRIST_QUAT_DIM: Final[int] = SMPL_WRIST_COUNT * 4
# Compatibility aliases for tools that import the generic reference names.
VR_POINT_COUNT: Final[int] = 0
VR_POINT_POS_DIM: Final[int] = 0
VR_POINT_QUAT_DIM: Final[int] = 0
VR_ANCHOR_DIM: Final[int] = 0
MIXED_REFERENCE_DIM: Final[int] = (
    REFERENCE_FRAMES * LBC_DIM
    + REFERENCE_FRAMES * SMPL_ARM_POS_DIM
    + REFERENCE_FRAMES * SMPL_WRIST_QUAT_DIM
)
ACTOR_OBS_DIM: Final[int] = 930
CRITIC_DIM: Final[int] = 446
TOKEN_COUNT: Final[int] = 2
TOKEN_DIM: Final[int] = 32

# Deployment is deliberately explicit about policy order.  It is the G1
# simulation/actor order used by ``JointPosition``; Unitree's hardware order
# is a different permutation and is recorded in every SONIC export manifest.
G1_POLICY_JOINT_NAMES: Final[tuple[str, ...]] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# This is the hardware (MuJoCo/Unitree low-command) order used by SONIC.
G1_HARDWARE_JOINT_NAMES: Final[tuple[str, ...]] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# KNEES_BENT_KEYFRAME, serialized in policy order.  A zero action must command
# this exact posture on robot, export, and training sides.
G1_POLICY_DEFAULT_JOINT_POS: Final[tuple[float, ...]] = (
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2,
    0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

G1_LBC_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "vx", "vy", "wz", "roll", "pitch", "yaw", "height",
)
G1_HEIGHT_NOMINAL: Final[float] = 0.75
G1_HEIGHT_MIN: Final[float] = 0.35
G1_HEIGHT_MAX: Final[float] = 0.80
G1_HUMAN_HEIGHT: Final[float] = 1.70
G1_ROBOT_HEIGHT: Final[float] = 1.30

V2_AUX_LOSS_COEFFICIENTS: Final[dict[str, float]] = {
    "depsic_arm_q_recon": 1.0 / 300.0,
    "depsic_arm_dq_recon": 1.0 / 300.0,
    "depsic_lbc_recon": 1.0 / 300.0,
    "depsic_mixed_token": 1.0,
}


__all__ = [
    "ACTOR_OBS_DIM",
    "CRITIC_DIM",
    "G1_HARDWARE_JOINT_NAMES",
    "G1_HEIGHT_MAX",
    "G1_HEIGHT_MIN",
    "G1_HEIGHT_NOMINAL",
    "G1_HUMAN_HEIGHT",
    "G1_LBC_FEATURE_NAMES",
    "G1_POLICY_DEFAULT_JOINT_POS",
    "G1_POLICY_JOINT_NAMES",
    "G1_ROBOT_HEIGHT",
    "LBC_DIM",
    "MIXED_REFERENCE_DIM",
    "REFERENCE_FRAMES",
    "SMPL_ARM_JOINT_COUNT",
    "SMPL_ARM_POS_DIM",
    "SMPL_WRIST_COUNT",
    "SMPL_WRIST_QUAT_DIM",
    "TOKEN_COUNT",
    "TOKEN_DIM",
    "V1_REFERENCE_DIM",
    "VR_ANCHOR_DIM",
    "VR_POINT_COUNT",
    "VR_POINT_POS_DIM",
    "VR_POINT_QUAT_DIM",
    "V2_AUX_LOSS_COEFFICIENTS",
]
