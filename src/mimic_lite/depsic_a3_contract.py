"""Canonical A3 T3D0 DEPSIC V1 joint and tensor contracts.

The policy controls 29 joints.  The two head joints are part of the robot and
the redirected motion PKL, but are externally replayed and deliberately never
enter the policy action/proprioception contract.
"""

from __future__ import annotations

from typing import Final


A3_POLICY_JOINT_NAMES: Final[tuple[str, ...]] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
A3_HEAD_JOINT_NAMES: Final[tuple[str, str]] = ("head_yaw_joint", "head_pitch_joint")
A3_31DOF_JOINT_NAMES: Final[tuple[str, ...]] = A3_POLICY_JOINT_NAMES + A3_HEAD_JOINT_NAMES
A3_ARM_JOINT_NAMES: Final[tuple[str, ...]] = A3_POLICY_JOINT_NAMES[3:17]

# The current A3 GMR redirector omits joint_names and serializes this exact
# MJCF-order layout.  Other nameless arrays are intentionally rejected.
A3_T3D0_MJCF_JOINT_NAMES: Final[tuple[str, ...]] = (
    A3_POLICY_JOINT_NAMES[:3] + A3_HEAD_JOINT_NAMES + A3_POLICY_JOINT_NAMES[3:]
)

A3_POLICY_DIM: Final[int] = len(A3_POLICY_JOINT_NAMES)
A3_ARM_DIM: Final[int] = len(A3_ARM_JOINT_NAMES)
A3_HEAD_DIM: Final[int] = len(A3_HEAD_JOINT_NAMES)
A3_LBC_DIM: Final[int] = 7
A3_REFERENCE_FRAMES: Final[int] = 10
A3_ACTOR_OBS_DIM: Final[int] = 10 * (3 + 3 + A3_POLICY_DIM * 3)
A3_TOKENIZER_DIM: Final[int] = A3_REFERENCE_FRAMES * (A3_ARM_DIM * 2 + A3_LBC_DIM)
A3_CRITIC_DIM: Final[int] = (
    A3_REFERENCE_FRAMES * A3_LBC_DIM
    + A3_REFERENCE_FRAMES * A3_ARM_DIM * 2
    + 3 + 3 + 3 + A3_POLICY_DIM * 3
)

