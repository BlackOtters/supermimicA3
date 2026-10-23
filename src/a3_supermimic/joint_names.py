"""Robot coordinate names, also usable by the NumPy-only data converter."""

POLICY_JOINTS = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in
           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")),
    *tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in
           ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
)
HEAD_JOINTS = ("head_yaw_joint", "ultra_head_pitch_joint")
