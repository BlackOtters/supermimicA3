"""SONIC 035 reference reset, expressed in the A3 generalized-state convention."""
from dataclasses import dataclass

import torch

from a3_supermimic.kinematics.rotations import (
    quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul,
)


@dataclass
class ResetCfg:
    pose_half_range: tuple[float, ...] = (.05, .05, .01, .1, .1, .2)
    velocity_half_range: tuple[float, ...] = (.5, .5, .2, .52, .52, .78)
    joint_position_half_range: float = .1


def reset_reference(q, v, layout, joint_qids, soft_limits, cfg: ResetCfg, *, noise=True):
    q, v = q.clone(), v.clone()
    # SONIC perturbs WORLD angular velocity independently of the pose. Convert
    # before changing orientation; MuJoCo stores free-joint angular velocity locally.
    omega_w = quat_apply(q[:, 3:7], v[:, 3:6])
    if noise:
        pose = (torch.rand(len(q), 6, device=q.device, dtype=q.dtype)*2-1) * q.new_tensor(cfg.pose_half_range)
        q[:, :3] += pose[:, :3]
        delta = quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
        q[:, 3:7] = quat_mul(delta, q[:, 3:7])
        velocity = (torch.rand_like(pose)*2-1) * q.new_tensor(cfg.velocity_half_range)
        v[:, :3] += velocity[:, :3]
        omega_w += velocity[:, 3:]
        q[:, layout.policy_q] += (torch.rand_like(q[:, layout.policy_q])*2-1)*cfg.joint_position_half_range
    v[:, 3:6] = quat_apply_inverse(q[:, 3:7], omega_w)
    # Also clamp deterministic evaluation resets, as SONIC does. The selected
    # four-ankle reset hook subsequently synchronizes rotors to the clamped outputs.
    q[:, joint_qids] = q[:, joint_qids].clamp(soft_limits[:, 0], soft_limits[:, 1])
    return q, v
