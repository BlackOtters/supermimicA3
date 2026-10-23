"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the full A3 simulator state
``s = [qpos | qvel | previous targets | fresh-reset flag]`` (root position expressed relative to the env origin,
see ``StateIO``), the previous clean action, the sim context ``xi`` and the
reference frame, so the same code serves the simulator-side tests and the
imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from a3_supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from a3_supermimic.kinematics.model import ModelKinematics
from a3_supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse, quat_inv, quat_mul

OBS_DIM = 93
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))


def anchor_relative_orientation(robot_anchor_quat: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference orientation in the robot anchor frame; first two rotation columns, row-major."""
    from a3_supermimic.kinematics.rotations import rot6d_from_quat

    return rot6d_from_quat(quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat))


ANCHOR_OBS_DIM = 6


def anchor_relative_pose(robot_pos, robot_quat, reference_pos, reference_quat):
    position = quat_apply_inverse(robot_quat, reference_pos - robot_pos)
    return torch.cat((position, anchor_relative_orientation(robot_quat, reference_quat)), -1)


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, N, 93) oldest→newest  ->  (N, 930) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)


@dataclass
class TerminationCfg:
    height_threshold: float = .25
    tilt_threshold: float = .8  # absolute difference of projected-gravity z components
    height_temperature: float = .02  # metres; imagination only
    tilt_temperature: float = .04  # gravity-z difference; imagination only

    def __post_init__(self):
        values = (self.height_threshold, self.tilt_threshold, self.height_temperature, self.tilt_temperature)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Termination thresholds and temperatures must be finite and positive")


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: ModelKinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
        layout=None,
        termination_cfg: TerminationCfg | None = None,
        observation_anchor_body_name: str | None = None,
    ):
        self.fk = fk
        self.layout = layout
        self.policy_q = layout.tensor_ids("policy_q", fk.device)
        self.policy_v = layout.tensor_ids("policy_v", fk.device)
        self.encoder_q = layout.tensor_ids("encoder_q", fk.device)
        self.encoder_v = layout.tensor_ids("encoder_v", fk.device)
        from a3_supermimic.robot import ANKLES, POLICY_JOINTS
        self.ankle_cols = torch.tensor([POLICY_JOINTS.index(n) for n in ANKLES], device=fk.device)
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.observation_anchor_index = tracked_body_names.index(observation_anchor_body_name or anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()
        self.termination_cfg = termination_cfg or TerminationCfg()

    def observation(self, s, prev_action, xi):
        q = s[..., self.policy_q].clone()
        dq = s[..., self.layout.nq + self.policy_v].clone()
        if self.layout.compact:
            q = q - self.default_joint_pos + xi[..., 1:30]
        else:
            elastic = xi[..., 29:30] > 0.5
            q[..., self.ankle_cols] = torch.where(elastic, s[..., self.encoder_q], q[..., self.ankle_cols])
            dq[..., self.ankle_cols] = torch.where(elastic, s[..., self.layout.nq + self.encoder_v], dq[..., self.ankle_cols])
            q = q - self.default_joint_pos - xi[..., :29]
        return torch.cat((q, dq, s[..., self.layout.nq+3:self.layout.nq+6],
                          projected_gravity(s[..., 3:7]), prev_action[..., :29]), -1)

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., :self.layout.nq], s[..., self.layout.nq:self.layout.physical_dim])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., self.policy_q],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    def termination_features(self, root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Paper Appendix A: height and gravity-z tilt errors normalized by their limits."""
        cfg = self.termination_cfg
        height = (ref.root_pos_w[..., 2] - root_pos[..., 2]) / cfg.height_threshold
        tilt = (projected_gravity(ref.root_quat_w)[..., 2] -
                projected_gravity(root_quat)[..., 2]) / cfg.tilt_threshold
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref)  # pelvis == free-joint root
        return (f.abs() > 1.).any(-1)

    def soft_termination(self, s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Smooth OR of the two hard failure thresholds; timeouts are not failures."""
        cfg = self.termination_cfg
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref).abs()
        z = (f[..., 0] - 1.) * (cfg.height_threshold / cfg.height_temperature)
        tilt = (f[..., 1] - 1.) * (cfg.tilt_threshold / cfg.tilt_temperature)
        # 1 - sigmoid(-z)*sigmoid(-tilt), without cancellation near zero failure probability.
        return -torch.expm1(F.logsigmoid(-z) + F.logsigmoid(-tilt))

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        from a3_supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, wrap_to_pi, yaw_quat

        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
