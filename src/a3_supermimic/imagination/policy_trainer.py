"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from a3_supermimic.data.replay_buffer import ReplayBuffer
from a3_supermimic.envs.mdp.motion_library import MotionLibrary
from a3_supermimic.imagination.objective import gae_targets, policy_objective
from a3_supermimic.imagination.rollout import ImagRollout, imagine
from a3_supermimic.policy.nets import Critic, Policy
from a3_supermimic.training.distributed import allreduce_grads
from a3_supermimic.world_model.world_model import WorldModel
from a3_supermimic.world_model.heads import flatten_history_term_major


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 2048  # e5 per-rank batch; 4096 environments per rank
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # Append 6D reference-relative anchor orientation to the goal.


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from a3_supermimic.imagination.compiled import ImagStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr * self.cfg.critic_lr / self.cfg.lr

    def _critic_inputs(self, ro: ImagRollout, xi):
        H1, n, _ = ro.states.shape
        context = xi[None].expand(H1, n, -1).reshape(H1*n, -1)
        feat = self.wm.features(ro.states.reshape(H1 * n, -1), context).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs)
        feat, goals, e_root = self._critic_inputs(ro, start.xi)
        if cfg.use_critic:
            v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        if not torch.isfinite(gn):
            raise FloatingPointError("Nonfinite actor gradient")
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        with torch.no_grad():
            # Diagnostics only; the actor objective is exactly the upstream -J/H.
            goals_start = ro.goals[0].detach()
            z = self.policy.encoders['motion'](goals_start)
            metrics['latent_std_mean'] = float((z.var(0, unbiased=False) + 1.e-4).sqrt().mean())
            metrics['latent_saturation_fraction'] = float((z.abs() > .999).float().mean())
            # Hold proprioception fixed: a changing latent is insufficient if
            # the decoder ignores it. Shuffling tests the full goal-to-action path.
            obs_fixed = flatten_history_term_major(torch.stack([
                self.wm.heads.observation(s, a, start.xi)
                for s, a in zip(start.hist_states, start.hist_actions)]))
            original = self.policy(obs_fixed, goals_start)
            shuffled = self.policy(obs_fixed, goals_start.roll(1, 0))
            metrics['goal_swap_action_rms'] = float((original - shuffled).square().mean().sqrt())
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = feat.detach(), goals.detach(), e_root.detach()
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1)).reshape(H1, n)
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_er = e_root_d[:-1].reshape(-1, 4)
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
