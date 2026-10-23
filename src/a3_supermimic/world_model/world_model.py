"""Learned dynamics with analytic observation, reward and soft termination heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from a3_supermimic.world_model.dynamics import DynamicsMLP, MLP
from a3_supermimic.world_model.heads import AnalyticHeads
from a3_supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    termination_mode: str = "analytic"  # analytic | learned (retargeted/e5)
    term_hidden: tuple[int, ...] = (64, 64)
    term_loss_weight: float = 1.0


class TerminationHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)

    def forward(self, normalized_features):
        return self.mlp(torch.cat((normalized_features, normalized_features.abs()), -1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device, xi_offset, xi_scale, terrain=None):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, heads.layout, xi_offset, xi_scale, device)
        self.terrain = terrain
        terrain_dim = terrain.feature_dim if terrain is not None else 0
        self.feature_dim = self.spec.feat_dim + terrain_dim
        self.dynamics = DynamicsMLP(feat_dim=self.feature_dim, act_dim=heads.layout.action_dim,
                                    xi_dim=len(xi_offset), hidden=cfg.hidden, out_dim=self.spec.delta_dim)
        self.heads = heads  # analytic observation / reward / termination / FK (no parameters)
        if cfg.termination_mode not in ('analytic', 'learned'):
            raise ValueError('termination_mode must be analytic or learned')
        self.term_head = TerminationHead(cfg.term_hidden) if cfg.termination_mode == 'learned' else None
        self.to(device)

    def termination_logits(self, s, ref):
        return self.term_head(self.heads.termination_features(s[..., :3], s[..., 3:7], ref))

    def termination_probability(self, s, ref):
        return (self.termination_logits(s, ref).sigmoid() if self.term_head is not None
                else self.heads.soft_termination(s, ref))

    # ------------------------------------------------------------------ one step

    def features(self, s, xi):
        feat = self.spec.canonical(s)
        if self.terrain is not None:
            feat = torch.cat((feat, self.terrain(s, xi)), -1)
        return feat

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.features(s, xi)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta, action)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true):
        """One autoregressive step of the dynamics loss; analytic heads have no fitted loss."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        return s_next, e * e

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step
        self._term_loss_step_fn = (torch.compile(self._term_loss_step, mode=None if mode == "default" else mode, dynamic=False)
                                  if mode else self._term_loss_step)

    def _term_loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        from a3_supermimic.world_model.heads import ReferenceFrame
        s_next, sq = self._loss_step(s, a, xi, s_true)
        ref = ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat)
        logits = self.termination_logits(s_next, ref)
        return s_next, sq, F.binary_cross_entropy_with_logits(logits, done), ((logits > 0) == done.bool()).float().mean()

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done=None, refs=None):
        """k-step autoregressive loss.

        states: (k+1, n, state_dim); a_exec: (k, n, action_dim); xi: (n, context_dim).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = total_acc = 0.0
        metrics = {}
        err_first = None
        err_last = None
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        term_fn = getattr(self, "_term_loss_step_fn", None) or self._term_loss_step
        if self.term_head is not None and (done is None or refs is None):
            raise ValueError('Learned termination requires next-state done labels and references')
        for j in range(k):
            if self.term_head is None:
                s, sq = step_fn(s, a_exec[j], xi, states[j + 1])
            else:
                s, sq, bce, accuracy = term_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w)
                total_term = total_term + bce
                total_acc = total_acc + accuracy.detach()
            total_state = total_state + sq.sum(-1).mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
        loss = total_state / k
        if self.term_head is not None:
            loss = loss + self.cfg.term_loss_weight * total_term / k
            metrics.update(term_bce=(total_term / k).detach(), term_acc=total_acc / k)
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
