"""Learned dynamics with analytic observation, reward and soft termination heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from a3_supermimic.world_model.dynamics import DynamicsMLP
from a3_supermimic.world_model.heads import AnalyticHeads
from a3_supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)


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
        self.to(device)

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

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor):
        """k-step autoregressive loss.

        states: (k+1, n, state_dim); a_exec: (k, n, action_dim); xi: (n, context_dim).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        metrics = {}
        err_first = None
        err_last = None
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq = step_fn(s, a_exec[j], xi, states[j + 1])
            total_state = total_state + sq.sum(-1).mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
        loss = total_state / k
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
