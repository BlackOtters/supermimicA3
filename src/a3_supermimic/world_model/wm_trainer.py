"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from a3_supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from a3_supermimic.envs.mdp.motion_library import MotionLibrary
from a3_supermimic.world_model.heads import ReferenceFrame
from a3_supermimic.training.distributed import allreduce_grads, rank, world_size
from a3_supermimic.training.gradients import clip_grad_norm_stable
from a3_supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg, *, failure_dir: Path | None = None):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.failure_dir = failure_dir
        self.norm_fallbacks = 0
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        return self.wm.window_loss(w.states, w.a_exec, w.xi)

    def _fail(self, reason, w, loss, batch_index, *, grad_norm=None, local_grad_finite=None):
        path = None
        if self.failure_dir is not None:
            self.failure_dir.mkdir(parents=True, exist_ok=True)
            path = self.failure_dir / f"wm_rank{rank()}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save({"wm": self.wm.state_dict(), "window": vars(w),
                        "loss": loss.detach(), "grad_norm": grad_norm,
                        "local_grad_finite": local_grad_finite, "batch_index": batch_index,
                        "reason": reason}, temporary)
            temporary.replace(path)
        if world_size() > 1:
            # Let every rank finish its snapshot before torchrun terminates siblings.
            dist.barrier()
        raise FloatingPointError(
            f"Nonfinite world-model {reason}: rank={rank()}, loss={float(loss.detach())}, "
            f"grad_norm={grad_norm}, local_grad_finite={local_grad_finite}, snapshot={path}")

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for batch_index in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss_finite = loss.detach().isfinite().to(torch.int32)
            if world_size() > 1:
                dist.all_reduce(loss_finite, op=dist.ReduceOp.MIN)
            if not bool(loss_finite):
                self._fail("loss", w, loss, batch_index)
            loss.backward()
            # Keep the local result before a bad rank contaminates the averaged gradients.
            local_finite = torch.stack([p.grad.isfinite().all() for p in self.wm.parameters()
                                        if p.grad is not None]).all().detach()
            allreduce_grads(self.wm)
            gn = clip_grad_norm_stable(self.wm.parameters(), self.cfg.grad_clip)
            if not torch.isfinite(gn):
                self._fail("gradient", w, loss, batch_index, grad_norm=gn.detach(),
                           local_grad_finite=bool(local_finite))
            if gn.dtype == torch.float64:
                self.norm_fallbacks += 1
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Convert aggregate metrics once per update; finite guards synchronize each minibatch.
        result = {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}
        result["grad_norm_fp64_fallbacks_total"] = float(self.norm_fallbacks)
        return result

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
