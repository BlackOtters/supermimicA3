"""Algorithm A.1 on the repository's A3 four-ankle simulator."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from a3_supermimic.config import TrainConfig
from a3_supermimic.data.replay_buffer import ReplayBuffer
from a3_supermimic.envs.mdp.motion_library import discover_npz
from a3_supermimic.envs.simulator import Simulator
from a3_supermimic.world_model.world_model import WorldModel
from a3_supermimic.world_model.wm_trainer import WorldModelTrainer
from a3_supermimic.imagination.policy_trainer import PolicyTrainer
from a3_supermimic.policy.nets import Policy, Critic
from a3_supermimic.training.distributed import init_distributed, broadcast_params, allreduce_mean, finalize
from a3_supermimic.training.schedule import cosine_lr
from a3_supermimic.training.tracking import SwanLabLogger, tracking_error_message
from a3_supermimic.robot import BASE_TASK, UPSTREAM_COMMIT, POLICY_JOINTS, CONTRACT_VERSION, TRACKING_ANCHOR, check_contract_version
from a3_supermimic.imagination.rollout import OBS_NOISE_HALF
from a3_supermimic.world_model.heads import ANCHOR_OBS_DIM


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank, local_rank, self.world = init_distributed()
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
        for section, name in ((cfg.wm_trainer, "lr"), (cfg.policy_trainer, "lr"), (cfg.policy_trainer, "critic_lr")):
            setattr(section, name, getattr(section, name)*(self.world/cfg.lr_world_size)**.5)
        cfg.lr_world_size = self.world
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.set_device(cfg.device)
        torch.manual_seed(cfg.seed+1000*self.rank)
        cfg.policy.goal_dim = 68*cfg.env.future_frames + (ANCHOR_OBS_DIM if cfg.env.anchor_obs else 0)
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.policy_trainer.history = cfg.policy.history
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.obs_noise = cfg.env.obs_noise
        if cfg.env.replay_capacity <= cfg.wm_trainer.window or cfg.policy_trainer.horizon < 1:
            raise ValueError("Replay capacity must exceed the WM window; horizon must be positive")
        if cfg.schedule.warmup_iterations < cfg.wm_trainer.window:
            raise ValueError("Warmup must fill a complete world-model window")
        if cfg.schedule.wm_only_updates < 0 or cfg.schedule.policy_updates_per_wm_update < 1:
            raise ValueError("WM-only updates must be nonnegative; policy updates per round must be positive")
        if cfg.policy_trainer.batch_size*cfg.policy_trainer.horizon < cfg.policy_trainer.critic_minibatches:
            raise ValueError("Critic minibatches would be empty")
        self.files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips)
        if not self.files:
            raise FileNotFoundError(f"No A3 training clips in {cfg.data.npz_dir}; run a3-sm-data validate first")
        self.log_dir = Path(cfg.log_root)/cfg.run_name
        if self.rank == 0 and self.log_dir.exists() and any(self.log_dir.iterdir()) and cfg.resume is None:
            raise FileExistsError(f"Refusing to overwrite run {self.log_dir}; use a new run name or --resume")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sim = Simulator(cfg, self.files)
        if not math.isclose(cfg.world_model.state_spec.dt, self.sim.env.step_dt, rel_tol=0., abs_tol=1.e-9):
            self.sim.close()
            raise ValueError("World-model integration dt must equal the A3 simulator policy dt")
        self.lib = self.sim.library
        self.buffer = ReplayBuffer(cfg.env.replay_capacity, cfg.env.num_envs, self.sim.layout.state_dim,
                                   len(self.sim.context.offset), cfg.device, action_dim=self.sim.layout.action_dim)
        self.sim.attach_buffer(self.buffer)
        self.wm = WorldModel(cfg.world_model, self.sim.heads, cfg.device, self.sim.context.offset,
                             self.sim.context.scale, self.sim.terrain)
        self.policy = Policy(cfg.policy).to(cfg.device)
        self.critic = Critic(cfg.critic, feat_dim=self.wm.feature_dim,
                             xi_dim=len(self.sim.context.offset), pcfg=cfg.policy).to(cfg.device)
        for module in (self.wm, self.policy, self.critic):
            broadcast_params(module)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer,
                                            failure_dir=self.log_dir / "failures")
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        task = OmegaConf.to_container(self.sim.env.cfg, resolve=True)
        self.contract = {"version": CONTRACT_VERSION, "upstream": UPSTREAM_COMMIT, "base_task": BASE_TASK,
                         "robot": asdict(self.sim.layout), "context_names": self.sim.context.names,
                         "joint_order": POLICY_JOINTS, "data_sha256": self.lib.fingerprint,
                         "task": task, "history": cfg.policy.history,
                         "goal_dim": cfg.policy.goal_dim, "anchor_obs": cfg.env.anchor_obs,
                         "anchor_observation": "relative_orientation_rot6d_row_major" if cfg.env.anchor_obs else "none",
                         "observation_anchor": cfg.env.observation_anchor,
                         "representation": cfg.env.representation, "policy_config": asdict(cfg.policy),
                         "state_spec": asdict(cfg.world_model.state_spec),
                         "termination": asdict(cfg.env.termination), "reset": asdict(cfg.env.reset),
                         "imagination_termination": "analytic_sigmoid_or",
                         "wm_only_updates": cfg.schedule.wm_only_updates,
                         "wm_batches_per_update": cfg.wm_trainer.batches_per_update,
                         "observation_noise_half": OBS_NOISE_HALF.tolist(),
                         "action_semantics": {"actor_history": "clean", "action_rate": "clean",
                                              "dynamics_input": "executed", "delay_state": "executed"},
                         "terrain_features": self.sim.terrain is not None,
                         "tracking_anchor": TRACKING_ANCHOR, "reward": asdict(self.sim.heads.reward_cfg)}
        self.contract_digest = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        self.iteration = self.num_updates = self.num_policy_updates = 0
        self.utd_acc = 0.
        self.writer = SummaryWriter(str(self.log_dir/"tb")) if self.rank == 0 else None
        if cfg.resume is not None:
            self.load(cfg.resume)
        if self.rank == 0:
            (self.log_dir/"config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str)+"\n")
            (self.log_dir/"contract.json").write_text(json.dumps(self.contract, indent=2)+"\n")
            print(f"A3 ready: state={self.sim.layout.state_dim}, context={len(self.sim.context.offset)}, "
                  f"motions={len(self.lib.names)}, physics={BASE_TASK}", flush=True)

    def save(self, name="ckpt_latest.pt"):
        if self.rank != 0:
            return
        path = self.log_dir/name
        payload = {"iteration": self.iteration, "num_updates": self.num_updates, "utd_acc": self.utd_acc,
                   "num_policy_updates": self.num_policy_updates,
                   "config": asdict(self.cfg), "contract": self.contract, "contract_sha256": self.contract_digest,
                   "wm": self.wm.state_dict(), "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
                   "wm_opt": self.wm_trainer.opt.state_dict(), "policy_opt": self.policy_trainer.opt.state_dict(),
                   "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                   "bin_return": self.sim.command.bin_return, "bin_memory": self.sim.command.bin_memory,
                   "synthetic_fixture": self.cfg.data.allow_synthetic,
                   "resume_semantics": "optimizer/sampler continuation; simulator and replay restart with warmup"}
        tmp = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, path):
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        check_contract_version(ck)
        if ck.get("contract_sha256") != self.contract_digest:
            raise ValueError("Checkpoint contract mismatch: robot/physics, data, context, or observation changed")
        self.wm.load_state_dict(ck["wm"]); self.policy.load_state_dict(ck["policy"]); self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.sim.command.bin_return.copy_(ck["bin_return"])
        self.sim.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration, self.num_updates, self.utd_acc = ck["iteration"], ck["num_updates"], ck["utd_acc"]
        self.num_policy_updates = ck["num_policy_updates"]

    def update(self):
        """One scheduled WM round, followed by actor/critic only after WM-only warmup."""
        cfg, sch = self.cfg, self.cfg.schedule
        lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr,
                       sch.lr_final_ratio, sch.lr_decay_fraction)
        self.wm_trainer.set_lr(lr)
        self.policy_trainer.set_lr(lr*cfg.policy_trainer.lr/cfg.wm_trainer.lr)
        wm_metrics = self.wm_trainer.update(self.buffer)
        policy_active = self.num_updates >= sch.wm_only_updates
        pi_metrics = {}
        if policy_active:
            for _ in range(sch.policy_updates_per_wm_update):
                pi_metrics = self.policy_trainer.update(self.buffer)
                self.num_policy_updates += 1
        self.num_updates += 1
        metrics = {**{f"wm/{k}": v for k, v in wm_metrics.items()},
                   **{f"policy/{k}": v for k, v in pi_metrics.items()}, "lr": lr,
                   "train/wm_only": float(not policy_active),
                   "train/wm_only_remaining_updates": max(0, sch.wm_only_updates-self.num_updates)}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Nonfinite training metrics at iteration {self.iteration}")
        return metrics

    def run(self):
        cfg, sch = self.cfg, self.cfg.schedule
        obs = self.sim.observation()
        start = time.monotonic()
        collected = 0
        ep_return = torch.zeros(cfg.env.num_envs, device=cfg.device)
        ep_length = torch.zeros_like(ep_return)
        recent_returns = []
        metrics = {}
        tracker = None
        tracking_state = "crashed"
        try:
            # All ranks fail together if cloud initialization fails on rank zero.
            tracking_error = None
            try:
                tracker = SwanLabLogger(cfg, self.log_dir, rank=self.rank, world=self.world,
                                        contract_digest=self.contract_digest)
            except Exception as error:
                tracking_error = tracking_error_message(error)
            if allreduce_mean(float(tracking_error is not None)):
                raise RuntimeError(f"SwanLab initialization failed on rank 0 ({tracking_error or 'see rank 0'})")
            while self.iteration < sch.total_iterations:
                with torch.no_grad():
                    action = self.policy(obs["policy"], obs["goal"])
                    obs, reward, term, timeout, extra = self.sim.step(action)
                ep_return += reward; ep_length += 1
                done = term | timeout
                if done.any():
                    recent_returns.extend((ep_return[done]/(.1*ep_length[done])).tolist())
                    ep_return[done] = 0.; ep_length[done] = 0.
                self.iteration += 1; collected += 1
                if collected >= sch.warmup_iterations:
                    self.utd_acc += sch.updates_per_env_step
                    while self.utd_acc >= 1.:
                        self.utd_acc -= 1.
                        metrics = self.update()
                if self.iteration % sch.log_interval == 0:
                    elapsed = time.monotonic()-start
                    ret_sum, count = allreduce_mean(sum(recent_returns)), allreduce_mean(len(recent_returns))
                    metrics.update({"train/episode_normalized_return": ret_sum/count if count else 0.,
                                    "train/finished_episodes": count*self.world, "time/hours": elapsed/3600,
                                    "train/updates": self.num_updates,
                                    "train/policy_updates": self.num_policy_updates,
                                    "replay/transitions": self.buffer.size})
                    tracking_error = None
                    if self.rank == 0:
                        for key, value in metrics.items():
                            self.writer.add_scalar(key, value, self.iteration)
                        with (self.log_dir/"metrics.jsonl").open("a") as f:
                            f.write(json.dumps({"iteration": self.iteration, **metrics})+"\n")
                        try:
                            tracker.log(metrics, step=self.iteration)
                        except Exception as error:
                            tracking_error = tracking_error_message(error)
                        phase = ("replay_fill" if collected < sch.warmup_iterations else
                                 "wm_only" if self.num_policy_updates == 0 and sch.wm_only_updates else "joint")
                        wm_loss = f"{metrics['wm/loss']:.4g}" if 'wm/loss' in metrics else "pending"
                        actor_grad = f"{metrics['policy/policy_grad_norm']:.4g}" if 'policy/policy_grad_norm' in metrics else "pending"
                        print(f"it={self.iteration} phase={phase} updates={self.num_updates} "
                              f"policy_updates={self.num_policy_updates} wm_loss={wm_loss} "
                              f"actor_grad={actor_grad}", flush=True)
                    if allreduce_mean(float(tracking_error is not None)):
                        raise RuntimeError(f"SwanLab metric logging failed ({tracking_error or 'see rank 0'}); local JSONL is preserved")
                    recent_returns.clear()
                    if sch.max_hours is not None and allreduce_mean(float(elapsed/3600 > sch.max_hours)) > 0:
                        break
                if self.iteration % sch.checkpoint_interval == 0:
                    self.save(); self.save(f"ckpt_{self.iteration:07d}.pt")
            self.save()
            tracking_state = "success"
        except KeyboardInterrupt:
            tracking_state = "aborted"
            raise
        finally:
            try:
                if tracker is not None:
                    tracker.finish(tracking_state)
            finally:
                if self.writer is not None:
                    self.writer.close()
                self.sim.close()
                finalize()
