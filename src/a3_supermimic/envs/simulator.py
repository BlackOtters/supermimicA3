"""Adapter around the existing A3 physics and randomization lifecycle."""
from pathlib import Path
from dataclasses import asdict
import math
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from a3_supermimic.robot import setup_backend, RobotLayout, POLICY_JOINTS, HEAD_JOINTS, TRACKED_BODIES, BASE_TASK, TRACKING_ANCHOR

setup_backend()
from a3_supermimic.envs import randomization as _randomization  # register A3 upstream DR
from a3_supermimic.envs import compact_robot as _compact_robot
from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.backends.mjlab.env import MjlabBackendEnv
from a3_supermimic.kinematics.model import ModelKinematics
from a3_supermimic.kinematics.rotations import quat_apply
from a3_supermimic.envs.mdp.reset import ResetCfg, reset_reference
from a3_supermimic.world_model.heads import AnalyticHeads, flatten_history_term_major, anchor_relative_obs
from a3_supermimic.world_model.wm_trainer import reference_at
from a3_supermimic.envs.mdp.motion_library import MotionLibrary
from a3_supermimic.envs.context import SimContext, CompactSimContext, TerrainFeatures
from a3_supermimic.imagination.rollout import OBS_NOISE_HALF, ANCHOR_NOISE_HALF


def task_config(cfg, files):
    if cfg.env.representation not in ('full', 'compact_rigid'):
        raise ValueError('representation must be full or compact_rigid')
    compact = cfg.env.representation == 'compact_rigid'
    if compact and (cfg.env.evaluation_physics != 'rigid' or
                    cfg.env.randomization_profile != 'upstream_rigid' or cfg.env.terrain != 'plane'):
        raise ValueError('compact_rigid requires rigid physics, upstream_rigid DR and plane terrain')
    # Snapshot the inherited A3 task instead of discovering another workspace.
    task = OmegaConf.load(Path(__file__).resolve().parents[1] / "assets/task_base.yaml")
    OmegaConf.set_struct(task, False)
    # Replace the task interface; retain A3 physics and physical-parameter DR.
    task.name = "A3SuperMimicFourAnkle"
    task.num_envs = cfg.env.num_envs
    # Full-corpus floor contacts can exceed the parent's 500 constraint slots.
    # Reserve capacity without changing solver iterations or physical parameters.
    task.sim.mjlab_njmax = max(1024, int(task.sim.get("mjlab_njmax", 500)))
    task.max_episode_length = max(1, round(cfg.env.episode_length_s / .02))
    task.command = {"_target_": "a3_supermimic.A3MotionCommand", "files": [str(p) for p in files],
                    "reset_noise": cfg.env.reset_noise, "sampling_mode": cfg.env.sampling_mode,
                    "allow_synthetic": cfg.data.allow_synthetic, "reset_cfg": asdict(cfg.env.reset)}
    task.observation = {}
    task.reward = {}
    task.termination = {}
    # Paper Appendix A omits interval velocity kicks; reset noise remains enabled.
    task.randomization.pop("perturb_root_vel", None)
    task.pop("v1_init_checkpoint", None)
    if cfg.env.terrain != "inherit":
        if cfg.env.terrain != "plane":
            raise ValueError("terrain must be inherit or plane")
        task.terrain = "plane"
        task.pop("terrain_generator", None)
    if cfg.env.evaluation_physics not in ("mixed", "rigid", "elastic"):
        raise ValueError("evaluation_physics must be mixed, rigid or elastic")
    # Topology selection also applies to training with domain randomization.
    # Keep the inherited mixture unless a fixed topology is explicitly selected.
    if cfg.env.evaluation_physics != "mixed":
        task.randomization.four_ankle_dr.rigid_probability = {"rigid": 1., "elastic": 0.}[cfg.env.evaluation_physics]
    if not cfg.env.domain_randomization:
        # The four-ankle reset hook is physics, not optional. Keep it with fixed
        # parameters and retain the mixed/rigid/elastic topology selector.
        dr = task.randomization.four_ankle_dr
        task.randomization = {"four_ankle_dr": OmegaConf.to_container(dr, resolve=True)}
        dr = task.randomization.four_ankle_dr
        dr.rigid_probability = {"mixed": .25, "rigid": 1., "elastic": 0.}[cfg.env.evaluation_physics]
        for name in ("ankle_inertia_scale", "armature_scale", "kp_scale", "damping_ratio_scale"):
            dr[name] = [1., 1.]
        dr.pitch_k_range = [math.sqrt(200.*2000.)]*2
        dr.roll_k_range = [math.sqrt(65.*400.)]*2
        dr.pitch_d_range = dr.roll_d_range = [1., 1.]
        dr.delay_ms = [5.]
        task.randomization.fixed_contact_friction = {
            "_target_": "mimic_lite.depsic_a3_ultra_contact_friction",
            "body_names": "^(?!.*head.*).+$", "static_friction_range": [1., 1.],
            "num_buckets": 1, "plane_only": False,
        }
    if cfg.env.randomization_profile not in ("a3", "upstream_rigid"):
        raise ValueError("randomization_profile must be a3 or upstream_rigid")
    if cfg.env.randomization_profile == "upstream_rigid":
        if cfg.env.evaluation_physics != "rigid":
            raise ValueError("upstream_rigid randomization requires evaluation_physics=rigid")
        # Keep the mandatory rigid-topology hook, with nominal motors and zero latency.
        dr = task.randomization.four_ankle_dr
        for name in ("ankle_inertia_scale", "armature_scale", "kp_scale", "damping_ratio_scale"):
            dr[name] = [1., 1.]
        dr.pitch_k_range = dr.roll_k_range = [1., 1.]
        dr.pitch_d_range = dr.roll_d_range = [1., 1.]
        dr.delay_ms = [0.]
        events = {"four_ankle_dr": OmegaConf.to_container(dr, resolve=True)}
        if cfg.env.domain_randomization:
            events["upstream_torso_com"] = {
                "_target_": "mimic_lite.depsic_a3_ultra_body_com", "body_names": ["torso_Link"],
                "x": [-.025, .025], "y": [-.05, .05], "z": [-.05, .05],
            }
        events["upstream_reset"] = {
            "_target_": "a3_supermimic.upstream_rigid_reset",
            "friction_range": [.3, 1.2] if cfg.env.domain_randomization else [1., 1.],
            "bias_range": [-.01, .01] if cfg.env.domain_randomization else [0., 0.],
        }
        task.randomization = events
    if compact:
        task.name = 'A3SuperMimicCompactRigid'
        task.robot.name = 'a3-supermimic-compact-rigid'
        task.randomization.pop('four_ankle_dr')  # native rigid mechanics, no topology switching hook
    return OmegaConf.create(OmegaConf.to_container(task, resolve=True))


class A3MotionCommand(Command, namespace="a3_supermimic"):
    def __init__(self, env, files, reset_noise=True, sampling_mode="return", allow_synthetic=False, reset_cfg=None):
        super().__init__(env)
        self.layout = RobotLayout.from_model(env.sim.mj_model)
        self.fk = ModelKinematics(env.sim.mj_model, device=self.device)
        self.fk.body_names = [n.split("/")[-1] for n in self.fk.body_names]
        self.policy_joint_ids = torch.tensor([self.asset.joint_names.index(n) for n in POLICY_JOINTS], device=self.device)
        self.head_joint_ids = torch.tensor([self.asset.joint_names.index(n) for n in HEAD_JOINTS if n in self.asset.joint_names], device=self.device, dtype=torch.long)
        self.joint_qids = self.asset.indexing.joint_q_adr
        self.joint_vids = self.asset.indexing.joint_v_adr
        self.reset_cfg = ResetCfg(**(reset_cfg or {}))
        self.reset_joint_limits = self.asset.data.soft_joint_pos_limits[0].clone()
        for jid, name in zip(self.policy_joint_ids, POLICY_JOINTS):
            hard = torch.tensor(env.sim.mj_model.joint("robot/"+name).range, device=self.device, dtype=torch.float32)
            factor = .8 if name.startswith("waist_") else .9
            self.reset_joint_limits[jid] = hard.mean() + (hard-hard.mean())*factor
        self.default_joint_pos = self.asset.data.default_joint_pos[0, self.policy_joint_ids].clone()
        self.library = MotionLibrary(files, env.sim.mj_model, self.layout, self.fk,
                                     self.fk.body_indices(TRACKED_BODIES), self.default_joint_pos,
                                     allow_synthetic=allow_synthetic)
        self.reset_noise, self.sampling_mode = reset_noise, sampling_mode
        if sampling_mode not in ("return", "uniform", "start"):
            raise ValueError("Unknown reset sampling mode")
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.forced_frame = torch.full_like(self.frame, -1)
        nb = len(self.library.bin_starts)
        self.bin_return = torch.full((nb,), 10., device=self.device)
        self.bin_memory = torch.full((nb,), 1./nb, device=self.device)
        self.ep_rewards = torch.zeros(self.num_envs, int(env.cfg.max_episode_length), device=self.device)
        self.ep_bins = torch.zeros_like(self.ep_rewards, dtype=torch.long)
        self.ep_len = torch.zeros_like(self.frame)

    def _sample(self, n):
        lib = self.library
        if self.sampling_mode == "start":
            return lib.starts[torch.randint(len(lib.names), (n,), device=self.device)]
        if self.sampling_mode == "uniform":
            frame = torch.randint(lib.num_frames, (n,), device=self.device)
        else:
            hard = torch.softmax(-25.*self.bin_return/10., 0)
            self.bin_memory.lerp_(hard, .01)
            p = .5/len(hard) + .4*hard + .1*self.bin_memory/self.bin_memory.sum()
            bins = torch.multinomial(p, n, replacement=True)
            frame = lib.bin_starts[bins] + (torch.rand(n, device=self.device)*(lib.bin_ends[bins]-lib.bin_starts[bins]+1)).long()
        return torch.maximum(torch.minimum(frame, lib.last_frame_of(frame)-1), lib.starts[lib.frame_clip[frame]])

    def sample_init(self, env_ids, reset_td=None):
        frame = self._sample(len(env_ids))
        forced = self.forced_frame[env_ids]
        frame = torch.where(forced >= 0, forced, frame)
        self.forced_frame[env_ids] = -1
        self.frame[env_ids] = frame
        self.origins[env_ids] = self.env.scene.get_spawn_origins(env_ids)
        q, v = self.library.generalized(frame)
        q, v = reset_reference(q, v, self.layout, self.joint_qids, self.reset_joint_limits,
                               self.reset_cfg, noise=self.reset_noise)
        self.asset.write_joint_state_to_sim(q[:, self.joint_qids], v[:, self.joint_vids], env_ids=env_ids)
        root = torch.cat((q[:, :3]+self.origins[env_ids], q[:, 3:7], v[:, :3], quat_apply(q[:, 3:7], v[:, 3:6])), -1)
        return root

    def reset(self, env_ids):
        self.ep_len[env_ids] = 0

    def pre_step(self, substep):
        if len(self.head_joint_ids):
            self.asset.set_joint_position_target(self.library.head_targets(self.frame), joint_ids=self.head_joint_ids)

    def record_reward(self, reward):
        ids = torch.arange(self.num_envs, device=self.device)
        self.ep_rewards[ids, self.ep_len] = reward
        self.ep_bins[ids, self.ep_len] = torch.searchsorted(self.library.bin_starts, self.frame, right=True)-1
        self.ep_len += 1

    def finalize(self, ids, terminated, clip_end):
        if not len(ids):
            return
        lens = self.ep_len[ids]
        steps = torch.arange(int(lens.max()), device=self.device)
        mask = steps[None] < lens[:, None]
        discount = .99**steps.double()
        r = self.ep_rewards[ids, :len(steps)].double()*mask
        rtg = (r*discount).flip(1).cumsum(1).flip(1)/discount
        # Match upstream: without sampler value bootstrapping, omit the final
        # 100 steps near a time-limit, but include failures and true clip ends.
        timeout = ~terminated[ids] & ~clip_end[ids]
        mask &= ~(timeout[:, None] & ((lens[:, None]-steps[None]) < 100))
        bins, values = self.ep_bins[ids, :len(steps)][mask], rtg[mask].float()
        sums = torch.zeros_like(self.bin_return).index_add_(0, bins, values)
        count = torch.zeros_like(self.bin_return).index_add_(0, bins, torch.ones_like(values))
        touched = count > 0
        self.bin_return[touched] = .95*self.bin_return[touched] + .05*sums[touched]/count[touched]


class Simulator:
    def __init__(self, cfg, files):
        self.cfg = cfg
        task = task_config(cfg, files)
        self.env = MjlabBackendEnv(task, device=cfg.device, headless=True)
        self.command = self.env.command_manager
        self.library, self.layout = self.command.library, self.command.layout
        self.device, self.num_envs = self.env.device, self.env.num_envs
        limits = self.command.reset_joint_limits[self.command.policy_joint_ids]
        self.heads = AnalyticHeads(self.command.fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link", limits,
                                   self.command.default_joint_pos, layout=self.layout,
                                   termination_cfg=cfg.env.termination,
                                   observation_anchor_body_name=cfg.env.observation_anchor)
        self.context = CompactSimContext(self.env) if self.layout.compact else SimContext(self.env)
        self.previous = torch.zeros(self.num_envs, self.layout.action_dim, device=self.device)
        self.previous_clean = torch.zeros(self.num_envs, 29, device=self.device)
        self.fresh = torch.ones(self.num_envs, 1, device=self.device)
        self.buffer = None
        self.history = torch.zeros(cfg.policy.history, self.num_envs, 93, device=self.device)
        self.env.reset()
        if not self.layout.compact and int(self.env._four_ankle_state.delay_lags.max()) >= self.env.decimation:
            raise ValueError("The state contract requires actuator delay shorter than one policy step")
        self.xi = self.context.read()
        self.terrain = None if self.env.terrain_type == "plane" else TerrainFeatures.from_env(self.env)
        self._reset_history(torch.arange(self.num_envs, device=self.device))

    def state(self):
        q = self.env.sim.data.qpos.clone()
        q[:, :3] -= self.command.origins
        physical = torch.cat((q, self.env.sim.data.qvel.clone()), -1)
        return physical if self.layout.compact else torch.cat((physical, self.previous, self.fresh), -1)

    def _observation(self, state):
        obs = self.heads.observation(state, self.previous_clean, self.xi)
        if self.cfg.env.obs_noise:
            obs = obs + (torch.rand_like(obs)*2-1)*OBS_NOISE_HALF.to(self.device)
        return obs

    def _reset_history(self, ids):
        self.history[:, ids] = self._observation(self.state())[ids][None]

    def observation(self):
        goal = self.library.goal_window(self.command.frame, self.cfg.env.future_frames)
        if self.cfg.env.anchor_obs:
            bodies = self.heads.robot_bodies(self.state())
            ref = self.library.bodies(self.command.frame)
            ai = self.heads.observation_anchor_index
            extra = anchor_relative_obs(bodies["pos"][:, ai], bodies["quat"][:, ai], ref.pos_w[:, ai], ref.quat_w[:, ai])
            if self.cfg.env.obs_noise:
                extra += (torch.rand_like(extra)*2-1)*ANCHOR_NOISE_HALF.to(self.device)
            goal = torch.cat((goal, extra), -1)
        return {"policy": flatten_history_term_major(self.history), "goal": goal}

    def attach_buffer(self, buffer):
        self.buffer = buffer
        buffer.register_context(torch.arange(self.num_envs, device=self.device), self.xi)

    def reset(self, ids=None, frames=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if frames is not None:
            self.command.forced_frame[ids] = frames
        mask = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.env.reset(TensorDict({"_reset": mask}, batch_size=[self.num_envs], device=self.device))
        self.previous[ids] = 0.; self.previous_clean[ids] = 0.; self.fresh[ids] = 1.
        self.xi = self.context.read()
        if self.buffer is not None:
            self.buffer.register_context(ids, self.xi[ids])
        self._reset_history(ids)
        return self.observation()

    @torch.no_grad()
    def step(self, action, *, auto_reset=True):
        state = self.state()
        frame, episode = self.command.frame.clone(), self.env.episode_id.clone()
        previous_clean = self.previous_clean.clone()
        executed = action + torch.randn_like(action)*self.cfg.env.action_noise_std
        full_action = executed if self.layout.compact else torch.cat((executed, self.library.head_targets(frame)), -1)
        result = self.env._step(TensorDict({"action": executed}, batch_size=[self.num_envs], device=self.device))
        self.previous = full_action.clone(); self.previous_clean = action.clone(); self.fresh.zero_()
        next_state = self.state()
        ref = reference_at(self.library, frame)
        terms = self.heads.reward(next_state, action, previous_clean, ref)
        reward = terms["total"]
        invalid = result["terminated"].squeeze(-1) | ~torch.isfinite(next_state).all(-1) | ~torch.isfinite(reward)
        terminated = self.heads.terminated(next_state, ref) | invalid
        clip_end = frame >= self.library.last_frame_of(frame)
        timeout = (self.env.episode_length_buf >= self.env.max_episode_length.squeeze(-1)) | clip_end
        done = terminated | timeout
        reward = torch.where(invalid, torch.zeros_like(reward), reward)
        if self.buffer is not None:
            self.buffer.append(state, full_action, next_state, frame, episode, terminated,
                               previous_clean_action=previous_clean,
                               valid=~invalid, start_valid=torch.isfinite(state).all(-1))
        self.command.record_reward(reward)
        ids = done.nonzero().flatten()
        self.command.finalize(ids, terminated, clip_end)
        self.command.frame = torch.minimum(frame+1, self.library.last_frame_of(frame))
        self.history = torch.cat((self.history[1:], self._observation(next_state)[None]), 0)
        extra = {"reward_terms": terms, "invalid": invalid, "terminal_state": next_state}
        if auto_reset and len(ids):
            self.reset(ids)
        return self.observation(), reward, terminated, timeout, extra

    def close(self):
        self.env.close()
