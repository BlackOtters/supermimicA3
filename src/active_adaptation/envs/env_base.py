import os
import math
import warnings
from collections import OrderedDict
from typing import Dict, Mapping, cast

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.envs.adapters import SimAdapter, SceneAdapter
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.video_recorder import (
    VideoRecorder,
    NullVideoRecorder,
    IsaacVideoRecorder,
    RgbArrayVideoRecorder,
)
from active_adaptation.envs.utils import GroundQuery
from active_adaptation.registry import RegistryMixin

if active_adaptation.get_backend() == "isaaclab":
    import isaacsim.core.utils.torch as torch_utils


EMA_DECAY = 0.99
PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
def parse_component_spec(name: str, cfg):
    if cfg is None or not hasattr(cfg, "items"):
        raise ValueError(f"Component '{name}' must be a mapping.")
    kwargs = dict(cfg)
    target = kwargs.pop("_target_", name)
    return name, target, kwargs


class ObsGroup:
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int = 0,
    ):
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.timestamp = -1

    @property
    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        if not hasattr(self, "_spec"):
            sample = self.compute({}, 0)
            spec = {
                self.name: Unbounded(
                    sample[self.name].shape,
                    dtype=sample[self.name].dtype,
                )
            }
            self._spec = Composite(spec, shape=[sample[self.name].shape[0]]).to(
                sample[self.name].device
            )
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> TensorDictBase:
        tensordict[self.name] = self._compute()
        return tensordict

    def _compute(self) -> torch.Tensor:
        return torch.cat([func.compute() for func in self.funcs.values()], dim=-1)

    def symmetry_transform(self):
        transforms = [
            func.symmetry_transform().to(func.device) for func in self.funcs.values()
        ]
        return symmetry_utils.SymmetryTransform.cat(transforms)


class RewardGroup:
    """Group of reward terms; per-term EMA logging lives on each :class:`~mdp.Reward`."""

    def __init__(
        self,
        env: "_EnvBase",
        name: str,
        funcs: OrderedDict[str, mdp.Reward],
        enabled: bool = True,
        compile: bool = False,
    ):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.enabled = enabled
        self.compile = compile

        self.enabled_rewards = sum(func.enabled for func in funcs.values())
        self.rew_buf = torch.zeros(
            env.num_envs, self.enabled_rewards, device=env.device
        )
        if compile:
            self.compute = torch.compile(self.compute, fullgraph=True)

    def compute(self) -> torch.Tensor:
        rewards = []
        if self.name in {"tracking", "tracking_metrics"}:
            print_enabled = True
            # print(f"Reward group '{self.name}':")
        else:
            print_enabled = False
        for key, func in self.funcs.items():
            reward = func.compute()
            self.env.stats[self.name, key].add_(reward)
            if func.enabled:
                rewards.append(reward)
        if len(rewards):
            self.rew_buf = torch.cat(rewards, 1)
        return self.rew_buf.sum(dim=1, keepdim=True)

    def get_ema_stats(self) -> Dict[str, float]:
        """Flatten per-term EMA metrics (e.g. mean, optional var) for logging."""
        result: Dict[str, float] = {}
        for key, func in self.funcs.items():
            mean, var = func.get_ema_stats()
            result[key] = mean.item()
            if var is not None:
                result[f"{key}_var"] = var.item()
        return result


class _EnvBase(EnvBase, RegistryMixin):
    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(
            device=device,
            batch_size=[cfg.num_envs],
            run_type_checks=False,
        )
        self.backend = active_adaptation.get_backend()
        self.cfg = cfg
        self.headless = headless

        self._setup_simulation()
        self._setup_mdp_managers()
        self._build_tensor_specs()

        self.timestamp: int = 0
        self.stats: TensorDict = self.reward_spec["stats"].zero()
        self.input_tensordict = None
        self.extra = {}
        self._startup_done = False

    @property
    def max_episode_length(self) -> torch.Tensor:
        return self._max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, value: int | torch.Tensor):
        if isinstance(value, int):
            value = torch.full((self.num_envs, 1), value, device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.dtype == torch.long, "Max episode length must be an integer tensor"
            assert value.shape == (self.num_envs, 1), "Max episode length must be a tensor of shape (num_envs, 1)"
        else:
            raise ValueError(f"Invalid type for max episode length: {type(value)}")
        self._max_episode_length = value.to(self.device)

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------
    def _setup_simulation(self):
        self.terrain_type = None
        self.setup_scene()
        self.sim = cast(SimAdapter, self.sim)
        self.scene = cast(SceneAdapter, self.scene)
        if self.terrain_type is None:
            warnings.warn(
                "Terrain type is not set. Please check if the scene is properly initialized."
            )
        self.step_dt = float(self.cfg.sim.step_dt)
        self.physics_dt = float(self.sim.get_physics_dt())
        if self.step_dt <= 0.0 or self.physics_dt <= 0.0:
            raise ValueError(
                "sim.step_dt and physics_dt must both be positive, got "
                f"step_dt={self.step_dt}, physics_dt={self.physics_dt}."
            )
        ratio = self.step_dt / self.physics_dt
        self.decimation = round(ratio)
        if self.decimation < 1 or not math.isclose(
            ratio, self.decimation, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "sim.step_dt must be an integer multiple of physics_dt. "
                f"Got step_dt={self.step_dt} and physics_dt={self.physics_dt} "
                f"(ratio={ratio:.12g}); silently truncating this ratio changes "
                "the control/reward time base."
            )

        self.max_episode_length = int(self.cfg.max_episode_length)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_count = 0
        self.current_iter = 0

        # MuJoCo Warp simulates all worlds in one batch.  A rare non-finite
        # state in one world must be isolated before command/reward/observation
        # code sees it; otherwise one corrupt privileged observation can poison
        # global normalization statistics and take down the entire PPO run.
        self._physics_invalid_mask = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        self._physics_invalid_any = False
        self._physics_nonfinite_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        # A finite simulator state does not guarantee that a task callback
        # produced finite command/reward/observation tensors.  Keep this
        # separate from the physics mask so the dashboard distinguishes a
        # MuJoCo failure from an MDP implementation failure.
        self._mdp_invalid_mask = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )

    def _setup_mdp_managers(self):
        self.randomizations: Mapping[str, mdp.Randomization] = OrderedDict()
        self.observation_funcs: Mapping[str, ObsGroup] = OrderedDict()
        self.reward_groups: Mapping[str, RewardGroup] = OrderedDict()
        self.input_managers: Mapping[str, mdp.Action] = OrderedDict()
        self.termination_funcs: Mapping[str, mdp.Termination] = OrderedDict()

        self._enabled_reward_groups = 0

        self._startup_callbacks = []
        self._reset_callbacks = []
        self._state_reset_callbacks = []
        # Resetting an observation history must happen after the simulator has
        # propagated reset root/joint writes.  In particular, mjlab entity
        # read properties are only current after ``sim.forward()``.  Keeping
        # these callbacks separate prevents the first returned observation of
        # an episode from being built from the previous episode's state.
        self._observation_reset_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []
        self._update_callbacks = []
        self._debug_draw_callbacks = []

        # MDP: command manager
        command_cfg = dict(self.cfg.command)
        class_name = command_cfg.pop("_target_", None)
        if class_name is None:
            raise ValueError("Command config must provide `_target_`.")
        command = mdp.Command.make(class_name, self, **command_cfg)
        if not command:
            raise ValueError(f"Command class '{class_name}' not found")
        self.command_manager = cast(mdp.Command, command)
        self._pre_step_callbacks.append(self.command_manager.pre_step)
        self._reset_callbacks.append(self.command_manager.reset)
        self._state_reset_callbacks.append(self.command_manager.reset)
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)

        # MDP: input managers
        for input_name, input_cfg in dict(self.cfg.get("input", {})).items():
            _, input_cls_name, input_kwargs = parse_component_spec(
                input_name, input_cfg
            )
            input_cls = mdp.Action.registry[input_cls_name]
            input_manager = cast(mdp.Action, input_cls(self, **input_kwargs))
            self.input_managers[input_name] = input_manager
            self._reset_callbacks.append(input_manager.reset)
            self._state_reset_callbacks.append(input_manager.reset)
            self._debug_draw_callbacks.append(input_manager.debug_draw)

        # MDP: randomizations
        for rand_name, rand_cfg in self.cfg.get("randomization", {}).items():
            # Same overlay switch as reward terms: ``{_enabled_: false}``
            # drops an inherited randomization without restating the table.
            if isinstance(rand_cfg, Mapping) and rand_cfg.get("_enabled_", True) is False:
                continue
            rand_name, cls_name, rand_kwargs = parse_component_spec(rand_name, rand_cfg)
            rand_kwargs.pop("_enabled_", None)
            rand = mdp.Randomization.make(cls_name, self, **rand_kwargs)
            if not rand:
                continue
            rand = cast(mdp.Randomization, rand)
            self.randomizations[rand_name] = rand
            self._add_mdp_component(rand)

        # MDP: observations
        for group_name, group_cfg in self.cfg.observation.items():
            funcs = OrderedDict()
            for obs_name, obs_cfg in group_cfg.items():
                obs_name, obs_cls_name, obs_kwargs = parse_component_spec(
                    obs_name, obs_cfg
                )
                obs = mdp.Observation.make(obs_cls_name, self, **obs_kwargs)
                if not obs:
                    continue
                obs = cast(mdp.Observation, obs)
                funcs[obs_name] = obs
                self._add_mdp_component(obs)
            self.observation_funcs[group_name] = ObsGroup(group_name, funcs)

        # MDP: rewards
        reward_cfg = dict(self.cfg.reward)
        self.mult_dt = reward_cfg.pop("_mult_dt_", True)
        for group_name, group_cfg in reward_cfg.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()

            group_cfg = dict(group_cfg)
            enabled = group_cfg.pop("_enabled_", True)
            compile = group_cfg.pop("_compile_", False)
            self._enabled_reward_groups += int(enabled)

            for rew_name, rew_cfg in group_cfg.items():
                # Allow task overlays to disable an inherited reward term
                # without having to duplicate the complete reward table.
                # Historically ``_enabled_`` was only consumed at group
                # level, which made a V1 AMP overlay pass that key into a
                # Reward constructor and fail at startup.
                if isinstance(rew_cfg, Mapping) and rew_cfg.get("_enabled_", True) is False:
                    continue
                rew_name, cls_name, rew_kwargs = parse_component_spec(rew_name, rew_cfg)
                rew_kwargs.pop("_enabled_", None)
                reward = mdp.Reward.make(cls_name, self, **rew_kwargs)
                if not reward:
                    continue
                reward = cast(mdp.Reward, reward)
                funcs[rew_name] = reward
                self._add_mdp_component(reward)
                print(f"\t{rew_name}: \t{reward.weight:.2f} \t enabled={reward.enabled}")

            self.reward_groups[group_name] = RewardGroup(
                self, group_name, funcs, enabled, compile
            )

        # MDP: terminations
        print("Termination functions:")
        for term_name, term_cfg in self.cfg.get("termination", {}).items():
            term_name, cls_name, term_kwargs = parse_component_spec(term_name, term_cfg)
            term = mdp.Termination.make(cls_name, self, **term_kwargs)
            if not term:
                continue
            term = cast(mdp.Termination, term)
            print(f"\t{term_name}: \t{'timeout' if term.is_timeout else 'termination'}")
            self.termination_funcs[term_name] = term
            self._add_mdp_component(term)

    def _build_tensor_specs(self):
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            shape=[self.num_envs],
            device=self.device,
        )

        action_spec = {
            input_name: Unbounded(
                [self.num_envs, input_manager.action_dim], device=self.device
            )
            for input_name, input_manager in self.input_managers.items()
        }
        self.action_spec = Composite(
            action_spec, shape=[self.num_envs], device=self.device
        )

        observation_spec = {}
        [
            observation_spec.update(group.spec)
            for group in self.observation_funcs.values()
        ]
        self.observation_spec = Composite(
            observation_spec, shape=[self.num_envs], device=self.device
        )
        self.observation_spec["episode_id"] = Unbounded(
            [self.num_envs], dtype=torch.long, device=self.device
        )

        reward_spec = Composite(
            {
                "stats": {
                    "episode_len": Unbounded([self.num_envs, 1]),
                    "success": Unbounded([self.num_envs, 1]),
                }
            },
            shape=[self.num_envs],
        ).to(self.device)
        reward_spec_extensions = Composite({})

        for group_name, reward_group in self.reward_groups.items():
            for rew_name in reward_group.funcs.keys():
                reward_spec_extensions["stats", group_name, rew_name] = Unbounded(
                    1, device=self.device
                )
            reward_spec_extensions["stats", group_name, "return"] = Unbounded(
                1, device=self.device
            )

        for term_name in self.termination_funcs.keys():
            reward_spec_extensions["stats", "termination", term_name] = Unbounded(
                1, device=self.device
            )
        # This is an environment integrity termination, rather than a task
        # reward/termination choice.  Always expose it so bad MuJoCo worlds
        # remain visible in W&B for every mjlab task.
        reward_spec_extensions["stats", "termination", "physics_nonfinite"] = Unbounded(
            1, device=self.device
        )
        reward_spec_extensions["stats", "termination", "mdp_nonfinite"] = Unbounded(
            1, device=self.device
        )

        reward_spec_extensions["reward"] = Unbounded(
            self._enabled_reward_groups, device=self.device
        )
        reward_spec_extensions["discount"] = Unbounded(1, device=self.device)
        reward_spec.update(reward_spec_extensions.expand(self.num_envs).to(self.device))
        self.reward_spec = reward_spec

    def _add_mdp_component(self, component: mdp.MDPComponent):
        if mdp.is_method_implemented(component, mdp.MDPComponent, "startup"):
            self._startup_callbacks.append(component.startup)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "reset"):
            # Keep the original all-component sequence for non-mjlab
            # backends.  mjlab additionally needs the state/observation split
            # to forward MuJoCo before any observation history samples state.
            self._reset_callbacks.append(component.reset)
            if isinstance(component, mdp.Observation):
                self._observation_reset_callbacks.append(component.reset)
            else:
                self._state_reset_callbacks.append(component.reset)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "pre_step"):
            self._pre_step_callbacks.append(component.pre_step)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "post_step"):
            self._post_step_callbacks.append(component.post_step)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "update"):
            cb = ScopedTimer(component.__class__.__name__)(component.update)
            self._update_callbacks.append(cb)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "debug_draw"):
            self._debug_draw_callbacks.append(component.debug_draw)

    def setup_scene(self):
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Runtime helpers
    # ---------------------------------------------------------------------
    def set_progress(self, progress: int):
        self.current_iter = progress

    @staticmethod
    def _callback_label(callback) -> str:
        owner = getattr(callback, "__self__", None)
        if owner is not None:
            return owner.__class__.__name__
        return getattr(callback, "__qualname__", getattr(callback, "__name__", "callback"))

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def action_manager(self):
        return self.input_managers["action"]

    @property
    def stats_ema(self) -> Dict[str, float]:
        """Aggregate EMA stats from all reward groups."""
        result = {}
        for group_key, group in self.reward_groups.items():
            for rew_key, value in group.get_ema_stats().items():
                result[f"reward.{group_key}/{rew_key}"] = value
        return result

    def reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if tensordict is not None:
            self._assert_tensordict_shape(tensordict)

        select_reset_only = kwargs.pop("select_reset_only", False)
        if select_reset_only and tensordict is not None:
            reset_input = tensordict.select(
                *self.reset_keys,
                "terminated",
                "truncated",
                strict=False,
            )
            tensordict_reset = self._reset(reset_input, **kwargs)
        else:
            tensordict_reset = self._reset(tensordict, **kwargs)

        if tensordict_reset is tensordict:
            raise RuntimeError(
                "EnvBase._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty()) "
                "inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, TensorDictBase):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )
        return self._reset_proc_data(tensordict, tensordict_reset)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if not self._startup_done:
            [callback() for callback in self._startup_callbacks]
            self._startup_done = True
            
        if tensordict is not None:
            env_mask = tensordict.get("_reset", None)
            if env_mask is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
                reset_td = None
            else:
                env_mask = env_mask.reshape(self.num_envs)
                env_ids = env_mask.nonzero().squeeze(-1)
                reset_td = tensordict[env_ids]
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
            reset_td = None

        if len(env_ids):
            if hasattr(self, "_physics_invalid_mask"):
                self._physics_invalid_mask[env_ids] = False
            if hasattr(self, "_mdp_invalid_mask"):
                self._mdp_invalid_mask[env_ids] = False
            num_envs = env_ids.numel()
            self.episode_length_buf[env_ids] = 0
            self.episode_id[env_ids] = self.episode_count + torch.arange(
                num_envs, device=self.device
            )
            self.episode_count += num_envs

            self._reset_idx(env_ids, reset_td)
            self.scene.reset(env_ids)
            if self.backend == "mjlab":
                [callback(env_ids) for callback in self._state_reset_callbacks]

                # mjlab writes reset root/joint state directly into MuJoCo's
                # qpos/qvel buffers.  Its derived entity quantities (poses,
                # projected gravity, COM angular velocity, etc.) remain stale
                # until forward propagation.  DEPSIC actor-history reset hooks
                # sample precisely those quantities, so run forward before them.
                # This shared path covers G1/A3 V1 and their V2 task subclasses.
                self.sim.forward()
                [callback(env_ids) for callback in self._observation_reset_callbacks]
            else:
                [callback(env_ids) for callback in self._reset_callbacks]

        tensordict = TensorDict({}, self.num_envs, device=self.device)
        if self.backend == "mjlab":
            # Return the real reset state.  Returning observation_spec.zero()
            # here made every newly reset DEPSIC environment feed an artificial
            # all-zero first policy input, despite the reset callbacks having
            # already initialized the underlying observation histories.
            self._compute_observation(tensordict)
        else:
            # Preserve the existing reset protocol for the other backends;
            # their reset-time simulation synchronization is backend-specific.
            tensordict.update(self.observation_spec.zero())
        tensordict.set("episode_id", self.episode_id.clone())
        return tensordict

    def _reset_idx(
        self, env_ids: torch.Tensor, reset_td: TensorDictBase | None = None
    ):
        init_state = self.command_manager.sample_init(env_ids, reset_td)
        if not isinstance(init_state, dict):
            init_state = {"robot": init_state}
        for key, value in init_state.items():
            if value is not None:
                self.scene.articulations[key].write_root_state_to_sim(
                    value, env_ids=env_ids
                )
        self.stats[env_ids] = 0.0

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # A normal TorchRL caller has already partially reset every terminal
        # row.  Clearing here also makes direct ``env.step`` callers safe: an
        # integrity marker belongs to exactly one transition.
        if hasattr(self, "_mdp_invalid_mask"):
            self._mdp_invalid_mask.zero_()
        if self.backend == "mjlab":
            # A normal caller resets every terminal world before the next
            # step.  Start the new transition clean without a GPU->CPU sync;
            # a new failure will set both fields below.
            self._physics_invalid_mask.zero_()
            self._physics_invalid_any = False
            # Log zero on healthy policy ticks instead of retaining the last
            # failure count indefinitely in the dashboard.
            self.extra["env/physics_nonfinite_env_count"] = 0.0
            self.extra["env/physics_nonfinite_total"] = self._physics_nonfinite_total.clone()
            self.extra["env/physics_nonfinite_last_substep"] = 0.0
        with ScopedTimer("simulation", sync=False):
            with ScopedTimer("process_action", sync=False):
                for input_key, input_manager in self.input_managers.items():
                    if (action := tensordict.get(input_key)) is not None:
                        input_manager.process_action(action)

            for substep in range(self.decimation):
                with ScopedTimer("simulation_pre_step", sync=False):
                    self.scene.zero_external_wrenches()
                    self._apply_action(substep)
                    [callback(substep) for callback in self._pre_step_callbacks]
                    self.scene.write_data_to_sim()
                with ScopedTimer("simulation_step", sync=PROFILE_SYNC_TIMERS):
                    self.sim.step(render=False)
                with ScopedTimer("simulation_post_step", sync=False):
                    # Worlds are independent in mjlab's batched MuJoCo step.
                    # Check once after the final physics substep, before any
                    # command/reward/observation code consumes simulator data.
                    # Checking every 1 ms substep would force 20 CUDA-to-host
                    # synchronizations per 20 ms DEPSIC policy tick.
                    if self.backend == "mjlab" and substep == self.decimation - 1:
                        self._isolate_mjlab_nonfinite_envs(substep)
                    with ScopedTimer("scene.update", sync=PROFILE_SYNC_TIMERS):
                        self.scene.update(self.physics_dt)
                    with ScopedTimer("post_step_callbacks", sync=False):
                        [callback(substep) for callback in self._post_step_callbacks]
            # TODO: test if this is needed
            # if self.backend == "mjlab":
            #     with ScopedTimer("simulation_forward", sync=PROFILE_SYNC_TIMERS):
            #         self.sim._sim.forward()

        if self.sim.has_gui() and self.backend != "mjlab":
            self.sim.render()

        self.episode_length_buf.add_(1)
        self.timestamp += 1

        tensordict = TensorDict({}, self.num_envs, device=self.device)

        with ScopedTimer("command_update", sync=False):
            self.command_manager.update()
        with ScopedTimer("update_callbacks", sync=False):
            [callback() for callback in self._update_callbacks]
            # for callback in self._update_callbacks:
            #     with ScopedTimer(f"{callback.__self__.__class__.__name__}", sync=False):
            #         callback()
        with ScopedTimer("reward", sync=False):
            tensordict = self._compute_reward(tensordict)
        with ScopedTimer("termination", sync=False):
            tensordict = self._compute_termination(tensordict)
        with ScopedTimer("command_step", sync=False):
            self.command_manager.step()
        with ScopedTimer("observation", sync=False):
            tensordict = self._compute_observation(tensordict)

        # Observation callbacks run after the normal termination pass.  A
        # callback can nevertheless produce a non-finite value, so finalize
        # the integrity termination after observations have been sanitized.
        self._finalize_mdp_nonfinite_transition(tensordict)

        tensordict.set("episode_id", self.episode_id.clone())
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui():
            if self.backend == "isaaclab":
                self.debug_draw.clear()
            elif self.backend == "mjlab":
                self.sim.viewer.clear()
            [callback() for callback in self._debug_draw_callbacks]
            if self.backend == "mjlab":
                self.sim.viewer.update()

        return tensordict

    def _apply_action(self, substep: int):
        [
            input_manager.apply_action(substep)
            for input_manager in self.input_managers.values()
        ]

    def _compute_reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.reward_groups:
            reward = torch.ones((self.num_envs, 1), device=self.device)
            physics_invalid = getattr(self, "_physics_invalid_mask", None)
            mdp_invalid = getattr(self, "_mdp_invalid_mask", None)
            invalid = None
            if physics_invalid is not None:
                invalid = physics_invalid
            if mdp_invalid is not None:
                invalid = mdp_invalid if invalid is None else invalid | mdp_invalid
            if invalid is None:
                invalid = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            reward.masked_fill_(invalid, 0.0)
            tensordict.set("reward", reward)
            return tensordict

        all_rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            self.stats[group, "return"].add_(reward)
            if reward_group.enabled:
                all_rewards.append(reward)
        rewards = torch.cat(all_rewards, dim=1)
        # A reward or later observation callback may have marked a row after
        # earlier reward terms were evaluated.  Exclude the *entire* synthetic
        # transition from PPO, not merely the term that first exposed it.
        physics_invalid = getattr(self, "_physics_invalid_mask", None)
        mdp_invalid = getattr(self, "_mdp_invalid_mask", None)
        invalid = physics_invalid
        if mdp_invalid is not None:
            invalid = mdp_invalid if invalid is None else invalid | mdp_invalid
        if invalid is not None:
            rewards.masked_fill_(invalid, 0.0)
        if self.mult_dt:
            rewards *= self.step_dt

        self.stats["episode_len"][:] = self.episode_length_buf.reshape(self.num_envs, 1)
        self.stats["success"][:] = (
            (self.episode_length_buf.reshape(self.num_envs, 1) >= self.max_episode_length * 0.9)
            .float()
        )
        tensordict.set("reward", rewards)
        return tensordict

    def _compute_termination(self, tensordict: TensorDictBase) -> TensorDictBase:
        truncated = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        discount = torch.ones((self.num_envs, 1), device=self.device)
        for key, func in self.termination_funcs.items():
            result = func.compute(terminated)
            if isinstance(result, tuple):
                term_value, term_discount = result
            else:
                term_value, term_discount = result, 1.0
            if not func.enabled:
                term_value.zero_()
            if func.is_timeout:
                truncated |= term_value
            else:
                terminated |= term_value
            discount *= term_discount
            self.stats["termination", key] = term_value.float()
        # This must be unconditional: the recovery protocol is only correct
        # when the corrupted transition ends here.  It intentionally does not
        # depend on an individual task YAML having remembered to add a term.
        physics_nonfinite = getattr(
            self,
            "_physics_invalid_mask",
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        )
        terminated |= physics_nonfinite
        self.stats["termination", "physics_nonfinite"] = physics_nonfinite.float()
        mdp_nonfinite = getattr(
            self,
            "_mdp_invalid_mask",
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        )
        terminated |= mdp_nonfinite
        self.stats["termination", "mdp_nonfinite"] = mdp_nonfinite.float()
        tensordict.set("truncated", truncated)
        tensordict.set("terminated", terminated)
        tensordict.set("done", terminated | truncated)
        tensordict.set("discount", discount)
        return tensordict

    def _isolate_mjlab_nonfinite_envs(self, substep: int) -> None:
        """Recover bad MuJoCo worlds before they can enter the MDP pipeline.

        ``mjwarp.step`` can very occasionally produce a NaN/Inf in a single
        high-contact world.  Waiting for a normal task termination is too late:
        command updates, rewards, observations and VecNorm would already have
        consumed the invalid tensors.  Reset just those worlds to a known
        finite default state now, mark their current transition terminal, and
        let TorchRL perform the regular command/DR/history reset afterwards.
        """
        from mjlab.utils.nan_guard import NanGuard

        invalid = NanGuard.detect_nans(self.sim.data)
        if not bool(invalid.any()):
            return

        env_ids = invalid.nonzero(as_tuple=False).squeeze(-1)
        self._physics_invalid_mask[env_ids] = True
        self._physics_invalid_any = True
        self._physics_nonfinite_total.add_(env_ids.numel())

        # reset_data clears MuJoCo's qacc/contact/sensor caches in addition to
        # qpos/qvel.  Merely nan_to_num-ing qpos/qvel leaves stale derived
        # buffers behind and can immediately re-infect observations.
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)

        asset = self.scene.articulations["robot"]
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.get_spawn_origins(env_ids)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        asset.write_joint_state_to_sim(
            asset.data.default_joint_pos[env_ids],
            asset.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.sim.forward()

        # A failed recovery means the model/default state itself is invalid;
        # continuing would silently corrupt a rollout, so fail loudly instead.
        still_invalid = NanGuard.detect_nans(self.sim.data)[env_ids]
        if bool(still_invalid.any()):
            bad_ids = env_ids[still_invalid].detach().cpu().tolist()
            raise RuntimeError(
                "mjlab non-finite-state recovery failed for environments "
                f"{bad_ids} at policy step {self.timestamp}, physics substep {substep}."
            )

        self.extra["env/physics_nonfinite_env_count"] = torch.tensor(
            float(env_ids.numel()), device=self.device
        )
        self.extra["env/physics_nonfinite_total"] = self._physics_nonfinite_total.clone()
        self.extra["env/physics_nonfinite_last_substep"] = torch.tensor(
            float(substep), device=self.device
        )

    def _compute_observation(self, tensordict: TensorDictBase) -> TensorDictBase:
        for group_name, group in self.observation_funcs.items():
            group.compute(tensordict, self.timestamp)
            value = tensordict.get(group_name, None)
            if isinstance(value, torch.Tensor) and (
                value.is_floating_point() or value.is_complex()
            ):
                # Keep the bad row out of both the rollout and subsequent
                # history updates.  The row mask is retained for terminal
                # handling below; no Python bool conversion is needed here.
                flat = value.reshape(self.num_envs, -1)
                bad = ~torch.isfinite(flat).all(dim=1)
                invalid_mask = getattr(self, "_mdp_invalid_mask", None)
                if invalid_mask is not None:
                    invalid_mask |= bad.unsqueeze(-1)
                tensordict.set(
                    group_name,
                    torch.where(torch.isfinite(value), value, torch.zeros_like(value)),
                )
        return tensordict

    def _finalize_mdp_nonfinite_transition(self, tensordict: TensorDictBase) -> None:
        """Make reward/observation non-finite values terminal and finite.

        Physics failures are already included in ``_physics_invalid_mask``;
        this method adds task-pipeline failures discovered while computing
        rewards or observations.  It runs after observation construction so
        the terminal flag covers every producer in the current transition.
        """
        physics = getattr(
            self,
            "_physics_invalid_mask",
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        )
        mdp = getattr(
            self,
            "_mdp_invalid_mask",
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        )
        if "reward" in tensordict.keys(True, True):
            reward = tensordict["reward"]
            if reward.is_floating_point() or reward.is_complex():
                invalid = physics | mdp
                tensordict.set(
                    "reward",
                    torch.where(
                        torch.isfinite(reward),
                        reward.masked_fill(invalid, 0.0),
                        torch.zeros_like(reward),
                    ),
                )
        terminated = tensordict.get("terminated", None)
        truncated = tensordict.get("truncated", None)
        if terminated is not None:
            terminated = terminated | physics | mdp
            tensordict.set("terminated", terminated)
            if truncated is not None:
                tensordict.set("done", terminated | truncated)
            else:
                tensordict.set("done", terminated)
        # Keep the task's existing terminal-discount contract.  GAE already
        # receives ``terminated`` above; changing ``discount`` here would
        # silently alter timeout/termination semantics across all tasks.
        self.stats["termination", "mdp_nonfinite"] = mdp.float()

    @property
    def ground(self):
        if not hasattr(self, "_ground"):
            self._ground = GroundQuery(
                self.terrain_type, self.device, self.ground_mesh
            )
        return self._ground

    @property
    def ground_mesh(self):
        """Warp ground mesh used for ray-based height queries.

        The concrete mesh construction is delegated to the backend-specific
        ``SceneAdapter`` implementations so that this environment stays
        agnostic to how ground geometry is represented per backend.
        """
        return self.scene.ground_mesh

    def get_ground_height_at(self, pos: torch.Tensor) -> torch.Tensor:
        return self.ground.height_at(pos)

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------
    def get_recorder(self, path, enabled: bool = True) -> VideoRecorder:
        """Return a backend-specific video recorder as a context manager.

        Usage:
            with env.get_recorder(\"video.mp4\", enabled) as rec:
                ...
                rec.add_frame()

        For backends with ``rgb_array`` rendering support, this returns a
        streaming recorder. Otherwise, or when ``enabled`` is False, this
        returns a no-op recorder so call sites don't need to branch.
        """
        if not enabled:
            return NullVideoRecorder()
        if self.backend == "isaaclab":
            return IsaacVideoRecorder(self, path, enabled=True)
        if self.backend == "mjlab":
            return RgbArrayVideoRecorder(self, path, enabled=True)
        # Other backends: return a no-op recorder by default.
        return NullVideoRecorder()

    def _set_seed(self, seed: int = -1):
        if self.backend == "isaaclab":
            seed = torch_utils.set_seed(seed)
            if hasattr(self, "_rgb_annotator"):
                try:
                    import omni.replicator.core as rep

                    rep.set_global_seed(seed)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to seed IsaacLab Replicator graph: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return seed
        elif self.backend == "mujoco":
            torch.manual_seed(seed)
            np.random.seed(seed)
        elif self.backend == "mjlab":
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def render(self, mode: str = "human"):
        self.sim.render()
        if mode == "human":
            return None
        if mode == "rgb_array":
            if hasattr(self, "_rgb_annotator"):
                rgb_data = self._rgb_annotator.get_data()
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
                    *rgb_data.shape
                )
                return rgb_data[:, :, :3]
            if self.backend == "mjlab":
                return self.sim.render_rgb_array()
            raise NotImplementedError(
                f"rgb_array mode not supported for backend '{self.backend}'. "
                "Only IsaacLab and mjlab backends support rgb_array rendering."
            )
        raise NotImplementedError(f"Render mode '{mode}' not supported.")

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["observation_spec"] = self.observation_spec
        state_dict["action_spec"] = self.action_spec
        state_dict["reward_spec"] = self.reward_spec
        return state_dict

    def diagnostics(self) -> dict:
        d = dict(self.extra)
        d.update(self.action_manager.diagnostics())
        return d

    def close(self, *, raise_if_closed: bool = True):
        if not self.is_closed:
            if self.backend == "isaaclab":
                del self.scene
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
            elif self.backend == "mjlab":
                if self.sim.has_gui():
                    self.sim.viewer.close()
                self.sim.close()
            super().close(raise_if_closed=raise_if_closed)
