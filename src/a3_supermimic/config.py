"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from a3_supermimic.imagination.policy_trainer import PolicyTrainerCfg
from a3_supermimic.policy.nets import CriticCfg, PolicyCfg
from a3_supermimic.world_model.wm_trainer import WMTrainerCfg
from a3_supermimic.world_model.world_model import WorldModelCfg
from a3_supermimic.world_model.heads import TerminationCfg
from a3_supermimic.envs.mdp.reset import ResetCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/a3")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    allow_synthetic: bool = False  # only for a smoke test, never a training result


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    reset: ResetCfg = field(default_factory=ResetCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    domain_randomization: bool = True
    randomization_profile: str = "upstream_rigid"  # a3 | upstream_rigid
    representation: str = "compact_rigid"  # compact_rigid: 71-state/33-context; full: legacy mixed topology
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    anchor_obs: bool = True  # 6D reference-relative orientation, no position feedback
    observation_anchor: str = "torso_Link"  # e5 uses the torso for both goal feedback and rewards
    terrain: str = "plane"  # upstream training terrain; inherit is an explicit A3 rough-terrain opt-in
    evaluation_physics: str = "rigid"  # legacy name: topology for training and evaluation; DR is independent


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    wm_only_updates: int = 500  # complete WM update rounds before the first actor/critic update
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class SwanLabCfg:
    enabled: bool = False
    project: str = "a3-supermimic"
    workspace: str | None = None
    mode: str = "online"


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    swanlab: SwanLabCfg = field(default_factory=SwanLabCfg)
    resume: Path | None = None
    lr_world_size: int = 1  # records applied sqrt(world_size) scaling in saved configurations
