"""Resolve the standalone recipe and robot without starting GPU simulation."""
from dataclasses import asdict
import json
import os
from pathlib import Path


def main():
    import argparse
    import tyro
    from omegaconf import OmegaConf
    from a3_supermimic.config import TrainConfig
    from a3_supermimic.robot import RobotLayout, load_robot_model, CONTRACT_VERSION, TRACKING_ANCHOR
    from a3_supermimic.envs.simulator import task_config
    from a3_supermimic.world_model.heads import ANCHOR_OBS_DIM

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume", type=Path)
    known, _ = pre.parse_known_args()
    defaults = TrainConfig()
    if known.resume:
        import torch
        from a3_supermimic.scripts.evaluate import restore_config
        from a3_supermimic.robot import check_contract_version
        saved = torch.load(known.resume, map_location="cpu", weights_only=False)
        check_contract_version(saved)
        defaults = restore_config(saved["config"])
    cfg = tyro.cli(TrainConfig, default=defaults)
    cfg.policy.goal_dim = 68 * cfg.env.future_frames + (ANCHOR_OBS_DIM if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    layout = RobotLayout.from_model(load_robot_model(compact=cfg.env.representation == "compact_rigid"))
    task = task_config(cfg, [Path(cfg.data.npz_dir) / "library.json"])
    world = int(os.environ.get("NPROC_PER_NODE", "4"))
    print(json.dumps({
        "config": asdict(cfg), "contract_version": CONTRACT_VERSION,
        "tracking_anchor": TRACKING_ANCHOR, "world_size": world,
        "effective_lr": cfg.policy_trainer.lr * (world / cfg.lr_world_size) ** .5,
        "global_envs": world * cfg.env.num_envs,
        "global_actor_batch": world * cfg.policy_trainer.batch_size,
        "global_wm_batch": world * cfg.wm_trainer.batch_size,
        "wm_only_optimizer_steps": cfg.schedule.wm_only_updates * cfg.wm_trainer.batches_per_update,
        "state_dim": layout.state_dim, "model_fingerprint": layout.fingerprint,
        "task": OmegaConf.to_container(task, resolve=True),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
