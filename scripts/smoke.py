"""Bounded compact-rigid train/resume/export check; generated data is synthetic."""
import argparse
from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from a3_supermimic.config import TrainConfig
from a3_supermimic.scripts.data import make_fixture
from a3_supermimic.scripts.evaluate import restore_config, evaluate
from a3_supermimic.training.loop import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--retargeted", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new smoke output directory")
    args.output.mkdir(parents=True)
    fixture = args.output / "data/standing.npz"
    make_fixture(fixture, frames=128)
    cfg = TrainConfig()
    if args.retargeted:
        cfg.env.anchor_position = True
        cfg.env.collision_profile = 'retargeted'
        cfg.env.waist_soft_limit_factor = .9
        cfg.env.solver_iterations = 10
        cfg.world_model.termination_mode = 'learned'
        cfg.tf32 = True
    cfg.run_name = "train"
    cfg.log_root = args.output
    cfg.seed = 42
    cfg.data.npz_dir = fixture.parent
    cfg.data.allow_synthetic = True
    cfg.env.num_envs = 8
    cfg.env.replay_capacity = 64
    cfg.env.episode_length_s = .24
    cfg.world_model.hidden = (32, 32)
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (32,)
    cfg.critic.hidden = (16,)
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = 8
    cfg.wm_trainer.batches_per_update = 1
    cfg.policy_trainer.horizon = 4
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 2
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = "default" if args.compile else None
    cfg.schedule.total_iterations = 50
    cfg.schedule.wm_only_updates = 5  # Save mid-warmup, then cross the boundary after resume.
    if args.retargeted:
        cfg.schedule.wm_only_updates = 0
    cfg.schedule.log_interval = 10
    cfg.schedule.checkpoint_interval = 50
    trainer = Trainer(cfg)
    assert trainer.sim.layout.state_dim == 71 and trainer.sim.xi.shape[-1] == 33
    assert trainer.sim.env.sim.mj_model.ntendon == 0
    assert cfg.policy.goal_dim == (689 if args.retargeted else 686)
    actor_before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    critic_before = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
    trainer.run()
    if args.retargeted:
        assert trainer.num_policy_updates == trainer.num_updates > 0
        assert trainer.wm.term_head is not None
    else:
        assert 0 < trainer.num_updates < cfg.schedule.wm_only_updates
        assert trainer.num_policy_updates == 0
        assert not trainer.policy_trainer.opt.state and not trainer.policy_trainer.critic_opt.state
        for module, before in ((trainer.policy, actor_before), (trainer.critic, critic_before)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    checkpoint = args.output / "train/ckpt_latest.pt"
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    resumed_cfg = restore_config(saved["config"])
    resumed_cfg.run_name = "resume"
    resumed_cfg.resume = checkpoint
    resumed_cfg.schedule.total_iterations = 100
    resumed = Trainer(resumed_cfg)
    for key, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, saved["policy"][key], rtol=0, atol=0)
    resumed.run()
    assert resumed.num_updates > saved["num_updates"] > 0
    assert resumed.num_policy_updates == resumed.num_updates - cfg.schedule.wm_only_updates > 0
    checkpoint = args.output / "resume/ckpt_latest.pt"
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    report = evaluate(restore_config(saved["config"]), saved, [fixture], "rigid",
                      export=args.output / "actor.pt")
    result = {"synthetic_fixture": True, "iterations": resumed.iteration,
              "updates": resumed.num_updates, "contract": saved["contract"]["version"],
              "policy_updates": resumed.num_policy_updates, "goal_dim": saved["config"]["policy"]["goal_dim"],
              "model_fingerprint": saved["contract"]["robot"]["fingerprint"],
              "compiled": args.compile, "evaluation": report,
              "module_files": {name: str(module.__file__) for name, module in sys.modules.items()
                               if name.startswith(("a3_supermimic", "active_adaptation", "mimic_lite"))
                               and getattr(module, "__file__", None)}}
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "module_files"}, indent=2))


if __name__ == "__main__":
    main()
