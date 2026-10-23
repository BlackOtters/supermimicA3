"""Small real-physics collect/fit/BPTT/checkpoint check, with synthetic data only."""
import argparse
from pathlib import Path
import torch
from a3_supermimic.scripts.data import make_fixture
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile", action="store_true", help="Also exercise the default Inductor training path")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh smoke directory: {args.output}")
    args.output.mkdir(parents=True)
    fixture = args.output/"motions"/"standing.npz"
    make_fixture(fixture, 128)
    from a3_supermimic.config import TrainConfig
    from a3_supermimic.training.loop import Trainer
    from a3_supermimic.scripts.evaluate import restore_config
    cfg = TrainConfig()
    cfg.env.representation = 'full'
    cfg.env.randomization_profile = 'a3'
    cfg.env.evaluation_physics = 'mixed'
    cfg.device = args.device
    cfg.data.npz_dir = fixture.parent
    cfg.data.allow_synthetic = True
    cfg.log_root = args.output
    cfg.run_name = "train"
    cfg.env.num_envs = 8
    cfg.env.replay_capacity = 32
    cfg.env.episode_length_s = .24
    cfg.env.reset_noise = True
    cfg.world_model.hidden = (32, 32)
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (32,)
    cfg.critic.hidden = (16,)
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = "default" if args.compile else None
    cfg.wm_trainer.window = 2; cfg.wm_trainer.batch_size = 4; cfg.wm_trainer.batches_per_update = 1
    cfg.policy_trainer.horizon = 3; cfg.policy_trainer.batch_size = 4
    cfg.policy_trainer.critic_epochs = 1; cfg.policy_trainer.critic_minibatches = 2
    cfg.schedule.total_iterations = 20; cfg.schedule.warmup_iterations = 8
    cfg.schedule.wm_only_updates = 2
    cfg.schedule.log_interval = 2; cfg.schedule.checkpoint_interval = 10
    trainer = Trainer(cfg)
    sim = trainer.sim
    env, asset = sim.env, sim.env.scene.articulations["robot"]
    # Exercise BOTH branches, rather than hoping eight random worlds include both.
    dr = env._four_ankle_state
    dr.rigid_probability = 1.; sim.reset(torch.arange(4, device=sim.device))
    dr.rigid_probability = 0.; sim.reset(torch.arange(4, 8, device=sim.device))
    dr.rigid_probability = .25
    assert not dr.is_elastic[:4].any() and dr.is_elastic[4:].all()
    state = sim.state()
    limits = sim.command.reset_joint_limits[sim.command.policy_joint_ids]
    assert ((state[:, sim.layout.policy_q] >= limits[:, 0]-1.e-6)
            & (state[:, sim.layout.policy_q] <= limits[:, 1]+1.e-6)).all()
    assert sim.heads.termination_cfg.height_threshold == .25
    assert sim.heads.termination_cfg.tilt_threshold == .8
    assert "perturb_root_vel" not in sim.env.cfg.randomization
    assert sim.heads.anchor_index == TRACKED_BODIES.index(TRACKING_ANCHOR)
    clean = sim.heads.observation(state, sim.previous_clean, sim.xi)
    q = asset.data.joint_pos[:, sim.command.policy_joint_ids].clone()
    dq = asset.data.joint_vel[:, sim.command.policy_joint_ids].clone()
    enc_q, enc_v = dr.encoder_state()
    q[:, dr.roll_cols] = enc_q; dq[:, dr.roll_cols] = enc_v
    q -= asset.data.default_joint_pos[:, sim.command.policy_joint_ids]
    expected = torch.cat((q, dq, asset.data.root_com_ang_vel_b, asset.data.projected_gravity_b, sim.previous_clean), -1)
    torch.testing.assert_close(clean, expected, atol=2.e-5, rtol=2.e-5)
    bodies = sim.heads.robot_bodies(state)
    body_ids = torch.tensor([asset.body_names.index(n) for n in TRACKED_BODIES], device=sim.device)
    torch.testing.assert_close(bodies["pos"], asset.data.body_link_pos_w[:, body_ids]-sim.command.origins[:, None], atol=2.e-5, rtol=2.e-5)
    print("A3_MIXED_ENCODER_AND_FK_ALIGNMENT_OK", flush=True)
    trainer.run()
    assert trainer.num_updates > 0
    checkpoint_path = args.output/"train"/"ckpt_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    resumed_cfg = restore_config(checkpoint["config"])
    resumed_cfg.resume = checkpoint_path
    resumed_cfg.run_name = "resume"
    resumed_cfg.schedule.total_iterations = 32
    restored = Trainer(resumed_cfg)
    for label, module in (("wm", restored.wm), ("policy", restored.policy), ("critic", restored.critic)):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, checkpoint[label][key], atol=0, rtol=0)
    assert restored.wm_trainer.opt.state_dict()["state"].keys() == checkpoint["wm_opt"]["state"].keys()
    restored.run()
    assert restored.num_updates > checkpoint["num_updates"]
    print(f"A3_SUPERMIMIC_SMOKE_AND_RESUME_OK updates={restored.num_updates}", flush=True)


if __name__ == "__main__":
    main()
