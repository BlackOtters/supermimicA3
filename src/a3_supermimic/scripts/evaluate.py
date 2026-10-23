"""Deterministic, whole-clip evaluation for A3 (separate rigid/elastic reports)."""
import argparse
from dataclasses import is_dataclass
from pathlib import Path
import json
import torch


def restore_config(values):
    from a3_supermimic.config import TrainConfig
    cfg = TrainConfig()
    def fill(obj, source):
        for key, value in source.items():
            if not hasattr(obj, key):
                raise ValueError(f"Unknown saved config field {key}")
            current = getattr(obj, key)
            if is_dataclass(current):
                fill(current, value)
            else:
                setattr(obj, key, value)
    fill(cfg, values)
    return cfg


@torch.no_grad()
def evaluate(cfg, checkpoint, files, physics, export=None):
    from a3_supermimic.robot import check_contract_version
    check_contract_version(checkpoint)
    from a3_supermimic.envs.simulator import Simulator
    from a3_supermimic.policy.nets import Policy
    cfg.env.num_envs = 1
    cfg.env.domain_randomization = False
    cfg.env.evaluation_physics = physics
    cfg.env.action_noise_std = 0.
    cfg.env.obs_noise = cfg.env.reset_noise = False
    cfg.env.sampling_mode = "start"
    cfg.env.terrain = "plane"
    cfg.env.episode_length_s = 1.e6
    # Avoid allocating million-step sampler episode logs: the longest clip determines it.
    import numpy as np
    lengths = []
    for file in files:
        if Path(file).name == "library.json":
            metadata = json.loads(Path(file).read_text())
            lengths.extend(c["frames"] for c in metadata["clips"])
            continue
        with np.load(file, allow_pickle=False) as z:
            lengths.append(len(z["root_pos"]))
    cfg.env.episode_length_s = max(lengths)/50. + .1
    sim = Simulator(cfg, files)
    try:
        if tuple(checkpoint["contract"]["robot"]["joint_names"]) != sim.layout.joint_names:
            raise ValueError("Checkpoint robot joint topology does not match this A3 asset")
        if checkpoint["contract"]["robot"]["fingerprint"] != sim.layout.fingerprint:
            raise ValueError("Checkpoint A3 physical asset changed; evaluation would use different mechanics")
        policy = Policy(cfg.policy).to(cfg.device).eval()
        policy.load_state_dict(checkpoint["policy"])
        if export is not None:
            if export.exists():
                raise FileExistsError(export)
            sample = sim.observation()
            traced = torch.jit.trace(policy, (sample["policy"], sample["goal"]))
            export.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(export))
            export.with_suffix(".json").write_text(json.dumps(checkpoint["contract"], indent=2)+"\n")
        rows = []
        for clip, name in enumerate(sim.library.names):
            obs = sim.reset(frames=sim.library.starts[clip:clip+1])
            total, failed, steps = 0., False, 0
            for _ in range(int(sim.library.lengths[clip])):
                action = policy(obs["policy"], obs["goal"])
                obs, reward, terminated, timeout, extra = sim.step(action, auto_reset=False)
                total += float(reward[0]); steps += 1
                if bool(terminated[0]):
                    failed = True
                    break
                if bool(timeout[0]):
                    break
            rows.append({"clip": name, "normalized_return": total/(.1*int(sim.library.lengths[clip])),
                         "executed_step_normalized_return": total/(.1*steps), "failure": failed, "steps": steps,
                         "total_clip_frames": int(sim.library.lengths[clip])})
        return {"physics": physics, "domain_randomization": False, "observation_noise": False,
                "terrain": "plane", "sliding_friction": 1., "synthetic_fixture": cfg.data.allow_synthetic,
                "mean_normalized_return": sum(r["normalized_return"] for r in rows)/len(rows),
                "failures": sum(r["failure"] for r in rows), "clips": rows}
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics", choices=("rigid", "elastic", "both"), default="rigid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--export-policy", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    from a3_supermimic.robot import check_contract_version
    check_contract_version(checkpoint)
    if checkpoint['config']['env'].get('representation') == 'compact_rigid' and args.physics != 'rigid':
        parser.error('The compact rigid checkpoint has no elastic joints; use --physics rigid')
    if checkpoint.get("synthetic_fixture") and not args.allow_synthetic:
        parser.error("This is a synthetic smoke checkpoint; pass --allow-synthetic only for validation")
    from a3_supermimic.envs.mdp.motion_library import discover_npz
    files = discover_npz(args.npz_dir)
    if not files:
        parser.error("No evaluation clips")
    results = []
    for index, physics in enumerate(("rigid", "elastic") if args.physics == "both" else (args.physics,)):
        cfg = restore_config(checkpoint["config"])
        cfg.device = args.device; cfg.data.allow_synthetic = args.allow_synthetic
        torch.manual_seed(cfg.seed)
        results.append(evaluate(cfg, checkpoint, files, physics, args.export_policy if index == 0 else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "results": results}, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
