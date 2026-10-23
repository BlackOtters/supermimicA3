"""Validate the explicit A3 data contract, or produce an unmistakable smoke fixture."""
import argparse
from pathlib import Path
import json
import re
import numpy as np
from a3_supermimic.robot import load_robot_model, RobotLayout, POLICY_JOINTS, HEAD_JOINTS
from a3_supermimic.envs.mdp.motion_library import read_motion


def make_fixture(path, frames=256):
    if frames < 2:
        raise ValueError("Need at least two frames")
    if path.exists():
        raise FileExistsError(path)
    model = load_robot_model()
    from mimic_lite.assets.a3_ultra_four_ankle import make_four_ankle_cfg
    cfg, _ = make_four_ankle_cfg("mjlab")
    names = POLICY_JOINTS + HEAD_JOINTS
    default = np.zeros(31, dtype=np.float32)
    for pattern, value in cfg.init_state.joint_pos.items():
        for i, name in enumerate(names):
            if re.fullmatch(pattern, name):
                default[i] = value
    pos = np.zeros((frames, 3), dtype=np.float32)
    pos[:, 2] = cfg.init_state.pos[2]
    quat = np.zeros((frames, 4), dtype=np.float32); quat[:, 0] = 1.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=np.array(50.), joint_names=np.array(names),
                        joint_pos=np.broadcast_to(default, (frames, 31)), joint_vel=np.zeros((frames, 31), dtype=np.float32),
                        root_pos=pos, root_quat_wxyz=quat, root_lin_vel_w=np.zeros((frames, 3), dtype=np.float32),
                        root_ang_vel_b=np.zeros((frames, 3), dtype=np.float32), synthetic_fixture=np.array(True))
    read_motion(path, model, RobotLayout.from_model(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Synthetic standing clip for smoke checks ONLY")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--frames", type=int, default=256)
    validate = sub.add_parser("validate")
    validate.add_argument("input", type=Path)
    args = parser.parse_args()
    if args.command == "fixture":
        make_fixture(args.output, args.frames)
        print(f"Synthetic fixture created: {args.output}")
        return
    if args.input.is_dir() and (args.input/"library.json").exists():
        from a3_supermimic.data.convert_corpus import read_packed_manifest
        m = read_packed_manifest(args.input/"library.json", verify=True)
        print(json.dumps({"clips": m["num_clips"], "frames": m["num_frames"], "hours": m["total_hours"]}))
        return
    model = load_robot_model(); layout = RobotLayout.from_model(model)
    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.npz"))
    if not files:
        parser.error("No NPZ clips found")
    summary = []
    for path in files:
        q, _, synthetic = read_motion(path, model, layout)
        summary.append({"path": str(path), "frames": len(q), "seconds": len(q)/50., "synthetic": synthetic})
    print(json.dumps({"clips": summary, "total_seconds": sum(c["seconds"] for c in summary)}, indent=2))


if __name__ == "__main__":
    main()
