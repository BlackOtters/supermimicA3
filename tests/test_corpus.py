import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from a3_supermimic.data.convert_corpus import (
    JOINTS, ROOT_NAMES, convert_qpos, convert_corpus, read_packed_manifest,
)
from a3_supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
from a3_supermimic.robot import TRACKED_BODIES


def source_clip(n=101):
    t = np.arange(n)*.02
    initial = Rotation.from_euler("xyz", [.4, -.2, .3])
    rotation = Rotation.from_rotvec(t[:, None]*np.array([0., 0., 1.7]))*initial
    q = rotation.as_quat(scalar_first=True)
    q[1::2] *= -1  # Arbitrary equivalent quaternion signs must not create speed spikes.
    names = tuple(reversed(JOINTS))
    qpos = np.concatenate((np.array([2., -3., 1.])+t[:, None]*np.array([.3, -.1, .05]),
                           q, t[:, None]*np.linspace(-.3, .3, 31)), -1)
    return qpos, names, rotation


@pytest.mark.parametrize("n", [2, 101])
def test_velocity_frames_signs_endpoints_and_named_mapping(n):
    qpos, names, rotation = source_clip(n)
    a, stats = convert_qpos(qpos, names, 50)
    np.testing.assert_allclose(a["root_v"][:, :3], np.broadcast_to([.3, -.1, .05], (n, 3)), atol=1.e-6)
    expected = rotation.inv().apply(np.broadcast_to([0., 0., 1.7], (n, 3)).copy())
    np.testing.assert_allclose(a["root_v"][:, 3:], expected, atol=1.e-6)
    np.testing.assert_allclose(a["joint_v"], np.broadcast_to(np.linspace(.3, -.3, 31), (n, 31)), atol=2.e-4)
    np.testing.assert_allclose(a["root_q"][0, :3], [0, 0, 1.])
    assert stats["origin_xy"] == [2., -3.]


def make_corpus(folder):
    q, names, _ = source_clip(51)
    group = folder/"amass"
    (group/"motions").mkdir(parents=True)
    np.savez(group/"motions"/"move.npz", qpos=q, joint_names=names, fps=50)
    np.savez(group/"motions"/"move2.npz", qpos=q, joint_names=names, fps=50)
    (group/"manifest.json").write_text(json.dumps({"format_version": 2, "qpos_dim": 38,
        "qpos_names": ROOT_NAMES+names, "timestep": .02, "motions_subdir": "motions",
        "num_motions": 2, "num_frames": 2*len(q)}))
    return q, names


@pytest.mark.parametrize('compact', [False, True])
def test_packed_library_matches_named_npz_and_checks_corruption(tmp_path, robot, compact):
    q, names = make_corpus(tmp_path/"source")
    output = tmp_path/"packed"
    convert_corpus(tmp_path/"source", output, workers=2)
    m = read_packed_manifest(output/"library.json", verify=True)
    assert m["num_clips"] == 2 and m["num_frames"] == 102
    a, _ = convert_qpos(q, names, 50)
    plain = tmp_path/"plain.npz"
    np.savez(plain, fps=50, joint_names=JOINTS, root_pos=a["root_q"][:, :3], root_quat_wxyz=a["root_q"][:, 3:],
             root_lin_vel_w=a["root_v"][:, :3], root_ang_vel_b=a["root_v"][:, 3:],
             joint_pos=a["joint_q"].astype(np.float32), joint_vel=a["joint_v"].astype(np.float32))
    model, layout, fk = robot
    if compact:
        from a3_supermimic.robot import load_robot_model, RobotLayout
        from a3_supermimic.kinematics.model import ModelKinematics
        model = load_robot_model(compact=True)
        layout, fk = RobotLayout.from_model(model), ModelKinematics(model, dtype=torch.float64)
    args = (model, layout, fk, fk.body_indices(TRACKED_BODIES), torch.zeros(29))
    lib = MotionLibrary(discover_npz(output), *args)
    original = MotionLibrary([plain], *args)
    frames = torch.tensor([0, 1, 25, 50])
    for got, want in zip(lib.generalized(frames), original.generalized(frames)):
        torch.testing.assert_close(got, want)
    torch.testing.assert_close(lib.goal_window(frames), original.goal_window(frames))
    assert lib.bin_starts.tolist() == [0, 50, 51, 101]
    assert lib.last_frame_of(torch.tensor([50, 51, 101])).tolist() == [50, 101, 101]
    # Array corruption must not silently retain the old checkpoint fingerprint.
    data = np.load(output/"joint_q.npy", mmap_mode="r+")
    data[0, 0] = 1.
    data.flush()
    with pytest.raises(ValueError, match="Checksum"):
        MotionLibrary(discover_npz(output), *args)


def test_invalid_source_is_rejected_without_publishing_library(tmp_path):
    q, names, _ = source_clip()
    with pytest.raises(ValueError, match="50 Hz"):
        convert_qpos(q, names, 30)
    with pytest.raises(ValueError, match="exactly once"):
        convert_qpos(q, names[:-1]+names[:1], 50)
    q[0, 3:7] *= 2
    with pytest.raises(ValueError, match="Non-unit"):
        convert_qpos(q, names, 50)
    make_corpus(tmp_path/"source")
    manifest = tmp_path/"source"/"amass"/"manifest.json"
    m = json.loads(manifest.read_text()); m["qpos_names"][3:7] = ["qx", "qy", "qz", "qw"]
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Root coordinate"):
        convert_corpus(tmp_path/"source", tmp_path/"bad")
    assert not (tmp_path/"bad"/"library.json").exists()
