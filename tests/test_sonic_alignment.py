from dataclasses import asdict

import pytest
import torch

from a3_supermimic.envs.mdp.reset import ResetCfg, reset_reference
from a3_supermimic.kinematics.rotations import (
    matrix_from_quat, quat_apply, quat_from_rotvec, quat_inv, quat_mul,
)
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR, check_contract_version
from a3_supermimic.world_model.heads import AnalyticHeads, ReferenceFrame


def make_heads(robot):
    _, layout, fk = robot
    return AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link",
                         torch.tensor([[-3., 3.]]*29), torch.zeros(29), layout=layout)


def test_reset_matches_sonic_world_euler_velocity_and_clamps(robot):
    model, layout, _ = robot
    torch.manual_seed(18)
    n = 4096
    q = torch.tensor(model.qpos0).expand(n, -1).clone()
    q[:, 3:7] = quat_from_rotvec(torch.tensor([.7, -.4, 1.2]))
    q[:, layout.policy_q] = .19
    v = torch.full((n, model.nv), .3, dtype=q.dtype)
    limits = q.new_tensor([[-.2, .2]]*29)
    ids = torch.tensor(layout.policy_q)
    out, vel = reset_reference(q, v, layout, ids, limits, ResetCfg())
    # Recover the WORLD Euler perturbation at a nontrivial reference pose.
    r = matrix_from_quat(quat_mul(out[:, 3:7], quat_inv(q[:, 3:7])))
    angles = torch.stack((torch.atan2(r[:, 2, 1], r[:, 2, 2]),
                          torch.asin(-r[:, 2, 0]), torch.atan2(r[:, 1, 0], r[:, 0, 0])), -1)
    delta_pose = torch.cat((out[:, :3]-q[:, :3], angles), -1)
    delta_velocity = torch.cat((vel[:, :3]-v[:, :3],
                               quat_apply(out[:, 3:7], vel[:, 3:6])-quat_apply(q[:, 3:7], v[:, 3:6])), -1)
    for actual, half in ((delta_pose, [.05, .05, .01, .1, .1, .2]),
                         (delta_velocity, [.5, .5, .2, .52, .52, .78])):
        normalized = actual / q.new_tensor(half)
        assert (normalized.abs() <= 1.+1.e-5).all()
        assert (normalized.min(0).values < -.98).all()
        assert (normalized.max(0).values > .98).all()
        assert (normalized.mean(0).abs() < .04).all()
    assert (out[:, ids] <= .2).all() and (out[:, ids] >= -.2).all()
    assert (out[:, ids] == .2).any()
    torch.testing.assert_close(vel[:, 6:], v[:, 6:])
    q[:, ids] = 10.
    out, vel = reset_reference(q, v, layout, ids, limits, ResetCfg(), noise=False)
    torch.testing.assert_close(out[:, ids], torch.full_like(out[:, ids], .2))
    torch.testing.assert_close(vel, v)


def test_paper_height_and_gravity_tilt_termination(robot):
    import math
    _, layout, _ = robot
    heads = make_heads(robot)
    errors = torch.tensor([.249, .25, .251, -.249, -.25, -.251], dtype=torch.float64)
    root = torch.zeros(6, 3, dtype=errors.dtype)
    quat = torch.zeros(6, 4, dtype=errors.dtype); quat[:, 0] = 1.
    s = torch.zeros(6, layout.state_dim, dtype=errors.dtype)
    s[:, 2] = -errors; s[:, 3:7] = quat
    # Low reference heights do not loosen the limit; both signs use strict > .25.
    ref = ReferenceFrame(None, None, None, None, root, quat)
    assert heads.terminated(s, ref).tolist() == [False, False, True, False, False, True]
    s[:, 2] = 0.
    threshold_angle = math.acos(.2)
    angles = errors.new_tensor([[0., 0., math.pi], [threshold_angle-1.e-6, 0., 0.],
                               [threshold_angle+1.e-6, 0., 0.], [0., 0., 0.],
                               [.6, 0., 0.], [-.6, 0., 0.]])
    s[:, 3:7] = quat_from_rotvec(angles)
    quat[4:] = quat_from_rotvec(errors.new_tensor([[0., .6, 0.], [.6, 0., 0.]]))
    # Yaw is ignored. Equal gravity-z for differently oriented tilted roots is valid.
    assert heads.terminated(s, ref).tolist() == [False, False, True, False, False, False]
    f = heads.termination_features(s[:, :3], s[:, 3:7], ref)
    flipped = ReferenceFrame(None, None, None, None, root, -quat)
    torch.testing.assert_close(f, heads.termination_features(s[:, :3], s[:, 3:7], flipped))
    theta = angles.clone().requires_grad_()
    features = heads.termination_features(s[:, :3], quat_from_rotvec(theta), ref)
    features.square().sum().backward()
    assert torch.isfinite(theta.grad).all()


def test_e5_reward_tracks_torso_while_termination_still_tracks_pelvis(robot):
    model, layout, _ = robot
    heads = make_heads(robot)
    assert TRACKING_ANCHOR == "torso_Link"
    s = torch.zeros(1, layout.state_dim, dtype=torch.float64)
    s[:, :layout.nq] = torch.tensor(model.qpos0)
    reference = heads.robot_bodies(s)
    ref = ReferenceFrame(reference["pos"], reference["quat"], reference["lin_vel"], reference["ang_vel"],
                         s[:, :3].clone(), s[:, 3:7].clone())
    # Articulate the waist: torso moves while the floating-base pelvis stays fixed.
    from a3_supermimic.robot import POLICY_JOINTS
    s[:, layout.policy_q[POLICY_JOINTS.index("waist_pitch_joint")]] = .3
    action = torch.zeros(1, 29)
    reward = heads.reward(s, action, action, ref)
    assert reward["motion_global_root_pos"].item() <= .01
    torch.testing.assert_close(reward["motion_global_root_ori"], torch.tensor([.01], dtype=s.dtype)*torch.exp(torch.tensor(-.3**2/.4**2, dtype=s.dtype)))
    assert reward["motion_body_ori"].item() < .02
    assert not heads.terminated(s, ref).any()
    cfg = asdict(heads.reward_cfg)
    assert [cfg[f"w_{key}"] for key in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel",
                                        "root_pos", "root_ori", "action_rate", "joint_limit")] == [1., 1., 1., 1., .5, .5, -.1, -2.]


def test_imagination_uses_current_reference_through_clip_end(robot, motion):
    from a3_supermimic.data.replay_buffer import StartBatch
    from a3_supermimic.envs.mdp.motion_library import MotionLibrary
    from a3_supermimic.imagination.compiled import ImagStep
    from a3_supermimic.imagination.rollout import imagine, OBS_NOISE_HALF
    from a3_supermimic.kinematics.model import ModelKinematics
    from a3_supermimic.policy.nets import Policy, PolicyCfg
    from a3_supermimic.world_model.world_model import WorldModel, WorldModelCfg
    model, layout, _ = robot
    fk = ModelKinematics(model)
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link",
                          torch.tensor([[-3., 3.]]*29), torch.zeros(29), layout=layout)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES), torch.zeros(29), allow_synthetic=True)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu", torch.zeros(34), torch.ones(34))
    actor = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), history=2, goal_dim=136))
    frame = torch.tensor([0, 61])
    q, v = lib.generalized(frame)
    s = torch.cat((q, v, torch.zeros(2, 32)), -1)
    xi = torch.zeros(2, 34); xi[:, -1] = .2
    start = StartBatch(s, torch.zeros(2, 29), xi, frame, s[None].expand(2, -1, -1),
                       torch.zeros(2, 2, 29))
    observed = []
    step = ImagStep(wm, torch.zeros(29))
    def capture(s, a, previous, context, ref):
        observed.append(ref.root_pos_w.clone())
        return step(s, a, previous, context, ref)
    imagine(wm, actor, lib, start, 4, future_frames=2, obs_noise=False, step_fn=capture)
    for t, actual in enumerate(observed):
        target = lib.root_q[torch.minimum(frame+t, lib.last_frame_of(frame)), :3]
        torch.testing.assert_close(actual, target)
    torch.testing.assert_close(OBS_NOISE_HALF[29:58], torch.full((29,), .5))


@pytest.mark.parametrize("version", [3, 4, 5, 6, 7])
def test_old_reward_termination_checkpoint_rejected(version):
    with pytest.raises(ValueError, match="start a new run"):
        check_contract_version({"contract": {"version": version}})
