from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from a3_supermimic.config import TrainConfig
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR
from a3_supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, anchor_relative_pose


def test_retargeted_contacts_preserve_ground_and_ultra_mechanics(robot):
    from a3_supermimic.envs.compact_robot import make_compact_rigid_cfg, make_retargeted_rigid_cfg
    from a3_supermimic.envs.simulator import task_config
    old, _ = make_compact_rigid_cfg('mjlab')
    new, _ = make_retargeted_rigid_cfg('mjlab')
    a, b = old.build().compile(), new.build().compile()
    for field in ('body_mass', 'body_inertia', 'jnt_range', 'dof_armature', 'actuator_gainprm'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert b.npair == 0
    active = (b.geom_contype | b.geom_conaffinity) != 0
    assert active.any() and np.all(b.geom_contype[active] == 1)
    assert np.all(b.geom_conaffinity[active] == 0)  # Robot pairs cannot pass the mask test.
    feet = np.array(['foot' in b.geom(i).name and 'collision' in b.geom(i).name for i in range(b.ngeom)])
    assert feet.sum() == 26
    assert np.all(b.geom_condim[feet] == 3) and np.all(b.geom_priority[feet] == 1)
    assert np.all(b.geom_condim[active & ~feet] == 1)
    cfg = TrainConfig()
    cfg.env.collision_profile = 'retargeted'
    cfg.env.solver_iterations = 10
    cfg.env.waist_soft_limit_factor = .9
    task = task_config(cfg, ['library.json'])
    assert task.robot.name == 'a3-supermimic-retargeted-rigid'
    assert task.sim.mjlab_solver_iterations == 10
    assert task.command.waist_soft_limit_factor == .9
    assert task.input.action.action_scaling.left_ankle_pitch_joint == .4925


def test_pose_goal_simulation_imagination_and_learned_termination(robot, motion):
    from a3_supermimic.data.replay_buffer import StartBatch
    from a3_supermimic.envs.mdp.motion_library import MotionLibrary
    from a3_supermimic.envs.simulator import Simulator
    from a3_supermimic.imagination.rollout import imagine
    from a3_supermimic.policy.nets import Policy, PolicyCfg
    from a3_supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from a3_supermimic.world_model.wm_trainer import reference_at
    model, layout, _ = robot
    from a3_supermimic.kinematics.model import ModelKinematics
    fk = ModelKinematics(model)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES), torch.zeros(29), allow_synthetic=True)
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, 'pelvis_link',
                          torch.tensor([[-3., 3.]]*29), torch.zeros(29), layout=layout)
    frames = torch.tensor([0, 61])
    q, v = lib.generalized(frames)
    s = torch.cat((q, v, torch.zeros(2, 32)), -1)
    sim = Simulator.__new__(Simulator)
    sim.cfg = TrainConfig()
    sim.cfg.env.anchor_position = True
    sim.cfg.env.obs_noise = False
    sim.cfg.env.future_frames = 2
    sim.library, sim.heads, sim.device = lib, heads, 'cpu'
    sim.command = SimpleNamespace(frame=frames)
    sim.history = torch.zeros(2, 2, 93)
    sim.state = lambda: s
    initial_goal = sim.observation()['goal']
    s[:, 0] += .4
    goal = sim.observation()['goal']
    assert goal.shape == (2, 145)
    torch.testing.assert_close(goal[:, :-9], initial_goal[:, :-9])
    torch.testing.assert_close(goal[:, -6:], initial_goal[:, -6:])
    assert not torch.allclose(goal[:, -9:-6], initial_goal[:, -9:-6])
    xi = torch.zeros(2, 34)
    start = StartBatch(s, torch.zeros(2, 29), xi, frames, s[None].expand(2,-1,-1), torch.zeros(2,2,29))
    wm = WorldModel(WorldModelCfg(hidden=(16,), termination_mode='learned', term_hidden=(8,)),
                    heads, 'cpu', torch.zeros(34), torch.ones(34))
    policy = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), history=2, goal_dim=145))
    ro = imagine(wm, policy, lib, start, 3, future_frames=2, obs_noise=False, anchor_obs=True, anchor_position=True)
    torch.testing.assert_close(ro.goals[0], goal)
    ref = reference_at(lib, frames)
    torch.testing.assert_close(ro.dhat[1], wm.termination_logits(ro.states[1], ref).sigmoid())
    assert all(p.grad is None for p in wm.term_head.parameters())
    loss, metrics = wm.window_loss(torch.stack((s,s)), torch.zeros(1,2,31), xi,
                                  torch.tensor([[0.,1.]]), [ref])
    loss.backward()
    assert 'term_bce' in metrics and 'term_acc' in metrics
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in wm.term_head.parameters())


def test_learned_termination_features_keep_signed_tilt(robot):
    from a3_supermimic.kinematics.rotations import quat_from_rotvec, projected_gravity
    _, layout, fk = robot
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, 'pelvis_link',
                          torch.tensor([[-3.,3.]]*29), torch.zeros(29), layout=layout)
    q = quat_from_rotvec(torch.tensor([[.5, 0., 0.]]))
    identity = torch.tensor([[1.,0.,0.,0.]])
    p = torch.tensor([[0.,0.,1.]])
    ref = ReferenceFrame(None,None,None,None,p,identity)
    features = heads.termination_features(p, q, ref)
    assert features[0,1] < 0
    torch.testing.assert_close(features[:,1], (projected_gravity(identity)[:,2]-projected_gravity(q)[:,2])/.8)


def test_replay_does_not_sample_overwritten_history():
    from a3_supermimic.data.replay_buffer import ReplayBuffer
    b = ReplayBuffer(12,1,1,1,'cpu',action_dim=29)
    b.register_context(torch.tensor([0]),torch.zeros(1,1))
    for t in range(20):
        b.append(torch.tensor([[float(t)]]),torch.zeros(1,29),torch.tensor([[float(t+1)]]),
                 torch.tensor([t]),torch.tensor([0]),torch.tensor([False]), previous_clean_action=torch.zeros(1,29))
    batch = b.sample_starts(128,3)
    torch.testing.assert_close(batch.hist_states[1:]-batch.hist_states[:-1],torch.ones(2,128,1))
    assert batch.state.min() >= 10
