import numpy as np
import torch
import mujoco

from a3_supermimic.robot import load_robot_model, RobotLayout, TRACKED_BODIES
from a3_supermimic.config import TrainConfig
from a3_supermimic.kinematics.model import ModelKinematics
from a3_supermimic.envs.mdp.motion_library import MotionLibrary
from a3_supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from a3_supermimic.policy.nets import Policy, PolicyCfg


def test_compact_fk_preserves_neutral_head_hands_and_inertia(robot, motion):
    full, old, _ = robot
    model = load_robot_model(compact=True)
    layout = RobotLayout.from_model(model)
    assert (model.nq, model.nv, layout.state_dim, layout.action_dim) == (36, 35, 71, 29)
    assert not model.ntendon
    from mimic_lite.assets.a3_ultra_four_ankle import rigid_ankle_spec
    nominal_rigid = rigid_ankle_spec().compile()
    np.testing.assert_allclose(model.body_mass.sum(), nominal_rigid.body_mass.sum(), rtol=0, atol=1.e-9)
    rigid_dofs = [int(nominal_rigid.joint(n).dofadr[0]) for n in layout.joint_names]
    # MjSpec.to_xml rounds numeric text; actuator construction below restores
    # the exact nominal armature before the runtime model is compiled.
    np.testing.assert_allclose(model.dof_armature[6:], nominal_rigid.dof_armature[rigid_dofs], rtol=2.e-6, atol=1.e-12)
    a, b = mujoco.MjData(full), mujoco.MjData(model)
    rng = np.random.default_rng(7)
    for _ in range(4):
        q = rng.uniform(-.25, .25, 29)
        a.qpos[list(old.policy_q)] = b.qpos[list(layout.policy_q)] = q
        mujoco.mj_forward(full, a)
        mujoco.mj_forward(model, b)
        for name in TRACKED_BODIES:
            np.testing.assert_allclose(a.xpos[full.body(name).id], b.xpos[model.body(name).id], atol=1.e-10)
    fk = ModelKinematics(model)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES), torch.zeros(29), allow_synthetic=True)
    q, v = lib.generalized(torch.tensor([0, 1]))
    assert q.shape == (2, 36) and v.shape == (2, 35)
    assert lib.goal_window(torch.tensor([0, 1])).shape == (2, 680)
    spec = StateSpec(StateSpecCfg(), layout, torch.zeros(33), torch.ones(33), 'cpu')
    assert spec.feat_dim == 71 and spec.delta_dim == 70
    state = torch.cat((q, v), -1)
    delta = torch.zeros(2, 70, requires_grad=True)
    pred = spec.integrate(state, delta, torch.zeros(2, 29))
    spec.normalized_sq_error(pred, state).sum().backward()
    assert pred.shape == (2, 71) and torch.isfinite(delta.grad).all()


def test_compact_default_and_fixed_asset_actuators():
    from a3_supermimic.envs.simulator import task_config
    from a3_supermimic.envs.compact_robot import make_compact_rigid_cfg
    cfg = TrainConfig()
    task = task_config(cfg, ['library.json'])
    assert task.robot.name == 'a3-supermimic-compact-rigid'
    assert cfg.env.anchor_obs and cfg.env.observation_anchor == 'torso_Link'
    assert set(task.randomization) == {'upstream_torso_com', 'upstream_reset'}
    asset, _ = make_compact_rigid_cfg('mjlab')
    assert len(asset.articulation.actuators) == 29
    assert all(a.delay_max_lag == 0 for a in asset.articulation.actuators)
    from mimic_lite.assets.a3_ultra_four_ankle import ANKLES, INERTIAS
    for actuator in asset.articulation.actuators:
        name, = actuator.target_names_expr
        if name in ANKLES:
            assert actuator.armature == INERTIAS[ANKLES.index(name)]


def test_upstream_actor_goal_gradients_and_export():
    torch.manual_seed(3)
    cfg = PolicyCfg(encoder_hidden=(64, 32), decoder_hidden=(32,), goal_dim=145, history=2)
    policy = Policy(cfg)
    goals = torch.randn(16, 145, requires_grad=True)
    obs = torch.randn(16, 186)
    action = policy(obs, goals)
    action.square().sum().backward()
    assert torch.isfinite(goals.grad).all() and goals.grad.abs().sum() > 0
    assert all(p.grad is not None for p in policy.encoders.parameters())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in policy.modules())
    assert not any('reconstruction' in key for key in policy.state_dict())
    with torch.no_grad():
        exported = torch.jit.trace(policy, (obs, goals.detach()))
        torch.testing.assert_close(exported(obs, goals), policy(obs, goals))
