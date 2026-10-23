import numpy as np
import torch
import mujoco
import pytest
from a3_supermimic.robot import TRACKED_BODIES, POLICY_JOINTS, ANKLES, TRACKING_ANCHOR
from a3_supermimic.world_model.heads import AnalyticHeads
from a3_supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from a3_supermimic.kinematics.rotations import quat_from_rotvec


def test_fk_all_bodies_and_independent_rotors(robot):
    model, layout, fk = robot
    rng = np.random.default_rng(42)
    q = np.broadcast_to(model.qpos0, (3, model.nq)).copy()
    q[:, :3] = rng.normal(size=(3, 3))
    q[:, 3:7] = quat_from_rotvec(torch.tensor(rng.normal(size=(3, 3))*.3)).numpy()
    q[:, 7:] += rng.normal(size=q[:, 7:].shape)*.1
    v = rng.normal(size=(3, model.nv))*.2
    pred = fk(torch.tensor(q), torch.tensor(v))
    data = mujoco.MjData(model)
    for b in range(3):
        data.qpos[:], data.qvel[:] = q[b], v[b]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(pred["pos"][b], data.xpos[1:], atol=1.e-9)
        dots = np.abs((pred["quat"][b].numpy()*data.xquat[1:]).sum(-1))
        np.testing.assert_allclose(dots, 1., atol=1.e-9)
        for i in range(1, model.nbody):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, i, vel, 0)
            # mj_objectVelocity's body velocity is at its COM, shift to link origin.
            expected = vel[3:] + np.cross(vel[:3], data.xpos[i]-data.xipos[i])
            np.testing.assert_allclose(pred["lin_vel"][b, i-1], expected, atol=1.e-9)
            np.testing.assert_allclose(pred["ang_vel"][b, i-1], vel[:3], atol=1.e-9)
    qt = torch.tensor(q, requires_grad=True)
    fk(qt)["pos"].square().sum().backward()
    assert torch.isfinite(qt.grad).all()


def test_state_has_rotors_head_fingers_and_deterministic_delay_memory(robot):
    model, layout, _ = robot
    spec = StateSpec(StateSpecCfg(), layout, torch.zeros(34), torch.ones(34), "cpu")
    s = torch.zeros(2, layout.state_dim)
    s[:, :layout.nq] = torch.tensor(model.qpos0)
    s[:, layout.nq:layout.physical_dim] = .1
    s[:, -1] = 1.
    action = torch.randn(2, 31, requires_grad=True)
    out = spec.integrate(s, torch.zeros(2, spec.delta_dim), action)
    assert out.shape == s.shape
    assert spec.canonical(s).shape[-1] == spec.feat_dim
    torch.testing.assert_close(out[:, 7:layout.nq], s[:, 7:layout.nq]+.002)
    torch.testing.assert_close(out[:, layout.physical_dim:-1], action)
    assert (out[:, -1] == 0).all()
    assert len(layout.encoder_q) == 4 and len(layout.joint_names) > 35
    out.sum().backward(); assert torch.isfinite(action.grad).all()
    torch.testing.assert_close(spec.normalized_sq_error(out, out), torch.zeros(2))


def test_encoder_observation_mixed_rigid_and_elastic(robot):
    model, layout, fk = robot
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link",
                          torch.tensor([[-10., 10.]]*29), torch.zeros(29), layout=layout)
    s = torch.zeros(2, layout.state_dim, dtype=torch.float64)
    s[:, 3] = 1.
    s[:, layout.policy_q] = .2
    s[:, layout.encoder_q] = .8
    s[:, layout.nq+torch.tensor(layout.policy_v)] = 1.
    s[:, layout.nq+torch.tensor(layout.encoder_v)] = 3.
    xi = torch.zeros(2, 34, dtype=torch.float64); xi[:, :29] = .01; xi[1, 29] = 1.
    obs = heads.observation(s, torch.zeros(2, 29), xi)
    cols = torch.tensor([POLICY_JOINTS.index(n) for n in ANKLES])
    torch.testing.assert_close(obs[0, cols], torch.full((4,), .19, dtype=obs.dtype))
    torch.testing.assert_close(obs[1, cols], torch.full((4,), .79, dtype=obs.dtype))
    torch.testing.assert_close(obs[0, 29+cols], torch.ones(4, dtype=obs.dtype))
    torch.testing.assert_close(obs[1, 29+cols], torch.full((4,), 3., dtype=obs.dtype))


def test_motion_names_reordered_and_bad_contract_rejected(robot, motion, tmp_path):
    from a3_supermimic.envs.mdp.motion_library import read_motion
    model, layout, _ = robot
    q, v, synthetic = read_motion(motion, model, layout)
    assert synthetic
    with np.load(motion) as z:
        payload = {k: z[k] for k in z.files}
    for key in ("joint_names",):
        payload[key] = payload[key][::-1]
    for key in ("joint_pos", "joint_vel"):
        payload[key] = payload[key][:, ::-1]
    reordered = tmp_path/"reordered.npz"; np.savez(reordered, **payload)
    q2, v2, _ = read_motion(reordered, model, layout)
    np.testing.assert_array_equal(q, q2); np.testing.assert_array_equal(v, v2)
    payload["fps"] = 60.
    bad = tmp_path/"bad.npz"; np.savez(bad, **payload)
    with pytest.raises(ValueError, match="50 Hz"):
        read_motion(bad, model, layout)
