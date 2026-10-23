import math
from types import SimpleNamespace

import pytest
import torch

from a3_supermimic.config import TrainConfig
from a3_supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, matrix_from_quat
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR
from a3_supermimic.world_model.heads import (
    AnalyticHeads, ReferenceFrame, TerminationCfg, anchor_relative_orientation,
)


def test_orientation_feedback_matches_rotation_and_common_frame():
    robot = quat_from_rotvec(torch.tensor([[.2, -.1, .4]], dtype=torch.float64))
    reference = quat_from_rotvec(torch.tensor([[-.3, .4, -.2]], dtype=torch.float64))
    actual = anchor_relative_orientation(robot, reference)
    expected = matrix_from_quat(robot).transpose(-1, -2) @ matrix_from_quat(reference)
    torch.testing.assert_close(actual, expected[..., :2].reshape(1, 6))
    world = quat_from_rotvec(torch.tensor([[.1, .2, -.8]], dtype=torch.float64))
    torch.testing.assert_close(anchor_relative_orientation(quat_mul(world, robot), quat_mul(world, reference)), actual)
    torch.testing.assert_close(anchor_relative_orientation(robot, -reference), actual)


def test_soft_termination_thresholds_survival_and_gradients(robot):
    model, layout, fk = robot
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link",
                          torch.tensor([[-3., 3.]] * 29), torch.zeros(29), layout=layout)
    base = torch.zeros(1, layout.state_dim, dtype=torch.float64)
    base[:, :layout.nq] = torch.tensor(model.qpos0)
    ref = ReferenceFrame(None, None, None, None, base[:, :3].clone(), base[:, 3:7].clone())
    probabilities = []
    for height in (0., .1, .24, .25, .26, .4):
        s = base.clone()
        s[:, 2] += height
        probability = heads.soft_termination(s, ref)
        expected = 1 - (1 / (1 + math.exp((height - .25) / .02))) / (1 + math.exp(-.8 / .04))
        torch.testing.assert_close(probability, torch.tensor([expected], dtype=s.dtype))
        if height != .25:
            assert bool(heads.terminated(s, ref)) == (height > .25)
        probabilities.append(float(probability))
    assert probabilities == sorted(probabilities)
    assert (1 - probabilities[0]) ** 32 > .9998
    assert abs(probabilities[3] - .5) < 1.e-7
    theta = math.acos(.2)  # gravity-z difference = .8
    s = base.clone()
    s[:, 3:7] = quat_from_rotvec(torch.tensor([[theta, 0., 0.]], dtype=s.dtype))
    assert abs(float(heads.soft_termination(s, ref)) - .5) < 3.e-6
    s[:, 2] += .25
    assert abs(float(heads.soft_termination(s, ref)) - .75) < 1.e-7

    def probability(x):
        s = base.clone()
        s[:, 2] += x[0]
        rot = torch.stack((x[1], x[1]*0, x[1]*0))[None]
        s[:, 3:7] = quat_from_rotvec(rot)
        return heads.soft_termination(s, ref)

    x = torch.tensor([.24, 1.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(probability, (x,))
    assert (torch.autograd.grad(probability(x).sum(), x)[0] > 0).all()


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_soft_termination_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        TerminationCfg(height_temperature=temperature)


def test_simulator_and_imagination_orientation_goals_have_no_position_feedback(robot, motion):
    from a3_supermimic.data.replay_buffer import StartBatch
    from a3_supermimic.envs.mdp.motion_library import MotionLibrary
    from a3_supermimic.envs.simulator import Simulator
    from a3_supermimic.imagination.rollout import imagine
    from a3_supermimic.kinematics.model import ModelKinematics
    from a3_supermimic.policy.nets import Policy, PolicyCfg
    from a3_supermimic.world_model.world_model import WorldModel, WorldModelCfg

    model, layout, _ = robot
    fk = ModelKinematics(model)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES), torch.zeros(29), allow_synthetic=True)
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link",
                          torch.tensor([[-3., 3.]] * 29), torch.zeros(29), layout=layout)
    frame = torch.tensor([0, 61])
    q, v = lib.generalized(frame)
    state = torch.cat((q, v, torch.zeros(2, 32)), -1)
    sim = Simulator.__new__(Simulator)
    sim.cfg = TrainConfig()
    sim.cfg.env.obs_noise = False
    sim.cfg.env.future_frames = 2
    sim.library, sim.heads, sim.device = lib, heads, "cpu"
    sim.command = SimpleNamespace(frame=frame)
    sim.history = torch.zeros(2, 2, 93)
    sim.state = lambda: state
    goal = sim.observation()["goal"]
    assert goal.shape == (2, 142)
    state[:, :3] += torch.tensor([3., -4., 2.])
    torch.testing.assert_close(sim.observation()["goal"], goal, atol=0, rtol=0)
    xi = torch.zeros(2, 34)
    start = StartBatch(state, torch.zeros(2, 29), xi, frame,
                       state[None].expand(2, -1, -1), torch.zeros(2, 2, 29))
    wm = WorldModel(WorldModelCfg(hidden=(16,)), heads, "cpu", torch.zeros(34), torch.ones(34))
    policy = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), history=2, goal_dim=142))
    ro = imagine(wm, policy, lib, start, 3, future_frames=2, obs_noise=False, anchor_obs=True)
    torch.testing.assert_close(ro.goals[0], goal)
    assert not any("term_head" in key for key in wm.state_dict())
    assert torch.isfinite(ro.dhat).all() and ((ro.dhat >= 0) & (ro.dhat <= 1)).all()
    torch.testing.assert_close(ro.survival[1:], torch.cumprod(1-ro.dhat[1:], dim=0))


def test_wm_only_freezes_actor_critic_and_resumes_at_update_boundary(tmp_path):
    from a3_supermimic.training.loop import Trainer
    from a3_supermimic.robot import CONTRACT_VERSION

    class TinyTrainer:
        def __init__(self, model, critic=None):
            self.model, self.critic = model, critic
            self.opt = torch.optim.Adam(model.parameters(), lr=.01)
            if critic is not None:
                self.critic_opt = torch.optim.Adam(critic.parameters(), lr=.01)

        def set_lr(self, lr):
            for group in self.opt.param_groups:
                group["lr"] = lr

        def update(self, buffer):
            self.opt.zero_grad()
            loss = self.model(torch.ones(1, 1)).square().sum()
            loss.backward()
            self.opt.step()
            if self.critic is not None:
                self.critic_opt.zero_grad()
                self.critic(torch.ones(1, 1)).square().sum().backward()
                self.critic_opt.step()
            return {"loss": float(loss.detach())}

    def make_trainer():
        t = Trainer.__new__(Trainer)
        t.cfg = TrainConfig()
        t.cfg.device = "cpu"
        t.rank, t.iteration, t.num_updates, t.num_policy_updates, t.utd_acc = 0, 1280, 499, 0, .4
        t.log_dir = tmp_path
        t.contract, t.contract_digest = {"version": CONTRACT_VERSION}, "test-contract"
        t.wm, t.policy, t.critic = [torch.nn.Linear(1, 1) for _ in range(3)]
        t.wm_trainer, t.policy_trainer = TinyTrainer(t.wm), TinyTrainer(t.policy, t.critic)
        t.buffer = None
        t.sim = SimpleNamespace(command=SimpleNamespace(bin_return=torch.zeros(1), bin_memory=torch.zeros(1)))
        return t

    t = make_trainer()
    assert t.cfg.schedule.wm_only_updates == 500
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (t.wm, t.policy, t.critic)]
    metrics = t.update()
    assert t.num_updates == 500 and t.num_policy_updates == 0
    assert metrics["train/wm_only"] == 1 and metrics["train/wm_only_remaining_updates"] == 0
    assert not any(k.startswith("policy/") for k in metrics)
    assert any(not torch.equal(before[0][k], v) for k, v in t.wm.state_dict().items())
    for m, saved in zip((t.policy, t.critic), before[1:]):
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, saved[k], atol=0, rtol=0)
    assert not t.policy_trainer.opt.state and not t.policy_trainer.critic_opt.state
    t.save()
    resumed = make_trainer()
    resumed.load(tmp_path / "ckpt_latest.pt")
    assert resumed.num_updates == 500 and resumed.num_policy_updates == 0
    assert resumed.utd_acc == .4
    metrics = resumed.update()
    assert resumed.num_updates == 501 and resumed.num_policy_updates == 1
    assert metrics["train/wm_only"] == 0
    assert resumed.policy_trainer.opt.state and resumed.policy_trainer.critic_opt.state
    for m, saved in zip((resumed.policy, resumed.critic), before[1:]):
        assert any(not torch.equal(v, saved[k]) for k, v in m.state_dict().items())
