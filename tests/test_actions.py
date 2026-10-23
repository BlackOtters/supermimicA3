"""Exercise the simulator adapter with deterministic noise and a minimal backend."""
from types import SimpleNamespace

import torch

from a3_supermimic.config import TrainConfig
from a3_supermimic.data.replay_buffer import ReplayBuffer
from a3_supermimic.envs.simulator import Simulator
from a3_supermimic.envs.mdp.motion_library import MotionLibrary
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR
from a3_supermimic.world_model.heads import AnalyticHeads


def test_clean_actor_history_and_reward_keep_executed_dynamics_input(robot, motion, monkeypatch):
    from a3_supermimic.kinematics.model import ModelKinematics
    model, layout, _ = robot
    fk = ModelKinematics(model)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES),
                        torch.zeros(29), allow_synthetic=True)
    sim = Simulator.__new__(Simulator)
    sim.cfg = TrainConfig(); sim.cfg.env.obs_noise = False
    sim.device = torch.device('cpu'); sim.num_envs = 2
    sim.layout, sim.library = layout, lib
    sim.heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, 'pelvis_link',
                             torch.tensor([[-3., 3.]]*29), torch.zeros(29), layout=layout)
    sim.previous = torch.zeros(2, 31); sim.previous_clean = torch.zeros(2, 29)
    sim.fresh = torch.ones(2, 1); sim.xi = torch.zeros(2, 34)
    sim.history = torch.zeros(10, 2, 93)
    sim.command = SimpleNamespace(frame=torch.zeros(2, dtype=torch.long), origins=torch.zeros(2, 3),
                                  record_reward=lambda r: None, finalize=lambda *args: None)
    q, v = lib.generalized(sim.command.frame)
    sim.env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q, qvel=v)),
                              episode_id=torch.zeros(2, dtype=torch.long),
                              episode_length_buf=torch.zeros(2, dtype=torch.long),
                              max_episode_length=torch.full((2, 1), 500))
    received = []
    def backend_step(td):
        received.append(td['action'].clone())
        sim.env.episode_length_buf += 1
        return {'terminated': torch.zeros(2, 1, dtype=torch.bool)}
    sim.env._step = backend_step
    sim.buffer = ReplayBuffer(8, 2, layout.state_dim, 34, 'cpu')
    sim.buffer.register_context(torch.arange(2), sim.xi)
    action = torch.full((2, 29), .2)
    for noise in (1., -1.):
        monkeypatch.setattr(torch, 'randn_like', lambda x: torch.full_like(x, noise))
        obs, _, _, _, extra = sim.step(action, auto_reset=False)
        torch.testing.assert_close(received[-1], action + noise*.1)
        torch.testing.assert_close(obs['policy'][:, -29:], action)
        torch.testing.assert_close(sim.state()[:, layout.physical_dim:layout.physical_dim+29],
                                   received[-1])
    # Identical clean commands incur no action-rate penalty despite opposite execution noise.
    torch.testing.assert_close(extra['reward_terms']['action_rate_l2'], torch.zeros(2))
    torch.testing.assert_close(sim.buffer.previous_clean_actions[0], torch.zeros(2, 29))
    torch.testing.assert_close(sim.buffer.previous_clean_actions[1], action)
    torch.testing.assert_close(sim.buffer.actions[0, :, :29], received[0])
    torch.testing.assert_close(sim.buffer.actions[1, :, :29], received[1])
    resets = []
    sim.reset = lambda ids: resets.append(ids.clone())
    for t, bad_value in enumerate((1.e16, float('nan'), float('inf')), start=2):
        bad_action = action.clone(); bad_action[0, 0] = bad_value
        _, reward, terminal, _, extra = sim.step(bad_action)
        assert torch.isfinite(received[-1]).all()
        assert (received[-1][0] == 0).all()
        torch.testing.assert_close(received[-1][1], action[1] - .1)
        assert terminal[0] and extra['invalid_action'][0] and reward[0] == 0
        assert resets[-1].tolist() == [0]
        assert not sim.buffer.valid[t, 0] and not sim.buffer.start_valid[t, 0]
        assert sim.buffer.valid[t, 1]
