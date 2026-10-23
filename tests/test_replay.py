import torch
import pytest
from a3_supermimic.data.replay_buffer import ReplayBuffer


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.e16])
def test_bad_actions_cannot_enter_world_model_or_actor_samples(bad):
    buffer = ReplayBuffer(12, 2, 40, 1, "cpu", action_dim=29)
    buffer.register_context(torch.arange(2), torch.zeros(2, 1))
    for t in range(10):
        s = torch.zeros(2, 40)
        action = torch.zeros(2, 29); action[0, 0] = bad
        buffer.append(s, action, s, torch.full((2,), t), torch.zeros(2, dtype=torch.long),
                      torch.zeros(2, dtype=torch.bool), previous_clean_action=action,
                      valid=torch.ones(2, dtype=torch.bool), start_valid=torch.ones(2, dtype=torch.bool))
    assert not buffer.valid[:10, 0].any() and not buffer.start_valid[:10, 0].any()
    assert buffer.valid[:10, 1].all()
    assert (buffer.sample_windows(128, 8).a_exec == 0).all()
    assert (buffer.sample_starts(128, 10).hist_actions == 0).all()


def test_windows_exclude_resets_invalid_transitions_and_overwritten_ring():
    buffer = ReplayBuffer(9, 2, 40, 4, "cpu")
    ids = torch.arange(2)
    for t in range(25):
        if t % 5 == 0:
            buffer.register_context(ids, torch.full((2, 4), float(t//5)))
        s = torch.full((2, 40), float(t))
        end = torch.full((2,), t % 5 == 4)
        buffer.append(s, torch.full((2, 31), float(t)), s+1, torch.full((2,), t),
                      torch.full((2,), t//5), end,
                      previous_clean_action=torch.full((2, 29), -float(t)),
                      valid=torch.full((2,), t % 5 != 2))
    w = buffer.sample_windows(256, 2)
    assert (w.frame >= 16).all()
    assert (w.frame % 5 != 2).all()
    assert ((w.frame[0]//5) == (w.frame[1]//5)).all()
    torch.testing.assert_close(w.xi[:, 0], (w.frame[0]//5).float())
    torch.testing.assert_close(w.states[0, :, 0], w.frame[0].float())
    torch.testing.assert_close(w.states[-1, :, 0], w.frame[-1].float()+1)
    assert not w.done[0].bool().any()
    torch.testing.assert_close(w.a_exec[..., 0], w.frame.float())


def test_short_episode_start_history_is_padded_at_reset():
    buffer = ReplayBuffer(8, 1, 40, 1, "cpu")
    buffer.register_context(torch.tensor([0]), torch.tensor([[1.]]))
    for t in range(5):
        if t == 3:
            buffer.register_context(torch.tensor([0]), torch.tensor([[2.]]))
        s = torch.full((1, 40), float(t))
        buffer.append(s, torch.zeros(1, 31), s+1, torch.tensor([t]), torch.tensor([int(t>=3)]),
                      torch.tensor([t == 2]), previous_clean_action=torch.full((1, 29), -float(t)))
    start = buffer.sample_starts(200, 10)
    torch.testing.assert_close(start.prev_action[:, 0], -start.frame.float())
    torch.testing.assert_close(start.hist_actions[..., 0], -start.hist_states[..., 0])
    for col, frame in enumerate(start.frame.tolist()):
        earliest = 3 if frame >= 3 else 0
        assert int(start.hist_states[:, col, 0].min()) == earliest
        assert int(start.hist_states[-1, col, 0]) == frame
        assert start.xi[col, 0] == (2 if frame >= 3 else 1)
