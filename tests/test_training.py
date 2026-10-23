import torch
import pytest
from a3_supermimic.robot import TRACKED_BODIES, TRACKING_ANCHOR
from a3_supermimic.kinematics.model import ModelKinematics
from a3_supermimic.world_model.heads import AnalyticHeads
from a3_supermimic.world_model.world_model import WorldModel, WorldModelCfg
from a3_supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg
from a3_supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
from a3_supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from a3_supermimic.data.replay_buffer import ReplayBuffer
from a3_supermimic.envs.mdp.motion_library import MotionLibrary


def _distributed_bad_loss_worker(rank, root):
    from pathlib import Path
    from types import SimpleNamespace
    from a3_supermimic.data.replay_buffer import WindowBatch
    import torch.distributed as dist

    root = Path(root)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(root / "rendezvous").as_uri())
    try:
        wm = torch.nn.Linear(1, 1)
        wm.enable_compile = lambda mode: None
        trainer = WorldModelTrainer(wm, None, WMTrainerCfg(batches_per_update=1, compile=None),
                                    failure_dir=root)
        window = WindowBatch(torch.ones(1), torch.zeros(1), torch.zeros(1),
                             torch.zeros(1), torch.zeros(1, dtype=torch.long))
        trainer.loss_on = lambda w: (wm(w.states[:, None]).sum() + (float("inf") if rank else 0.), {})
        with pytest.raises(FloatingPointError, match="Nonfinite world-model loss"):
            trainer.update(SimpleNamespace(sample_windows=lambda *args: window))
        assert not trainer.opt.state
    finally:
        dist.destroy_process_group()


def test_bad_loss_on_one_rank_stops_all_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_bad_loss_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in (0, 1):
        snapshot = torch.load(tmp_path / f"wm_rank{rank}.pt", weights_only=True)
        assert snapshot["reason"] == "loss"
        assert bool(snapshot["loss"].isfinite()) == (rank == 0)


def test_finite_loss_with_overflowed_norm_can_update_and_reports_fallback():
    from types import SimpleNamespace
    wm = torch.nn.Linear(1, 1)
    wm.enable_compile = lambda mode: None
    trainer = WorldModelTrainer(wm, None, WMTrainerCfg(batches_per_update=1, compile=None))
    trainer.loss_on = lambda w: (wm(torch.tensor([[3.e20]])).sum(), {})
    result = trainer.update(SimpleNamespace(sample_windows=lambda *args: None))
    assert result["grad_norm_fp64_fallbacks_total"] == 1
    assert torch.isfinite(torch.tensor(result["grad_norm"], dtype=torch.float64))
    assert trainer.opt.state
    assert all(bool(p.isfinite().all()) for p in wm.parameters())


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_world_model_update_preserves_failure_and_parameters(tmp_path, failure):
    from a3_supermimic.data.replay_buffer import WindowBatch
    from types import SimpleNamespace

    wm = torch.nn.Linear(1, 1)
    wm.enable_compile = lambda mode: None
    trainer = WorldModelTrainer(wm, None, WMTrainerCfg(batches_per_update=1, compile=None),
                               failure_dir=tmp_path)
    window = WindowBatch(torch.tensor([1.]), torch.zeros(1), torch.zeros(1),
                         torch.zeros(1), torch.zeros(1, dtype=torch.long))
    buffer = SimpleNamespace(sample_windows=lambda *args: window)
    if failure == "loss":
        # Infinite loss with finite derivatives must also stop before Adam updates weights.
        trainer.loss_on = lambda w: (wm(w.states[:, None]).sum() + float("inf"), {})
    else:
        wm.weight.register_hook(lambda grad: grad * float("nan"))
        trainer.loss_on = lambda w: (wm(w.states[:, None]).sum(), {})
    before = {k: v.clone() for k, v in wm.state_dict().items()}
    with pytest.raises(FloatingPointError, match=f"Nonfinite world-model {failure}"):
        trainer.update(buffer)
    for key, value in wm.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert not trainer.opt.state
    snapshot = torch.load(tmp_path / "wm_rank0.pt", weights_only=True)
    torch.testing.assert_close(snapshot["window"]["states"], window.states)
    assert snapshot["reason"] == failure
    assert snapshot["local_grad_finite"] is (False if failure == "gradient" else None)
    for key, value in snapshot["wm"].items():
        torch.testing.assert_close(value, before[key])


def test_world_model_fit_and_frozen_model_actor_update(robot, motion):
    model, layout, _ = robot
    torch.manual_seed(7)
    fk = ModelKinematics(model)
    q0 = torch.zeros(29)
    lib = MotionLibrary([motion], model, layout, fk, fk.body_indices(TRACKED_BODIES), q0, allow_synthetic=True)
    heads = AnalyticHeads(fk, TRACKED_BODIES, TRACKING_ANCHOR, "pelvis_link", torch.tensor([[-3., 3.]]*29), q0, layout=layout)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32)), heads, "cpu", torch.zeros(34), torch.ones(34))
    buffer = ReplayBuffer(32, 4, layout.state_dim, 34, "cpu")
    buffer.register_context(torch.arange(4), torch.zeros(4, 34))
    for t in range(16):
        q, v = lib.generalized(torch.tensor([t]))
        s = torch.cat((q.expand(4, -1), v.expand(4, -1), torch.zeros(4, 32)), -1)
        a = torch.randn(4, 31)*.05
        target = s.clone(); target[:, layout.policy_q] += a[:, :29]*.01
        target[:, layout.physical_dim:-1] = a
        buffer.append(s, a, target, torch.full((4,), t), torch.arange(4), torch.zeros(4, dtype=torch.bool),
                      previous_clean_action=torch.zeros(4, 29))
    wt = WorldModelTrainer(wm, lib, WMTrainerCfg(window=2, batches_per_update=2, batch_size=4, compile=None))
    before = [p.clone() for p in wm.parameters()]
    metrics = wt.update(buffer)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, wm.parameters()))
    pcfg = PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,), history=3, goal_dim=136)
    actor = Policy(pcfg); critic = Critic(CriticCfg(hidden=(16,)), feat_dim=wm.spec.feat_dim, xi_dim=34, pcfg=pcfg)
    pt = PolicyTrainer(wm, actor, critic, lib, PolicyTrainerCfg(horizon=3, batch_size=4, history=3,
                       future_frames=2, critic_epochs=1, critic_minibatches=2, compile=None, obs_noise=False))
    actor_before = [p.clone() for p in actor.parameters()]
    wm_before = [p.clone() for p in wm.parameters()]
    result = pt.update(buffer)
    assert result["policy_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(wm_before, wm.parameters()))
