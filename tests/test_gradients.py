import torch
from a3_supermimic.training.gradients import clip_grad_norm_stable


def test_large_finite_gradients_are_clipped_without_overflow():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.e20, 4.e20])
    assert torch.isinf(torch.nn.utils.get_total_norm([p.grad]))
    norm = clip_grad_norm_stable([p], 1.)
    torch.testing.assert_close(norm, torch.tensor(5.e20, dtype=torch.float64))
    torch.testing.assert_close(p.grad, torch.tensor([.6, .8]))

    # The coefficient would underflow/flush to zero if cast back to FP32 before scaling.
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), 3.e38)
    norm = clip_grad_norm_stable([p], 1.)
    assert torch.isfinite(norm)
    torch.testing.assert_close(p.grad, torch.full((4,), .5))


def test_invalid_gradients_remain_detectable():
    for value in [float('nan'), float('inf')]:
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([value, 1.])
        norm = clip_grad_norm_stable([p], 1.)
        assert not torch.isfinite(norm)
        assert not torch.isfinite(p.grad[0])
        assert p.grad[1] == 1.


def test_regular_clipping_matches_torch():
    generator = torch.Generator().manual_seed(42)
    params = [torch.nn.Parameter(torch.zeros(n)) for n in (5, 17)]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    for a, b in zip(params, copies):
        a.grad = torch.randn(a.shape, generator=generator)
        b.grad = a.grad.clone()
    actual = clip_grad_norm_stable(params, 1.)
    expected = torch.nn.utils.clip_grad_norm_(copies, 1.)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(params, copies):
        torch.testing.assert_close(a.grad, b.grad)
