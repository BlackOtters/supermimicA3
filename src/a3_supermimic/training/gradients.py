"""Gradient clipping without overflowing the norm of finite FP32 gradients."""
import torch


@torch.no_grad()
def clip_grad_norm_stable(parameters, max_norm: float) -> torch.Tensor:
    parameters = tuple(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.nn.utils.get_total_norm(grads)
    if not torch.isfinite(norm):
        # Squaring large, finite FP32 elements may overflow even though the norm fits.
        # Recompute only the reduction in FP64; never replace invalid gradient elements.
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads]))
        if torch.isfinite(norm):
            coefficient = (max_norm / (norm + 1.e-6)).clamp(max=1.)
            for g in grads:
                g.copy_(g.to(torch.float64) * coefficient)
        return norm
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
    return norm
