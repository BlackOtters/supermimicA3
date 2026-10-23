"""Complete generalized state, including elastic rotors and command delay memory."""
from dataclasses import dataclass
import torch
from a3_supermimic.kinematics.rotations import (
    heading_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul,
    quat_normalize, quat_from_rotvec, rot6d_from_quat, rotvec_from_quat,
)


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    explicit_integration: bool = True


class StateSpec(torch.nn.Module):
    def __init__(self, cfg, layout, xi_offset, xi_scale, device):
        super().__init__()
        self.cfg, self.layout, self.dt = cfg, layout, cfg.dt
        self.nj = layout.nv - 6
        memory = 0 if layout.compact else 32
        self.feat_dim = 13 + 2 * self.nj + memory
        self.delta_dim = 12 + 2 * self.nj
        def rep(scales, sizes):
            return torch.cat([torch.full((n,), float(s), device=device) for s, n in zip(scales, sizes)])
        self.register_buffer("err_scale", rep(cfg.err_scale, (3, 3, 3, 3, self.nj, self.nj)))
        self.register_buffer("out_scale", rep(cfg.out_scale, (3, 3, self.nj, 3, 3, self.nj)))
        self.register_buffer("in_scale", rep((1., 1., 1., 2., 1., 5., 1.), (1, 6, 3, 3, self.nj, self.nj, memory)))
        self.register_buffer("xi_offset", xi_offset.to(device).clone())
        self.register_buffer("xi_scale", xi_scale.to(device).clone())

    def split(self, s):
        nq, nv = self.layout.nq, self.layout.nv
        return s[..., :3], s[..., 3:7], s[..., 7:nq], s[..., nq:nq+3], s[..., nq+3:nq+6], s[..., nq+6:nq+nv]

    def canonical(self, s):
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        h = heading_quat(q)
        return torch.cat((p[..., 2:3], rot6d_from_quat(quat_mul(quat_inv(h), q)),
                          quat_apply_inverse(h, v), w, qj, qd, s[..., self.layout.physical_dim:]), -1) / self.in_scale

    def normalize_xi(self, xi):
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s, delta, action):
        dv, dw, dqd, dp, dr, dq = (delta * self.out_scale).split((3, 3, self.nj, 3, 3, self.nj), -1)
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        h = heading_quat(q)
        v, w, qd = v + quat_apply(h, dv), w + dw, qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.
        physical = torch.cat((p + dt*v + quat_apply(h, dp),
                          quat_normalize(quat_mul(q, quat_from_rotvec(dt*w + dr))),
                          qj + dt*qd + dq, v, w, qd), -1)
        return physical if self.layout.compact else torch.cat((physical, action, torch.zeros_like(action[..., :1])), -1)

    def error_vector(self, pred, true):
        p1, q1, j1, v1, w1, d1 = self.split(pred)
        p2, q2, j2, v2, w2, d2 = self.split(true)
        rot = rotvec_from_quat(quat_mul(quat_inv(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat((p1-p2, rot, v1-v2, w1-w2, j1-j2, d1-d2), -1) / self.err_scale

    def normalized_sq_error(self, pred, true):
        return self.error_vector(pred, true).square().mean(-1)
