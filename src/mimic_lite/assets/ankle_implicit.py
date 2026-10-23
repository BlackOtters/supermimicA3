"""Linearly implicit fixed-tendon elasticity, without reducing the physics step.

MuJoCo's implicitfast treats velocity derivatives, not spring position
derivatives. Warp's tree-sparse derivative also omits cross-tree tendon terms.
For the four independent encoder/output pairs, use a rank-four Woodbury solve:

  (A0 + J.T @ diag(h*D+h*h*K) @ J) a
      = M*a_forward - h*J.T @ diag(K) @ J*v

A0 is the normal implicitfast matrix excluding these tendon dampers. This is
backward Euler for the linear springs/dampers, with the existing MuJoCo contact
force solve retained (not a fully implicit nonlinear contact solve). Native
passive forces remain exactly -J.T@(K*x+D*J*v); no physical damping is added.
"""

import copy

import numpy as np
import warp as wp
from mujoco_warp._src import derivative, forward, smooth
from mujoco_warp._src.types import IntegratorType, DisableBit


@wp.kernel
def _rhs(timestep: wp.array[float], stiffness: wp.array2d[float],
         damping: wp.array2d[float], tids: wp.array[int], enc: wp.array[int],
         out: wp.array[int], velocity: wp.array2d[float], ma: wp.array2d[float],
         rhs: wp.array2d[float], scales: wp.array2d[float]):
    w, i = wp.tid()
    h = timestep[w % timestep.shape[0]]
    value = ma[w, i]
    for k in range(4):
        tid = tids[k]
        K = stiffness[w % stiffness.shape[0], tid]
        D = damping[w % damping.shape[0], tid]
        v = velocity[w, enc[k]] - velocity[w, out[k]]
        if i == enc[k]:
            value -= h * K * v
        if i == out[k]:
            value += h * K * v
        if i == 0:
            scales[w, k] = wp.sqrt(h * D + h * h * K)
    rhs[w, i] = value


@wp.kernel
def _column(k: int, enc: wp.array[int], out: wp.array[int],
            scales: wp.array2d[float], rhs: wp.array2d[float]):
    w, i = wp.tid()
    value = float(0.)
    if i == enc[k]:
        value = scales[w, k]
    if i == out[k]:
        value = -scales[w, k]
    rhs[w, i] = value


@wp.kernel
def _save_column(k: int, column: wp.array2d[float], columns: wp.array3d[float]):
    w, i = wp.tid()
    columns[w, i, k] = column[w, i]


@wp.kernel
def _schur(enc: wp.array[int], out: wp.array[int], scales: wp.array2d[float],
           columns: wp.array3d[float], free_acc: wp.array2d[float],
           weights: wp.array2d[float]):
    w = wp.tid()
    A = wp.mat44(0.)
    b = wp.vec4(0.)
    for i in range(4):
        b[i] = scales[w, i] * (free_acc[w, enc[i]] - free_acc[w, out[i]])
        for j in range(4):
            # Symmetrize roundoff in independent sparse back-substitutions.
            aij = scales[w, i] * (columns[w, enc[i], j] - columns[w, out[i], j])
            aji = scales[w, j] * (columns[w, enc[j], i] - columns[w, out[j], i])
            A[i, j] = .5 * (aij + aji)
        A[i, i] += 1.
    L = wp.mat44(0.)
    for i in range(4):
        for j in range(i + 1):
            value = A[i, j]
            for k in range(j):
                value -= L[i, k] * L[j, k]
            if i == j:
                L[i, j] = wp.sqrt(value)
            else:
                L[i, j] = value / L[j, j]
    y = wp.vec4(0.)
    for i in range(4):
        value = b[i]
        for j in range(i):
            value -= L[i, j] * y[j]
        y[i] = value / L[i, i]
    x = wp.vec4(0.)
    for reverse in range(4):
        i = 3 - reverse
        value = y[i]
        for j in range(i + 1, 4):
            value -= L[j, i] * x[j]
        x[i] = value / L[i, i]
        weights[w, i] = x[i]


@wp.kernel
def _correct(columns: wp.array3d[float], weights: wp.array2d[float],
             free_acc: wp.array2d[float], acc: wp.array2d[float]):
    w, i = wp.tid()
    value = free_acc[w, i]
    for k in range(4):
        value -= columns[w, i, k] * weights[w, k]
    acc[w, i] = value


class FixedTendonImplicit:
    def __init__(self, sim, tendon_ids, encoder_dofs, output_dofs):
        m, d = sim.wp_model, sim.wp_data
        if m.opt.integrator != IntegratorType.IMPLICITFAST or not m.is_sparse:
            raise ValueError("Four-ankle implicit solve requires sparse implicitfast")
        if m.ntendon != 4 or set(tendon_ids) != set(range(4)):
            raise ValueError("Four-ankle implicit solve requires exactly four fixed tendons")
        if m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER):
            raise ValueError("Four-ankle springs and dampers must be enabled")
        self.tids = wp.array(np.asarray(tendon_ids), dtype=int, device=sim.wp_device)
        self.enc = wp.array(np.asarray(encoder_dofs), dtype=int, device=sim.wp_device)
        self.out = wp.array(np.asarray(output_dofs), dtype=int, device=sim.wp_device)
        self.zero_damping = wp.zeros((1, 4), device=sim.wp_device)
        self.zero_poly = wp.zeros((1, 4), dtype=wp.vec2, device=sim.wp_device)
        self.matrix = wp.empty((d.nworld, 1, m.nM), device=sim.wp_device)
        self.factor = wp.empty((d.nworld, 1, m.nC), device=sim.wp_device)
        self.diagonal = wp.empty((d.nworld, m.nv), device=sim.wp_device)
        self.rhs = wp.empty((d.nworld, m.nv), device=sim.wp_device)
        self.free_acc = wp.empty_like(self.rhs)
        self.column = wp.empty_like(self.rhs)
        self.acc = wp.empty_like(self.rhs)
        self.scales = wp.empty((d.nworld, 4), device=sim.wp_device)
        self.weights = wp.empty_like(self.scales)
        self.columns = wp.empty((d.nworld, m.nv, 4), device=sim.wp_device)

    def __call__(self, m, d):
        # Copy at graph capture time so later model-field expansion cannot leave
        # stale pointers. Only the derivative sees zero damping, never forces.
        base = copy.copy(m)
        base.tendon_damping = self.zero_damping
        base.tendon_dampingpoly = self.zero_poly
        derivative.deriv_smooth_vel(base, d, self.matrix)
        wp.launch(_rhs, dim=(d.nworld, m.nv), inputs=[m.opt.timestep,
            m.tendon_stiffness, m.tendon_damping, self.tids, self.enc, self.out,
            d.qvel, d.efc.Ma], outputs=[self.rhs, self.scales])
        smooth.factor_solve_i(m, d, self.matrix, self.factor, self.diagonal,
                             self.free_acc, self.rhs)
        for k in range(4):
            wp.launch(_column, dim=(d.nworld, m.nv),
                inputs=[k, self.enc, self.out, self.scales], outputs=[self.rhs])
            smooth.solve_LD(m, d, self.factor, self.diagonal, self.column, self.rhs)
            wp.launch(_save_column, dim=(d.nworld, m.nv),
                inputs=[k, self.column], outputs=[self.columns])
        wp.launch(_schur, dim=d.nworld, inputs=[self.enc, self.out, self.scales,
            self.columns, self.free_acc], outputs=[self.weights])
        wp.launch(_correct, dim=(d.nworld, m.nv),
            inputs=[self.columns, self.weights, self.free_acc], outputs=[self.acc])
        forward._advance(m, d, self.acc)


def install_implicit_ankles(sim, tendon_ids, encoder_dofs, output_dofs):
    """Dispatch only opted-in models; ordinary/legacy models keep Warp's path."""
    if not getattr(forward.implicit, "_a3_dispatch", False):
        original = forward.implicit

        def implicit(m, d):
            handler = getattr(m, "_a3_fixed_tendon_implicit", None)
            if handler is None:
                return original(m, d)
            return handler(m, d)

        implicit._a3_dispatch = True
        forward.implicit = implicit
    handler = FixedTendonImplicit(sim, tendon_ids, encoder_dofs, output_dofs)
    sim.wp_model._a3_fixed_tendon_implicit = handler
    sim.create_graph()
    return handler
