"""Differentiable MuJoCo FK, adapted from SuperMimic-Reproduced for A3 and independent rotors."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from a3_supermimic.kinematics.rotations import quat_apply, quat_from_angle_axis, quat_mul, quat_normalize


@dataclass
class _BodyLink:
    body_id: int  # MuJoCo body id (1..nbody-1)
    parent_index: int  # index into our body list (0..nb-1), -1 for the root
    joint_type: int  # mjJNT_FREE / mjJNT_HINGE / -1 (fixed)
    qpos_adr: int
    dof_adr: int


class ModelKinematics:
    """Batched FK including world-parented elastic motor rotors and fixed terrain bodies."""

    def __init__(
        self,
        model: mujoco.MjModel,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.nbody = model.nbody - 1  # exclude world
        self.nq = model.nq
        self.nv = model.nv
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

        links: list[_BodyLink] = []
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b])
            parent_index = -1 if parent == 0 else parent - 1
            assert parent < b, "MuJoCo bodies are topologically ordered"
            njnt = int(model.body_jntnum[b])
            if njnt == 0:
                links.append(_BodyLink(b, parent_index, -1, -1, -1))
                continue
            assert njnt == 1, f"body {b} has {njnt} joints; only 0/1 supported"
            j = int(model.body_jntadr[b])
            jt = int(model.jnt_type[j])
            assert jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE), f"joint type {jt}"
            links.append(_BodyLink(b, parent_index, jt, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))
        self.links = links

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)  # noqa: E731
        self.body_pos = t(model.body_pos[1:])  # (nb,3)
        self.body_quat = t(model.body_quat[1:])  # (nb,4) wxyz
        jnt_pos = np.zeros((self.nbody, 3))
        jnt_axis = np.zeros((self.nbody, 3))
        qpos0 = np.zeros(self.nbody)
        for i, l in enumerate(links):
            if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                j = int(model.body_jntadr[l.body_id])
                jnt_pos[i] = model.jnt_pos[j]
                jnt_axis[i] = model.jnt_axis[j]
                qpos0[i] = model.qpos0[l.qpos_adr]
        self.jnt_pos = t(jnt_pos)
        self.jnt_axis = t(jnt_axis)
        self.qpos0 = t(qpos0)
        # Hinge bookkeeping in body order (skip root / fixed bodies).
        self.hinge_body_idx = [i for i, l in enumerate(links) if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE]
        self.hinge_qpos_adr = torch.tensor([links[i].qpos_adr for i in self.hinge_body_idx], device=self.device)
        self.hinge_dof_adr = torch.tensor([links[i].dof_adr for i in self.hinge_body_idx], device=self.device)

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def body_indices(self, names) -> torch.Tensor:
        return torch.tensor([self.body_names.index(n) for n in names], device=self.device)

    # ------------------------------------------------------------------ core

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        """FK for a batch.

        Args:
          qpos: (..., nq) MuJoCo generalized positions.
          qvel: (..., nv) MuJoCo generalized velocities (free joint: lin world, ang body).
        Returns:
          dict with 'pos' (..., nb, 3), 'quat' (..., nb, 4) and, if qvel is given,
          'lin_vel' (..., nb, 3), 'ang_vel' (..., nb, 3) in world frame at the link origin.
        """
        batch = qpos.shape[:-1]
        nb = self.nbody
        pos = [None] * nb
        quat = [None] * nb
        lin = [None] * nb
        ang = [None] * nb
        compute_vel = qvel is not None

        zero = torch.zeros(*batch, 3, dtype=qpos.dtype, device=qpos.device)
        identity = torch.cat((torch.ones_like(zero[..., :1]), zero), -1)
        for i in range(nb):
            l = self.links[i]
            if l.joint_type == mujoco.mjtJoint.mjJNT_FREE:
                pos[i] = qpos[..., l.qpos_adr:l.qpos_adr+3]
                quat[i] = quat_normalize(qpos[..., l.qpos_adr+3:l.qpos_adr+7])
                if compute_vel:
                    lin[i] = qvel[..., l.dof_adr:l.dof_adr+3]
                    ang[i] = quat_apply(quat[i], qvel[..., l.dof_adr+3:l.dof_adr+6])
                continue
            p = l.parent_index
            pq, pp = (quat[p], pos[p]) if p >= 0 else (identity, zero)
            pv, pw = (lin[p], ang[p]) if p >= 0 and compute_vel else (zero, zero)
            # Body frame from parent frame (fixed offset).
            b_pos = pp + quat_apply(pq, self.body_pos[i].expand(*batch, 3))
            b_quat = quat_mul(pq, self.body_quat[i].expand(*batch, 4))
            if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                anchor = b_pos + quat_apply(b_quat, self.jnt_pos[i].expand(*batch, 3))
                axis_w = quat_apply(b_quat, self.jnt_axis[i].expand(*batch, 3))
                angle = qpos[..., l.qpos_adr] - self.qpos0[i]
                q_loc = quat_from_angle_axis(angle, self.jnt_axis[i].expand(*batch, 3))
                b_quat = quat_mul(b_quat, q_loc)
                b_pos = anchor - quat_apply(b_quat, self.jnt_pos[i].expand(*batch, 3))
                if compute_vel:
                    qd = qvel[..., l.dof_adr].unsqueeze(-1)
                    w_joint = axis_w * qd
                    ang[i] = pw + w_joint
                    lin[i] = pv + torch.cross(pw, b_pos - pp, dim=-1) + torch.cross(w_joint, b_pos - anchor, dim=-1)
            else:  # fixed body
                if compute_vel:
                    ang[i] = pw
                    lin[i] = pv + torch.cross(pw, b_pos - pp, dim=-1)
            pos[i] = b_pos
            quat[i] = b_quat

        out = {"pos": torch.stack(pos, dim=-2), "quat": torch.stack(quat, dim=-2)}
        if compute_vel:
            out["lin_vel"] = torch.stack(lin, dim=-2)
            out["ang_vel"] = torch.stack(ang, dim=-2)
        return out

    __call__ = forward
