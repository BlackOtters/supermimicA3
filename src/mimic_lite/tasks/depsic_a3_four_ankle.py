"""Episode-level physical four-axis elastic/rigid switch, with encoder state API."""

import math
import torch

from mimic_lite.assets.a3_ultra_four_ankle import ANKLES, ENCODERS, ROTORS, TRANSMISSIONS
from mimic_lite.tasks.depsic_a3_ankle_roll_elastic import depsic_elastic_motor_dr
from active_adaptation.envs.mdp.randomizations.common import _mjlab_recompute_constants


class depsic_four_ankle_dr(depsic_elastic_motor_dr, namespace="mimic_lite"):
    def __init__(self, env, rigid_probability=.25,
                 pitch_k_range=(200., 2000.), roll_k_range=(65., 400.),
                 k_distribution="log_uniform", pitch_d_range=(.5, 2.), roll_d_range=(.5, 2.),
                 d_distribution="uniform", implicit_springs=True, **kwargs):
        super().__init__(env, elastic_joints=ANKLES, rotor_bodies=ROTORS, **kwargs)
        if not math.isfinite(rigid_probability) or not 0 <= rigid_probability <= 1:
            raise ValueError("rigid_probability must lie in [0, 1]")
        self.rigid_probability = float(rigid_probability)
        self.k_bounds = (pitch_k_range, roll_k_range) * 2
        self.d_bounds = (pitch_d_range, roll_d_range) * 2
        self.k_distribution, self.d_distribution = k_distribution, d_distribution
        for distribution, bounds in ((k_distribution, self.k_bounds), (d_distribution, self.d_bounds)):
            if distribution not in ("uniform", "log_uniform"):
                raise ValueError("Elastic distribution must be uniform or log_uniform")
            for lo, hi in bounds:
                if not all(map(math.isfinite, (lo, hi))) or not 0 < lo <= hi:
                    raise ValueError("Elastic parameter bounds must be finite, positive and ordered")
        sim, asset = env.sim, self.asset
        self.output_ids = torch.tensor([asset.joint_names.index(n) for n in ANKLES], device=self.device)
        self.encoder_ids = torch.tensor([asset.joint_names.index(n) for n in ENCODERS], device=self.device)
        self.output_qids = asset.indexing.joint_q_adr[self.output_ids]
        self.encoder_qids = asset.indexing.joint_q_adr[self.encoder_ids]
        self.output_dofs = asset.indexing.joint_v_adr[self.output_ids]
        self.encoder_dofs = asset.indexing.joint_v_adr[self.encoder_ids]
        self.direct_ctrl = self.ctrl_ids[self.roll_cols]
        self.encoder_ctrl = torch.tensor([sim.mj_model.actuator(f"robot/{n}_encoder_drive").id for n in ANKLES], device=self.device)
        self.tendon_ids = torch.tensor([sim.mj_model.tendon(f"robot/{n}").id for n in TRANSMISSIONS], device=self.device)
        sim.expand_model_fields(("tendon_stiffness", "tendon_damping"))
        self.is_elastic = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sampled_k = torch.zeros(self.num_envs, 4, device=self.device)
        self.sampled_d = torch.zeros_like(self.sampled_k)
        env._four_ankle_state = self
        if implicit_springs:
            from mimic_lite.assets.ankle_implicit import install_implicit_ankles
            self.implicit_integrator = install_implicit_ankles(sim,
                self.tendon_ids.cpu().tolist(), self.encoder_dofs.cpu().tolist(),
                self.output_dofs.cpu().tolist())

    def _sample(self, n, bounds, distribution):
        limits = torch.as_tensor(bounds, device=self.device)
        if distribution == "log_uniform":
            limits = limits.log()
        samples = limits[:, 0] + torch.rand(n, 4, device=self.device) * (limits[:, 1] - limits[:, 0])
        return samples.exp() if distribution == "log_uniform" else samples

    def reset(self, env_ids):
        if not env_ids.numel():
            return
        super().reset(env_ids)
        rows = env_ids[:, None]
        model, data = self.env.sim.model, self.env.sim.data
        elastic = torch.rand(len(env_ids), device=self.device) >= self.rigid_probability
        self.is_elastic[env_ids] = elastic
        self.sampled_k[env_ids] = self._sample(len(env_ids), self.k_bounds, self.k_distribution) * elastic[:, None]
        self.sampled_d[env_ids] = self._sample(len(env_ids), self.d_bounds, self.d_distribution) * elastic[:, None]
        model.tendon_stiffness[rows, self.tendon_ids] = self.sampled_k[env_ids]
        model.tendon_damping[rows, self.tendon_ids] = self.sampled_d[env_ids]
        # Same randomized equivalent inertia, placed on exactly one ACTIVE side.
        # In rigid rows rotors are independent world-fixed, unforced and parked;
        # they exert no torque/inertia on the robot. No stiff-spring approximation.
        inertia = model.body_inertia[rows, self.rotor_bodies, 2]
        model.dof_armature[rows, self.output_dofs] = inertia * (~elastic[:, None])
        for field in ("actuator_gainprm", "actuator_biasprm"):
            values = getattr(model, field)
            nominal_sample = values[rows, self.direct_ctrl].clone()
            values[rows, self.encoder_ctrl] = nominal_sample * elastic[:, None, None]
            values[rows, self.direct_ctrl] = nominal_sample * (~elastic[:, None, None])
        data.qpos[rows, self.encoder_qids] = torch.where(elastic[:, None], data.qpos[rows, self.output_qids], 0.)
        data.qvel[rows, self.encoder_dofs] = torch.where(elastic[:, None], data.qvel[rows, self.output_dofs], 0.)
        data.qacc_warmstart[rows, self.encoder_dofs] = 0.
        _mjlab_recompute_constants(self.env, self.recompute_level)

    def _park_inactive(self):
        # Enforce a disabled internal state even if a generic reset/perturbation
        # touches it. Never copy these parked values into policy observations.
        data = self.env.sim.data
        for field, ids in (("qpos", self.encoder_qids), ("qvel", self.encoder_dofs),
                           ("qacc", self.encoder_dofs), ("qacc_warmstart", self.encoder_dofs)):
            values = getattr(data, field)
            values[:, ids] = torch.where(self.is_elastic[:, None], values[:, ids], 0.)

    def pre_step(self, substep):
        self._park_inactive()

    def post_step(self, substep):
        self._park_inactive()

    def encoder_state(self):
        """Actual encoder coordinates for elastic rows; identity for rigid rows."""
        data = self.env.sim.data
        return (torch.where(self.is_elastic[:, None], data.qpos[:, self.encoder_qids], data.qpos[:, self.output_qids]),
                torch.where(self.is_elastic[:, None], data.qvel[:, self.encoder_dofs], data.qvel[:, self.output_dofs]))

    def motor_effort(self):
        model, data = self.env.sim.model, self.env.sim.data
        q, dq = self.encoder_state()
        gain = model.actuator_gainprm[:, self.direct_ctrl, 0] + model.actuator_gainprm[:, self.encoder_ctrl, 0]
        bias = model.actuator_biasprm[:, self.direct_ctrl] + model.actuator_biasprm[:, self.encoder_ctrl]
        ctrl = torch.where(self.is_elastic[:, None], data.ctrl[:, self.encoder_ctrl], data.ctrl[:, self.direct_ctrl])
        return gain * ctrl + bias[:, :, 1] * q + bias[:, :, 2] * dq
