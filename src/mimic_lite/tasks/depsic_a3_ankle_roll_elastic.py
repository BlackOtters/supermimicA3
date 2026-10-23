"""Episode-constant serial stiffness DR and strain-free reset."""

import torch
import math
from active_adaptation.envs.mdp.randomizations.base import Randomization
from mimic_lite.assets.a3_ultra_ankle_roll_elastic import ROLL_JOINTS, MOTOR_JOINTS


class depsic_ankle_roll_elastic(Randomization, namespace="mimic_lite"):
    supported_backends = ("mjlab",)

    def __init__(self, env, stiffness_range=(65., 90.), damping=1.):
        super().__init__(env)
        self.low, self.high = map(float, stiffness_range)
        self.damping = float(damping)
        if not 0 < self.low <= self.high or self.damping < 0:
            raise ValueError("Invalid serial elasticity parameters")
        asset = env.scene.articulations["robot"]
        ids = torch.tensor([asset.joint_names.index(n) for n in (*ROLL_JOINTS, *MOTOR_JOINTS)], device=self.device)
        self.qids = asset.indexing.joint_q_adr[ids]
        self.vids = asset.indexing.joint_v_adr[ids]
        self.tids = torch.tensor([env.sim.mj_model.tendon(f"robot/{side}_ankle_roll_transmission").id
                                 for side in ("left", "right")], device=self.device)
        env.sim.expand_model_fields(("tendon_stiffness", "tendon_damping"))

    def reset(self, env_ids):
        if not env_ids.numel():
            return
        rows = env_ids[:, None]
        if self.low == self.high:
            # Fixed playback settings must not consume the command sampler's RNG.
            self.env.sim.model.tendon_stiffness[rows, self.tids] = self.low
        else:
            self.env.sim.model.tendon_stiffness[rows, self.tids] = torch.empty(
                len(env_ids), 2, device=self.device).uniform_(self.low, self.high)
        self.env.sim.model.tendon_damping[rows, self.tids] = self.damping
        data = self.env.sim.data
        data.qpos[rows, self.qids[2:]] = data.qpos[rows, self.qids[:2]]
        data.qvel[rows, self.vids[2:]] = data.qvel[rows, self.vids[:2]]
        data.qacc_warmstart[rows, self.vids[2:]] = 0


class depsic_elastic_motor_dr(Randomization, namespace="mimic_lite"):
    """Episode-constant equivalent motor inertia, PD and shared bus delay.

    Pitch inertia lives on the pitch joint, roll inertia ONLY on the upstream
    rotor. These are equivalent actuator coordinates, not a parallel linkage.
    All 29 policy actuators form one delay group, including custom roll drives.
    """

    supported_backends = ("mjlab",)

    def __init__(self, env, ankle_inertia_scale=(0.7, 2.0), armature_scale=(0.5, 1.5),
                 kp_scale=(0.8, 1.2), damping_ratio_scale=(0.5, 2.0),
                 delay_ms=(0., 5., 10.), elastic_joints=None, rotor_bodies=None):
        super().__init__(env)
        from mjlab.managers.event_manager import RecomputeLevel
        from active_adaptation.envs.mdp.randomizations.common import _mjlab_ensure_recompute_fields_expanded
        from mimic_lite.depsic_a3_contract import A3_POLICY_JOINT_NAMES

        self.ranges = tuple(tuple(map(float, b)) for b in
                            (ankle_inertia_scale, armature_scale, kp_scale, damping_ratio_scale))
        for bounds in self.ranges:
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("Motor DR scales must be finite, positive and ordered")
        sim = env.sim
        asset = env.scene.articulations["robot"]
        self.asset = asset
        model = sim.mj_model
        self.joint_names = tuple(A3_POLICY_JOINT_NAMES)
        joint_ids = torch.tensor([asset.joint_names.index(n) for n in self.joint_names], device=self.device)
        self.dof_ids = asset.indexing.joint_v_adr[joint_ids]
        self.ctrl_ids = torch.tensor([model.actuator(f"robot/{n}").id for n in self.joint_names], device=self.device)
        elastic_joints = ROLL_JOINTS if elastic_joints is None else elastic_joints
        rotor_bodies = [f"{s}_virtual_ankle_rotor" for s in ("left", "right")] if rotor_bodies is None else rotor_bodies
        if len(elastic_joints) != len(rotor_bodies):
            raise ValueError("Each elastic joint needs one rotor body")
        self.roll_cols = torch.tensor([self.joint_names.index(n) for n in elastic_joints], device=self.device, dtype=torch.long)
        self.ankle_cols = torch.tensor([i for i, n in enumerate(self.joint_names) if "ankle" in n], device=self.device)
        self.rotor_bodies = torch.tensor([model.body(f"robot/{n}").id for n in rotor_bodies], device=self.device, dtype=torch.long)
        sim.expand_model_fields(("dof_armature", "body_inertia", "actuator_gainprm", "actuator_biasprm"))
        self.nominal_armature = sim.get_default_field("dof_armature")[self.dof_ids].to(self.device).clone()
        self.nominal_rotor = sim.get_default_field("body_inertia")[self.rotor_bodies].to(self.device).clone()
        self.nominal_kp = sim.get_default_field("actuator_gainprm")[self.ctrl_ids, 0].to(self.device).clone()
        self.nominal_kd = -sim.get_default_field("actuator_biasprm")[self.ctrl_ids, 2].to(self.device).clone()
        self.recompute_level = RecomputeLevel.set_const_0
        _mjlab_ensure_recompute_fields_expanded(env, self.recompute_level)
        delay_ms = tuple(map(float, delay_ms))
        if not delay_ms or not all(math.isfinite(x) and x >= 0 for x in delay_ms):
            raise ValueError("Delay choices must be finite nonnegative milliseconds")
        lag = torch.tensor(delay_ms, dtype=torch.float64) / (1000 * model.opt.timestep)
        if not torch.allclose(lag, lag.round(), atol=1e-6, rtol=0):
            raise ValueError("Delay must be an integer number of physics substeps")
        self.delay_lags = lag.round().long().to(self.device)
        # Public set_lags also updates fused builtin buffers. The custom roll
        # actuators have separate buffers; writing the SAME draw covers both.
        self.actuators = [a for a in asset.actuators if any(n in self.joint_names for n in a.target_names)]
        if not self.actuators:
            raise ValueError("No policy actuator delay group")
        for actuator in self.actuators:
            if actuator.cfg.delay_max_lag < int(self.delay_lags.max()) or actuator.cfg.delay_hold_prob != 1.0:
                raise ValueError("Actuator delay capacity/hold does not match reset-only DR")
        self.sampled_delay_lags = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if getattr(env, "_depsic_episode_delay_owner", None) is not None:
            raise ValueError("Only one reset component may own the policy actuator delay")
        env._depsic_episode_delay_owner = self

    def reset(self, env_ids):
        if not env_ids.numel():
            return
        from active_adaptation.envs.mdp.randomizations.common import _mjlab_recompute_constants

        n = len(env_ids)
        rows = env_ids[:, None]
        model = self.env.sim.model
        scales = torch.empty(n, len(self.joint_names), device=self.device).uniform_(*self.ranges[1])
        scales[:, self.ankle_cols] = torch.empty(n, len(self.ankle_cols), device=self.device).uniform_(*self.ranges[0])
        model.dof_armature[rows, self.dof_ids] = self.nominal_armature * scales
        model.body_inertia[rows, self.rotor_bodies] = self.nominal_rotor * scales[:, self.roll_cols, None]
        kp_scale = torch.empty_like(scales).uniform_(*self.ranges[2])
        ratio = torch.empty_like(scales).uniform_(*self.ranges[3])
        kp = self.nominal_kp * kp_scale
        kd = self.nominal_kd * kp_scale.sqrt() * ratio
        model.actuator_gainprm[rows, self.ctrl_ids, 0] = kp
        model.actuator_biasprm[rows, self.ctrl_ids, 1] = -kp
        model.actuator_biasprm[rows, self.ctrl_ids, 2] = -kd
        lags = self.delay_lags[torch.randint(len(self.delay_lags), (n,), device=self.device)]
        self.sampled_delay_lags[env_ids] = lags
        for actuator in self.actuators:
            actuator.set_lags(lags, env_ids)
        _mjlab_recompute_constants(self.env, self.recompute_level)
