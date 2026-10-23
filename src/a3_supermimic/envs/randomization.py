"""Upstream reset DR adapted to A3's action-offset and collision-geometry APIs."""
import re
import math
import torch
from active_adaptation.envs.mdp.randomizations.base import Randomization
from a3_supermimic.joint_names import POLICY_JOINTS


class upstream_rigid_reset(Randomization, namespace="a3_supermimic"):
    supported_backends = ("mjlab",)

    def __init__(self, env, friction_range=(.3, 1.2), bias_range=(-.01, .01)):
        super().__init__(env)
        self.friction_range = tuple(friction_range)
        self.bias_range = tuple(bias_range)
        if (not all(map(math.isfinite, (*friction_range, *bias_range)))
                or not 0 < friction_range[0] <= friction_range[1]
                or bias_range[0] > bias_range[1]):
            raise ValueError("Invalid upstream friction or encoder bias range")
        asset = env.scene.articulations["robot"]
        self.joint_ids = torch.tensor([asset.joint_names.index(n) for n in POLICY_JOINTS], device=self.device)
        model = env.sim.mj_model
        robot_geoms = set(asset.indexing.geom_ids.tolist())
        geoms = []
        sides = set()
        for g in robot_geoms:
            name = model.body(int(model.geom_bodyid[g])).name.split("/")[-1]
            if re.fullmatch(r"(left|right)_(ankle_roll|foot_forefoot|foot_toe)_Link", name) and (
                    model.geom_contype[g] or model.geom_conaffinity[g]):
                geoms.append(g)
                sides.add(name.split('_')[0])
        if sides != {'left', 'right'}:
            raise ValueError('Upstream friction must match collision geometries on both A3 feet')
        self.geom_ids = torch.tensor(sorted(geoms), device=self.device)
        env.sim.expand_model_fields(("geom_friction",))
        ground = [g for g in range(model.ngeom) if g not in robot_geoms and (
            model.geom_contype[g] or model.geom_conaffinity[g])]
        if not ground or any(model.body_weldid[model.geom_bodyid[g]] != 0 for g in ground):
            raise ValueError("Upstream foot friction requires static world-welded terrain")
        if any(model.geom_priority[g] > model.geom_priority[r] for g in ground for r in geoms):
            raise ValueError("Ground priority must not override foot friction")
        # A3 uses equal contact priorities: MuJoCo's max rule must not clamp low foot samples.
        equal_priority = [g for g in ground if any(model.geom_priority[g] == model.geom_priority[r] for r in geoms)]
        if equal_priority:
            ground_ids = torch.tensor(equal_priority, device=self.device, dtype=torch.long)
            env.sim.model.geom_friction[:, ground_ids, 0] = env.sim.model.geom_friction[:, ground_ids, 0].clamp_max(friction_range[0])
            model.geom_friction[equal_priority, 0] = model.geom_friction[equal_priority, 0].clip(max=friction_range[0])
        env._supermimic_upstream_reset = self

    def reset(self, env_ids):
        n = len(env_ids)
        if not n:
            return
        bias = torch.empty(n, len(self.joint_ids), device=self.device).uniform_(*self.bias_range)
        # Upstream observes q+bias and applies target-bias. A3's stored offset has opposite sign.
        self.env.input_managers['action'].offset[env_ids[:, None], self.joint_ids] = -bias
        friction = torch.empty(n, 1, device=self.device).uniform_(*self.friction_range)
        self.env.sim.model.geom_friction[env_ids[:, None], self.geom_ids, 0] = friction
