"""Measured simulator parameters and a differentiable terrain height lookup."""
import numpy as np
import torch
from torch import nn
from a3_supermimic.kinematics.rotations import heading_quat, quat_apply


class CompactSimContext:
    """G1's exact 33-value layout: foot friction, encoder bias, torso COM offset."""
    def __init__(self, env):
        self.env = env
        self.event = env._supermimic_upstream_reset
        self.torso = env.sim.mj_model.body('robot/torso_Link').id
        self.com_nominal = torch.tensor(env.sim.mj_model.body_ipos[self.torso].copy(),
                                        device=env.device, dtype=torch.float32)
        self.names = ['foot_friction', *(f'encoder_bias/{i}' for i in range(29)),
                      'torso_com/x', 'torso_com/y', 'torso_com/z']
        self.offset = torch.tensor([.75]+[0.]*32, device=env.device)
        self.scale = torch.tensor([.45]+[.01]*29+[.05]*3, device=env.device)

    def read(self):
        env = self.env
        friction = env.sim.model.geom_friction[:, self.event.geom_ids[0], :1].expand(env.num_envs, -1)
        bias = -env.input_managers['action'].offset[:, self.event.joint_ids]
        com = (env.sim.model.body_ipos[:, self.torso]-self.com_nominal).expand(env.num_envs, -1)
        return torch.cat((friction, bias, com), -1).clone()


class SimContext:
    def __init__(self, env):
        self.env = env
        self.asset = env.scene.articulations["robot"]
        self.layout = env.command_manager.layout
        self.policy_ids = env.input_managers["action"].joint_ids
        model = env.sim.mj_model
        robot_bodies = np.array([i for i in range(model.nbody) if model.body(i).name.startswith("robot/")])
        robot_geoms = np.array([i for i in range(model.ngeom) if model.geom_bodyid[i] in robot_bodies])
        robot_dofs = np.arange(6, model.nv)
        # Only robot geoms are randomized; a terrain friction floor is fixed by
        # the inherited randomization event and saved with the task config.
        self.fields = (
            ("body_mass", robot_bodies), ("body_inertia", robot_bodies), ("body_ipos", robot_bodies),
            ("geom_friction", robot_geoms), ("dof_armature", robot_dofs), ("dof_frictionloss", robot_dofs),
            ("actuator_gainprm", np.arange(model.nu)), ("actuator_biasprm", np.arange(model.nu)),
            ("tendon_stiffness", np.arange(model.ntendon)), ("tendon_damping", np.arange(model.ntendon)),
        )
        self.selections = []
        self.names = [*(f"default_offset/{i}" for i in range(29)), "is_elastic", "delay_substeps"]
        offsets = [torch.zeros(31, device=env.device)]
        scales = [torch.tensor([.01]*29 + [1., 2.], device=env.device)]
        for name, ids in self.fields:
            # PD needs only gain[0] and bias[1:3], not the unused ten coefficients.
            component = slice(0, 1) if name == "actuator_gainprm" else slice(1, 3) if name == "actuator_biasprm" else None
            default = np.asarray(getattr(model, name))[ids]
            if component is not None:
                default = default[..., component]
            value = torch.tensor(default.copy(), device=env.device, dtype=torch.float32).flatten()
            ids_t = torch.tensor(ids, device=env.device, dtype=torch.long)
            self.selections.append((name, ids_t, component))
            offsets.append(value)
            floor = .05 if name == "body_ipos" else .001 if name in ("body_inertia", "dof_armature") else 1.
            if name == "tendon_stiffness":
                floor = 1000.  # nominal mixed-asset springs are zero, sampled K reaches 2000 N m/rad
            scales.append(value.abs().clamp_min(floor))
            self.names.extend(f"{name}/{i}" for i in range(len(value)))
        # Origin is used by terrain queries, never exposed to the deployed actor.
        offsets.append(torch.zeros(3, device=env.device)); scales.append(torch.full((3,), 10., device=env.device))
        self.names.extend(("origin/x", "origin/y", "origin/z"))
        self.offset, self.scale = torch.cat(offsets), torch.cat(scales)

    def read(self):
        env = self.env
        dr = env._four_ankle_state
        parts = [env.input_managers["action"].offset[:, self.policy_ids],
                 dr.is_elastic[:, None].float(), dr.sampled_delay_lags[:, None].float()]
        for name, ids, component in self.selections:
            x = getattr(env.sim.model, name)
            # mjlab fields are either shared (1, ...) or expanded (N, ...).
            value = x[:, ids]
            if component is not None:
                value = value[..., component]
            parts.append(value.reshape(value.shape[0], -1).expand(env.num_envs, -1))
        parts.append(env.command_manager.origins)
        return torch.cat(parts, -1).clone()


class TerrainFeatures(nn.Module):
    """World heightmap queried around the predicted root, with gradients in xy/yaw.

    The physical simulator keeps the inherited rough terrain. The world model
    sees a 5x5 patch at 0.4 m spacing, interpolated from a 0.1 m height grid.
    """
    feature_dim = 25

    def __init__(self, heights, bounds):
        super().__init__()
        self.register_buffer("heights", heights[None, None])
        self.register_buffer("bounds", bounds)
        y, x = torch.meshgrid(torch.linspace(-.8, .8, 5, device=heights.device),
                              torch.linspace(-.8, .8, 5, device=heights.device), indexing="ij")
        self.register_buffer("patch", torch.stack((x.flatten(), y.flatten(), torch.zeros(25, device=heights.device)), -1))

    @classmethod
    def from_env(cls, env, resolution=.1):
        origins = env.command_manager.origins
        if env.terrain_type == "plane":
            return cls(torch.zeros(2, 2, device=env.device), torch.tensor([-1., -1., 1., 1.], device=env.device))
        # Cover the full mesh, including its border; no origin-dependent grid per rank.
        mesh = env.ground_mesh
        import warp as wp
        vertices = wp.to_torch(mesh.points)
        low, high = vertices[:, :2].min(0).values, vertices[:, :2].max(0).values
        nx, ny = (torch.ceil((high-low)/resolution).long()+1).tolist()
        ys, xs = torch.linspace(float(low[1]), float(high[1]), ny, device=env.device), torch.linspace(float(low[0]), float(high[0]), nx, device=env.device)
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((x, y, torch.full_like(x, float(vertices[:, 2].max())+10.)), -1)
        height = env.get_ground_height_at(points)
        if not torch.isfinite(height).all() or height.min() < -20:
            raise ValueError("Terrain height grid has missing rays; cannot condition the world model")
        return cls(height, torch.cat((low, high)))

    def forward(self, state, xi):
        leading = state.shape[:-1]
        s, context = state.reshape(-1, state.shape[-1]), xi.reshape(-1, xi.shape[-1])
        hq = heading_quat(s[:, 3:7])[:, None].expand(-1, 25, -1)
        p = s[:, None, :3] + context[:, None, -3:] + quat_apply(hq, self.patch[None].expand(len(s), -1, -1))
        grid = 2 * (p[..., :2] - self.bounds[:2]) / (self.bounds[2:] - self.bounds[:2]) - 1
        # One shared image, with all environments as query rows; no expanded image copies.
        z = torch.nn.functional.grid_sample(self.heights, grid[None], align_corners=True, padding_mode="border").reshape(len(s), 25)
        return ((z - context[:, -1:]) / .05).reshape(*leading, 25)
