import math
import mujoco
from typing import cast

from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.mjlab.adapter import (
    MjlabSceneAdapter,
    MjlabSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class MjlabBackendEnv(_EnvBase):
    """MjLab backend env: scene/sim construction and viewer glue."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
        if self.sim.has_gui():
            self.sim.viewer.setup()
            self.sim.viewer.update()

    def setup_scene(self):
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.scene import Scene, SceneCfg
        from mjlab.utils.nan_guard import NanGuardCfg
        import mjlab.terrains as terrain_gen
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
        from mjlab.viewer import ViewerConfig
        from mjlab.utils.spec_config import CollisionCfg, _GEOM_ATTR_DEFAULTS

        def edit_spec(self: CollisionCfg, spec: mujoco.MjSpec):
            from mjlab.utils.spec import disable_collision
            from mjlab.utils.string import filter_exp, resolve_field

            self.validate()

            all_geoms: list[mujoco.MjsGeom] = spec.geoms
            all_geom_names = tuple(g.name for g in all_geoms)
            geom_subset = filter_exp(self.geom_names_expr, all_geom_names)

            resolved_fields = {
                name: resolve_field(getattr(self, name), geom_subset, default)
                for name, default in _GEOM_ATTR_DEFAULTS.items()
            }

            # raise error if any of the resolved fields are None
            if any(not len(field) for field in resolved_fields.values()):
                raise ValueError("Resolved fields cannot be empty")

            for i, geom_name in enumerate(geom_subset):
                geom = spec.geom(geom_name)

                geom.condim = resolved_fields["condim"][i]
                geom.contype = resolved_fields["contype"][i]
                geom.conaffinity = resolved_fields["conaffinity"][i]
                geom.priority = resolved_fields["priority"][i]

                CollisionCfg.set_array_field(geom.friction, resolved_fields["friction"][i])
                CollisionCfg.set_array_field(geom.solref, resolved_fields["solref"][i])
                CollisionCfg.set_array_field(geom.solimp, resolved_fields["solimp"][i])

                if resolved_fields["margin"][i] is not None:
                    geom.margin = resolved_fields["margin"][i]
                if resolved_fields["gap"][i] is not None:
                    geom.gap = resolved_fields["gap"][i]
                if resolved_fields["solmix"][i] is not None:
                    geom.solmix = resolved_fields["solmix"][i]

            other_geoms = ()
            if self.disable_other_geoms:
                other_geoms = set(all_geom_names).difference(geom_subset)
                for geom_name in other_geoms:
                    geom = spec.geom(geom_name)
                    disable_collision(geom)
        
        # replace the edit_spec method of CollisionCfg with our own
        CollisionCfg.edit_spec = edit_spec

        from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))
        if callable(asset_cfg):
            asset_cfg, sensors = asset_cfg(backend="mjlab")
        elif isinstance(asset_cfg, AssetCfg):
            sensors = tuple(sensor.mjlab() for sensor in asset_cfg.sensors_mjlab)
            asset_cfg = asset_cfg.mjlab()
        else:
            raise ValueError(
                "Asset configuration must be an instance of AssetCfg or callable, "
                f"got {type(asset_cfg)}"
            )
        terrain = self.cfg.get("terrain", "plane")
        self.terrain_type = terrain
        # A3 024 ships a calibrated floor as part of the released passive-foot
        # MJCF.  A second, coincident mjlab plane produces a different contact
        # model; replacing the XML floor with the generic plane changes its
        # tangential friction and contact compliance.  Keep the asset floor as
        # the sole physical ground.  Plane height queries and grid origins do
        # not require a TerrainEntity, so the normal plane-task API remains
        # unchanged.
        embedded_ground = self.cfg.robot.name in {
            "a3-depsic-024",
            "a3-depsic-024-serial",
        }
        if embedded_ground and terrain != "plane":
            raise ValueError(
                f"{self.cfg.robot.name} embeds its calibrated flat floor and "
                f"only supports terrain=plane, got {terrain!r}."
            )

        env_spacing = 2.5
        if embedded_ground:
            terrain_cfg = None
        elif terrain == "plane":
            terrain_cfg = TerrainEntityCfg(terrain_type="plane")
        elif terrain == "rough":
            from hydra.utils import instantiate

            terrain_cfg = TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    size=(5.0, 5.0),
                    border_width=20.0,
                    num_rows=8,
                    num_cols=8,
                    sub_terrains={
                        "boxes": terrain_gen.BoxRandomGridTerrainCfg(
                            proportion=1.0,
                            grid_width=0.5,
                            grid_height_range=(0.001, 0.005),
                            platform_width=0.5,
                        )
                    },
                    add_lights=False,
                ),
                env_spacing=env_spacing,
                num_envs=self.cfg.num_envs,
            )
            if self.cfg.get("terrain_generator") is not None:
                terrain_cfg.terrain_generator = instantiate(self.cfg.terrain_generator)
        else:
            raise ValueError(
                f"Unsupported terrain `{terrain}`. Expected one of: `plane`, `rough`."
            )

        if self.cfg.get("terrain_contact") is not None:
            if terrain_cfg is None:
                raise ValueError("terrain_contact cannot override an embedded ground")
            terrain_cfg.collisions = (CollisionCfg(
                geom_names_expr=(r".*",), disable_other_geoms=False,
                **dict(self.cfg.terrain_contact),
            ),)

        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=env_spacing,
            entities={"robot": asset_cfg},
            sensors=sensors,
            terrain=terrain_cfg,
        )
        scene = Scene(scene_cfg, device=str(self.device))
        nan_guard_cfg = self.cfg.sim.get("mjlab_nan_guard", {}) or {}
        if not hasattr(nan_guard_cfg, "get"):
            raise TypeError("task.sim.mjlab_nan_guard must be a mapping")
        nan_guard = NanGuardCfg(
            enabled=bool(nan_guard_cfg.get("enabled", False)),
            buffer_size=int(nan_guard_cfg.get("buffer_size", 100)),
            output_dir=str(nan_guard_cfg.get("output_dir", "/tmp/mjlab/nan_dumps")),
            max_envs_to_dump=int(nan_guard_cfg.get("max_envs_to_dump", 5)),
        )
        if nan_guard.enabled:
            print(
                "mjlab NanGuard enabled: "
                f"buffer_size={nan_guard.buffer_size}, output_dir={nan_guard.output_dir}"
            )
        sim = Simulation(
            num_envs=scene.num_envs,
            cfg=SimulationCfg(
                nconmax=200,
                # Closed-chain robots can require substantially more active
                # equality/contact constraints than the rigid-body default.
                njmax=int(self.cfg.sim.get("mjlab_njmax", 500)),
                contact_sensor_maxmatch=80,
                nan_guard=nan_guard,
                mujoco=MujocoCfg(
                    # Keep the mjlab integration step a task-level choice.
                    # Five milliseconds is the established default; assets
                    # that need a finer step must opt in explicitly.
                    timestep=float(self.cfg.sim.get("mjlab_physics_dt", 0.005)),
                    # Elastic transmissions with tiny finger inertias can make
                    # Warp's float32 Newton factorization fail on self-contact.
                    # Opt in per task; keep all other tasks on their old solver.
                    solver=self.cfg.sim.get("mjlab_solver", "newton"),
                    iterations=int(self.cfg.sim.get("mjlab_solver_iterations", 10)),
                    ls_iterations=20,
                    # Warp's MULTICCD path rejects the non-zero margins used
                    # by the fitted passive-foot A3 024 mesh.  Keep this
                    # compatibility switch scoped to fitted-foot assets; existing
                    # G1/A3 tasks retain their prior global CCD contract.
                    disableflags=(
                        ("multiccd",)
                        if self.cfg.robot.name in {
                            "a3-depsic-024", "a3-depsic-024-serial",
                            "a3-ultra-depsic-v1-s2r-plus",
                        }
                        else ()
                    ),
                ),
            ),
            model=scene.compile(),
            device=str(self.device),
        )

        if "mjlab_solver" in self.cfg.sim:
            print(f"mjlab solver={sim.cfg.mujoco.solver}, "
                  f"iterations={sim.cfg.mujoco.iterations}")
        scene.initialize(sim.mj_model, sim.model, sim.data)
        sim.create_graph()

        self.scene = MjlabSceneAdapter(scene, sim)
        viewer_cfg = self._make_viewer_cfg(ViewerConfig)
        viewer = MjLabViewer(self, sim) if not self.headless else None
        self.sim = MjlabSimAdapter(sim, viewer, viewer_cfg=viewer_cfg, scene=scene)

    def _make_viewer_cfg(self, viewer_config_cls):
        lookat = tuple(float(v) for v in self.cfg.viewer.lookat)
        eye = tuple(float(v) for v in self.cfg.viewer.eye)
        resolution = tuple(int(v) for v in self.cfg.viewer.resolution)

        delta = [eye_i - lookat_i for eye_i, lookat_i in zip(eye, lookat)]
        distance = math.sqrt(sum(v * v for v in delta))
        if distance <= 1e-8:
            distance = 5.0
            azimuth = 90.0
            elevation = -45.0
        else:
            planar = math.hypot(delta[0], delta[1])
            azimuth = math.degrees(math.atan2(delta[1], delta[0]))
            elevation = -math.degrees(math.atan2(delta[2], planar))

        return viewer_config_cls(
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            width=resolution[0],
            height=resolution[1],
            env_idx=0,
            max_extra_envs=max(0, self.cfg.num_envs - 1),
        )


__all__ = ["MjlabBackendEnv"]
