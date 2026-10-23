import torch
import subprocess
import sys
from omegaconf import OmegaConf
from a3_supermimic.config import TrainConfig
from a3_supermimic.robot import POLICY_JOINTS


def legacy_config():
    cfg = TrainConfig()
    cfg.env.representation = 'full'
    cfg.env.randomization_profile = 'a3'
    cfg.env.evaluation_physics = 'mixed'
    return cfg


def test_config_resolves_without_workstation_project_manifest(tmp_path):
    code = '''
from pathlib import Path
import active_adaptation.project_loading.plugin as plugin
plugin.PROJECTS_FILE = Path("/nonexistent/a3_sm_projects.json")
from a3_supermimic.config import TrainConfig
from a3_supermimic.envs.simulator import task_config
task = task_config(TrainConfig(), ["library.json"])
assert task.robot.name == "a3-supermimic-compact-rigid"
assert task.sim.mjlab_solver == "newton"
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_inherits_selected_a3_physics_without_motion_or_amp_dependencies(monkeypatch):
    for name in ("DEPSIC_A3_V1_MOTION_PATH", "DEPSIC_SMPL_PATH", "DEPSIC_A3_ULTRA_AMP_CACHE"):
        monkeypatch.delenv(name, raising=False)
    from a3_supermimic.envs.simulator import task_config
    cfg = legacy_config()
    task = task_config(cfg, ["future_data.npz"])
    assert task.robot.name == "a3-ultra-four-ankle-mixed"
    assert task.terrain == "plane"
    assert "terrain_generator" not in task
    assert task.sim.step_dt == .02 and task.sim.mjlab_physics_dt == .005
    assert task.sim.mjlab_solver == "newton"
    assert task.sim.mjlab_njmax >= 1024
    assert "perturb_root_vel" not in task.randomization
    assert cfg.env.reset_noise
    assert cfg.env.future_frames == 10
    assert cfg.env.termination.height_threshold == .25
    assert cfg.env.termination.tilt_threshold == .8
    assert task.randomization.four_ankle_dr.rigid_probability == .25
    assert list(task.randomization.four_ankle_dr.delay_ms) == [0., 5., 10.]
    assert task.randomization.s2r_joint_params.randomize_armature is False
    assert set(task.input.action.action_scaling) == set(POLICY_JOINTS)
    assert len(task.observation) == len(task.reward) == len(task.termination) == 0
    OmegaConf.to_container(task, resolve=True)
    from mimic_lite.depsic_a3_contract import A3_POLICY_JOINT_NAMES
    assert POLICY_JOINTS == A3_POLICY_JOINT_NAMES


def test_training_topology_override_preserves_other_randomization():
    from a3_supermimic.envs.simulator import task_config
    cfg = legacy_config()
    expected = OmegaConf.to_container(task_config(cfg, ["future_data.npz"]).randomization, resolve=True)
    for label, probability in (("rigid", 1.), ("elastic", 0.), ("mixed", .25)):
        cfg.env.evaluation_physics = label
        actual = OmegaConf.to_container(task_config(cfg, ["future_data.npz"]).randomization, resolve=True)
        expected["four_ankle_dr"]["rigid_probability"] = probability
        assert actual == expected


def test_eval_freezes_randomization_but_keeps_elastic_dynamics():
    from a3_supermimic.envs.simulator import task_config
    cfg = legacy_config(); cfg.env.domain_randomization = False
    for label, probability in (("rigid", 1.), ("elastic", 0.)):
        cfg.env.evaluation_physics = label
        task = task_config(cfg, ["future_data.npz"])
        assert set(task.randomization) == {"four_ankle_dr", "fixed_contact_friction"}
        assert task.randomization.four_ankle_dr.rigid_probability == probability
        assert list(task.randomization.four_ankle_dr.kp_scale) == [1., 1.]
        assert list(task.randomization.fixed_contact_friction.static_friction_range) == [1., 1.]


def test_upstream_rigid_has_only_upstream_randomization():
    import pytest
    from a3_supermimic.envs.simulator import task_config
    cfg = legacy_config()
    cfg.env.randomization_profile = "upstream_rigid"
    with pytest.raises(ValueError, match="requires evaluation_physics=rigid"):
        task_config(cfg, ["future_data.npz"])
    cfg.env.evaluation_physics = "rigid"
    task = task_config(cfg, ["future_data.npz"])
    assert set(task.randomization) == {"four_ankle_dr", "upstream_torso_com", "upstream_reset"}
    dr = task.randomization.four_ankle_dr
    assert dr.rigid_probability == 1.
    assert list(dr.delay_ms) == [0.]
    for name in ("ankle_inertia_scale", "armature_scale", "kp_scale", "damping_ratio_scale"):
        assert list(dr[name]) == [1., 1.]
    assert list(task.randomization.upstream_reset.friction_range) == [.3, 1.2]
    assert list(task.randomization.upstream_reset.bias_range) == [-.01, .01]
    assert list(task.randomization.upstream_torso_com.body_names) == ["torso_Link"]
    assert list(task.randomization.upstream_torso_com.x) == [-.025, .025]
    assert list(task.randomization.upstream_torso_com.y) == [-.05, .05]
    assert list(task.randomization.upstream_torso_com.z) == [-.05, .05]
    cfg.env.domain_randomization = False
    task = task_config(cfg, ["future_data.npz"])
    assert set(task.randomization) == {"four_ankle_dr", "upstream_reset"}
    assert list(task.randomization.upstream_reset.friction_range) == [1., 1.]
    assert list(task.randomization.upstream_reset.bias_range) == [0., 0.]
    assert list(task.randomization.four_ankle_dr.delay_ms) == [0.]


def test_terrain_features_have_xy_gradient_and_normalizers_are_checkpointed(robot):
    from a3_supermimic.envs.context import TerrainFeatures
    from a3_supermimic.world_model.state_spec import StateSpec, StateSpecCfg
    _, layout, _ = robot
    field = torch.arange(100, dtype=torch.float32).reshape(10, 10)*.001
    terrain = TerrainFeatures(field, torch.tensor([-2., -2., 2., 2.]))
    state = torch.zeros(2, layout.state_dim, requires_grad=True)
    with torch.no_grad(): state[:, 3] = 1.
    xi = torch.zeros(2, 34)
    feature = terrain(state, xi)
    feature.sum().backward()
    assert feature.shape == (2, 25)
    assert state.grad[:, :2].abs().sum() > 0
    assert torch.isfinite(state.grad).all()
    spec = StateSpec(StateSpecCfg(), layout, torch.arange(34).float(), torch.ones(34)*2, "cpu")
    restored = StateSpec(StateSpecCfg(), layout, torch.zeros(34), torch.ones(34), "cpu")
    restored.load_state_dict(spec.state_dict())
    torch.testing.assert_close(spec.normalize_xi(xi), restored.normalize_xi(xi), atol=0, rtol=0)
