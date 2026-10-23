# Source provenance

This repository is a standalone export of the A3 training implementation on
2026-10-23. The source workspace's Git base was
`502c0876629ee22fd7a06ab50e9627a219d4e76f`; the exported A3 implementation also
included uncommitted changes. `source_origin.json` records per-file SHA256 values
of the original SuperMimic implementation, so the base commit is not presented
as a complete description of this snapshot.

## Components

| Component | Source |
|---|---|
| World model, imagination, actor/critic, analytic rewards | [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), adaptation lineage `66f44a4f8f2c8dd31d00434c80e979eea2572622` |
| e5 comparison | Same upstream repository at `d9be385b3eeb13a638fb33fb200d3bfd9be9839f` |
| Simulation framework | A3 workspace's modified [active-adaptation](https://github.com/btx0424/active-adaptation) runtime |
| A3 task/actuator adapters | A3 workspace's modified [MimicLite](https://github.com/EGalahad/mimic-lite) runtime |
| A3 Ultra model | [AgibotTech/A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model), local checkout at `b8c892347a61cc14d4ca3c21b3d54d64ab95c979` |

Only the Python dependency closure needed by SuperMimic and its tests is bundled.
Unrelated robot assets, project discovery metadata, experiments, training data,
checkpoints, credentials and the original Git history are excluded.
Robot files have their own hashes in `src/a3_supermimic/assets/asset_sha256.json`.

## Standalone changes

1. `task_config()` loads the resolved A3 base task from the packaged
   `assets/task_base.yaml` instead of composing another workspace's Hydra tree.
   Fields replaced by SuperMimic are empty in that base snapshot. Final training
   and evaluation task dictionaries are checked against the original workspace.
2. The default Ultra model path points inside the installed package.
   `DEPSIC_A3_ULTRA_MODEL_ROOT` remains an explicit override.
3. Dependency package initializers register only the required A3 components;
   unrelated algorithm registration and automatic workspace discovery are disabled.
4. Installation, launcher, configuration checks and a compact-rigid smoke test
   are new portable entry points. The distributed-launch test targets this launcher.

These changes do not alter reward formulas, reset/sampling rules, network
architecture, optimizer math, robot parameters or the four-GPU recipe.
The latest source includes the SwanLab environment-variable fix and the
`mjlab_njmax >= 1024` contact-buffer capacity fix. Contract version remains 8;
the task hash still distinguishes capacity/configuration changes.

## Attribution and licensing

The SuperMimic-derived portions retain Yuxi Liu's MIT notice in
[`licenses/SuperMimic-MIT.txt`](licenses/SuperMimic-MIT.txt).
Third-party framework code and robot assets retain their original ownership and
terms; that MIT notice does not relicense them. This export does not assign a
new blanket license to the entire collection. Refer to the original component
repositories for their distribution terms.
