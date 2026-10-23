# Validation

## v9 update, 2026-10-23

- CPU suite: **59 passed** in 23.42 s, including analytic soft-termination
  threshold/gradient checks, 6D feedback parity and translation independence,
  the 500/501 WM-update boundary, frozen actor/critic and checkpoint counters.
- Compiled CUDA smoke on local RTX 4090: passed with 8 environments and compact
  test networks. Saved mid-WM-only warmup, resumed through the transition, then
  evaluated and exported TorchScript. 100 environment iterations, 8 WM rounds,
  5 warmup rounds and 3 actor/critic updates; goal width 686, contract 9.
- CPU four-rank configuration: 500 WM-only rounds / 5,000 optimizer steps per
  rank, global 16,384 environments and 8,192 actor/WM batch; effective LR 4e-4.
- An initial combined CPU test invocation encountered a native segmentation
  fault during import/garbage collection. The isolated v9 checks, configuration
  check, full-suite rerun and compiled train/resume/eval all completed afterward.

The smoke uses a synthetic standing fixture and shortened warmup; it verifies
execution, not tracking quality. No four-GPU production run was launched for
this update. See [LAFAN v9 configuration](docs/lafan_training.md).

## Original v8 standalone export

Validated on 2026-10-23. These checks establish execution and configuration
parity, not reproduction of the upstream G1 tracking scores.

| Check | Result |
|---|---|
| CPU regression suite | 50 passed in 22.78 s |
| Final train/eval task dictionaries | Exactly equal to the source workspace |
| Bundled robot assets | All 75 files match their recorded SHA256 |
| Wheel installation outside the workspace | Pass; 125 internal modules came from installed site-packages |
| Compiled CUDA training | Pass; 8 environments, compact test networks, synthetic data |
| Checkpoint save/restore and continued updates | Pass; exact restored policy weights, 100 total iterations / 8 updates |
| Rigid evaluation and TorchScript export | Pass; outputs produced |
| Four-rank launcher configuration | 16,384 global environments, 8,192 actor/WM batch, effective initial LR 4e-4 |
| Pinned dependency resolution | Pass; pip dry-run with installed packages ignored, using PyPI |

Reproduction commands after installation:

```bash
python -m pytest tests -q --disable-warnings
bash scripts/train.sh --check-config
python -m pip install --dry-run --ignore-installed -r requirements.txt --index-url https://pypi.org/simple
# After checking that a GPU is idle:
python scripts/smoke.py --output /tmp/supermimic_a3_smoke_unique --compile
```

The GPU check ran on one RTX 4090 using an installed wheel from `/tmp`, with
`PYTHONPATH` and the external model override unset. The test environment inherited
ML dependencies from an existing Python 3.11 environment; it was not a fresh
full CUDA dependency installation. The exact training-runtime model fingerprint
was `1a89cba59f5762955fce81e4d9088d5307f688093c0006f5808f92be062294ee`.
The bare-model CPU probe has a different fingerprint before runtime actuators
are attached; that probe also matched the original workspace.

The generated standing fixture is explicitly marked synthetic. It terminated
at 80/128 frames with normalized return 0.342 after only eight optimizer updates;
this is an execution test, not evidence of successful tracking. Machine-readable
results are in [validation_smoke.json](validation_smoke.json).

A four-GPU training run was not started for this export. The original recipe was
previously capacity-tested on one 48 GB DSW GPU with 4096 environments, actor/WM
batch 2048, the full replay buffer and complete updates: whole-device peak
34363 MiB. That is historical evidence for the source recipe, not a new four-GPU
benchmark. The launcher performs an actual NCCL preflight on the destination.
