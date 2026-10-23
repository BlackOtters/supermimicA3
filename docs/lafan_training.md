# Filtered LAFAN-only training

The LAFAN subset is selected from the already screened v2 motion library.
It retains 32 clips (16 original, 16 mirrored), 293,758 frames at 50 Hz,
or 1.632 hours. AMASS and BONES clips are excluded. Coordinates, velocities,
clip lengths and existing filtering decisions are preserved exactly.

After installing this repository, prepare a separate packed subset once:

```bash
python scripts/subset_corpus.py /path/to/filtered_v2/library \
  data/lafan_filtered_v2 --corpus lafan1
```

The command validates the parent checksums, repacks only the selected clips,
checks every output slice against its parent, and validates the new library.
It refuses to overwrite an existing destination. The output includes a source
manifest hash and a `subset_report.json` provenance report.

Launch a fresh four-GPU run:

```bash
export SWANLAB_API_KEY_FILE=/path/to/private/swanlab_api_key
env -u RESUME \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  ENVS_PER_GPU=4096 ACTOR_BATCH_PER_GPU=2048 WM_BATCH_PER_GPU=2048 \
  bash scripts/train_lafan.sh
```

The entrypoint defaults to `data/lafan_filtered_v2` inside this repository,
verifies that every clip belongs to filtered LAFAN, and uses a distinct run name.
`DATA_DIR`, `PYTHON` and `LOG_ROOT` remain explicit environment overrides.
The v9 recipe retains the e5 rigid model, rewards, encoder architecture, UTD,
learning rates and SwanLab settings, with three changes:

- Goal feedback is the reference torso orientation relative to the robot torso,
  encoded as the first two rotation-matrix columns, flattened in row-major order.
  The former three position-error inputs are removed: goal width is now 686.
  Simulator and imagination share the same implementation and +/-0.05 noise.
- Imagination uses analytic soft termination: with absolute root height error z
  and gravity-z difference t, `p = 1 - sigmoid((0.25-z)/0.02) * sigmoid((0.8-t)/0.04)`.
  The implementation uses log-sigmoid/expm1 to retain precision at small p.
  Real simulation keeps the hard thresholds; timeouts are still nonterminal.
  The learned termination MLP and its BCE loss are removed.
- After 40 environment steps of replay filling, the first 500 WM update rounds
  freeze actor and critic. Each round has 10 WM minibatches, so this is 5,000 WM
  optimizer steps per rank. Real data collection continues at UTD 0.4; policy
  updates start with round 501. These are not 500 environment steps.

The soft-termination widths and WM-only warmup follow the design described in
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), while the torso
anchor and placement in the goal encoder retain this repository's architecture.
The default remains 320,000 environment iterations and a checkpoint every 500.
Logs are saved under `logs/a3_supermimic_lafan_v9_4gpu_<timestamp>/`.

SwanLab records `train/wm_only`, `train/wm_only_remaining_updates`,
`train/updates` (WM rounds), and `train/policy_updates`. Policy metrics start only
after the warmup. The completed counters are restored from v9 checkpoints.

The dataset contract differs from the mixed corpus; this entrypoint starts from
scratch. v8 checkpoints also have an incompatible input/termination contract.
To continue a later v9 LAFAN checkpoint, use the generic `scripts/train.sh`
with that same LAFAN data path and `RESUME` pointing to its checkpoint.

Use `bash scripts/train_lafan.sh --check-config` or `--print-command` for a CPU-only
configuration check. Multi-GPU communication is checked on the training machine
when launching the job.
