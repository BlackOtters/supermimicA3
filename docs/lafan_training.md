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
The v8/e5 rigid model, losses, network, UTD, learning rates and SwanLab settings
are unchanged. The default remains 320,000 iterations and a checkpoint every 500.
Logs are saved under `logs/a3_supermimic_lafan_v8_4gpu_<timestamp>/`.

The dataset contract differs from the mixed corpus; this entrypoint starts from
scratch. To continue a later LAFAN checkpoint, use the generic `scripts/train.sh`
with that same LAFAN data path and `RESUME` pointing to its checkpoint.

Use `bash scripts/train_lafan.sh --check-config` or `--print-command` for a CPU-only
configuration check. Multi-GPU communication is checked on the training machine
when launching the job.
