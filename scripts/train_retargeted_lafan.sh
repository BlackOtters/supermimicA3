#!/usr/bin/env bash
# Ultra29 adaptation of tianshuwu/supermimic-retargeted ff69a0e, fresh LAFAN run.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RUN_NAME="${RUN_NAME:-a3_supermimic_lafan_v10_retargeted_4gpu_$(date +%Y%m%d_%H%M%S)}"
# Match the published TWO-card run's global 8192 envs / 4096 batch on four cards.
export ENVS_PER_GPU="${ENVS_PER_GPU:-2048}"
export WM_BATCH_PER_GPU="${WM_BATCH_PER_GPU:-1024}"
export ACTOR_BATCH_PER_GPU="${ACTOR_BATCH_PER_GPU:-1024}"
export REPLAY_CAPACITY=512 WM_ONLY_UPDATES=0
# Trainer applies sqrt(4); effective LR is 2e-4*sqrt(2), as in their two-GPU recipe.
export BASE_LR=0.0001414213562373095
exec bash "$repo/scripts/train_lafan.sh" "$@" \
  --env.anchor-obs --env.anchor-position --env.collision-profile retargeted \
  --env.waist-soft-limit-factor 0.9 --env.solver-iterations 10 \
  --world-model.termination-mode learned --tf32
