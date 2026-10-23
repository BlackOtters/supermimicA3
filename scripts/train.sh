#!/usr/bin/env bash
# One process per GPU; v9 adds analytic termination, WM-only warmup and 6D feedback.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$repo/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python3}"; fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export RUN_NAME="${RUN_NAME:-a3_supermimic_v9_${NPROC_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export A3_SM_CACHE_ROOT="${A3_SM_CACHE_ROOT:-/tmp/a3_supermimic_${RUN_NAME}_$$}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# SDK0.10.2 expects a structured SWANLAB_PROJECT; use our own plain-string name.
unset SWANLAB_PROJECT
log_root="${LOG_ROOT:-$repo/logs}"
tracking_mode="${A3_SM_SWANLAB_MODE:-online}"
data_dir="${DATA_DIR:-$repo/data/library}"
recipe=(
  --run-name "$RUN_NAME" --log-root "$log_root" --seed 42
  --data.npz-dir "$data_dir" --env.num-envs "${ENVS_PER_GPU:-4096}"
  --env.replay-capacity "${REPLAY_CAPACITY:-1000}" --wm-trainer.batch-size "${WM_BATCH_PER_GPU:-2048}"
  --policy-trainer.batch-size "${ACTOR_BATCH_PER_GPU:-2048}"
  --wm-trainer.lr "${BASE_LR:-0.0002}" --policy-trainer.lr "${BASE_LR:-0.0002}" --policy-trainer.critic-lr "${BASE_LR:-0.0002}"
  --lr-world-size 1 --schedule.total-iterations "${TOTAL_ITERATIONS:-320000}"
  --schedule.log-interval 50 --schedule.checkpoint-interval 500
  --schedule.wm-only-updates "${WM_ONLY_UPDATES:-500}"
)
if [[ -n "${RESUME:-}" ]]; then
  recipe=(--resume "$RESUME" --run-name "$RUN_NAME" --log-root "$log_root"
          --data.npz-dir "$data_dir" --schedule.total-iterations "${TOTAL_ITERATIONS:-320000}")
fi
case "$tracking_mode" in
  online|offline) recipe+=(--swanlab.enabled --swanlab.project "${A3_SM_SWANLAB_PROJECT:-a3-supermimic}" --swanlab.mode "$tracking_mode") ;;
  disabled) recipe+=(--swanlab.no-enabled) ;;
  *) echo 'A3_SM_SWANLAB_MODE must be online, offline or disabled' >&2; exit 1 ;;
esac
mode="${1:-}"
if [[ "$mode" == --check-config || "$mode" == --print-command || "$mode" == --check-tracking || "$mode" == --check-distributed ]]; then shift; else mode=train; fi
if [[ "$mode" == --check-config ]]; then
  exec "$python_bin" -m a3_supermimic.scripts.check_config "${recipe[@]}" "$@"
fi
if [[ "$mode" == --print-command ]]; then
  printf '%q ' "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" -m a3_supermimic.scripts.train "${recipe[@]}" "$@"
  printf '\n'; exit 0
fi
if [[ "$mode" == --check-tracking ]]; then
  exec "$python_bin" -m a3_supermimic.scripts.check_tracking
fi
cd "$repo"
nvidia-smi
pgrep -af '[t]orchrun|[p]ython.*train' || true
if command -v tmux >/dev/null 2>&1; then tmux ls || true; fi
"$python_bin" - <<'PY'
import os, subprocess, torch
n = int(os.environ['NPROC_PER_NODE'])
if n < 1 or torch.cuda.device_count() < n:
    raise SystemExit(f'Need {n} visible GPUs; found {torch.cuda.device_count()}')
busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if busy:
    raise SystemExit('GPU compute processes already exist:\n'+busy)
print(f'GPU preflight OK: {n} processes; torch {torch.__version__}')
PY
if (( NPROC_PER_NODE > 1 )); then
  "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" -m a3_supermimic.scripts.check_distributed
fi
if [[ "$mode" == --check-distributed ]]; then exit 0; fi
if [[ "$tracking_mode" == online ]]; then
  "$python_bin" -m a3_supermimic.scripts.check_tracking
fi
"$python_bin" -m a3_supermimic.scripts.data validate "$data_dir"
mkdir -p "$log_root"
log_file="$log_root/${RUN_NAME}_launcher.log"
if [[ -n "${RESUME:-}" ]]; then log_file="$log_root/${RUN_NAME}_resume_$(date +%Y%m%d_%H%M%S)_launcher.log"; fi
if [[ -e "$log_file" ]]; then echo 'Choose a new RUN_NAME: launcher log exists' >&2; exit 1; fi
{
  echo "Start: $(date -Is); host=$(hostname); GPUs=${CUDA_VISIBLE_DEVICES:-all}; ranks=$NPROC_PER_NODE; seed=42"
  printf 'Command: '; printf '%q ' "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" -m a3_supermimic.scripts.train "${recipe[@]}" "$@"; printf '\n'
  "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" -m a3_supermimic.scripts.train "${recipe[@]}" "$@"
} 2>&1 | tee "$log_file"
