#!/usr/bin/env bash
# Fresh four-GPU training on the filtered LAFAN-only packed subset.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_DIR="${DATA_DIR:-$repo/data/lafan_filtered_v2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RUN_NAME="${RUN_NAME:-a3_supermimic_lafan_v8_4gpu_$(date +%Y%m%d_%H%M%S)}"
if [[ "$NPROC_PER_NODE" != 4 ]]; then
  echo 'This LAFAN entrypoint requires NPROC_PER_NODE=4.' >&2; exit 1
fi
if [[ -n "${RESUME:-}" ]]; then
  echo 'This entrypoint starts a fresh LAFAN run; unset RESUME. Use train.sh to resume an existing LAFAN run.' >&2
  exit 1
fi
python_bin="${PYTHON:-$repo/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python3}"; fi
"$python_bin" - "$DATA_DIR/library.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
clips = m['clips']
if not clips or len(clips) != m['num_clips']:
    raise SystemExit('Empty or inconsistent LAFAN library')
if not m.get('filtering') or m.get('subset', {}).get('corpus') != 'lafan1':
    raise SystemExit('Expected a lafan1 subset of the already filtered library')
if not all(c['source'].split('/')[0] == 'lafan1' and c['name'].startswith('lafan1__') for c in clips):
    raise SystemExit('Library contains clips outside LAFAN1')
print(f"LAFAN data: {len(clips)} clips, {m['num_frames']} frames, {m['total_hours']:.3f} hours", file=sys.stderr)
PY
exec bash "$repo/scripts/train.sh" "$@"
