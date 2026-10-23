#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3.11}"
"$python_bin" -m venv "$repo/.venv"
"$repo/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$repo/.venv/bin/python" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$repo/.venv/bin/python" -m pip install -e "$repo" pytest==9.1.1
"$repo/.venv/bin/python" -m a3_supermimic.scripts.check_config
