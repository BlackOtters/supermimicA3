"""Worker pinning must happen before Torch/Warp initialize CUDA."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("visible,rank,expected", [("6,2,4", "1", "2"), ("GPU-a,GPU-b", "0", "GPU-a")])
def test_worker_setup_before_cuda_import(tmp_path, visible, rank, expected):
    env = {**os.environ, "WORLD_SIZE": "3", "LOCAL_RANK": rank,
           "CUDA_VISIBLE_DEVICES": visible, "A3_SM_CACHE_ROOT": str(tmp_path)}
    code = '''
import os, sys
from a3_supermimic.scripts.train import prepare_process
assert "torch" not in sys.modules and "warp" not in sys.modules
prepare_process()
assert "torch" not in sys.modules and "warp" not in sys.modules
assert os.environ["CUDA_VISIBLE_DEVICES"] == sys.argv[1]
assert os.environ["SM_GPU_PINNED"] == "1"
from pathlib import Path
assert Path(os.environ["WARP_CACHE_PATH"]).parent.name == "rank-"+os.environ["LOCAL_RANK"]
'''
    result = subprocess.run([sys.executable, "-c", code, expected], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize("probe_exit,check_only", [(19, False), (0, True)])
def test_launcher_stops_before_data_and_training(tmp_path, probe_exit, check_only):
    repo = Path(__file__).resolve().parents[1]
    launcher = repo/"scripts/train.sh"
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    for name in ("nvidia-smi", "pgrep", "tmux", "df"):
        path = bin_dir/name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    fake_python = bin_dir/"fake_python"
    fake_python.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["CALL_LOG"], "a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "cumem_host": os.environ.get("NCCL_CUMEM_HOST_ENABLE")})+"\\n")
if "a3_supermimic.scripts.check_distributed" in sys.argv:
    sys.exit(int(os.environ["PROBE_EXIT"]))
if sys.argv[1] == "-":
    sys.stdin.read()
''')
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir)+os.pathsep+os.environ["PATH"],
           "PYTHON": str(fake_python), "CALL_LOG": str(tmp_path/"calls.jsonl"),
           "PROBE_EXIT": str(probe_exit), "LOG_ROOT": str(tmp_path/"logs"), "RUN_NAME": "check"}
    env.pop("NCCL_CUMEM_HOST_ENABLE", None)
    result = subprocess.run(["bash", str(launcher)]+(["--check-distributed"] if check_only else []),
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == probe_exit, result.stdout+result.stderr
    import json
    calls = [json.loads(line) for line in (tmp_path/"calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2  # GPU inventory then communication; no dataset/model/training.
    assert "a3_supermimic.scripts.check_distributed" in calls[1]["args"]
    assert all(call["cumem_host"] == "0" for call in calls)
