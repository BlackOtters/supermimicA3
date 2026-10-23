import json
import sys
from types import SimpleNamespace

from a3_supermimic.config import TrainConfig
from a3_supermimic.training.tracking import SwanLabLogger


def test_tracking_rank_zero_and_resume(tmp_path, monkeypatch):
    calls = []
    events = []
    class Settings:
        Terminal = SimpleNamespace
        def __init__(self, **kwargs):
            self.values = kwargs
    run = SimpleNamespace(id="test-run", url="https://example.invalid/run",
                          log=lambda values, step: events.append((step, values)),
                          finish=lambda state: events.append(state))
    sdk = SimpleNamespace(__version__="0.10.2", Settings=Settings,
                          init=lambda **kwargs: calls.append(kwargs) or run)
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    key = tmp_path / "credential"
    key.write_text("test-only-secret")
    monkeypatch.setenv("SWANLAB_API_KEY_FILE", str(key))
    cfg = TrainConfig()
    cfg.swanlab.enabled = True
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    worker = SwanLabLogger(cfg, log_dir, rank=1, world=4, contract_digest="abc")
    worker.log({"wm/loss": 1.}, 50)
    worker.finish()
    assert not calls and not events and not (log_dir / "swanlab.json").exists()
    logger = SwanLabLogger(cfg, log_dir, rank=0, world=4, contract_digest="abc")
    logger.log({"wm/loss": .2}, 100)
    logger.finish("crashed")
    assert calls[0]["config"]["distributed"]["global_envs"] == 16384
    assert calls[0]["public"] is False and calls[0]["resume"] == "never"
    assert "test-only-secret" not in json.dumps(calls[0]["config"])
    assert "test-only-secret" not in (log_dir / "swanlab.json").read_text()
    assert events == [(100, {"wm/loss": .2}), "crashed"]
    cfg.resume = log_dir / "ckpt_latest.pt"
    SwanLabLogger(cfg, log_dir, rank=0, world=4, contract_digest="abc")
    assert calls[-1]["id"] == "test-run" and calls[-1]["resume"] == "must"


def test_tracking_disabled_does_not_import_sdk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    logger = SwanLabLogger(TrainConfig(), tmp_path, rank=0, world=1, contract_digest="abc")
    logger.log({}, 1)
    logger.finish()
    assert not list(tmp_path.iterdir())


def test_tracking_error_keeps_cause_and_redacts_credentials(tmp_path, monkeypatch):
    from a3_supermimic.training.tracking import tracking_error_message
    key = tmp_path / 'credential'
    key.write_text('file-test-secret')
    monkeypatch.setenv('SWANLAB_API_KEY_FILE', str(key))
    monkeypatch.setenv('SWANLAB_API_KEY', 'env-test-secret')
    result = tracking_error_message(ValueError('project invalid: env-test-secret file-test-secret'))
    assert result == 'ValueError: project invalid: <redacted> <redacted>'
