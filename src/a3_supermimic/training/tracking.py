"""Optional rank-zero SwanLab metrics; credentials never enter TrainConfig."""
from dataclasses import asdict
import json
import os
from pathlib import Path


def load_tracking_credentials(mode):
    if mode not in ("online", "offline"):
        raise ValueError("SwanLab mode must be online or offline")
    if mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
        credential = os.environ.get("SWANLAB_API_KEY_FILE")
        if credential:
            os.environ["SWANLAB_API_KEY"] = Path(credential).read_text().strip()
        if not os.environ.get("SWANLAB_API_KEY"):
            raise RuntimeError("Set SWANLAB_API_KEY_FILE or SWANLAB_API_KEY before online training")


def tracking_settings():
    import swanlab
    return swanlab.Settings(interactive=False,
                           terminal=swanlab.Settings.Terminal(proxy_type="none"))


def tracking_error_message(error):
    """Keep actionable SDK error text without exposing the supplied credential."""
    message = f"{type(error).__name__}: {error}"
    secrets = [os.environ.get("SWANLAB_API_KEY", "")]
    credential = os.environ.get("SWANLAB_API_KEY_FILE")
    if credential:
        try:
            secrets.append(Path(credential).read_text().strip())
        except OSError:
            pass
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


class SwanLabLogger:
    def __init__(self, cfg, log_dir: Path, *, rank: int, world: int, contract_digest: str):
        self.run = None
        if rank != 0 or not cfg.swanlab.enabled:
            return
        import swanlab

        load_tracking_credentials(cfg.swanlab.mode)
        config = json.loads(json.dumps(asdict(cfg), default=str))
        config["distributed"] = {
            "world_size": world, "global_envs": cfg.env.num_envs * world,
            "global_actor_batch": cfg.policy_trainer.batch_size * world,
            "global_wm_batch": cfg.wm_trainer.batch_size * world,
            "loss_metrics": "rank 0; episode return aggregates all ranks",
        }
        config["contract_sha256"] = contract_digest
        metadata = log_dir / "swanlab.json"
        previous = json.loads(metadata.read_text()) if cfg.resume and metadata.exists() else {}
        self.run = swanlab.init(
            project=cfg.swanlab.project, workspace=cfg.swanlab.workspace,
            name=cfg.run_name, mode=cfg.swanlab.mode, public=False,
            log_dir=str(log_dir / "swanlog"), config=config,
            id=previous.get("id"), resume="must" if previous else "never",
            settings=tracking_settings(),
        )
        metadata.write_text(json.dumps({"id": self.run.id, "url": self.run.url,
                                       "project": cfg.swanlab.project, "mode": cfg.swanlab.mode,
                                       "sdk_version": swanlab.__version__}, indent=2) + "\n")
        print(f"SwanLab: {self.run.url or metadata}", flush=True)

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def finish(self, state="success"):
        if self.run is not None:
            self.run.finish(state=state)
