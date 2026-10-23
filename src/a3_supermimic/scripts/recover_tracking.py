"""Re-upload durable JSONL metrics without touching a running trainer or its run state."""
import argparse
import json
from pathlib import Path
import time

from a3_supermimic.training.tracking import load_tracking_credentials, tracking_settings


def metric_rows(stream):
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.endswith("\n"):
            stream.seek(position)
            return
        row = json.loads(line)
        step = row.pop("iteration")
        yield step, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--idle-timeout", type=float, default=1800.)
    args = parser.parse_args()
    import swanlab
    load_tracking_credentials("online")
    cfg = json.loads((args.run_dir / "config.json").read_text())
    original = json.loads((args.run_dir / "swanlab.json").read_text())
    metadata = args.run_dir / "tracking_recovery.json"
    previous = json.loads(metadata.read_text()) if metadata.exists() else {}
    run = swanlab.init(
        project=original["project"], workspace=cfg["swanlab"].get("workspace"),
        name=cfg["run_name"] + "_metrics_recovery", mode="online", public=False,
        id=previous.get("id"), resume="must" if previous else "never",
        log_dir=str(args.run_dir / "swanlog_recovery"),
        config={**cfg, "metrics_source_run": original["id"],
                "metrics_recovery": "JSONL replay; no new training"},
        settings=tracking_settings(),
    )
    metadata.write_text(json.dumps({"id": run.id, "url": run.url,
                                   "source_run": original["id"]}, indent=2) + "\n")
    print(f"Metrics recovery: {run.url}", flush=True)
    last_step = -1
    last_update = time.monotonic()
    try:
        with (args.run_dir / "metrics.jsonl").open() as stream:
            while True:
                count = 0
                for step, values in metric_rows(stream):
                    if step <= last_step:
                        continue
                    run.log(values, step=step)
                    last_step = step
                    count += 1
                if count:
                    last_update = time.monotonic()
                    print(f"Queued {count} rows, through step {last_step}", flush=True)
                if not args.follow or time.monotonic() - last_update >= args.idle_timeout:
                    break
                time.sleep(10)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
