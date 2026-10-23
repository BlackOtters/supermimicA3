"""Train or resume A3 SuperMimic. Pin each torchrun worker before importing CUDA."""
import os


def prepare_process():
    """Pin the worker before any CUDA import; shared with the communication check."""
    from pathlib import Path
    cache_root = os.environ.get("A3_SM_CACHE_ROOT")
    if cache_root:
        cache = Path(cache_root)/("rank-"+os.environ.get("LOCAL_RANK", "0"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_PATH"] = str(cache/"warp")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache/"inductor")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank = int(os.environ["LOCAL_RANK"])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = visible.split(",")
            if rank >= len(devices):
                raise ValueError("torchrun needs one distinct visible GPU per worker")
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        os.environ["SM_GPU_PINNED"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    prepare_process()
    import tyro
    from a3_supermimic.config import TrainConfig
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume")
    resume, _ = pre.parse_known_args()
    defaults = TrainConfig()
    if resume.resume is not None:
        import torch
        from pathlib import Path
        from a3_supermimic.scripts.evaluate import restore_config
        saved = torch.load(resume.resume, map_location="cpu", weights_only=False)
        from a3_supermimic.robot import check_contract_version
        check_contract_version(saved)
        defaults = restore_config(saved["config"])
        defaults.resume = Path(resume.resume)
    cfg = tyro.cli(TrainConfig, default=defaults)
    from a3_supermimic.training.loop import Trainer
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
