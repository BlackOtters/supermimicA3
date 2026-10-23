"""Check the training GPU mapping and collectives without loading data or MuJoCo."""
import json
import os
from datetime import timedelta


def check_collectives(device):
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    # Exercise broadcast and the exact AVG reduction used for training gradients,
    # with small and larger messages over several repetitions.
    for count in (1, 262144):
        for step in range(3):
            x = torch.full((count,), rank+step, device=device, dtype=torch.float32)
            dist.broadcast(x, src=0)
            torch.testing.assert_close(x, torch.full_like(x, step), rtol=0, atol=0)
            x.fill_(rank+step)
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
            torch.testing.assert_close(x, torch.full_like(x, (world-1)/2+step), rtol=0, atol=0)
    dist.barrier()


def main():
    from a3_supermimic.scripts.train import prepare_process
    prepare_process()
    import torch
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or torch.cuda.device_count() != 1:
        raise RuntimeError("Use torchrun with one distinct GPU pinned per worker")
    torch.cuda.set_device(0)
    print("NCCL_PREFLIGHT_RANK "+json.dumps({
        "rank": int(os.environ["RANK"]), "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": torch.cuda.get_device_name(0), "nccl_version": torch.cuda.nccl.version(),
        "transport_env": {k: os.environ.get(k) for k in (
            "NCCL_P2P_DISABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_CUMEM_ENABLE",
            "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME")},
    }), flush=True)
    try:
        dist.init_process_group("nccl", device_id=torch.device("cuda:0"), timeout=timedelta(seconds=90))
        check_collectives("cuda:0")
        if dist.get_rank() == 0:
            print(f"NCCL_PREFLIGHT_OK ranks={dist.get_world_size()} broadcast+allreduce_avg", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
