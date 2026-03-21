"""Minimal DDP smoke test using torch.multiprocessing.spawn (no torchrun).

Instead of torchrun's TCP-store rendezvous (which segfaults on this cluster),
we use mp.spawn + file-based init_method to avoid c10d_rendezvous entirely.
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )

    print(
        f"[rank {rank}/{world_size}] device=cuda:{rank}  "
        f"gpu={torch.cuda.get_device_name(rank)}",
        flush=True,
    )

    # simple all-reduce to verify communication
    t = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(t)
    expected = sum(range(world_size))
    print(
        f"[rank {rank}] all_reduce result = {t.item():.0f}  (expected {expected})",
        flush=True,
    )

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs", flush=True)
    if world_size < 2:
        print("Need at least 2 GPUs for DDP test, aborting.", flush=True)
        return

    # Use a unique temp file per run; must NOT pre-exist (PyTorch creates it)
    init_file = f"/tmp/ddp_spawn_init_{os.getpid()}"
    if os.path.exists(init_file):
        os.remove(init_file)

    mp.spawn(worker, args=(world_size, init_file), nprocs=world_size, join=True)
    print("mp.spawn test PASSED", flush=True)


if __name__ == "__main__":
    main()
