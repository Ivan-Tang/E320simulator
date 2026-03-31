"""Minimal DDP smoke test — launched via torchrun."""
import os
import torch
import torch.distributed as dist

def main():
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    print(f"[rank {rank}/{world_size}] device=cuda:{local_rank}  "
          f"gpu={torch.cuda.get_device_name(local_rank)}", flush=True)

    # simple all-reduce to verify communication
    t = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{local_rank}")
    dist.all_reduce(t)
    print(f"[rank {rank}] all_reduce result = {t.item():.0f} "
          f"(expected {sum(range(world_size))})", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
