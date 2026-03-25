"""Distributed Data Parallel (DDP) utilities for E320simulator.

All training entry points (train.py, train_embedder.py, train_trackformer.py,
train_hit_filter.py) use this module for DDP setup and helpers.

Single-GPU / CPU usage is fully backward-compatible: when NOT launched via
``torchrun``, every function degrades to a no-op and ``setup_ddp`` returns
``(0, 1, False)``.

Typical launch:
    torchrun --standalone --nproc_per_node=2 -m src.train --task edge ...
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────

def is_ddp_launched() -> bool:
    """Return True when the process was started by torchrun / torch.distributed.launch."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def setup_ddp() -> tuple[int, int, bool]:
    """Initialise the NCCL process group.

    Returns
    -------
    rank        : global rank of this process (0 on single-GPU / CPU)
    world_size  : total number of processes (1 on single-GPU / CPU)
    is_ddp      : True when actually running in distributed mode
    """
    if not is_ddp_launched():
        return 0, 1, False

    # Already initialised (e.g. by mp.spawn wrapper that called init_process_group directly)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    init_method = os.environ.get("DDP_INIT_METHOD", "env://")
    dist.init_process_group(backend="nccl", init_method=init_method,
                            rank=rank, world_size=world_size)
    return rank, world_size, True


def cleanup_ddp() -> None:
    """Destroy the process group if one was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Device resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str = "auto") -> torch.device:
    """Return the torch.device for the current process.

    In DDP mode the device is always ``cuda:<LOCAL_RANK>``.
    Outside DDP the function replicates the original ``_resolve_device`` logic.
    """
    if is_ddp_launched():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Process-rank helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_main_process() -> bool:
    """Return True for rank-0 process (or when not in DDP)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def ddp_print(*args, **kwargs) -> None:
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Data sharding
# ──────────────────────────────────────────────────────────────────────────────

def shard_event_list(events: list, rank: int, world_size: int) -> list:
    """Return the slice of *events* that belongs to *rank* (round-robin)."""
    return events[rank::world_size]


# ──────────────────────────────────────────────────────────────────────────────
# Metric reduction
# ──────────────────────────────────────────────────────────────────────────────

def all_reduce_scalar(value: float, world_size: int) -> float:
    """Average *value* across all DDP ranks.  No-op when not in DDP."""
    if not dist.is_initialized() or world_size <= 1:
        return value
    t = torch.tensor(value, dtype=torch.float64,
                     device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapping
# ──────────────────────────────────────────────────────────────────────────────

def maybe_wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Move *model* to *device* and optionally wrap in DDP.

    Returns
    -------
    wrapped_model   : DDP-wrapped model (or plain model outside DDP)
    unwrapped_model : original model reference — use this for ``state_dict()``
                      to avoid the ``module.`` prefix added by DDP
    """
    model = model.to(device)
    if dist.is_initialized():
        wrapped = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
        )
        return wrapped, model
    return model, model
