"""Unit tests for src/ddp.py — all run in single-process / non-DDP mode."""
import os

import torch
import pytest

import src.ddp as ddp


# ──────────────────────────────────────────────────────────────────────────────
# is_ddp_launched
# ──────────────────────────────────────────────────────────────────────────────

def test_is_ddp_launched_false_by_default():
    """Without RANK/WORLD_SIZE env vars, is_ddp_launched() is False."""
    for key in ("RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)
    assert ddp.is_ddp_launched() is False


# ──────────────────────────────────────────────────────────────────────────────
# setup_ddp / cleanup_ddp
# ──────────────────────────────────────────────────────────────────────────────

def test_setup_ddp_returns_single_process():
    """setup_ddp without torchrun returns (0, 1, False)."""
    for key in ("RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)
    rank, world_size, is_ddp_flag = ddp.setup_ddp()
    assert rank == 0
    assert world_size == 1
    assert is_ddp_flag is False


def test_cleanup_ddp_is_safe_without_init():
    """cleanup_ddp should not raise when no process group was initialised."""
    ddp.cleanup_ddp()  # no-op; must not raise


# ──────────────────────────────────────────────────────────────────────────────
# resolve_device
# ──────────────────────────────────────────────────────────────────────────────

def test_resolve_device_cpu():
    assert ddp.resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_device():
    """resolve_device('auto') returns a valid torch.device (cpu/cuda/mps)."""
    for key in ("RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)
    dev = ddp.resolve_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda", "mps")


# ──────────────────────────────────────────────────────────────────────────────
# is_main_process / ddp_print
# ──────────────────────────────────────────────────────────────────────────────

def test_is_main_process_without_dist():
    assert ddp.is_main_process() is True


def test_ddp_print_does_not_raise(capsys):
    ddp.ddp_print("hello from ddp_print")
    captured = capsys.readouterr()
    assert "hello from ddp_print" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# shard_event_list
# ──────────────────────────────────────────────────────────────────────────────

def test_shard_event_list_single_rank():
    events = list(range(10))
    shard = ddp.shard_event_list(events, rank=0, world_size=1)
    assert shard == events


def test_shard_event_list_two_ranks():
    events = list(range(10))
    shard0 = ddp.shard_event_list(events, rank=0, world_size=2)
    shard1 = ddp.shard_event_list(events, rank=1, world_size=2)
    assert set(shard0) | set(shard1) == set(events)
    assert set(shard0) & set(shard1) == set()
    assert len(shard0) + len(shard1) == len(events)


def test_shard_event_list_round_robin():
    events = [0, 1, 2, 3, 4, 5]
    assert ddp.shard_event_list(events, rank=0, world_size=3) == [0, 3]
    assert ddp.shard_event_list(events, rank=1, world_size=3) == [1, 4]
    assert ddp.shard_event_list(events, rank=2, world_size=3) == [2, 5]


def test_shard_event_list_uneven():
    events = list(range(7))
    shards = [ddp.shard_event_list(events, rank=r, world_size=3) for r in range(3)]
    combined = [x for s in shards for x in s]
    assert sorted(combined) == events


# ──────────────────────────────────────────────────────────────────────────────
# all_reduce_scalar
# ──────────────────────────────────────────────────────────────────────────────

def test_all_reduce_scalar_noop_without_dist():
    val = ddp.all_reduce_scalar(3.14, world_size=1)
    assert abs(val - 3.14) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# maybe_wrap_ddp
# ──────────────────────────────────────────────────────────────────────────────

def test_maybe_wrap_ddp_returns_same_model_without_dist():
    model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")
    wrapped, raw = ddp.maybe_wrap_ddp(model, device)
    assert wrapped is raw  # no DDP wrapping without a process group
    assert next(wrapped.parameters()).device == device


def test_maybe_wrap_ddp_model_on_correct_device():
    model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")
    wrapped, raw = ddp.maybe_wrap_ddp(model, device)
    for p in raw.parameters():
        assert p.device == device


def test_maybe_wrap_ddp_state_dict_has_no_module_prefix():
    """Outside DDP, state_dict keys must NOT have 'module.' prefix."""
    model = torch.nn.Linear(4, 2)
    _, raw = ddp.maybe_wrap_ddp(model, torch.device("cpu"))
    keys = list(raw.state_dict().keys())
    assert all(not k.startswith("module.") for k in keys)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient accumulation equivalence (single-process sanity check)
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient_accumulation_equivalent_to_batch():
    """Accumulating over K steps should give the same gradient as one big step."""
    torch.manual_seed(0)
    model_ref = torch.nn.Linear(4, 1, bias=False)
    model_acc = torch.nn.Linear(4, 1, bias=False)
    model_acc.weight.data.copy_(model_ref.weight.data)

    K = 4
    inputs = [torch.randn(2, 4) for _ in range(K)]

    # Reference: compute mean loss over all K inputs, single backward
    optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.0)
    optimizer_ref.zero_grad()
    total_loss = sum(model_ref(x).sum() for x in inputs) / K
    total_loss.backward()
    grad_ref = model_ref.weight.grad.clone()

    # Accumulation: K separate backward passes, each scaled by 1/K
    optimizer_acc = torch.optim.SGD(model_acc.parameters(), lr=0.0)
    optimizer_acc.zero_grad()
    for x in inputs:
        loss = model_acc(x).sum() / K
        loss.backward()
    grad_acc = model_acc.weight.grad.clone()

    assert torch.allclose(grad_ref, grad_acc, atol=1e-6), (
        f"Gradient mismatch: ref={grad_ref}  acc={grad_acc}"
    )
