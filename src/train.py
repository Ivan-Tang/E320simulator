"""Unified training entry for ML models in E320simulator.

Tasks
-----
1) edge      : edge-classification (GNN / MLP)
2) embedder  : metric-learning hit embedder

Usage (script)
--------------
    python -m src.train --task edge --clusters /path/to/sim_clusters.parquet
    python -m src.train --task embedder --clusters /path/to/sim_clusters.parquet
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models import (
    EdgeMLP, InteractionNet, TransformerEdgeClassifier,
    ResGNN, EggNet, HierarchicalGNN,
)
from src.losses import FocalLoss, HingeLoss
import torch.distributed as dist
import src.ddp as ddp
from src.train_embedder import EmbedderTrainConfig, train_embedder
from src.utils import (
    EDGE_DIM,
    EDGE_FEAT_COLS,
    NODE_DIM,
    NODE_FEAT_COLS_SRC,
    ParquetEdgeSource,
    build_labeled_edges_from_sim,
    build_edges_to_parquet,
    edge_label_stats,
    event_to_tensors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # model
    model_type: Literal["gnn", "mlp", "transformer", "interaction_net", "eggnet", "hgnn"] = "gnn"
    hidden: int = 64
    n_mp: int = 2                        # message-passing rounds (gnn / hgnn interaction iters)

    # eggnet-specific parameters
    n_gnns_per_iter: int = 2             # inner GNN rounds per iteration (eggnet only)
    recurrent: bool = True               # share weights across iterations (eggnet only)

    # hgnn-specific parameters
    n_hierarchical_iters: int = 3        # hierarchical message-passing rounds (hgnn only)
    n_detector_layers: int = 5           # number of detector layers, default E320
    hgnn_emb_dim: int = 8               # intermediate embedding dimension (hgnn only)
    hgnn_emb_loss_weight: float = 0.0   # weight for HingeLoss on intermediate embeddings (0=disabled)

    # transformer-specific parameters
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0
    max_seeds: int = 100

    # loss
    focal_alpha: float = 0.995
    focal_gamma: float = 2.0

    # optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # scheduler
    n_epochs: int = 50
    warmup_epochs: int = 0           # linear LR warmup epochs (0 = disabled)
    lr_eta_min: float = 1e-5             # CosineAnnealingLR floor

    # data
    val_fraction: float = 0.2
    seed: int = 42
    skip_zero_pos_events: bool = True    # skip events with no positive edges

    # output
    checkpoint_dir: str | None = None   # save best model here
    log_every: int = 1                  # print val metrics every N epochs

    # pretrained embedder for feature augmentation (two-stage pipeline)
    embedder_checkpoint: str | None = None  # path to best_embedder.pt; raw features are augmented with embedder output

    # balanced mini-batch sampling to counter extreme class imbalance (pos_frac≈0.00002).
    # Enabled by default: each event batch keeps all positives + neg_pos_ratio×n_pos negatives,
    # preventing gradient collapse where the model outputs near-zero for all edges.
    # Focal-loss defaults (alpha=0.995) were tuned for pos_frac≈0.0002; with the actual
    # pos_frac=0.00002 (10× worse), balanced sampling is required for reliable convergence.
    balanced_sampling: bool = True
    neg_pos_ratio: int = 100        # negatives per positive in balanced batch

    # hardware
    device: str = "auto"                 # "auto" | "cpu" | "mps" | "cuda"
    gradient_accumulation_steps: int = 1  # accumulate gradients over N events before optimizer step


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_embedder(cfg: TrainConfig) -> dict | None:
    """Load pretrained embedder for feature augmentation (two-stage pipeline)."""
    if cfg.embedder_checkpoint is None:
        return None
    from src.train_embedder import load_embedder_checkpoint
    info = load_embedder_checkpoint(cfg.embedder_checkpoint, device="cpu")
    # Cache tensors for fast augmentation
    info["_emb_mean"] = torch.tensor(info["mean"], dtype=torch.float32)
    info["_emb_std"] = torch.tensor(info["std"], dtype=torch.float32)
    # Infer embedding dimension from model output layer
    info["_emb_dim"] = list(info["model"].mlp.children())[-1].out_features
    return info


def _augment_with_embedder(
    raw_nf: torch.Tensor,
    embedder_info: dict,
) -> torch.Tensor:
    """Replace raw node features with pretrained embedder output.

    The embedder is applied with its own normalisation (stored at training time).
    Returns shape ``(N, emb_dim)`` — the embedding REPLACES the raw features;
    it is not concatenated to them.
    """
    dev = raw_nf.device
    model = embedder_info["model"].eval().to(dev)
    emb_mean = embedder_info["_emb_mean"].to(dev)
    emb_std = embedder_info["_emb_std"].to(dev)
    with torch.no_grad():
        nf_norm = (raw_nf - emb_mean) / emb_std
        return model(nf_norm)  # (N, emb_dim)


def _build_model(cfg: TrainConfig, node_dim: int = NODE_DIM) -> nn.Module:
    if cfg.model_type == "mlp":
        return EdgeMLP(node_dim=node_dim, hidden=cfg.hidden)
    if cfg.model_type == "gnn":
        return ResGNN(node_dim=node_dim, hidden=cfg.hidden, n_graph_iters=cfg.n_mp)
    if cfg.model_type == "interaction_net":
        return InteractionNet(node_dim=node_dim, hidden=cfg.hidden, n_mp=cfg.n_mp)
    if cfg.model_type == "eggnet":
        return EggNet(
            node_dim=node_dim,
            hidden=cfg.hidden,
            n_iters=cfg.n_mp,
            n_gnns_per_iter=cfg.n_gnns_per_iter,
            recurrent=cfg.recurrent,
        )
    if cfg.model_type == "transformer":
        return TransformerEdgeClassifier(
            node_dim=node_dim,
            edge_dim=EDGE_DIM,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_encoder_layers=cfg.n_encoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
    if cfg.model_type == "hgnn":
        return HierarchicalGNN(
            node_dim=node_dim,
            hidden_dim=cfg.hidden,
            n_interaction_iters=cfg.n_mp,
            n_hierarchical_iters=cfg.n_hierarchical_iters,
            n_layers=cfg.n_detector_layers,
            emb_dim=cfg.hgnn_emb_dim,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def _compute_normalisation(
    train_df: pl.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std from training edges (in-memory path)."""
    node_np = train_df.select(NODE_FEAT_COLS_SRC).to_numpy(allow_copy=True).astype(np.float32)
    edge_np = train_df.select(EDGE_FEAT_COLS).to_numpy(allow_copy=True).astype(np.float32)
    node_mean = torch.tensor(node_np.mean(0), dtype=torch.float32)
    node_std  = torch.tensor(node_np.std(0).clip(1e-6), dtype=torch.float32)
    edge_mean = torch.tensor(edge_np.mean(0), dtype=torch.float32)
    edge_std  = torch.tensor(edge_np.std(0).clip(1e-6), dtype=torch.float32)
    return node_mean, node_std, edge_mean, edge_std


def _compute_normalisation_from_parquet(
    edges_dir: Path,
    train_eids: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std via streaming scan (parquet path).

    Uses Polars lazy scan so the full dataset is never loaded into RAM.
    """
    all_cols = NODE_FEAT_COLS_SRC + EDGE_FEAT_COLS
    lf = pl.scan_parquet(edges_dir / "chunk_*.parquet")
    lf = lf.filter(pl.col("event_id").is_in(list(train_eids)))
    stats = lf.select(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in all_cols],
        *[pl.col(c).std().alias(f"{c}_std")   for c in all_cols],
    ).collect()

    def _t(names: list[str], suffix: str) -> torch.Tensor:
        vals = [float(stats[f"{c}_{suffix}"][0] or 0.0) for c in names]
        return torch.tensor(vals, dtype=torch.float32)

    node_mean = _t(NODE_FEAT_COLS_SRC, "mean")
    node_std  = torch.tensor(
        [max(float(stats[f"{c}_std"][0] or 0.0), 1e-6) for c in NODE_FEAT_COLS_SRC],
        dtype=torch.float32,
    )
    edge_mean = _t(EDGE_FEAT_COLS, "mean")
    edge_std  = torch.tensor(
        [max(float(stats[f"{c}_std"][0] or 0.0), 1e-6) for c in EDGE_FEAT_COLS],
        dtype=torch.float32,
    )
    return node_mean, node_std, edge_mean, edge_std


def _normalise(
    nf: torch.Tensor,
    ef: torch.Tensor,
    node_mean: torch.Tensor,
    node_std: torch.Tensor,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # When an embedder is used, nf is already the embedder output and does not match
    # node_mean (which covers the original 7-dim raw features). Skip node normalisation
    # in that case; the embedder applies its own normalisation internally.
    if nf.shape[1] == node_mean.shape[0]:
        nf = (nf - node_mean) / node_std
    return nf, (ef - edge_mean) / edge_std


def _evaluate(
    model: nn.Module,
    df: pl.DataFrame,
    device: torch.device,
    node_mean: torch.Tensor,
    node_std: torch.Tensor,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
    embedder_info: dict | None = None,
) -> dict[str, float]:
    """Evaluate edge classifier.  Accepts either an in-memory DataFrame or a
    ParquetEdgeSource (with an optional list of event IDs to evaluate)."""
    model.eval()
    all_scores, all_labels = [], []
    # Use sort-based slicing instead of group_by to avoid Polars Rust thread panic.
    _df = df.sort("event_id").rechunk()
    _eids = _df["event_id"].to_numpy()
    _u_eids, _bdry = np.unique(_eids, return_index=True)
    _n = len(_df)
    with torch.no_grad():
        for i, eid in enumerate(_u_eids):
            start = int(_bdry[i])
            end = int(_bdry[i + 1]) if i + 1 < len(_u_eids) else _n
            ev_df = _df[start:end]
            nf, ei, ef, lab, _ = event_to_tensors(ev_df)
            if embedder_info is not None:
                nf = _augment_with_embedder(nf, embedder_info)
            nf, ef = _normalise(nf, ef, node_mean, node_std, edge_mean, edge_std)
            scores = model(nf.to(device), ei.to(device), ef.to(device)).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(lab.numpy())

    y_score = np.concatenate(all_scores)
    y_true  = np.concatenate(all_labels)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap":  float(average_precision_score(y_true, y_score)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    edges_df: pl.DataFrame | None = None,
    cfg: TrainConfig | None = None,
    *,
    edges_dir: str | Path | None = None,
) -> dict:
    """Train an edge-classification model and return results dict.

    Parameters
    ----------
    edges_df:
        Output of ``build_labeled_edges_from_sim`` (in-memory path).
        Mutually exclusive with ``edges_dir``.
    cfg:
        Training hyper-parameters.  Defaults to ``TrainConfig()``.
    edges_dir:
        Directory of parquet chunk files produced by
        ``build_edges_to_parquet`` (low-memory path for large datasets).
        Mutually exclusive with ``edges_df``.

    Returns
    -------
    dict with keys:
        model          – trained nn.Module (on CPU)
        history        – list of per-epoch dicts
        node_mean/std  – normalisation tensors
        edge_mean/std  – normalisation tensors
        best_ap        – best validation Average Precision
        checkpoint     – path to saved checkpoint (or None)
        train_eids     – set of training event IDs
        val_eids       – set of validation event IDs
    """
    if edges_df is None and edges_dir is None:
        raise ValueError("Provide either edges_df or edges_dir")
    if edges_df is not None and edges_dir is not None:
        raise ValueError("Provide edges_df OR edges_dir, not both")

    use_parquet = edges_dir is not None
    if use_parquet:
        edges_dir = Path(edges_dir)
        edge_source = ParquetEdgeSource(edges_dir)
    else:
        edge_source = None

    if cfg is None:
        cfg = TrainConfig()

    rank, world_size, is_ddp = ddp.setup_ddp()
    device = ddp.resolve_device(cfg.device)
    ddp.ddp_print(f"[train] device={device}  model={cfg.model_type}  epochs={cfg.n_epochs}")

    # ── train / val split ────────────────────────────────────────────────────
    if use_parquet:
        all_events = edge_source.event_ids.copy()
    else:
        all_events = np.array(edges_df["event_id"].unique().sort().to_numpy(), copy=True)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_events)
    n_train = int(len(all_events) * (1 - cfg.val_fraction))
    train_eids = set(all_events[:n_train].tolist())
    val_eids   = set(all_events[n_train:].tolist())

    train_df = edges_df.filter(pl.col("event_id").is_in(list(train_eids)))
    val_df   = edges_df.filter(pl.col("event_id").is_in(list(val_eids)))

    stats = edge_label_stats(edges_df)
    ddp.ddp_print(f"[train] total edges={stats['n_total']:,}  "
                  f"pos={stats['n_positive']}  "
                  f"pos_frac={stats['positive_fraction']:.5f}")
    ddp.ddp_print(f"[train] train events={len(train_eids)}  val events={len(val_eids)}")
    del edges_df  # free original DataFrame; train_df/val_df are independent rechunked copies

    # per-split positive counts
    n_pos_train = int(train_df.filter(pl.col("edge_label") == 1).height)
    n_pos_val = int(val_df.filter(pl.col("edge_label") == 1).height)
    ddp.ddp_print(f"[train] pos_train={n_pos_train}  pos_val={n_pos_val}")

    # ── normalisation ────────────────────────────────────────────────────────
    if use_parquet:
        print("[train] computing normalisation stats via streaming scan …")
        node_mean, node_std, edge_mean, edge_std = _compute_normalisation_from_parquet(
            edges_dir, train_eids
        )
    else:
        node_mean, node_std, edge_mean, edge_std = _compute_normalisation(train_df)

    # ── pretrained embedder (two-stage pipeline) ─────────────────────────────
    embedder_info = _load_embedder(cfg)
    node_dim_eff = NODE_DIM
    if embedder_info is not None:
        node_dim_eff = embedder_info["_emb_dim"]
        ddp.ddp_print(f"[train] embedder pipeline: node_dim {NODE_DIM} → {node_dim_eff} (embedder replaces raw features)")

    # ── model / optimiser ────────────────────────────────────────────────────
    # In DDP mode always enable find_unused_parameters: several models (e.g.
    # InteractionNet) compute emb_output in forward() but only HGNN uses it in
    # the loss.  Those parameters produce no gradient, which causes DDP to raise
    # "Expected to have finished reduction in prior iteration" without this flag.
    _find_unused = is_ddp or cfg.model_type == "hgnn"
    model, raw_model = ddp.maybe_wrap_ddp(
        _build_model(cfg, node_dim=node_dim_eff), device,
        find_unused_parameters=_find_unused,
    )
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    emb_criterion = HingeLoss(margin=1.0) if cfg.model_type == "hgnn" and cfg.hgnn_emb_loss_weight > 0 else None
    optimizer = optim.Adam(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_eta_min
    )
    n_params = sum(p.numel() for p in raw_model.parameters())
    ddp.ddp_print(f"[train] params={n_params:,}")

    # ── checkpoint dir (rank-0 only) ─────────────────────────────────────────
    ckpt_dir: Path | None = None
    if cfg.checkpoint_dir and ddp.is_main_process():
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = ckpt_dir / "config.json"
        cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    # ── training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_ap   = -1.0
    best_path: Path | None = None
    t0 = time.time()

    # Pre-build training event list using sort-based slicing.
    # DO NOT use train_df.group_by("event_id") here: on large DataFrames (>100M rows)
    # Polars dispatches group_by to background Rust threads (polars-N) which hit an
    # internal assertion failure (frame/mod.rs assertion `left == right`) due to a
    # length inconsistency in rechunked Arrow buffers.  That Rust thread panic causes
    # a SIGSEGV in the subprocess with no Python traceback, making it appear as a
    # mysterious "Segmentation fault" in the PBS job log.  Sort + searchsorted avoids
    # the parallel group_by entirely.
    train_df = train_df.sort("event_id").rechunk()
    _eids_np = train_df["event_id"].to_numpy()
    _unique_eids, _boundaries = np.unique(_eids_np, return_index=True)
    _n_rows = len(train_df)
    all_train_events = [
        (int(eid), train_df[int(_boundaries[i]) : (int(_boundaries[i + 1]) if i + 1 < len(_unique_eids) else _n_rows)])
        for i, eid in enumerate(_unique_eids)
    ]
    local_train_events = ddp.shard_event_list(all_train_events, rank, world_size)
    accum_steps = max(1, cfg.gradient_accumulation_steps)

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        optimizer.zero_grad()
        accum_count = 0  # events processed since last optimizer step

        for i, (_, ev_df) in enumerate(local_train_events):
            nf, ei, ef, lab, _ = event_to_tensors(ev_df)
            # In DDP mode, skipping must be disabled: different ranks would skip
            # different events → unequal backward() counts → NCCL AllReduce deadlock.
            if cfg.skip_zero_pos_events and lab.sum() == 0 and not is_ddp:
                continue

            # Balanced mini-batch: sample all positives + neg_pos_ratio×n_pos negatives
            if cfg.balanced_sampling and int(lab.sum()) > 0:
                pos_idx = torch.where(lab == 1)[0]
                neg_idx = torch.where(lab == 0)[0]
                n_neg = min(int(neg_idx.numel()), int(pos_idx.numel()) * cfg.neg_pos_ratio)
                if n_neg > 0:
                    neg_perm = torch.randperm(neg_idx.numel())[:n_neg]
                    sel = torch.cat([pos_idx, neg_idx[neg_perm]])
                    ei = ei[:, sel]
                    ef = ef[sel]
                    lab = lab[sel]

            if embedder_info is not None:
                nf = _augment_with_embedder(nf, embedder_info)
            nf, ef = _normalise(nf, ef, node_mean, node_std, edge_mean, edge_std)
            nf  = nf.to(device)
            ei  = ei.to(device)
            ef  = ef.to(device)
            lab = lab.to(device)

            pred = model(nf, ei, ef)
            if cfg.balanced_sampling:
                # Give each positive edge neg_pos_ratio× more weight to counteract
                # the 1:neg_pos_ratio imbalance within the balanced batch.
                # Without pos_weight, the equilibrium score for positives is ~0.2
                # (not 0.5), causing all inference scores to fall below threshold.
                sample_weight = torch.where(
                    lab == 1,
                    torch.full_like(lab, float(cfg.neg_pos_ratio)),
                    torch.ones_like(lab),
                )
                loss = nn.functional.binary_cross_entropy(pred, lab.float(), weight=sample_weight)
            else:
                loss = criterion(pred, lab)
            # Optional embedding loss for HGNN (two-stage training from Liu et al. 2023)
            _inner_model = raw_model  # unwrapped reference for attribute access
            if emb_criterion is not None and hasattr(_inner_model, "last_embeddings") and _inner_model.last_embeddings is not None:
                emb = _inner_model.last_embeddings   # (N, emb_dim), L2-normalised
                e_src = emb[ei[0]]                   # (E, emb_dim)
                e_dst = emb[ei[1]]                   # (E, emb_dim)
                dist_ = (e_src - e_dst).norm(dim=-1) # (E,)
                loss = loss + cfg.hgnn_emb_loss_weight * emb_criterion(dist_, lab)
            # Release last_embeddings immediately after use (or non-use) to avoid
            # accumulation of stale CUDA tensors across events, which can fragment
            # GPU memory and trigger a segfault in later epochs.
            if hasattr(_inner_model, "last_embeddings"):
                _inner_model.last_embeddings = None

            is_last_event = (i + 1) == len(local_train_events)
            will_step = (accum_count + 1 >= accum_steps) or is_last_event

            # In DDP, suppress AllReduce for intermediate accumulation steps.
            # Without no_sync(), every backward() triggers NCCL AllReduce even
            # when we haven't reached the accumulation boundary — causing O(n_events)
            # syncs per epoch (e.g. 4000) instead of O(n_events/accum_steps) (e.g. 40).
            backward_ctx = (model.no_sync() if is_ddp and not will_step
                            else contextlib.nullcontext())
            with backward_ctx:
                (loss / accum_steps).backward()
            accum_count += 1

            if will_step:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        row: dict = {"epoch": epoch, "train_loss": avg_loss}

        # ── validation (rank-0 only) ──────────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            if ddp.is_main_process():
                metrics = _evaluate(
                    raw_model, val_df, device,
                    node_mean, node_std, edge_mean, edge_std,
                    embedder_info=embedder_info,
                )
                row.update(metrics)
                elapsed = time.time() - t0
                ddp.ddp_print(
                    f"Epoch {epoch:>3}/{cfg.n_epochs} | "
                    f"loss={avg_loss:.6f} | "
                    f"AUC={metrics['auc']:.4f} | "
                    f"AP={metrics['ap']:.4f} | "
                    f"t={elapsed:.0f}s"
                )

                # ── save best checkpoint ──────────────────────────────────────
                if metrics["ap"] > best_ap:
                    best_ap = metrics["ap"]
                    if ckpt_dir is not None:
                        best_path = ckpt_dir / "best_model.pt"
                        torch.save(
                            {
                                "epoch":           epoch,
                                "model_state":     raw_model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "best_ap":         best_ap,
                                "node_mean":       node_mean,
                                "node_std":        node_std,
                                "edge_mean":       edge_mean,
                                "edge_std":        edge_std,
                                "node_dim":        node_dim_eff,
                            },
                            best_path,
                        )
            # Sync all ranks after validation
            if dist.is_initialized():
                dist.barrier()

        history.append(row)

    ddp.ddp_print(f"[train] done  best_AP={best_ap:.4f}  checkpoint={best_path}")
    ddp.cleanup_ddp()

    return {
        "model":      raw_model.cpu(),
        "history":    history,
        "node_mean":  node_mean,
        "node_std":   node_std,
        "edge_mean":  edge_mean,
        "edge_std":   edge_std,
        "best_ap":    best_ap,
        "checkpoint": str(best_path) if best_path else None,
        "train_eids": train_eids,
        "val_eids":   val_eids,
        # Kept for backward compatibility when using in-memory path
        "train_df":   train_df,
        "val_df":     val_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    cfg: TrainConfig | None = None,
    device: str = "cpu",
) -> dict:
    """Load a saved checkpoint.

    Returns dict with keys: model, node_mean, node_std, edge_mean, edge_std, epoch, best_ap
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # try to reload config from sibling config.json
    if cfg is None:
        cfg_path = Path(checkpoint_path).parent / "config.json"
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text())
            cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__})
        else:
            cfg = TrainConfig()

    # Determine effective node_dim (may be augmented with embedder features)
    node_dim = ckpt.get("node_dim", NODE_DIM)
    embedder_info = None
    if cfg.embedder_checkpoint:
        embedder_info = _load_embedder(cfg)
        node_dim = embedder_info["_emb_dim"]

    model = _build_model(cfg, node_dim=node_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    result = {
        "model":     model,
        "node_mean": ckpt["node_mean"],
        "node_std":  ckpt["node_std"],
        "edge_mean": ckpt["edge_mean"],
        "edge_std":  ckpt["edge_std"],
        "epoch":     ckpt["epoch"],
        "best_ap":   ckpt["best_ap"],
    }
    if embedder_info is not None:
        result["embedder_info"] = embedder_info
    return result


def train_model(
    clusters_df: pl.DataFrame,
    task: Literal["edge", "embedder"] = "edge",
    edge_cfg: TrainConfig | None = None,
    embed_cfg: EmbedderTrainConfig | None = None,
    edges_dir: str | Path | None = None,
) -> dict:
    """Unified programmatic training API.

    Parameters
    ----------
    clusters_df:
        Simulator cluster table.
    task:
        - ``edge``: build candidate edges and train edge classifier.
        - ``embedder``: build hit pairs and train metric-learning embedder.
    edges_dir:
        Optional pre-built parquet edge directory (avoids OOM for large
        datasets).  If provided for task="edge", ``clusters_df`` is used only
        to build edges when the directory does not yet exist.
    """
    if task == "edge":
        if edges_dir is not None:
            edges_dir = Path(edges_dir)
            if not edges_dir.exists() or not any(edges_dir.glob("chunk_*.parquet")):
                build_edges_to_parquet(clusters_df, edges_dir,
                                       cfg=edge_cfg and getattr(edge_cfg, "_baseline_cfg", None))
            return train(cfg=edge_cfg, edges_dir=edges_dir)
        edges_df = build_labeled_edges_from_sim(clusters_df)
        return train(edges_df, edge_cfg)

    if embed_cfg is None:
        embed_cfg = EmbedderTrainConfig()
    return train_embedder(clusters_df, embed_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unified trainer for edge models and embedder")
    parser.add_argument("--clusters", default=None, help="Path to sim_clusters.parquet")
    parser.add_argument("--edges", default=None,
                        help="Path to pre-built edges.parquet (skips cluster loading and "
                             "build_labeled_edges_from_sim; required when --clusters is omitted)")
    parser.add_argument("--task", default="edge", choices=["edge", "embedder"],
                        help="Training task: edge classifier or hit embedder")

    # parquet edge cache options (avoids OOM for large datasets)
    parser.add_argument("--edges-dir", default=None,
                        help="Pre-built edges parquet directory (skips edge building, trains directly)")
    parser.add_argument("--build-edges-to", default=None,
                        help="Build edges to this directory then train from it (low-memory path)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Events per parquet chunk when using --build-edges-to (default=200)")

    # edge-model options
    parser.add_argument("--model",    default="gnn",
                        choices=["gnn", "mlp", "interaction_net", "eggnet", "hgnn", "transformer"],
                        help="Edge model type (used when --task=edge)")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--n_mp",     type=int, default=2)
    parser.add_argument("--n_gnns_per_iter", type=int, default=2,
                        help="Inner GNN rounds per iteration (eggnet only)")
    parser.add_argument("--no-recurrent", action="store_true",
                        help="Disable weight sharing across iterations (eggnet only)")
    # transformer-specific options
    parser.add_argument("--d-model",          type=int,   default=256)
    parser.add_argument("--n-heads",          type=int,   default=8)
    parser.add_argument("--n-encoder-layers", type=int,   default=6)
    parser.add_argument("--dim-feedforward",  type=int,   default=1024)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--warmup-epochs",    type=int,   default=0,
                        help="Linear LR warmup epochs (0=disabled)")
    parser.add_argument("--focal-alpha",      type=float, default=0.995)
    parser.add_argument("--focal-gamma",      type=float, default=2.0)
    parser.add_argument("--grad-clip",        type=float, default=1.0)
    parser.add_argument("--weight-decay",     type=float, default=1e-5)
    parser.add_argument("--layers",   type=int, default=3,
                        help="Embedder MLP depth (used when --task=embedder)")
    parser.add_argument("--emb-dim",  type=int, default=8,
                        help="Embedding dimension (used when --task=embedder)")
    parser.add_argument("--nb-particles-per-sample", type=int, default=2000,
                        help="Pair sampling count per event for embedder training")
    parser.add_argument("--max-pairs", type=int, default=500000,
                        help="Maximum sampled pairs for embedder training")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--checkpoint", default=None, help="Directory to save checkpoints")
    parser.add_argument("--embedder-checkpoint", default=None,
                        help="Path to pretrained embedder .pt for node feature augmentation")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Run validation and print metrics every N epochs (use >1 in DDP to reduce barrier overhead)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate gradients over N events before optimizer step (DDP / memory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility (default 42)")
    args = parser.parse_args()

    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    _torch.cuda.manual_seed_all(args.seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    print(f"[seed] global seed set to {args.seed}", flush=True)

    if args.edges is None and args.clusters is None:
        parser.error("one of --clusters or --edges is required")

    # In DDP mode: rank 0 builds the edge table and writes a temp parquet;
    # other ranks wait at a barrier then read the pre-built file.
    # This avoids concurrent Lustre reads + Polars group_by triggering an
    # internal assertion failure, and cuts edge-building work to 1× instead of N×.
    rank, world_size, is_ddp = ddp.setup_ddp()

    if args.task == "edge":
        if args.edges is not None:
            # Pre-built edge table supplied directly — all ranks load it.
            # rechunk() immediately: on Lustre two concurrent readers can produce
            # parquet chunks with misaligned boundaries, causing a Polars
            # ShapeError ("filter's length differs from series") during train/val
            # split.  A single rechunk consolidates all chunks before any filter.
            edges_df = pl.read_parquet(args.edges).rechunk()
            print(f"[cli] rank {rank}: loaded {len(edges_df):,} edges from {args.edges}", flush=True)
            _edge_cache = None
        else:
            clusters_df = pl.read_parquet(args.clusters)
            print(f"[cli] rank {rank}: loaded {len(clusters_df):,} clusters", flush=True)
            _edge_cache = os.environ.get("DDP_INIT_METHOD", "").replace("file://", "") + "_edges.parquet"
            if is_ddp:
                if rank == 0:
                    edges_df = build_labeled_edges_from_sim(clusters_df)
                    print(f"[cli] rank 0: built {len(edges_df):,} edges → {_edge_cache}", flush=True)
                    edges_df.write_parquet(_edge_cache)
                dist.barrier()  # all ranks wait for rank 0 to finish writing
                if rank != 0:
                    edges_df = pl.read_parquet(_edge_cache)
                    print(f"[cli] rank {rank}: loaded {len(edges_df):,} edges from cache", flush=True)
            else:
                edges_df = build_labeled_edges_from_sim(clusters_df)
                print(f"[cli] built {len(edges_df):,} candidate edges", flush=True)

        cfg = TrainConfig(
            model_type       = args.model,
            n_epochs         = args.epochs,
            hidden           = args.hidden,
            n_mp             = args.n_mp,
            n_gnns_per_iter  = args.n_gnns_per_iter,
            recurrent        = not args.no_recurrent,
            lr               = args.lr,
            val_fraction     = args.val_fraction,
            device           = args.device,
            checkpoint_dir   = args.checkpoint,
            embedder_checkpoint = args.embedder_checkpoint,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            log_every                   = args.log_every,
        )
        train(edges_df, cfg)
        if is_ddp and rank == 0 and _edge_cache and os.path.exists(_edge_cache):
            os.remove(_edge_cache)
        return

    if args.clusters is None:
        parser.error("--clusters is required for --task=embedder")
    clusters_df = pl.read_parquet(args.clusters)
    print(f"[cli] rank {rank}: loaded {len(clusters_df):,} clusters", flush=True)
    embed_cfg = EmbedderTrainConfig(
        n_epochs=args.epochs,
        batch_size=4096,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        nb_particles_per_sample=args.nb_particles_per_sample,
        max_pairs=args.max_pairs,
        val_fraction=args.val_fraction,
        lr=args.lr,
        checkpoint_dir=args.checkpoint,
        device=args.device,
    )
    train_embedder(clusters_df, embed_cfg)


if __name__ == "__main__":
    _cli()
