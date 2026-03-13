"""
Edge-classification GNN training pipeline.

Usage (script)
--------------
    python -m src.train [--model gnn|mlp] [--epochs 20] [--checkpoint runs/exp1]

Usage (from notebook / other code)
-----------------------------------
    from src.train import TrainConfig, train
    cfg = TrainConfig(model_type="gnn", n_epochs=30, checkpoint_dir="runs/exp1")
    results = train(edges_df, cfg)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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

from src.models import EdgeMLP, InteractionNet
from src.losses import FocalLoss
from src.utils import (
    EDGE_FEAT_COLS,
    NODE_FEAT_COLS_SRC,
    build_labeled_edges_from_sim,
    edge_label_stats,
    event_to_tensors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # model
    model_type: Literal["gnn", "mlp"] = "gnn"
    hidden: int = 64
    n_mp: int = 2                        # message-passing rounds (gnn only)

    # loss
    focal_alpha: float = 0.995
    focal_gamma: float = 2.0

    # optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # scheduler
    n_epochs: int = 50
    lr_eta_min: float = 1e-5             # CosineAnnealingLR floor

    # data
    val_fraction: float = 0.2
    seed: int = 42
    skip_zero_pos_events: bool = True    # skip events with no positive edges

    # output
    checkpoint_dir: str | None = None   # save best model here
    log_every: int = 1                  # print val metrics every N epochs

    # hardware
    device: str = "auto"                 # "auto" | "cpu" | "mps" | "cuda"


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


def _build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_type == "mlp":
        return EdgeMLP(hidden=cfg.hidden)
    return InteractionNet(hidden=cfg.hidden, n_mp=cfg.n_mp)


def _compute_normalisation(
    train_df: pl.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std from training edges."""
    node_np = train_df.select(NODE_FEAT_COLS_SRC).to_numpy(allow_copy=True).astype(np.float32)
    edge_np = train_df.select(EDGE_FEAT_COLS).to_numpy(allow_copy=True).astype(np.float32)
    node_mean = torch.tensor(node_np.mean(0), dtype=torch.float32)
    node_std  = torch.tensor(node_np.std(0).clip(1e-6), dtype=torch.float32)
    edge_mean = torch.tensor(edge_np.mean(0), dtype=torch.float32)
    edge_std  = torch.tensor(edge_np.std(0).clip(1e-6), dtype=torch.float32)
    return node_mean, node_std, edge_mean, edge_std


def _normalise(
    nf: torch.Tensor,
    ef: torch.Tensor,
    node_mean: torch.Tensor,
    node_std: torch.Tensor,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (nf - node_mean) / node_std, (ef - edge_mean) / edge_std


def _evaluate(
    model: nn.Module,
    df: pl.DataFrame,
    device: torch.device,
    node_mean: torch.Tensor,
    node_std: torch.Tensor,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for _, ev_df in df.group_by("event_id"):
            nf, ei, ef, lab, _ = event_to_tensors(ev_df)
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
    edges_df: pl.DataFrame,
    cfg: TrainConfig | None = None,
) -> dict:
    """Train an edge-classification model and return results dict.

    Parameters
    ----------
    edges_df:
        Output of ``build_labeled_edges_from_sim``.
    cfg:
        Training hyper-parameters.  Defaults to ``TrainConfig()``.

    Returns
    -------
    dict with keys:
        model          – trained nn.Module (on CPU)
        history        – list of per-epoch dicts
        node_mean/std  – normalisation tensors
        edge_mean/std  – normalisation tensors
        best_ap        – best validation Average Precision
        checkpoint     – path to saved checkpoint (or None)
    """
    if cfg is None:
        cfg = TrainConfig()

    device = _resolve_device(cfg)
    print(f"[train] device={device}  model={cfg.model_type}  epochs={cfg.n_epochs}")

    # ── train / val split ────────────────────────────────────────────────────
    all_events = np.array(edges_df["event_id"].unique().sort().to_numpy(), copy=True)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_events)
    n_train = int(len(all_events) * (1 - cfg.val_fraction))
    train_eids = set(all_events[:n_train].tolist())
    val_eids   = set(all_events[n_train:].tolist())

    train_df = edges_df.filter(pl.col("event_id").is_in(list(train_eids)))
    val_df   = edges_df.filter(pl.col("event_id").is_in(list(val_eids)))

    stats = edge_label_stats(edges_df)
    print(f"[train] total edges={stats['n_total']:,}  "
          f"pos={stats['n_positive']}  "
          f"pos_frac={stats['positive_fraction']:.5f}")
    print(f"[train] train events={len(train_eids)}  val events={len(val_eids)}")

    # per-split positive counts
    n_pos_train = int(train_df.filter(pl.col("edge_label") == 1).height)
    n_pos_val = int(val_df.filter(pl.col("edge_label") == 1).height)
    print(f"[train] pos_train={n_pos_train}  pos_val={n_pos_val}")

    # ── normalisation ────────────────────────────────────────────────────────
    node_mean, node_std, edge_mean, edge_std = _compute_normalisation(train_df)

    # ── model / optimiser ────────────────────────────────────────────────────
    model     = _build_model(cfg).to(device)
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_eta_min
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,}")

    # ── checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir: Path | None = None
    if cfg.checkpoint_dir:
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # save config alongside weights
        cfg_path = ckpt_dir / "config.json"
        cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    # ── training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_ap   = -1.0
    best_path: Path | None = None
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for _, ev_df in train_df.group_by("event_id"):
            nf, ei, ef, lab, _ = event_to_tensors(ev_df)
            if cfg.skip_zero_pos_events and lab.sum() == 0:
                continue

            nf, ef = _normalise(nf, ef, node_mean, node_std, edge_mean, edge_std)
            nf  = nf.to(device)
            ei  = ei.to(device)
            ef  = ef.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()
            pred = model(nf, ei, ef)
            loss = criterion(pred, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        row: dict = {"epoch": epoch, "train_loss": avg_loss}

        # ── validation ───────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            metrics = _evaluate(
                model, val_df, device,
                node_mean, node_std, edge_mean, edge_std,
            )
            row.update(metrics)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{cfg.n_epochs} | "
                f"loss={avg_loss:.6f} | "
                f"AUC={metrics['auc']:.4f} | "
                f"AP={metrics['ap']:.4f} | "
                f"t={elapsed:.0f}s"
            )

            # ── save best checkpoint ──────────────────────────────────────────
            if metrics["ap"] > best_ap:
                best_ap = metrics["ap"]
                if ckpt_dir is not None:
                    best_path = ckpt_dir / "best_model.pt"
                    torch.save(
                        {
                            "epoch":      epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_ap":    best_ap,
                            "node_mean":  node_mean,
                            "node_std":   node_std,
                            "edge_mean":  edge_mean,
                            "edge_std":   edge_std,
                        },
                        best_path,
                    )

        history.append(row)

    print(f"[train] done  best_AP={best_ap:.4f}  checkpoint={best_path}")

    return {
        "model":      model.cpu(),
        "history":    history,
        "node_mean":  node_mean,
        "node_std":   node_std,
        "edge_mean":  edge_mean,
        "edge_std":   edge_std,
        "best_ap":    best_ap,
        "checkpoint": str(best_path) if best_path else None,
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

    model = _build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return {
        "model":     model,
        "node_mean": ckpt["node_mean"],
        "node_std":  ckpt["node_std"],
        "edge_mean": ckpt["edge_mean"],
        "edge_std":  ckpt["edge_std"],
        "epoch":     ckpt["epoch"],
        "best_ap":   ckpt["best_ap"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train edge-classification GNN")
    parser.add_argument("--clusters", required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--model",    default="gnn", choices=["gnn", "mlp"])
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--n_mp",     type=int, default=2)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--checkpoint", default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    clusters_df = pl.read_parquet(args.clusters)
    print(f"[cli] loaded {len(clusters_df):,} clusters")

    edges_df = build_labeled_edges_from_sim(clusters_df)
    print(f"[cli] built {len(edges_df):,} candidate edges")

    cfg = TrainConfig(
        model_type     = args.model,
        n_epochs       = args.epochs,
        hidden         = args.hidden,
        n_mp           = args.n_mp,
        lr             = args.lr,
        device         = args.device,
        checkpoint_dir = args.checkpoint,
    )
    train(edges_df, cfg)


if __name__ == "__main__":
    _cli()
