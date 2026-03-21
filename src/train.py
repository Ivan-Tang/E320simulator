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

from src.models import (
    EdgeMLP, InteractionNet, TransformerEdgeClassifier,
    ResGNN, EggNet, HierarchicalGNN,
)
from src.losses import FocalLoss, HingeLoss
from src.train_embedder import EmbedderTrainConfig, train_embedder
from src.utils import (
    EDGE_DIM,
    EDGE_FEAT_COLS,
    NODE_DIM,
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
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
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
    model = embedder_info["model"].eval()
    emb_mean = embedder_info["_emb_mean"]
    emb_std = embedder_info["_emb_std"]
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
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for _, ev_df in df.group_by("event_id"):
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
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    print(f"[train] device={device}  model={cfg.model_type}  epochs={cfg.n_epochs}  seed={cfg.seed}")

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

    # ── pretrained embedder (two-stage pipeline) ─────────────────────────────
    embedder_info = _load_embedder(cfg)
    node_dim_eff = NODE_DIM
    if embedder_info is not None:
        node_dim_eff = embedder_info["_emb_dim"]
        print(f"[train] embedder pipeline: node_dim {NODE_DIM} → {node_dim_eff} (embedder replaces raw features)")

    # ── model / optimiser ────────────────────────────────────────────────────
    model     = _build_model(cfg, node_dim=node_dim_eff).to(device)
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    emb_criterion = HingeLoss(margin=1.0) if cfg.model_type == "hgnn" and cfg.hgnn_emb_loss_weight > 0 else None
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

            if embedder_info is not None:
                nf = _augment_with_embedder(nf, embedder_info)
            nf, ef = _normalise(nf, ef, node_mean, node_std, edge_mean, edge_std)
            nf  = nf.to(device)
            ei  = ei.to(device)
            ef  = ef.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()
            pred = model(nf, ei, ef)
            loss = criterion(pred, lab)
            # Optional embedding loss for HGNN (two-stage training from Liu et al. 2023)
            if emb_criterion is not None and hasattr(model, "last_embeddings") and model.last_embeddings is not None:
                emb = model.last_embeddings          # (N, emb_dim), L2-normalised
                e_src = emb[ei[0]]                   # (E, emb_dim)
                e_dst = emb[ei[1]]                   # (E, emb_dim)
                dist = (e_src - e_dst).norm(dim=-1)  # (E,)
                loss = loss + cfg.hgnn_emb_loss_weight * emb_criterion(dist, lab)
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
                embedder_info=embedder_info,
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
                            "node_dim":   node_dim_eff,
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
) -> dict:
    """Unified programmatic training API.

    Parameters
    ----------
    clusters_df:
        Simulator cluster table.
    task:
        - ``edge``: build candidate edges and train edge classifier.
        - ``embedder``: build hit pairs and train metric-learning embedder.
    """
    if task == "edge":
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
    parser.add_argument("--clusters", required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--task", default="edge", choices=["edge", "embedder"],
                        help="Training task: edge classifier or hit embedder")

    # edge-model options
    parser.add_argument("--model",    default="gnn", choices=["gnn", "mlp", "interaction_net", "eggnet", "hgnn"],
                        help="Edge model type (used when --task=edge)")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--n_mp",     type=int, default=2)
    parser.add_argument("--n_gnns_per_iter", type=int, default=2,
                        help="Inner GNN rounds per iteration (eggnet only)")
    parser.add_argument("--no-recurrent", action="store_true",
                        help="Disable weight sharing across iterations (eggnet only)")
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
    args = parser.parse_args()

    clusters_df = pl.read_parquet(args.clusters)
    print(f"[cli] loaded {len(clusters_df):,} clusters")

    if args.task == "edge":
        edges_df = build_labeled_edges_from_sim(clusters_df)
        print(f"[cli] built {len(edges_df):,} candidate edges")

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
        )
        train(edges_df, cfg)
        return

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
