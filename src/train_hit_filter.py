"""Training for E320HitFilter — Stage 1 of the two-stage TrackFormer pipeline.

Adapted from the hit filtering stage of Van Stroud et al. (2025)
"Transformers for Charged Particle Track Reconstruction in High Energy Physics".

The hit filter classifies each of the ~3500 hits per event as signal
(track_id >= 0) or noise (track_id == -1).  E320 has an extreme class
imbalance (~0.017% signal), so Focal Loss is used.

A saved checkpoint stores:
  - model weights
  - per-feature normalisation (mean / std) computed on training hits
  - config dict

Usage
-----
    from src.train_hit_filter import HitFilterConfig, train_hit_filter
    cfg = HitFilterConfig(n_epochs=50, device="mps",
                          checkpoint_dir="runs/transformer")
    result = train_hit_filter(clusters_df, cfg)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import E320HitFilter
from src.losses import FocalLoss
from src.utils import NODE_DIM
import torch.distributed as dist
import src.ddp as ddp


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HitFilterConfig:
    # model
    d_model:         int   = 64
    n_heads:         int   = 4
    n_layers:        int   = 3
    dim_feedforward: int   = 128
    window_size:     int   = 256
    dropout:         float = 0.1

    # loss  — alpha very close to 1 because signal fraction ≈ 0.017%
    focal_alpha: float = 0.999
    focal_gamma: float = 2.0

    # filter threshold used for downstream evaluation / inference
    # low value → high recall; matches the paper's choice of 0.1
    filter_threshold: float = 0.1

    # optimiser
    lr:           float = 3e-4
    weight_decay: float = 1e-5
    grad_clip:    float = 1.0

    # scheduler
    n_epochs:   int   = 50
    lr_eta_min: float = 1e-6

    # data
    val_fraction: float = 0.2
    seed:         int   = 42

    # output
    checkpoint_dir: str | None = None
    checkpoint_name: str       = "hit_filter.pt"
    log_every:      int        = 1

    # hardware
    device: str = "auto"
    gradient_accumulation_steps: int = 1  # accumulate gradients over N events before optimizer step


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_FEAT_COLS = ["layer_id", "x_trk_mm", "y_trk_mm", "z_trk_mm",
              "size_x", "size_y", "size"]


def _resolve_device(cfg: HitFilterConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_normalisation(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    arr = df.select(_FEAT_COLS).to_numpy(allow_copy=True).astype(np.float32)
    mean = torch.tensor(arr.mean(0), dtype=torch.float32)
    std  = torch.tensor(arr.std(0).clip(1e-6), dtype=torch.float32)
    return mean, std


def _preprocess_events(
    df: pl.DataFrame,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-build per-event (node_feat, labels) tensors."""
    out = []
    for _, ev in df.group_by("event_id"):
        ev = ev.sort("node_id")
        feat = ev.select(_FEAT_COLS).to_numpy(allow_copy=True).astype(np.float32)
        labels = (ev["track_id"].to_numpy() >= 0).astype(np.float32)
        out.append((
            torch.from_numpy(feat),
            torch.from_numpy(labels),
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Validation metrics
# ──────────────────────────────────────────────────────────────────────────────

def _eval_filter(
    model: nn.Module,
    events: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    node_mean: torch.Tensor,
    node_std:  torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Compute hit efficiency (recall) and purity (precision) at threshold."""
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for nf_raw, labels in events:
            nf   = ((nf_raw - node_mean.cpu()) / node_std.cpu()).to(device)
            pred = model(nf).sigmoid().cpu()
            kept = pred >= threshold
            pos  = labels.bool()

            tp += int((kept & pos).sum())
            fp += int((kept & ~pos).sum())
            fn += int((~kept & pos).sum())

    efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    purity     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    reduction  = 1.0 - (tp + fp) / sum(len(lbl) for _, lbl in events)
    return {"efficiency": efficiency, "purity": purity,
            "hit_reduction": reduction, "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_hit_filter(
    clusters_df: pl.DataFrame,
    cfg: HitFilterConfig | None = None,
) -> dict:
    """Train E320HitFilter with Focal Loss.

    Parameters
    ----------
    clusters_df : pl.DataFrame
        Simulator cluster output (must have track_id column).
    cfg : HitFilterConfig | None
        Training config.  Defaults to HitFilterConfig().

    Returns
    -------
    dict with keys: model, history, node_mean, node_std, best_eff, checkpoint
    """
    if cfg is None:
        cfg = HitFilterConfig()

    rank, world_size, is_ddp = ddp.setup_ddp()
    device = ddp.resolve_device(cfg.device)
    ddp.ddp_print(f"[train_hit_filter] device={device}  epochs={cfg.n_epochs}  "
                  f"d_model={cfg.d_model}  window={cfg.window_size}")

    # ── Class balance stats ───────────────────────────────────────────────────
    n_total  = len(clusters_df)
    n_signal = int((clusters_df["track_id"] >= 0).sum())
    ddp.ddp_print(f"[train_hit_filter] total hits={n_total:,}  signal={n_signal:,}  "
                  f"purity={n_signal/n_total*100:.4f}%")

    # ── Train / val split ────────────────────────────────────────────────────
    all_events = np.array(clusters_df["event_id"].unique().sort().to_numpy(), copy=True)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_events)
    n_train   = int(len(all_events) * (1 - cfg.val_fraction))
    train_df  = clusters_df.filter(pl.col("event_id").is_in(all_events[:n_train].tolist()))
    val_df    = clusters_df.filter(pl.col("event_id").is_in(all_events[n_train:].tolist()))
    ddp.ddp_print(f"[train_hit_filter] train events={n_train}  val events={len(all_events)-n_train}")

    # ── Normalisation ────────────────────────────────────────────────────────
    node_mean, node_std = _compute_normalisation(train_df)

    # ── Pre-build tensors ────────────────────────────────────────────────────
    ddp.ddp_print("[train_hit_filter] preprocessing events...")
    all_train_events = _preprocess_events(train_df)
    local_train_events = ddp.shard_event_list(all_train_events, rank, world_size)
    val_events   = _preprocess_events(val_df)

    # ── Model ────────────────────────────────────────────────────────────────
    model, raw_model = ddp.maybe_wrap_ddp(
        E320HitFilter(
            node_dim        = NODE_DIM,
            d_model         = cfg.d_model,
            n_heads         = cfg.n_heads,
            n_layers        = cfg.n_layers,
            dim_feedforward = cfg.dim_feedforward,
            window_size     = cfg.window_size,
            dropout         = cfg.dropout,
        ),
        device,
    )

    n_params = sum(p.numel() for p in raw_model.parameters())
    ddp.ddp_print(f"[train_hit_filter] params={n_params:,}")

    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = optim.AdamW(raw_model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_eta_min
    )

    # ── Checkpoint dir (rank-0 only) ──────────────────────────────────────────
    ckpt_dir: Path | None = None
    if cfg.checkpoint_dir and ddp.is_main_process():
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "hit_filter_config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2)
        )

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_eff  = -1.0
    best_path: Path | None = None
    node_mean_cpu = node_mean
    node_std_cpu  = node_std
    accum_steps = max(1, cfg.gradient_accumulation_steps)
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        rng.shuffle(local_train_events)
        optimizer.zero_grad()
        accum_count = 0

        for i, (nf_raw, labels) in enumerate(local_train_events):
            nf  = ((nf_raw - node_mean_cpu) / node_std_cpu).to(device)
            lbl = labels.to(device)

            logits = model(nf)
            loss   = criterion(logits.sigmoid(), lbl)
            (loss / accum_steps).backward()
            accum_count += 1

            is_last = (i + 1) == len(local_train_events)
            if accum_count >= accum_steps or is_last:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        row: dict = {"epoch": epoch, "train_loss": avg_loss}

        # ── Validation (rank-0 only) ──────────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            if ddp.is_main_process():
                metrics = _eval_filter(raw_model, val_events, device,
                                       node_mean_cpu, node_std_cpu,
                                       cfg.filter_threshold)
                row.update(metrics)
                elapsed = time.time() - t0
                ddp.ddp_print(
                    f"Epoch {epoch:>3}/{cfg.n_epochs} | loss={avg_loss:.6f} | "
                    f"eff={metrics['efficiency']:.4f} | "
                    f"purity={metrics['purity']:.4f} | "
                    f"reduction={metrics['hit_reduction']:.4f} | "
                    f"t={elapsed:.0f}s"
                )

                if metrics["efficiency"] > best_eff:
                    best_eff = metrics["efficiency"]
                    if ckpt_dir is not None:
                        best_path = ckpt_dir / cfg.checkpoint_name
                        torch.save({
                            "epoch":       epoch,
                            "model_state": raw_model.state_dict(),
                            "best_eff":    best_eff,
                            "node_mean":   node_mean,
                            "node_std":    node_std,
                            "config":      dataclasses.asdict(cfg),
                        }, best_path)
            if dist.is_initialized():
                dist.barrier()

        history.append(row)

    ddp.ddp_print(f"[train_hit_filter] done  best_eff={best_eff:.4f}  checkpoint={best_path}")
    ddp.cleanup_ddp()
    return {
        "model":      raw_model.cpu(),
        "history":    history,
        "node_mean":  node_mean,
        "node_std":   node_std,
        "best_eff":   best_eff,
        "checkpoint": str(best_path) if best_path else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def load_hit_filter_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """Load a saved E320HitFilter checkpoint.

    Returns dict with keys: model, node_mean, node_std, config, epoch, best_eff
    """
    # Always load on CPU first to avoid MPS placeholder-storage errors,
    # then move to target device.
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})

    model = E320HitFilter(
        node_dim        = NODE_DIM,
        d_model         = cfg_dict.get("d_model",         64),
        n_heads         = cfg_dict.get("n_heads",          4),
        n_layers        = cfg_dict.get("n_layers",         3),
        dim_feedforward = cfg_dict.get("dim_feedforward", 128),
        window_size     = cfg_dict.get("window_size",     256),
        dropout         = cfg_dict.get("dropout",         0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return {
        "model":     model,
        "node_mean": ckpt["node_mean"].cpu(),
        "node_std":  ckpt["node_std"].cpu(),
        "config":    cfg_dict,
        "epoch":     ckpt.get("epoch"),
        "best_eff":  ckpt.get("best_eff"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train E320HitFilter")
    parser.add_argument("--clusters",    required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--checkpoint",  default=None, help="Directory to save checkpoint")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()

    import polars as pl
    clusters_df = pl.read_parquet(args.clusters)
    cfg = HitFilterConfig(
        n_epochs                    = args.epochs,
        device                      = args.device,
        checkpoint_dir              = args.checkpoint,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
    )
    train_hit_filter(clusters_df, cfg)


if __name__ == "__main__":
    _cli()
