"""Training for E320TrackFormer — MaskFormer-style set-prediction tracker.

Based on Van Stroud et al. (2025) "Transformers for Charged Particle Track
Reconstruction in High Energy Physics" (PRX 15, 041046).

Key differences from standard edge-classification training:
- No edge graph required: hits are fed directly as a set
- Hungarian matching between Q predicted track queries and n_true truth tracks
- Set-prediction loss: class BCE + mask BCE + mask Dice + auxiliary losses
  from intermediate decoder layers

Usage
-----
    from src.train_trackformer import TrackFormerConfig, train_trackformer
    cfg = TrackFormerConfig(n_epochs=100, device="mps", checkpoint_dir="runs/transformer")
    result = train_trackformer(clusters_df, cfg)
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
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

from src.models import E320TrackFormer
from src.utils import NODE_FEAT_COLS_SRC, NODE_DIM
from src.train_hit_filter import load_hit_filter_checkpoint
import torch.distributed as dist
import src.ddp as ddp


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackFormerConfig:
    # model architecture
    d_model:          int   = 128
    n_heads:          int   = 4
    n_encoder_layers: int   = 4
    n_decoder_layers: int   = 4
    dim_feedforward:  int   = 256
    max_queries:      int   = 30    # upper bound on tracks per event
    dropout:          float = 0.1

    # loss weights (following DETR/MaskFormer conventions)
    class_loss_weight: float = 1.0
    mask_bce_weight:   float = 5.0
    mask_dice_weight:  float = 5.0
    aux_loss_weight:   float = 0.5   # multiplier on each intermediate decoder layer
    pos_class_weight:  float = 10.0  # up-weight positive class (track vs. no-track)

    # optimiser
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    grad_clip:    float = 1.0

    # scheduler
    n_epochs:   int   = 50
    lr_eta_min: float = 1e-6

    # data
    val_fraction: float = 0.2
    seed:         int   = 42
    min_hits_per_track: int = 3  # skip truth tracks with fewer hits

    # hit filter (Stage 1) — frozen during MaskFormer training
    hit_filter_checkpoint: str | None = None   # path to hit_filter.pt
    hit_filter_threshold:  float      = 0.1    # low threshold → high signal recall

    # output
    checkpoint_dir: str | None = None
    log_every:      int        = 1

    # hardware
    device: str = "auto"
    gradient_accumulation_steps: int = 1  # accumulate gradients over N events before optimizer step


# ──────────────────────────────────────────────────────────────────────────────
# Per-event data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_hit_filter(cfg: TrackFormerConfig, device: torch.device):
    """Load frozen hit filter if configured; returns (model, mean, std) or None."""
    if cfg.hit_filter_checkpoint is None:
        return None
    info = load_hit_filter_checkpoint(cfg.hit_filter_checkpoint, device=str(device))
    info["model"].eval()
    for p in info["model"].parameters():
        p.requires_grad_(False)
    return info


def _apply_hit_filter(
    nf_raw: torch.Tensor,       # (N, 7) CPU
    truth_masks: torch.Tensor,  # (n_true, N)
    hf_info: dict,
    threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply frozen hit filter; return (nf_filtered_raw, truth_masks_filtered).

    Hits predicted as noise are removed.  Truth masks are re-indexed to the
    kept hits.  Truth tracks that lose all hits are also removed.
    """
    model     = hf_info["model"]
    hf_mean   = hf_info["node_mean"].cpu()
    hf_std    = hf_info["node_std"].cpu()

    with torch.no_grad():
        nf_norm = (nf_raw - hf_mean) / hf_std
        logits  = model(nf_norm.to(device)).cpu()
    keep = logits.sigmoid() >= threshold   # (N,) bool

    nf_filtered = nf_raw[keep]            # (N_kept, 7)

    if truth_masks.shape[0] > 0:
        tm_filtered = truth_masks[:, keep]          # (n_true, N_kept)
        valid_tracks = tm_filtered.any(dim=1)       # drop tracks with 0 kept hits
        tm_filtered  = tm_filtered[valid_tracks]
    else:
        tm_filtered = truth_masks[:, keep]

    return nf_filtered, tm_filtered


def _resolve_device(cfg: TrackFormerConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_hit_features(
    ev: pl.DataFrame,
) -> tuple[torch.Tensor, np.ndarray, dict[int, int]]:
    """Return (node_feat (N,7), node_ids (N,), nid_to_local).

    node_feat columns: [layer_id, x_trk_mm, y_trk_mm, z_trk_mm, size_x, size_y, size]
    """
    ev = ev.sort("node_id")
    node_ids  = ev["node_id"].to_numpy()
    feat_cols = ["layer_id", "x_trk_mm", "y_trk_mm", "z_trk_mm", "size_x", "size_y", "size"]
    node_feat = torch.from_numpy(
        ev.select(feat_cols).to_numpy(allow_copy=True).astype(np.float32)
    )
    nid_to_local = {int(n): i for i, n in enumerate(node_ids)}
    return node_feat, node_ids, nid_to_local


def _extract_truth_masks(
    ev: pl.DataFrame,
    nid_to_local: dict[int, int],
    min_hits: int = 3,
) -> torch.Tensor:
    """Build binary truth mask tensor (n_true, N).

    Each row corresponds to one signal track (track_id >= 0).
    Rows where fewer than min_hits nodes appear are skipped.
    Returns empty tensor (0, N) if no valid signal tracks.
    """
    N = len(nid_to_local)
    signal = ev.filter(pl.col("track_id") >= 0)
    track_ids = signal["track_id"].unique().sort().to_list()

    masks: list[torch.Tensor] = []
    for tid in track_ids:
        hits = signal.filter(pl.col("track_id") == tid)
        local_idx = [nid_to_local[int(n)] for n in hits["node_id"].to_list()
                     if int(n) in nid_to_local]
        if len(local_idx) < min_hits:
            continue
        m = torch.zeros(N)
        m[local_idx] = 1.0
        masks.append(m)

    if not masks:
        return torch.zeros(0, N)
    return torch.stack(masks)   # (n_true, N)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_normalisation(
    clusters_df: pl.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_cols = ["layer_id", "x_trk_mm", "y_trk_mm", "z_trk_mm", "size_x", "size_y", "size"]
    arr = clusters_df.select(feat_cols).to_numpy(allow_copy=True).astype(np.float32)
    mean = torch.tensor(arr.mean(0), dtype=torch.float32)
    std  = torch.tensor(arr.std(0).clip(1e-6), dtype=torch.float32)
    return mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Hungarian matching + set-prediction loss
# ──────────────────────────────────────────────────────────────────────────────

def _hungarian_match(
    pred_mask_probs: torch.Tensor,   # (Q, N)  detached sigmoid probs
    pred_class_probs: torch.Tensor,  # (Q,)    detached sigmoid probs
    truth_masks: torch.Tensor,       # (n_true, N)
) -> tuple[list[int], list[int]]:
    """Bipartite matching: minimise cost between Q predictions and n_true tracks.

    Cost per pair (q, t):
        BCE(pred_mask_q, truth_mask_t)  +  Dice(pred_mask_q, truth_mask_t)
        - log(pred_class_q)              (prefer high-confidence matches)

    Returns matched (pred_indices, gt_indices).
    """
    Q  = pred_mask_probs.shape[0]
    T  = truth_masks.shape[0]

    p = pred_mask_probs.unsqueeze(1).expand(-1, T, -1)  # (Q, T, N)
    t = truth_masks.unsqueeze(0).expand(Q, -1, -1)      # (Q, T, N)

    # BCE cost
    bce = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t, reduction="none").mean(-1)  # (Q, T)

    # Dice cost
    inter      = (p * t).sum(-1)                                          # (Q, T)
    dice_cost  = 1.0 - 2.0 * inter / (p.sum(-1) + t.sum(-1) + 1e-6)      # (Q, T)

    # Class cost: reward matching a high-confidence query to a real track
    class_cost = -torch.log(pred_class_probs.clamp(1e-6) ).unsqueeze(1).expand(-1, T)  # (Q, T)

    cost = bce + dice_cost + 0.1 * class_cost   # (Q, T)
    row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
    return list(row_idx), list(col_idx)


def _set_prediction_loss(
    outputs: dict,
    truth_masks: torch.Tensor,   # (n_true, N) on same device as model
    cfg: TrackFormerConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute the full MaskFormer set-prediction loss.

    Components (following Van Stroud et al. §III.D):
    - Class loss : binary cross-entropy (all Q queries)
    - Mask BCE   : binary cross-entropy on matched pair masks
    - Mask Dice  : dice loss on matched pair masks
    - Aux losses : same mask losses at each intermediate decoder layer

    Returns scalar loss tensor.
    """
    Q = outputs["track_logits"].shape[0]
    T = truth_masks.shape[0]

    pred_mask_probs  = outputs["mask_logits"].sigmoid()   # (Q, N)
    pred_class_probs = outputs["track_logits"].sigmoid()  # (Q,)

    # ── Hungarian matching ────────────────────────────────────────────────────
    if T == 0:
        # No truth tracks: push all queries to zero class
        class_loss = F.binary_cross_entropy_with_logits(
            outputs["track_logits"],
            torch.zeros(Q, device=device),
        )
        return cfg.class_loss_weight * class_loss

    with torch.no_grad():
        pred_idx, gt_idx = _hungarian_match(
            pred_mask_probs.detach(), pred_class_probs.detach(), truth_masks
        )

    # ── Class loss (all queries) ──────────────────────────────────────────────
    class_target = torch.zeros(Q, device=device)
    for pi in pred_idx:
        class_target[pi] = 1.0
    pos_w = torch.tensor(cfg.pos_class_weight, device=device)
    class_loss = F.binary_cross_entropy_with_logits(
        outputs["track_logits"], class_target,
        pos_weight=pos_w,
    )

    if not pred_idx:
        return cfg.class_loss_weight * class_loss

    # ── Mask loss (matched queries only) ─────────────────────────────────────
    def _mask_loss(mask_logits: torch.Tensor) -> torch.Tensor:
        matched_pred = mask_logits[pred_idx]                             # (M, N)
        matched_gt   = truth_masks[gt_idx].to(device)                   # (M, N)
        bce  = F.binary_cross_entropy_with_logits(matched_pred, matched_gt)
        probs = matched_pred.sigmoid()
        inter = (probs * matched_gt).sum(-1)
        dice = (1.0 - 2.0 * inter / (probs.sum(-1) + matched_gt.sum(-1) + 1e-6)).mean()
        return cfg.mask_bce_weight * bce + cfg.mask_dice_weight * dice

    main_mask_loss = _mask_loss(outputs["mask_logits"])

    # Auxiliary losses from intermediate decoder layers
    aux_loss = torch.tensor(0.0, device=device)
    for aux_ml in outputs["aux_mask_logits"]:
        aux_loss = aux_loss + cfg.aux_loss_weight * _mask_loss(aux_ml)

    return (
        cfg.class_loss_weight * class_loss
        + main_mask_loss
        + aux_loss
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation metric: per-event track efficiency (loose)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_efficiency(
    model: nn.Module,
    events: list[tuple[torch.Tensor, torch.Tensor, dict]],
    device: torch.device,
    node_mean: torch.Tensor,
    node_std:  torch.Tensor,
    cfg: TrackFormerConfig,
    conf_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_match_frac: float = 0.5,
) -> dict[str, float]:
    """Compute tracking efficiency and fake rate on a set of events."""
    model.eval()
    n_truth = 0
    n_matched = 0
    n_fake = 0

    with torch.no_grad():
        for nf_raw, truth_masks, _ in events:
            nf = ((nf_raw - node_mean.cpu()) / node_std.cpu()).to(device)
            out = model(nf)

            conf = out["track_logits"].sigmoid()        # (Q,)
            mask = out["mask_logits"].sigmoid()          # (Q, N)
            N    = nf.shape[0]
            T    = truth_masks.shape[0]
            n_truth += T

            kept_queries = (conf >= conf_threshold).nonzero(as_tuple=True)[0]

            truth_matched = set()
            for qi in kept_queries.tolist():
                hit_mask = mask[qi] >= mask_threshold   # (N,) bool
                if not hit_mask.any():
                    continue
                # Check if this candidate matches any truth track
                matched = False
                if T > 0:
                    overlap = (hit_mask.float().cpu().unsqueeze(0) * truth_masks).sum(-1)  # (T,)
                    best_t  = int(overlap.argmax())
                    n_pred_hits  = hit_mask.sum().item()
                    n_truth_hits = truth_masks[best_t].sum().item()
                    iou = overlap[best_t].item() / max(n_pred_hits + n_truth_hits - overlap[best_t].item(), 1)
                    if iou >= min_match_frac and best_t not in truth_matched:
                        n_matched += 1
                        truth_matched.add(best_t)
                        matched = True
                if not matched:
                    n_fake += 1

    eff  = n_matched / n_truth      if n_truth else 0.0
    fake = n_fake / (n_matched + n_fake) if (n_matched + n_fake) else 0.0
    return {"efficiency": eff, "fake_rate": fake, "n_truth": n_truth}


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_trackformer(
    clusters_df: pl.DataFrame,
    cfg: TrackFormerConfig | None = None,
) -> dict:
    """Train E320TrackFormer with set-prediction loss.

    Parameters
    ----------
    clusters_df : pl.DataFrame
        Simulator cluster output (with track_id column).
    cfg : TrackFormerConfig | None
        Training configuration.  Defaults to TrackFormerConfig().

    Returns
    -------
    dict with keys: model, history, node_mean, node_std, best_eff, checkpoint
    """
    if cfg is None:
        cfg = TrackFormerConfig()

    rank, world_size, is_ddp = ddp.setup_ddp()
    device = ddp.resolve_device(cfg.device)
    ddp.ddp_print(f"[train_trackformer] device={device}  epochs={cfg.n_epochs}  "
                  f"d_model={cfg.d_model}  queries={cfg.max_queries}")

    # ── Load frozen hit filter (Stage 1) ─────────────────────────────────────
    hf_info = _load_hit_filter(cfg, device)
    if hf_info is not None:
        hf_info["model"].to(device)
        ddp.ddp_print(f"[train_trackformer] hit filter loaded  "
                      f"threshold={cfg.hit_filter_threshold}")

    # ── Train / val split ────────────────────────────────────────────────────
    all_events = np.array(clusters_df["event_id"].unique().sort().to_numpy(), copy=True)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_events)
    n_train = int(len(all_events) * (1 - cfg.val_fraction))
    train_eids = set(all_events[:n_train].tolist())
    val_eids   = set(all_events[n_train:].tolist())

    train_df = clusters_df.filter(pl.col("event_id").is_in(list(train_eids)))
    val_df   = clusters_df.filter(pl.col("event_id").is_in(list(val_eids)))
    ddp.ddp_print(f"[train_trackformer] train events={len(train_eids)}  val events={len(val_eids)}")

    # ── Normalisation (computed from training hits) ───────────────────────────
    node_mean, node_std = _compute_normalisation(train_df)

    # ── Pre-build per-event tensors (avoid repeated Polars overhead) ──────────
    def _preprocess_events(df: pl.DataFrame) -> list:
        out = []
        for _, ev in df.group_by("event_id"):
            nf, nids, n2l = _extract_hit_features(ev)
            tm = _extract_truth_masks(ev, n2l, cfg.min_hits_per_track)
            out.append((nf, tm, n2l))
        return out

    ddp.ddp_print("[train_trackformer] preprocessing events...")
    train_events_raw = _preprocess_events(train_df)
    val_events_raw   = _preprocess_events(val_df)

    # Apply hit filter (if provided) to reduce 3500 → signal candidates
    if hf_info is not None:
        def _filter_events(events):
            out = []
            for nf, tm, n2l in events:
                nf_f, tm_f = _apply_hit_filter(
                    nf, tm, hf_info, cfg.hit_filter_threshold, device
                )
                out.append((nf_f, tm_f, n2l))
            return out
        train_events = _filter_events(train_events_raw)
        val_events   = _filter_events(val_events_raw)
        kept = np.mean([nf.shape[0] for nf, _, _ in train_events])
        ddp.ddp_print(f"[train_trackformer] after hit filter: mean kept hits/event={kept:.1f}")
    else:
        train_events = train_events_raw
        val_events   = val_events_raw

    # Only train on events that have at least one signal track after filtering
    all_train_events_with_truth = [(nf, tm, n2l) for nf, tm, n2l in train_events
                                   if tm.shape[0] > 0 and nf.shape[0] > 0]
    local_train_events = ddp.shard_event_list(all_train_events_with_truth, rank, world_size)
    ddp.ddp_print(f"[train_trackformer] train events with signal: {len(all_train_events_with_truth)}")

    # ── Model + optimiser ────────────────────────────────────────────────────
    model, raw_model = ddp.maybe_wrap_ddp(
        E320TrackFormer(
            node_dim         = NODE_DIM,
            d_model          = cfg.d_model,
            n_heads          = cfg.n_heads,
            n_encoder_layers = cfg.n_encoder_layers,
            n_decoder_layers = cfg.n_decoder_layers,
            dim_feedforward  = cfg.dim_feedforward,
            max_queries      = cfg.max_queries,
            dropout          = cfg.dropout,
        ),
        device,
    )

    n_params = sum(p.numel() for p in raw_model.parameters())
    ddp.ddp_print(f"[train_trackformer] params={n_params:,}")

    optimizer = optim.AdamW(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_eta_min
    )

    # ── Checkpoint dir (rank-0 only) ──────────────────────────────────────────
    ckpt_dir: Path | None = None
    if cfg.checkpoint_dir and ddp.is_main_process():
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = ckpt_dir / "config.json"
        cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_eff  = -1.0
    best_path: Path | None = None
    node_mean_dev = node_mean.to(device)
    node_std_dev  = node_std.to(device)
    accum_steps = max(1, cfg.gradient_accumulation_steps)
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        rng.shuffle(local_train_events)
        optimizer.zero_grad()
        accum_count = 0

        for i, (nf_raw, truth_masks, _) in enumerate(local_train_events):
            nf = ((nf_raw - node_mean) / node_std).to(device)
            tm = truth_masks.to(device)

            out  = model(nf)
            loss = _set_prediction_loss(out, tm, cfg, device)
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
                metrics = _eval_efficiency(
                    raw_model, val_events, device, node_mean_dev, node_std_dev, cfg
                )
                row.update(metrics)
                elapsed = time.time() - t0
                ddp.ddp_print(
                    f"Epoch {epoch:>3}/{cfg.n_epochs} | loss={avg_loss:.5f} | "
                    f"eff={metrics['efficiency']:.3f} | fake={metrics['fake_rate']:.3f} | "
                    f"t={elapsed:.0f}s"
                )

                if metrics["efficiency"] > best_eff:
                    best_eff = metrics["efficiency"]
                    if ckpt_dir is not None:
                        best_path = ckpt_dir / "best_model.pt"
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

    ddp.ddp_print(f"[train_trackformer] done  best_eff={best_eff:.3f}  checkpoint={best_path}")
    ddp.cleanup_ddp()

    return {
        "model":     raw_model.cpu(),
        "history":   history,
        "node_mean": node_mean,
        "node_std":  node_std,
        "best_eff":  best_eff,
        "checkpoint": str(best_path) if best_path else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def load_trackformer_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """Load a saved E320TrackFormer checkpoint.

    Returns dict with keys: model, node_mean, node_std, config, epoch, best_eff
    """
    # Always load on CPU first to avoid MPS placeholder-storage errors,
    # then move to target device.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})

    model = E320TrackFormer(
        node_dim         = NODE_DIM,
        d_model          = cfg_dict.get("d_model",          128),
        n_heads          = cfg_dict.get("n_heads",           4),
        n_encoder_layers = cfg_dict.get("n_encoder_layers",  4),
        n_decoder_layers = cfg_dict.get("n_decoder_layers",  4),
        dim_feedforward  = cfg_dict.get("dim_feedforward",   256),
        max_queries      = cfg_dict.get("max_queries",       30),
        dropout          = cfg_dict.get("dropout",           0.1),
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
    parser = argparse.ArgumentParser(description="Train E320TrackFormer")
    parser.add_argument("--clusters",              required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--epochs",                type=int,   default=50)
    parser.add_argument("--device",                default="auto")
    parser.add_argument("--checkpoint",            default=None, help="Directory to save checkpoint")
    parser.add_argument("--hit-filter-checkpoint", default=None, help="Path to hit_filter.pt")
    parser.add_argument("--hit-filter-threshold",  type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()

    import polars as pl
    clusters_df = pl.read_parquet(args.clusters)
    cfg = TrackFormerConfig(
        n_epochs                    = args.epochs,
        device                      = args.device,
        checkpoint_dir              = args.checkpoint,
        hit_filter_checkpoint       = args.hit_filter_checkpoint,
        hit_filter_threshold        = args.hit_filter_threshold,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
    )
    train_trackformer(clusters_df, cfg)


if __name__ == "__main__":
    _cli()
