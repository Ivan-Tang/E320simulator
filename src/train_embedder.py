"""Metric-learning hit embedder training/inference for E320simulator.

Pipeline
--------
1) Build hit pairs from simulated clusters via ``src.utils.build_pairs``.
2) Train an embedding network with hinge embedding loss on pair distances.
3) Save checkpoint (model + normalisation stats).
4) Provide inference helpers for hit embeddings and radius-neighbor query.
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
from sklearn.neighbors import KDTree

from src.models import Embedder
from src.losses import HingeLoss
from src.utils import build_pairs


DEFAULT_HIT_FEAT_COLS = [
    "layer_id",
    "x_trk_mm",
    "y_trk_mm",
    "z_trk_mm",
    "size_x",
    "size_y",
    "size",
]


@dataclass
class EmbedderTrainConfig:
    # data
    hit_feature_cols: list[str] = dataclasses.field(default_factory=lambda: DEFAULT_HIT_FEAT_COLS.copy())
    nb_particles_per_sample: int = 2000
    max_pairs: int | None = 500_000
    val_fraction: float = 0.2
    seed: int = 42

    # model
    emb_dim: int = 3
    hidden_dim: int = 64
    n_layers: int = 3

    # optimisation
    batch_size: int = 4096
    n_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    hinge_margin: float = 1.0

    # evaluation
    distance_threshold: float = 0.5

    # output
    checkpoint_dir: str | None = None
    log_every: int = 1

    # hardware
    device: str = "auto"  # auto/cpu/mps/cuda


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, hits_a: np.ndarray, hits_b: np.ndarray, target: np.ndarray):
        self.hits_a = torch.from_numpy(hits_a.astype(np.float32))
        self.hits_b = torch.from_numpy(hits_b.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.hits_a[idx], self.hits_b[idx], self.target[idx]


def _resolve_device(cfg: EmbedderTrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hit_normalisation_stats(hits_a: np.ndarray, hits_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([hits_a, hits_b], axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def _accuracy_from_distance(pred_dist: torch.Tensor, target: torch.Tensor, distance_threshold: float) -> float:
    pred_pos = pred_dist < distance_threshold
    true_pos = target > 0.5
    return float((pred_pos == true_pos).float().mean().item())


def build_pair_dataset_from_clusters(
    clusters_df: pl.DataFrame,
    cfg: EmbedderTrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (hits_a, hits_b, target) from simulator clusters using build_pairs."""
    needed = {"event_id", "track_id", "layer_id", *cfg.hit_feature_cols}
    missing = needed - set(clusters_df.columns)
    if missing:
        raise ValueError(f"clusters_df missing columns for embedder training: {missing}")

    rng = np.random.default_rng(cfg.seed)
    hits_a_all: list = []
    hits_b_all: list = []
    target_all: list[int] = []

    for _, ev in clusters_df.group_by("event_id"):
        ev = ev.sort("node_id") if "node_id" in ev.columns else ev

        hits = ev.select(cfg.hit_feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        pids = ev["track_id"].to_numpy()

        # E320 simulation uses layers 0..4. Keep a synthetic volume id=0 for compatibility.
        layers = ev["layer_id"].to_numpy().astype(np.int32)
        vols = np.zeros_like(layers, dtype=np.int32)

        h_a, h_b, t = build_pairs(
            hits=hits,
            particle_ids=pids,
            vols=vols,
            layers=layers,
            nb_particles_per_sample=cfg.nb_particles_per_sample,
            rng=rng,
        )
        if len(t) == 0:
            continue

        hits_a_all.extend(h_a)
        hits_b_all.extend(h_b)
        target_all.extend(t)

        if cfg.max_pairs is not None and len(target_all) >= cfg.max_pairs:
            break

    if len(target_all) == 0:
        return (
            np.empty((0, len(cfg.hit_feature_cols)), dtype=np.float32),
            np.empty((0, len(cfg.hit_feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    if cfg.max_pairs is not None and len(target_all) > cfg.max_pairs:
        hits_a_all = hits_a_all[: cfg.max_pairs]
        hits_b_all = hits_b_all[: cfg.max_pairs]
        target_all = target_all[: cfg.max_pairs]

    return (
        np.asarray(hits_a_all, dtype=np.float32),
        np.asarray(hits_b_all, dtype=np.float32),
        np.asarray(target_all, dtype=np.float32),
    )


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: HingeLoss,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    distance_threshold: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []

    with torch.no_grad():
        for hits_a, hits_b, target in loader:
            hits_a = hits_a.to(device)
            hits_b = hits_b.to(device)
            target = target.to(device)

            hits_a = (hits_a - mean_t) / std_t
            hits_b = (hits_b - mean_t) / std_t

            emb_a = model(hits_a)
            emb_b = model(hits_b)
            pred_dist = nn.functional.pairwise_distance(emb_a, emb_b)

            loss = criterion(pred_dist, target)
            acc = _accuracy_from_distance(pred_dist, target, distance_threshold)

            losses.append(float(loss.item()))
            accs.append(acc)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
    }


def train_embedder(
    clusters_df: pl.DataFrame,
    cfg: EmbedderTrainConfig | None = None,
) -> dict:
    """Train metric-learning embedder from simulated clusters."""
    if cfg is None:
        cfg = EmbedderTrainConfig()

    device = _resolve_device(cfg)
    print(f"[embed-train] device={device}  epochs={cfg.n_epochs}")

    hits_a, hits_b, target = build_pair_dataset_from_clusters(clusters_df, cfg)
    if len(target) == 0:
        raise ValueError("No training pairs were produced. Check signal tracks and sampling settings.")

    n_pos = int(target.sum())
    n_total = len(target)
    print(f"[embed-train] pairs={n_total:,}  pos={n_pos:,}  pos_frac={n_pos/max(n_total,1):.4f}")

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)

    n_train = int(n_total * (1.0 - cfg.val_fraction))
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:]

    x_a_tr, x_b_tr, y_tr = hits_a[tr_idx], hits_b[tr_idx], target[tr_idx]
    x_a_va, x_b_va, y_va = hits_a[va_idx], hits_b[va_idx], target[va_idx]

    mean, std = _hit_normalisation_stats(x_a_tr, x_b_tr)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    train_ds = PairDataset(x_a_tr, x_b_tr, y_tr)
    val_ds = PairDataset(x_a_va, x_b_va, y_va)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = Embedder(
        in_dim=len(cfg.hit_feature_cols),
        out_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
    ).to(device)

    criterion = HingeLoss(margin=cfg.hinge_margin)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt_dir: Path | None = None
    if cfg.checkpoint_dir:
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    history: list[dict] = []
    best_val_loss = float("inf")
    best_path: Path | None = None
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for h_a, h_b, tgt in train_loader:
            h_a = h_a.to(device)
            h_b = h_b.to(device)
            tgt = tgt.to(device)

            h_a = (h_a - mean_t) / std_t
            h_b = (h_b - mean_t) / std_t

            optimizer.zero_grad()
            emb_a = model(h_a)
            emb_b = model(h_b)
            pred_dist = nn.functional.pairwise_distance(emb_a, emb_b)

            loss = criterion(pred_dist, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_losses.append(float(loss.item()))
            batch_accs.append(_accuracy_from_distance(pred_dist.detach(), tgt, cfg.distance_threshold))

        tr_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        tr_acc = float(np.mean(batch_accs)) if batch_accs else float("nan")

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc}

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            val_metrics = _evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                mean_t=mean_t,
                std_t=std_t,
                distance_threshold=cfg.distance_threshold,
            )
            row["val_loss"] = val_metrics["loss"]
            row["val_acc"] = val_metrics["acc"]

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{cfg.n_epochs} | "
                f"tr_loss={tr_loss:.5f} tr_acc={tr_acc:.4f} | "
                f"va_loss={val_metrics['loss']:.5f} va_acc={val_metrics['acc']:.4f} | "
                f"t={elapsed:.0f}s"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if ckpt_dir is not None:
                    best_path = ckpt_dir / "best_embedder.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "best_val_loss": best_val_loss,
                            "feature_cols": cfg.hit_feature_cols,
                            "mean": mean,
                            "std": std,
                            "emb_dim": cfg.emb_dim,
                            "hidden_dim": cfg.hidden_dim,
                            "n_layers": cfg.n_layers,
                        },
                        best_path,
                    )

        history.append(row)

    print(f"[embed-train] done  best_val_loss={best_val_loss:.5f}  checkpoint={best_path}")
    return {
        "model": model.cpu(),
        "history": history,
        "best_val_loss": best_val_loss,
        "checkpoint": str(best_path) if best_path else None,
        "mean": mean,
        "std": std,
        "feature_cols": cfg.hit_feature_cols,
    }


def load_embedder_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Embedder(
        in_dim=len(ckpt["feature_cols"]),
        out_dim=int(ckpt["emb_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        n_layers=int(ckpt["n_layers"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return {
        "model": model,
        "feature_cols": ckpt["feature_cols"],
        "mean": np.asarray(ckpt["mean"], dtype=np.float32),
        "std": np.asarray(ckpt["std"], dtype=np.float32),
        "epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
    }


def embed_hits(
    model: nn.Module,
    hits: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cpu",
    batch_size: int = 8192,
) -> np.ndarray:
    """Run embedding inference on hit features."""
    model = model.to(device)
    model.eval()

    x = np.asarray(hits, dtype=np.float32)
    x = (x - mean) / np.clip(std, 1e-6, None)

    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[s: s + batch_size]).to(device)
            emb = model(xb).cpu().numpy()
            out_chunks.append(emb)

    if not out_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


def infer_embedding_neighbors(
    clusters_df: pl.DataFrame,
    checkpoint_path: str,
    radius: float = 1.0,
    feature_cols: list[str] | None = None,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Infer per-event embedding neighbors with radius query.

    Returns
    -------
    dict[event_id -> np.ndarray of object], where each value is the output of
    ``KDTree.query_radius(embeddings, r=radius)`` for that event.
    """
    loaded = load_embedder_checkpoint(checkpoint_path, device=device)
    model = loaded["model"]

    cols = feature_cols if feature_cols is not None else loaded["feature_cols"]
    missing = set(cols) - set(clusters_df.columns)
    if missing:
        raise ValueError(f"clusters_df missing feature columns for inference: {missing}")

    out: dict[int, np.ndarray] = {}
    for eid, ev in clusters_df.group_by("event_id"):
        event_id = int(eid[0]) if isinstance(eid, (list, tuple)) else int(eid)
        hits = ev.select(cols).to_numpy(allow_copy=True).astype(np.float32)

        emb = embed_hits(
            model=model,
            hits=hits,
            mean=loaded["mean"],
            std=loaded["std"],
            device=device,
        )
        tree = KDTree(emb)
        out[event_id] = tree.query_radius(emb, r=radius)

    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train metric-learning embedder for E320simulator")
    parser.add_argument("--clusters", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nb-particles-per-sample", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=500000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    from src.config import SIM_DIR, RUNS_DIR
    clusters_path = args.clusters or str(SIM_DIR / "sim_clusters_train.parquet")
    checkpoint_dir = args.checkpoint or str(RUNS_DIR / "embedder")

    clusters_df = pl.read_parquet(clusters_path)

    cfg = EmbedderTrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        nb_particles_per_sample=args.nb_particles_per_sample,
        max_pairs=args.max_pairs,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )
    train_embedder(clusters_df, cfg)


if __name__ == "__main__":
    _cli()
