"""
Horizontal benchmark: compare non-ML and ML track-finding algorithms on E320 simulation.

Non-ML (run on test only):
  - Baseline  (slope-window + chain seeding)
  - Hough Transform

ML (train on train, evaluate on test):
  - MLP, GNN, ResGNN, MPNN, AGNN, EggNet

Usage:
    cd /Users/IvanTang/hep/E320simulator
    python scripts/run_benchmark.py [--device mps] [--epochs 50] [--force-retrain]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import TrainConfig, train
from src.utils import build_labeled_edges_from_sim
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.run_hough import evaluate_hough_on_sim
from scripts.run_model import run_edge_classifier_reco
from scripts.compare_reco import compare

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path.home() / "hep/data_Run502"
SIM_DIR        = _BASE / "simulation"
RUNS_DIR       = _BASE / "runs"
OUT_DIR        = _BASE / "outputs"

TRAIN_CLUSTERS = SIM_DIR / "sim_clusters_train.parquet"
TEST_CLUSTERS  = SIM_DIR / "sim_clusters_test.parquet"
TEST_TRACKS    = SIM_DIR / "sim_tracks_test.parquet"

# ML model types to benchmark
ML_MODELS = ["mlp", "gnn", "resgnn", "mpnn", "agnn", "eggnet"]


# ── Training helper ───────────────────────────────────────────────────────────

def train_ml_model(
    model_type: str,
    clusters_df: pl.DataFrame,
    checkpoint_dir: Path,
    device: str = "mps",
    n_epochs: int = 50,
    force: bool = False,
) -> str:
    """Build edges, train model, and save checkpoint.  Returns checkpoint path."""
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists() and not force:
        print(f"  checkpoint exists → skipping training: {ckpt_path}")
        return str(ckpt_path)

    print(f"  Building labeled edges from train clusters...")
    edges_df = build_labeled_edges_from_sim(clusters_df)
    print(f"  Built {len(edges_df):,} candidate edges")

    cfg = TrainConfig(
        model_type     = model_type,
        n_epochs       = n_epochs,
        hidden         = 64,
        n_mp           = 2,
        lr             = 3e-4,
        device         = device,
        checkpoint_dir = str(checkpoint_dir),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train(edges_df, cfg)
    return str(ckpt_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(device: str = "mps", force_retrain: bool = False, epochs: int = 50) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Loading data...")
    clusters_train = pl.read_parquet(TRAIN_CLUSTERS)
    clusters_test  = pl.read_parquet(TEST_CLUSTERS)
    tracks_test    = pl.read_parquet(TEST_TRACKS)
    print(f"  Train clusters : {len(clusters_train):,}")
    print(f"  Test  clusters : {len(clusters_test):,}")
    print(f"  Test  tracks   : {len(tracks_test):,}")

    reco_paths: dict[str, str] = {}

    # ── Non-ML algorithms ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Non-ML algorithms (test set only)")

    for name, fn in [
        ("Baseline", lambda: evaluate_baseline_on_sim(clusters_test, tracks_test)),
        ("Hough",    lambda: evaluate_hough_on_sim(clusters_test, tracks_test)),
    ]:
        print(f"\n[{name}]")
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        out = OUT_DIR / f"{name.lower()}_test.parquet"
        result.write_parquet(out)
        reco_paths[name] = str(out)
        print(f"  → {out}  ({dt:.1f}s)")

    # ── ML algorithms ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ML algorithms  (train={epochs} epochs on {device}, then test)")

    for model_type in ML_MODELS:
        print(f"\n[{model_type.upper()}]")
        ckpt_dir  = RUNS_DIR / model_type
        ckpt_path = str(ckpt_dir / "best_model.pt")

        # Train
        t0 = time.perf_counter()
        try:
            ckpt_path = train_ml_model(
                model_type, clusters_train, ckpt_dir,
                device=device, n_epochs=epochs, force=force_retrain,
            )
        except Exception as e:
            print(f"  [ERROR] training failed: {e}")
            continue
        train_dt = time.perf_counter() - t0

        # Inference
        t0 = time.perf_counter()
        try:
            result = run_edge_classifier_reco(
                clusters_test, tracks_test, ckpt_path,
                threshold=0.5, device=device,
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            continue
        infer_dt = time.perf_counter() - t0

        out = OUT_DIR / f"{model_type}_test.parquet"
        result.write_parquet(out)
        reco_paths[model_type.upper()] = str(out)
        print(f"  → {out}  (train {train_dt:.0f}s  infer {infer_dt:.1f}s)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    compare(str(TEST_TRACKS), reco_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark E320 tracking algorithms")
    parser.add_argument("--device",        default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train even if checkpoint already exists")
    args = parser.parse_args()
    main(device=args.device, force_retrain=args.force_retrain, epochs=args.epochs)
