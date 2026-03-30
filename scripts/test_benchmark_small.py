"""
Small-scale end-to-end benchmark smoke test.

Runs the full pipeline on tiny synthetic data (no large Parquet files needed):
  1. Generate train/test simulation data (toy mode, no real background)
  2. Build labeled edge table
  3. Train embedder (few epochs)
  4. Train each ML model (few epochs, subprocess, same as production)
  5. Run inference (subprocess)
  6. Print comparison table

Usage:
    conda run -n e320root python scripts/test_benchmark_small.py
    conda run -n e320root python scripts/test_benchmark_small.py --epochs 3 --n-train 100 --n-test 50
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import SimConfig, simulate
from src.utils import build_labeled_edges_from_sim
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.compare_reco import compare


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Small-scale benchmark smoke test")
    p.add_argument("--n-train", type=int, default=300,
                   help="Number of training events (default 300)")
    p.add_argument("--n-test",  type=int, default=100,
                   help="Number of test events (default 100)")
    p.add_argument("--epochs",  type=int, default=5,
                   help="Training epochs per model (default 5)")
    p.add_argument("--device",  default="cuda",
                   choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--workdir", default=None,
                   help="Directory for temp checkpoints/outputs (default: /tmp/e320_small_test)")
    p.add_argument("--models", nargs="+",
                   default=["mlp", "gnn", "interaction_net", "eggnet", "hgnn"],
                   help="ML model types to benchmark")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spawn(cmd: list[str], label: str) -> None:
    """Run command in subprocess; raise on failure."""
    print(f"  [spawn] {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"{label} subprocess exited with code {result.returncode}")
    print(f"  [spawn] {label} done in {dt:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / "e320_small_test"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\nWork directory: {workdir}")

    train_clusters_path = workdir / "train_clusters.parquet"
    train_tracks_path   = workdir / "train_tracks.parquet"
    test_clusters_path  = workdir / "test_clusters.parquet"
    test_tracks_path    = workdir / "test_tracks.parquet"
    edges_train_path    = workdir / "edges_train.parquet"
    runs_dir            = workdir / "runs"
    outputs_dir         = workdir / "outputs"
    runs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    # ── 1. Generate data ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 1: Generating data  (train={args.n_train}  test={args.n_test})")
    if not train_clusters_path.exists():
        t0 = time.perf_counter()
        train_cfg = SimConfig(
            background_mode="synthetic",
            mean_n_signal=0.5,
            synthetic_bg_n_per_layer=100,
            n_events=args.n_train,
            seed=42,
        )
        clusters_train, tracks_train = simulate(train_cfg)
        clusters_train.write_parquet(train_clusters_path)
        tracks_train.write_parquet(train_tracks_path)
        print(f"  train: {len(clusters_train):,} clusters  "
              f"{clusters_train['event_id'].n_unique()} events  ({time.perf_counter()-t0:.1f}s)")
    else:
        clusters_train = pl.read_parquet(train_clusters_path)
        print(f"  train: loaded {len(clusters_train):,} clusters (cached)")

    if not test_clusters_path.exists():
        t0 = time.perf_counter()
        test_cfg = SimConfig(
            background_mode="synthetic",
            mean_n_signal=0.5,
            synthetic_bg_n_per_layer=100,
            n_events=args.n_test,
            seed=123,
        )
        clusters_test, tracks_test = simulate(test_cfg)
        clusters_test.write_parquet(test_clusters_path)
        tracks_test.write_parquet(test_tracks_path)
        print(f"  test:  {len(clusters_test):,} clusters  "
              f"{clusters_test['event_id'].n_unique()} events  ({time.perf_counter()-t0:.1f}s)")
    else:
        clusters_test = pl.read_parquet(test_clusters_path)
        tracks_test   = pl.read_parquet(test_tracks_path)
        print(f"  test:  loaded {len(clusters_test):,} clusters (cached)")

    # ── 2. Build edge table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Building labeled edge table")
    if not edges_train_path.exists():
        t0 = time.perf_counter()
        edges_train = build_labeled_edges_from_sim(clusters_train)
        print(f"  {len(edges_train):,} edges  ({time.perf_counter()-t0:.1f}s)")
        edges_train.write_parquet(edges_train_path)
        del edges_train
    else:
        print(f"  edge cache exists (cached at {edges_train_path})")

    # ── 3. Baseline (non-ML) ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Baseline (non-ML)")
    from src.baseline import BaselineConfig
    baseline_cfg = BaselineConfig(n_workers=1)
    reco_paths: dict[str, str] = {}
    t0 = time.perf_counter()
    baseline_result = evaluate_baseline_on_sim(clusters_test, tracks_test, baseline_cfg)
    baseline_out = outputs_dir / "baseline_test.parquet"
    baseline_result.write_parquet(baseline_out)
    reco_paths["Baseline"] = str(baseline_out)
    print(f"  Baseline done  ({time.perf_counter()-t0:.1f}s)")

    # ── 4. Embedder training ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Embedder training")
    embedder_dir  = runs_dir / "embedder"
    embedder_ckpt = embedder_dir / "best_embedder.pt"
    embedder_dir.mkdir(exist_ok=True)
    if not embedder_ckpt.exists():
        _spawn([
            sys.executable, "-m", "src.train",
            "--task", "embedder",
            "--clusters", str(train_clusters_path),
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--checkpoint", str(embedder_dir),
        ], "embedder")
    else:
        print(f"  checkpoint exists → skipping: {embedder_ckpt}")

    # ── 5. ML model training + inference ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 5: ML models  ({args.epochs} epochs each)")

    for model_type in args.models:
        print(f"\n  [{model_type.upper()}]")
        ckpt_dir  = runs_dir / model_type
        ckpt_path = ckpt_dir / "best_model.pt"
        ckpt_dir.mkdir(exist_ok=True)

        # Training
        if not ckpt_path.exists():
            t0 = time.perf_counter()
            try:
                _spawn([
                    sys.executable, "-m", "src.train",
                    "--task", "edge",
                    "--edges", str(edges_train_path),
                    "--model", model_type,
                    "--epochs", str(args.epochs),
                    "--device", args.device,
                    "--checkpoint", str(ckpt_dir),
                    "--embedder-checkpoint", str(embedder_ckpt),
                ], f"{model_type} train")
            except RuntimeError as e:
                print(f"  [ERROR] training failed: {e}")
                continue
            print(f"  training: {time.perf_counter()-t0:.1f}s")
        else:
            print(f"  checkpoint exists → skipping training")

        # Inference
        out = outputs_dir / f"{model_type}_test.parquet"
        t0 = time.perf_counter()
        try:
            _spawn([
                sys.executable, "-m", "scripts.run_model",
                "--mode", "edge",
                "--clusters", str(test_clusters_path),
                "--tracks", str(test_tracks_path),
                "--edge-checkpoint", str(ckpt_path),
                "--device", args.device,
                "--output", str(out),
            ], f"{model_type} infer")
        except RuntimeError as e:
            print(f"  [ERROR] inference failed: {e}")
            continue
        reco_paths[model_type.upper()] = str(out)
        print(f"  inference: {time.perf_counter()-t0:.1f}s")

    # ── 6. Comparison ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Comparison table")
    compare(str(test_tracks_path), reco_paths)

    print(f"\nAll outputs in: {workdir}")
    print("DONE")


if __name__ == "__main__":
    main()
