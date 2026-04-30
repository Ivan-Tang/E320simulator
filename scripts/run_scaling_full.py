"""Full benchmark sweep for a single (bg_per_layer, mean_n_signal) operating point.

Generates fresh simulation data from SimConfig, then runs the complete benchmark
(Baseline, Hough, MLP, GNN, InteractionNet, EggNet, HGNN) and saves a JSON
summary.  One PBS job per sweep point; submit at most 2 in parallel.

Output layout (rooted at DATA_ROOT):
    scaling/<tag>/sim_clusters_train.parquet
    scaling/<tag>/sim_clusters_test.parquet
    scaling/<tag>/sim_tracks_test.parquet
    scaling/<tag>/edges_train.parquet

Checkpoints:  RUNS_DIR/scaling/<tag>/<model>/best_model.pt
Results:      OUTPUTS_DIR/scaling/<tag>/results.json

Usage:
    python scripts/run_scaling_full.py \
        --tag bg700_sig012 \
        --bg-per-layer 700 \
        --mean-n-signal 0.12 \
        --device cuda \
        --epochs 200 \
        --workers 4 \
        --n-events-train 5000 \
        --n-events-test 2000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_ROOT, RUNS_DIR, OUTPUTS_DIR
from src.simulator import SimConfig, simulate_train_test
from src.baseline import BaselineConfig
from src.hough_baseline import HoughConfig
from src.utils import build_labeled_edges_from_sim
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.run_hough import evaluate_hough_on_sim
from scripts.run_model import run_edge_classifier_reco
from scripts.compare_reco import compute_metrics


ML_MODELS = ["mlp", "gnn", "interaction_net", "eggnet", "hgnn"]


def _f1(m: dict) -> float:
    recall = m.get("efficiency_%", 0.0) / 100.0
    fake_rate = m.get("fake_rate_%", 0.0) / 100.0
    precision = 1.0 - fake_rate
    denom = precision + recall
    if denom < 1e-9:
        return 0.0
    return 2.0 * precision * recall / denom


def train_ml_model(
    model_type: str,
    checkpoint_dir: Path,
    edges_file: Path,
    device: str,
    n_epochs: int,
    force: bool,
) -> str:
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists() and not force:
        print(f"  checkpoint exists → skipping: {ckpt_path}")
        return str(ckpt_path)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    extra_args = [
        "--task", "edge",
        "--edges", str(edges_file),
        "--model", model_type,
        "--epochs", str(n_epochs),
        "--device", device,
        "--checkpoint", str(checkpoint_dir),
    ]
    cmd = [sys.executable, "-m", "src.train", *extra_args]
    print(f"  [spawn] {' '.join(cmd)}")
    import os as _os
    env = _os.environ.copy()
    env["POLARS_MAX_THREADS"] = "1"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return str(ckpt_path)


def run_inference(
    model_type: str,
    ckpt_path: str,
    clusters_test: pl.DataFrame,
    tracks_test: pl.DataFrame,
    out_path: Path,
    device: str,
) -> dict:
    if not Path(ckpt_path).exists():
        print(f"  [SKIP] checkpoint missing: {ckpt_path}")
        return {}
    try:
        reco = run_edge_classifier_reco(
            clusters_test, tracks_test,
            checkpoint_path=ckpt_path,
            threshold=0.1,
            device=device,
        )
        if reco.is_empty():
            return {}
        reco.write_parquet(out_path)
        return compute_metrics(reco, tracks_test)
    except Exception as e:
        print(f"  [ERROR] inference failed for {model_type}: {e}")
        return {}


def main(
    tag: str,
    bg_per_layer: int,
    mean_n_signal: float,
    device: str,
    epochs: int,
    workers: int,
    n_events_train: int,
    n_events_test: int,
    force_retrain: bool,
) -> None:
    t_start = time.perf_counter()

    # ── Output dirs ─────────────────────────────────────────────────────────────
    sim_dir  = DATA_ROOT / "scaling" / tag
    runs_dir = RUNS_DIR / "scaling" / tag
    out_dir  = OUTPUTS_DIR / "scaling" / tag
    edges_train_path = sim_dir / "edges_train.parquet"
    train_clusters_path = sim_dir / "sim_clusters_train.parquet"
    test_clusters_path  = sim_dir / "sim_clusters_test.parquet"
    test_tracks_path    = sim_dir / "sim_tracks_test.parquet"
    sim_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Scaling sweep point: tag={tag}")
    print(f"  bg_per_layer={bg_per_layer}  mean_n_signal={mean_n_signal}")
    print(f"  n_events_train={n_events_train}  n_events_test={n_events_test}")
    print(f"  device={device}  epochs={epochs}  workers={workers}")
    print("=" * 65)

    # ── Generate sim data ────────────────────────────────────────────────────────
    if train_clusters_path.exists() and not force_retrain:
        print("\nLoading cached simulation data...")
        clusters_train = pl.read_parquet(train_clusters_path)
        clusters_test  = pl.read_parquet(test_clusters_path)
        tracks_test    = pl.read_parquet(test_tracks_path)
    else:
        print("\nGenerating simulation data...")
        t0 = time.perf_counter()
        cfg_train = SimConfig(
            n_events=n_events_train,
            mean_n_signal=mean_n_signal,
            synthetic_bg_n_per_layer=bg_per_layer,
            background_mode="synthetic",
            cluster_size_mode="fixed",
            seed=42,
            mode="train",
        )
        cfg_test = SimConfig(
            n_events=n_events_test,
            mean_n_signal=mean_n_signal,
            synthetic_bg_n_per_layer=bg_per_layer,
            background_mode="synthetic",
            cluster_size_mode="fixed",
            seed=123,
            mode="test",
        )
        from src.simulator import simulate
        clusters_train, tracks_train = simulate(cfg_train)
        clusters_test,  tracks_test  = simulate(cfg_test)

        clusters_train.write_parquet(train_clusters_path)
        clusters_test.write_parquet(test_clusters_path)
        tracks_test.write_parquet(test_tracks_path)
        print(f"  Train: {len(clusters_train):,} clusters / {clusters_train['event_id'].n_unique():,} events")
        print(f"  Test : {len(clusters_test):,} clusters / {clusters_test['event_id'].n_unique():,} events")
        print(f"  Data generation: {time.perf_counter() - t0:.1f}s")

    n_test_events = clusters_test["event_id"].n_unique()

    results: dict[str, dict] = {}

    # ── Non-ML ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Non-ML algorithms")
    baseline_cfg = BaselineConfig(n_workers=workers)
    hough_cfg    = HoughConfig(n_workers=workers)

    for name, fn in [
        ("Baseline", lambda: evaluate_baseline_on_sim(clusters_test, tracks_test, baseline_cfg)),
        ("Hough",    lambda: evaluate_hough_on_sim(clusters_test, tracks_test, hough_cfg)),
    ]:
        out_path = out_dir / f"{name.lower()}_test.parquet"
        if out_path.exists() and not force_retrain:
            print(f"  [{name}] cached → loading metrics")
            reco = pl.read_parquet(out_path)
            metrics = compute_metrics(reco, tracks_test)
        else:
            print(f"\n[{name}]")
            t0 = time.perf_counter()
            reco = fn()
            reco.write_parquet(out_path)
            metrics = compute_metrics(reco, tracks_test)
            dt = time.perf_counter() - t0
            print(f"  → {dt:.1f}s  eff={metrics.get('efficiency_%', 0):.1f}%  fake={metrics.get('fake_rate_%', 0):.1f}%")
        metrics["f1"] = _f1(metrics)
        results[name] = metrics

    # ── Build edge table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Building/loading edge table...")
    _REQUIRED_COLS = {"edge_label", "event_id", "is_signal_edge"}
    if edges_train_path.exists() and not force_retrain:
        cached_cols = set(pl.read_parquet_schema(edges_train_path).names())
        if not _REQUIRED_COLS.issubset(cached_cols):
            print("  Stale edge cache — rebuilding...")
            edges_train_path.unlink()
    if not edges_train_path.exists():
        print(f"  Building labeled edges from {clusters_train['event_id'].n_unique():,} events...")
        t0 = time.perf_counter()
        edges_train = build_labeled_edges_from_sim(clusters_train)
        print(f"  Built {len(edges_train):,} edges  ({time.perf_counter() - t0:.1f}s)")
        edges_train.write_parquet(edges_train_path)
        del edges_train
    else:
        print(f"  Edge cache valid → {edges_train_path}")

    # ── ML models ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ML models  ({epochs} epochs, {device})")

    for model_type in ML_MODELS:
        print(f"\n[{model_type.upper()}]")
        ckpt_dir  = runs_dir / model_type
        ckpt_path = str(ckpt_dir / "best_model.pt")

        t0 = time.perf_counter()
        try:
            ckpt_path = train_ml_model(
                model_type, ckpt_dir, edges_train_path,
                device=device, n_epochs=epochs, force=force_retrain,
            )
        except Exception as e:
            print(f"  [ERROR] training failed: {e}")
            results[model_type] = {}
            continue
        train_dt = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = run_inference(
            model_type, ckpt_path, clusters_test, tracks_test,
            out_path=out_dir / f"{model_type}_test.parquet",
            device=device,
        )
        infer_dt = time.perf_counter() - t0

        if metrics:
            metrics["f1"] = _f1(metrics)
            print(f"  eff={metrics.get('efficiency_%', 0):.1f}%  fake={metrics.get('fake_rate_%', 0):.1f}%"
                  f"  F1={metrics['f1']:.3f}  (train {train_dt:.0f}s  infer {infer_dt:.1f}s)")
        results[model_type] = metrics

    # ── Summary ──────────────────────────────────────────────────────────────────
    total_dt = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print(f"RESULTS  tag={tag}  bg_per_layer={bg_per_layer}  mean_n_signal={mean_n_signal}")
    print(f"{'Model':20s}  {'Efficiency':>10s}  {'Fake Rate':>9s}  {'F1':>7s}")
    print("-" * 52)
    for name, m in results.items():
        if not m:
            print(f"{name:20s}  {'FAILED':>10s}")
            continue
        print(f"{name:20s}  {m.get('efficiency_%', float('nan')):9.1f}%"
              f"  {m.get('fake_rate_%', float('nan')):8.1f}%"
              f"  {m.get('f1', float('nan')):6.3f}")

    summary = {
        "tag": tag,
        "bg_per_layer": bg_per_layer,
        "mean_n_signal": mean_n_signal,
        "n_events_train": n_events_train,
        "n_events_test": n_events_test,
        "epochs": epochs,
        "total_hits_per_event_approx": bg_per_layer * 5,
        "total_time_s": total_dt,
        "results": results,
    }
    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda v: None if (isinstance(v, float) and v != v) else v)
    print(f"\nSaved summary → {summary_path}")
    print(f"Total elapsed: {total_dt/3600:.1f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full benchmark for one scaling sweep point")
    parser.add_argument("--tag",             required=True, help="Unique label for this sweep point (e.g. bg700_sig012)")
    parser.add_argument("--bg-per-layer",    type=int,   required=True, help="Synthetic background clusters per layer per event")
    parser.add_argument("--mean-n-signal",   type=float, default=0.12,  help="Poisson mean signal tracks per event (default 0.12)")
    parser.add_argument("--device",          default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--workers",         type=int,   default=4,     help="Workers for non-ML algorithms")
    parser.add_argument("--n-events-train",  type=int,   default=5000)
    parser.add_argument("--n-events-test",   type=int,   default=2000)
    parser.add_argument("--force-retrain",   action="store_true")
    args = parser.parse_args()

    main(
        tag=args.tag,
        bg_per_layer=args.bg_per_layer,
        mean_n_signal=args.mean_n_signal,
        device=args.device,
        epochs=args.epochs,
        workers=args.workers,
        n_events_train=args.n_events_train,
        n_events_test=args.n_events_test,
        force_retrain=args.force_retrain,
    )
