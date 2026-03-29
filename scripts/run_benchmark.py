"""
Horizontal benchmark: compare non-ML and ML track-finding algorithms on E320 simulation.

Non-ML (run on test only):
  - Baseline  (slope-window + chain seeding)
  - Hough Transform

ML (train on train, evaluate on test):
  - MLP, GNN (ResGNN), InteractionNet, EggNet, HGNN

Usage:
    cd /Users/IvanTang/hep/E320simulator
    python scripts/run_benchmark.py [--device mps] [--epochs 200] [--workers 1] [--force-retrain]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import BaselineConfig
from src.hough_baseline import HoughConfig
from src.train import TrainConfig, train
from src.train_embedder import EmbedderTrainConfig, train_embedder
from src.train_trackformer import TrackFormerConfig, train_trackformer
from src.train_hit_filter import HitFilterConfig, train_hit_filter
from src.utils import build_labeled_edges_from_sim
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.run_hough import evaluate_hough_on_sim
from scripts.run_model import run_edge_classifier_reco, run_trackformer_reco
from scripts.compare_reco import compare



# ── Paths ─────────────────────────────────────────────────────────────────────
from src.config import DATA_ROOT, SIM_DIR, RUNS_DIR, OUTPUTS_DIR as OUT_DIR

TRAIN_CLUSTERS = SIM_DIR / "sim_clusters_train.parquet"
TEST_CLUSTERS  = SIM_DIR / "sim_clusters_test.parquet"
TEST_TRACKS    = SIM_DIR / "sim_tracks_test.parquet"
EDGES_TRAIN    = SIM_DIR / "edges_train.parquet"

# ML model types to benchmark
ML_MODELS = ["mlp", "gnn", "interaction_net", "eggnet", "hgnn"]


# ── DDP subprocess helper ──────────────────────────────────────────────────────

def _run_ddp_spawn(
    module: str,
    extra_args: list[str],
    nproc: int,
    accum_steps: int,
) -> None:
    """Launch a training module with DDP via subprocess + file:// rendezvous.

    Avoids ``torchrun --standalone`` which segfaults on this cluster due to a
    bug in ``c10d_rendezvous_backend._call_store``.  Instead we manually spawn
    one subprocess per rank, set RANK / LOCAL_RANK / WORLD_SIZE, and use a
    temporary file as the rendezvous store.
    """
    import os, tempfile, time as _time
    init_file = os.path.join(tempfile.gettempdir(),
                             f"ddp_init_{int(_time.time())}.store")
    env_base = os.environ.copy()
    env_base["WORLD_SIZE"] = str(nproc)
    env_base["MASTER_ADDR"] = "localhost"
    env_base["MASTER_PORT"] = "29400"
    env_base["DDP_INIT_METHOD"] = f"file://{init_file}"
    env_base["NCCL_P2P_DISABLE"] = "1"   # required on this cluster

    cmd_suffix = [
        sys.executable, "-m", module,
        *extra_args,
        "--gradient-accumulation-steps", str(accum_steps),
    ]
    print(f"  [ddp-spawn] {nproc}× {' '.join(cmd_suffix)}")
    procs = []
    for rank in range(nproc):
        env = env_base.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        procs.append(subprocess.Popen(cmd_suffix, env=env))

    failed = False
    for rank, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            print(f"  [ddp-spawn] rank {rank} exited with code {code}")
            failed = True

    # Clean up rendezvous file
    try:
        os.remove(init_file)
    except FileNotFoundError:
        pass

    if failed:
        raise subprocess.CalledProcessError(1, cmd_suffix)


# ── Training helper ───────────────────────────────────────────────────────────

def train_ml_model(
    model_type: str,
    edges_df: pl.DataFrame,
    checkpoint_dir: Path,
    device: str = "mps",
    n_epochs: int = 50,
    force: bool = False,
    embedder_checkpoint: str | None = None,
    ddp_nproc: int = 1,
    accum_steps: int = 1,
) -> str:
    """Train model on pre-built edges and save checkpoint.  Returns checkpoint path."""
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists() and not force:
        print(f"  checkpoint exists → skipping training: {ckpt_path}")
        return str(ckpt_path)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Always use a subprocess (single- or multi-GPU) so that each model's GPU
    # memory is fully released when its subprocess exits.  Without this, a failed
    # or OOM-ing model leaves 21+ GiB of PyTorch CUDA allocations in the parent
    # process, causing every subsequent model to OOM as well.
    edges_file = EDGES_TRAIN if EDGES_TRAIN.exists() else checkpoint_dir / "_tmp_edges.parquet"
    if not edges_file.exists():
        print(f"  Writing edges cache → {edges_file}")
        edges_df.write_parquet(edges_file)

    extra_args = [
        "--task", "edge",
        "--edges", str(edges_file),
        "--model", model_type,
        "--epochs", str(n_epochs),
        "--device", "auto" if ddp_nproc > 1 else device,
        "--checkpoint", str(checkpoint_dir),
    ]
    if embedder_checkpoint:
        extra_args += ["--embedder-checkpoint", embedder_checkpoint]

    if ddp_nproc > 1:
        _run_ddp_spawn("src.train", extra_args, nproc=ddp_nproc, accum_steps=accum_steps)
    else:
        cmd = [sys.executable, "-m", "src.train", *extra_args]
        print(f"  [spawn] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    return str(ckpt_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    device: str = "mps",
    force_retrain: bool = False,
    epochs: int = 200,
    workers: int = 1,
    ddp_nproc: int = 1,
    accum_steps: int = 1,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Loading data...")
    clusters_train = pl.read_parquet(TRAIN_CLUSTERS)
    clusters_test  = pl.read_parquet(TEST_CLUSTERS)
    tracks_test    = pl.read_parquet(TEST_TRACKS)

    print(f"  Train clusters : {len(clusters_train):,}  "
          f"({clusters_train['event_id'].n_unique():,} events)")
    print(f"  Test  clusters : {len(clusters_test):,}  "
          f"({clusters_test['event_id'].n_unique():,} events)")
    print(f"  Test  tracks   : {len(tracks_test):,}")
    print(f"  workers        : {workers}")
    if ddp_nproc > 1:
        print(f"  DDP nproc      : {ddp_nproc}  accum_steps={accum_steps}")

    baseline_cfg = BaselineConfig(n_workers=workers)
    hough_cfg    = HoughConfig(n_workers=workers)

    n_test_events = clusters_test["event_id"].n_unique()

    reco_paths: dict[str, str] = {}

    # ── Non-ML algorithms ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Non-ML algorithms (test set only)")

    for name, fn in [
        ("Baseline", lambda: evaluate_baseline_on_sim(clusters_test, tracks_test, baseline_cfg)),
        ("Hough",    lambda: evaluate_hough_on_sim(clusters_test, tracks_test, hough_cfg)),
    ]:
        print(f"\n[{name}]")
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        per_evt_ms = dt / n_test_events * 1e3
        out = OUT_DIR / f"{name.lower()}_test.parquet"
        result.write_parquet(out)
        reco_paths[name] = str(out)
        print(f"  → {out}  ({dt:.1f}s total  {per_evt_ms:.2f} ms/event)")

    # ── Stage 0: Shared metric-learning embedder (hinge loss) ────────────────
    print("\n" + "=" * 65)
    print("Stage 0: Metric-learning embedder (hinge loss)")
    embedder_dir  = RUNS_DIR / "embedder"
    embedder_ckpt = embedder_dir / "best_embedder.pt"
    embedder_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if embedder_ckpt.exists() and not force_retrain:
        print(f"  checkpoint exists → skipping: {embedder_ckpt}")
    elif ddp_nproc > 1:
        _run_ddp_spawn(
            "src.train",
            ["--task", "embedder",
             "--clusters", str(TRAIN_CLUSTERS),
             "--epochs", str(epochs),
             "--device", "auto",
             "--checkpoint", str(embedder_dir)],
            nproc=ddp_nproc,
            accum_steps=accum_steps,
        )
    else:
        emb_cfg = EmbedderTrainConfig(
            emb_dim        = 8,
            n_epochs       = epochs,
            device         = device,
            checkpoint_dir = str(embedder_dir),
        )
        train_embedder(clusters_train, emb_cfg)
    print(f"  embedder → {embedder_ckpt}  ({time.perf_counter() - t0:.0f}s)")

    # ── Shared edge table (built once, reused by all ML models) ──────────────
    print("\n" + "=" * 65)
    print("Building/loading edge table for ML training...")
    if EDGES_TRAIN.exists():
        print(f"  Loading pre-built edges from {EDGES_TRAIN} ...")
        t0 = time.perf_counter()
        edges_train = pl.read_parquet(EDGES_TRAIN)
        print(f"  Loaded {len(edges_train):,} edges  ({time.perf_counter() - t0:.1f}s)")
    else:
        print(f"  Building labeled edges from "
              f"{clusters_train['event_id'].n_unique():,} events ...")
        t0 = time.perf_counter()
        edges_train = build_labeled_edges_from_sim(clusters_train)
        dt = time.perf_counter() - t0
        print(f"  Built {len(edges_train):,} candidate edges  ({dt:.1f}s)")
        print(f"  Writing to {EDGES_TRAIN} ...")
        edges_train.write_parquet(EDGES_TRAIN)
        print(f"  Done  ({time.perf_counter() - t0:.1f}s total)")

    # ── ML algorithms ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ML algorithms  (train={epochs} epochs on {device}, then test)")

    for model_type in ML_MODELS:
        print(f"\n[{model_type.upper()}]")
        ckpt_dir  = RUNS_DIR / model_type
        ckpt_path = str(ckpt_dir / "best_model.pt")

        # ── TrackFormer: two-stage train + infer path ─────────────────────
        if model_type == "transformer":
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            hf_ckpt_path = str(ckpt_dir / "hit_filter.pt")
            tf_ckpt_path = str(ckpt_dir / "best_model.pt")

            # Stage 1: Hit Filter
            t0 = time.perf_counter()
            try:
                if Path(hf_ckpt_path).exists() and not force_retrain:
                    print(f"  [Stage 1] hit filter checkpoint exists → skipping: {hf_ckpt_path}")
                elif ddp_nproc > 1:
                    print("  [Stage 1] Training hit filter (DDP)...")
                    _run_ddp_spawn(
                        "src.train_hit_filter",
                        ["--clusters", str(TRAIN_CLUSTERS),
                         "--epochs", str(epochs),
                         "--device", "auto",
                         "--checkpoint", str(ckpt_dir)],
                        nproc=ddp_nproc,
                        accum_steps=accum_steps,
                    )
                else:
                    print("  [Stage 1] Training hit filter...")
                    hf_cfg = HitFilterConfig(
                        n_epochs       = epochs,
                        device         = device,
                        checkpoint_dir = str(ckpt_dir),
                    )
                    train_hit_filter(clusters_train, hf_cfg)
            except Exception as e:
                print(f"  [ERROR] hit filter training failed: {e}")
                continue
            hf_train_dt = time.perf_counter() - t0

            # Stage 2: MaskFormer (with frozen hit filter)
            t0 = time.perf_counter()
            try:
                if Path(tf_ckpt_path).exists() and not force_retrain:
                    print(f"  [Stage 2] trackformer checkpoint exists → skipping: {tf_ckpt_path}")
                elif ddp_nproc > 1:
                    print("  [Stage 2] Training MaskFormer on filtered hits (DDP)...")
                    _run_ddp_spawn(
                        "src.train_trackformer",
                        ["--clusters", str(TRAIN_CLUSTERS),
                         "--epochs", str(epochs),
                         "--device", "auto",
                         "--checkpoint", str(ckpt_dir),
                         "--hit-filter-checkpoint", hf_ckpt_path,
                         "--hit-filter-threshold", "0.1"],
                        nproc=ddp_nproc,
                        accum_steps=accum_steps,
                    )
                else:
                    print("  [Stage 2] Training MaskFormer on filtered hits...")
                    tf_cfg = TrackFormerConfig(
                        n_epochs               = epochs,
                        device                 = device,
                        checkpoint_dir         = str(ckpt_dir),
                        hit_filter_checkpoint  = hf_ckpt_path,
                        hit_filter_threshold   = 0.1,
                    )
                    train_trackformer(clusters_train, tf_cfg)
            except Exception as e:
                print(f"  [ERROR] maskformer training failed: {e}")
                continue
            tf_train_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                result = run_trackformer_reco(
                    clusters_test, tracks_test, tf_ckpt_path,
                    hit_filter_checkpoint=hf_ckpt_path,
                    hit_filter_threshold=0.1,
                    conf_threshold=0.5, mask_threshold=0.5,
                    min_layers=4, device=device,
                )
            except Exception as e:
                print(f"  [ERROR] inference failed: {e}")
                continue
            infer_dt = time.perf_counter() - t0

            out = OUT_DIR / f"{model_type}_test.parquet"
            result.write_parquet(out)
            reco_paths[model_type.upper()] = str(out)
            per_evt_ms = infer_dt / n_test_events * 1e3
            print(f"  → {out}  (hf_train {hf_train_dt:.0f}s  tf_train {tf_train_dt:.0f}s  infer {infer_dt:.1f}s  {per_evt_ms:.2f} ms/event)")
            continue

        # ── Edge-classification models ─────────────────────────────────────
        t0 = time.perf_counter()
        try:
            ckpt_path = train_ml_model(
                model_type, edges_train, ckpt_dir,
                device=device, n_epochs=epochs, force=force_retrain,
                embedder_checkpoint=str(embedder_ckpt),
                ddp_nproc=ddp_nproc, accum_steps=accum_steps,
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
        per_evt_ms = infer_dt / n_test_events * 1e3
        print(f"  → {out}  (train {train_dt:.0f}s  infer {infer_dt:.1f}s  {per_evt_ms:.2f} ms/event)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    compare(str(TEST_TRACKS), reco_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark E320 tracking algorithms")
    parser.add_argument("--device",           default="mps", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--epochs",           type=int, default=200)
    parser.add_argument("--workers",          type=int, default=1,
                        help="Parallel workers for non-ML algorithms (default 1 to limit memory)")
    parser.add_argument("--force-retrain",    action="store_true",
                        help="Re-train even if checkpoint already exists")
    parser.add_argument("--ddp-nproc",        type=int, default=1,
                        help="Number of GPUs for DDP training (1 = single-GPU, no DDP)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps for DDP training")
    args = parser.parse_args()
    main(
        device       = args.device,
        force_retrain= args.force_retrain,
        epochs       = args.epochs,
        workers      = args.workers,
        ddp_nproc    = args.ddp_nproc,
        accum_steps  = args.gradient_accumulation_steps,
    )
