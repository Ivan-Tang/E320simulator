"""
Benchmark script — DDP via torch.multiprocessing.spawn + file:// init.

Identical logic to run_benchmark.py but replaces torchrun (_run_torchrun) with
mp.spawn so that the c10d elastic rendezvous TCP store is never touched.
This works around the SIGSEGV in c10d_rendezvous_backend.py on this cluster.

Usage (single-GPU):
    python scripts/run_benchmark_spawn.py --device cuda --epochs 200

Usage (4-GPU DDP):
    python scripts/run_benchmark_spawn.py --device cuda --epochs 200 \
        --ddp-nproc 4 --gradient-accumulation-steps 2 --force-retrain
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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

from src.config import DATA_ROOT, SIM_DIR, RUNS_DIR, OUTPUTS_DIR as OUT_DIR

TRAIN_CLUSTERS = SIM_DIR / "sim_clusters_train.parquet"
TEST_CLUSTERS  = SIM_DIR / "sim_clusters_test.parquet"
TEST_TRACKS    = SIM_DIR / "sim_tracks_test.parquet"

ML_MODELS = ["mlp", "gnn", "interaction_net", "eggnet", "hgnn", "transformer"]


# ── mp.spawn DDP helper ────────────────────────────────────────────────────────

def _spawn_worker(
    rank: int,
    world_size: int,
    init_file: str,
    module_name: str,
    argv: list[str],
) -> None:
    """Worker launched by mp.spawn.  Initialises NCCL via file:// then runs CLI."""
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )

    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        if module_name == "src.train":
            from src.train import _cli
            _cli()
        elif module_name == "src.train_embedder":
            from src.train_embedder import _cli
            _cli()
        elif module_name == "src.train_trackformer":
            from src.train_trackformer import _cli
            _cli()
        elif module_name == "src.train_hit_filter":
            from src.train_hit_filter import _cli
            _cli()
        else:
            raise ValueError(f"Unknown module: {module_name}")
    finally:
        sys.argv = old_argv
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_spawn(
    module: str,
    extra_args: list[str],
    nproc: int,
    accum_steps: int,
) -> None:
    """Launch a training module via mp.spawn (blocks until done)."""
    argv = extra_args + ["--gradient-accumulation-steps", str(accum_steps)]
    init_file = f"/tmp/ddp_bench_init_{os.getpid()}_{module.replace('.', '_')}"
    if os.path.exists(init_file):
        os.remove(init_file)
    print(f"  [mp.spawn] {module}  nproc={nproc}  args={argv}")
    mp.spawn(
        _spawn_worker,
        args=(nproc, init_file, module, argv),
        nprocs=nproc,
        join=True,
    )


# ── Training helper ───────────────────────────────────────────────────────────

def train_ml_model(
    model_type: str,
    clusters_df: pl.DataFrame,
    checkpoint_dir: Path,
    device: str = "cuda",
    n_epochs: int = 50,
    force: bool = False,
    embedder_checkpoint: str | None = None,
    ddp_nproc: int = 1,
    accum_steps: int = 1,
) -> str:
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists() and not force:
        print(f"  checkpoint exists → skipping training: {ckpt_path}")
        return str(ckpt_path)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if ddp_nproc > 1:
        extra = [
            "--task", "edge",
            "--clusters", str(TRAIN_CLUSTERS),
            "--model", model_type,
            "--epochs", str(n_epochs),
            "--device", "auto",
            "--checkpoint", str(checkpoint_dir),
        ]
        if embedder_checkpoint:
            extra += ["--embedder-checkpoint", embedder_checkpoint]
        _run_spawn("src.train", extra, nproc=ddp_nproc, accum_steps=accum_steps)
    else:
        print(f"  Building labeled edges from {clusters_df['event_id'].n_unique():,} events ...")
        edges_df = build_labeled_edges_from_sim(clusters_df)
        print(f"  Built {len(edges_df):,} candidate edges")
        cfg = TrainConfig(
            model_type          = model_type,
            n_epochs            = n_epochs,
            hidden              = 64,
            n_mp                = 2,
            lr                  = 3e-4,
            device              = device,
            checkpoint_dir      = str(checkpoint_dir),
            embedder_checkpoint = embedder_checkpoint,
        )
        train(edges_df, cfg)
    return str(ckpt_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    device: str = "cuda",
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
        print(f"  DDP nproc      : {ddp_nproc}  accum_steps={accum_steps}  (mp.spawn + file://)")

    baseline_cfg = BaselineConfig(n_workers=workers)
    hough_cfg    = HoughConfig(n_workers=workers)
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
        out = OUT_DIR / f"{name.lower()}_test.parquet"
        result.write_parquet(out)
        reco_paths[name] = str(out)
        print(f"  → {out}  ({dt:.1f}s)")

    # ── Stage 0: Shared metric-learning embedder ──────────────────────────────
    print("\n" + "=" * 65)
    print("Stage 0: Metric-learning embedder (hinge loss)")
    embedder_dir  = RUNS_DIR / "embedder"
    embedder_ckpt = embedder_dir / "best_embedder.pt"
    embedder_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if embedder_ckpt.exists() and not force_retrain:
        print(f"  checkpoint exists → skipping: {embedder_ckpt}")
    elif ddp_nproc > 1:
        _run_spawn(
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

    # ── ML algorithms ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ML algorithms  (train={epochs} epochs on {device}, then test)")

    for model_type in ML_MODELS:
        print(f"\n[{model_type.upper()}]")
        ckpt_dir  = RUNS_DIR / model_type
        ckpt_path = str(ckpt_dir / "best_model.pt")

        # ── TrackFormer ────────────────────────────────────────────────────
        if model_type == "transformer":
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            hf_ckpt_path = str(ckpt_dir / "hit_filter.pt")
            tf_ckpt_path = str(ckpt_dir / "best_model.pt")

            # Stage 1: Hit Filter
            t0 = time.perf_counter()
            try:
                if Path(hf_ckpt_path).exists() and not force_retrain:
                    print(f"  [Stage 1] hit filter checkpoint exists → skipping")
                elif ddp_nproc > 1:
                    print("  [Stage 1] Training hit filter (DDP)...")
                    _run_spawn(
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

            # Stage 2: MaskFormer
            t0 = time.perf_counter()
            try:
                if Path(tf_ckpt_path).exists() and not force_retrain:
                    print(f"  [Stage 2] trackformer checkpoint exists → skipping")
                elif ddp_nproc > 1:
                    print("  [Stage 2] Training MaskFormer (DDP)...")
                    _run_spawn(
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
                    print("  [Stage 2] Training MaskFormer...")
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
            print(f"  → {out}  (hf_train {hf_train_dt:.0f}s  tf_train {tf_train_dt:.0f}s  infer {infer_dt:.1f}s)")
            continue

        # ── Edge-classification models ─────────────────────────────────────
        t0 = time.perf_counter()
        try:
            ckpt_path = train_ml_model(
                model_type, clusters_train, ckpt_dir,
                device=device, n_epochs=epochs, force=force_retrain,
                embedder_checkpoint=str(embedder_ckpt),
                ddp_nproc=ddp_nproc, accum_steps=accum_steps,
            )
        except Exception as e:
            print(f"  [ERROR] training failed: {e}")
            continue
        train_dt = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            result = run_edge_classifier_reco(
                clusters_test, tracks_test, ckpt_path,
                threshold=0.1, device=device,
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
    parser = argparse.ArgumentParser(description="Benchmark E320 (DDP via mp.spawn)")
    parser.add_argument("--device",           default="cuda", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--epochs",           type=int, default=200)
    parser.add_argument("--workers",          type=int, default=1)
    parser.add_argument("--force-retrain",    action="store_true")
    parser.add_argument("--ddp-nproc",        type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    main(
        device        = args.device,
        force_retrain = args.force_retrain,
        epochs        = args.epochs,
        workers       = args.workers,
        ddp_nproc     = args.ddp_nproc,
        accum_steps   = args.gradient_accumulation_steps,
    )
