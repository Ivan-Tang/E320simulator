"""Unified inference entry for comparing multiple ML models.

Supported modes
---------------
- edge             : edge-classifier only (GNN/MLP checkpoint from src.train)
- edge+embedder    : embedder pre-filter + edge-classifier scoring
- embedder         : embedder-only pre-filtered reconstruction
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import BaselineConfig
from scripts.run_gnn_on_sim import run_gnn_reco, run_embedder_reco


def run_model_reco(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame | None,
    mode: str,
    *,
    edge_checkpoint: str | None = None,
    edge_threshold: float = 0.5,
    embedder_checkpoint: str | None = None,
    embedder_radius: float = 1.0,
    device: str = "cpu",
    embedder_device: str | None = None,
    baseline_cfg: BaselineConfig | None = None,
) -> pl.DataFrame:
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    if mode == "edge":
        if edge_checkpoint is None:
            raise ValueError("mode='edge' requires edge_checkpoint")
        return run_gnn_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            checkpoint_path=edge_checkpoint,
            threshold=edge_threshold,
            baseline_cfg=baseline_cfg,
            device=device,
        )

    if mode == "edge+embedder":
        if edge_checkpoint is None or embedder_checkpoint is None:
            raise ValueError("mode='edge+embedder' requires both edge_checkpoint and embedder_checkpoint")
        return run_gnn_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            checkpoint_path=edge_checkpoint,
            threshold=edge_threshold,
            baseline_cfg=baseline_cfg,
            device=device,
            embedder_checkpoint_path=embedder_checkpoint,
            embedder_radius=embedder_radius,
            embedder_device=embedder_device,
        )

    if mode == "embedder":
        if embedder_checkpoint is None:
            raise ValueError("mode='embedder' requires embedder_checkpoint")
        return run_embedder_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            embedder_checkpoint_path=embedder_checkpoint,
            embedder_radius=embedder_radius,
            baseline_cfg=baseline_cfg,
            device=(embedder_device or device),
        )

    raise ValueError(f"Unsupported mode: {mode}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unified model inference for E320simulator")
    parser.add_argument("--mode", default="edge", choices=["edge", "edge+embedder", "embedder"])
    parser.add_argument("--clusters", required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--tracks", default=None, help="Path to sim_tracks.parquet (optional)")

    parser.add_argument("--edge-checkpoint", default=None,
                        help="Path to best_model.pt from src.train")
    parser.add_argument("--edge-threshold", type=float, default=0.5)

    parser.add_argument("--embedder-checkpoint", default=None,
                        help="Path to best_embedder.pt from src.train_embedder")
    parser.add_argument("--embedder-radius", type=float, default=1.0)

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--embedder-device", default=None, choices=["cpu", "cuda", "mps"])

    parser.add_argument("--output", default=None,
                        help="Output parquet path. Default: <clusters_dir>/<mode>_result.parquet")
    args = parser.parse_args()

    clusters_df = pl.read_parquet(args.clusters)
    tracks_df = pl.read_parquet(args.tracks) if args.tracks else None

    print(f"[run_model] mode={args.mode}  clusters={len(clusters_df):,}")

    result = run_model_reco(
        clusters_df=clusters_df,
        tracks_df=tracks_df,
        mode=args.mode,
        edge_checkpoint=args.edge_checkpoint,
        edge_threshold=args.edge_threshold,
        embedder_checkpoint=args.embedder_checkpoint,
        embedder_radius=args.embedder_radius,
        device=args.device,
        embedder_device=args.embedder_device,
    )

    if args.output is None:
        safe_mode = args.mode.replace("+", "_plus_")
        out = os.path.join(os.path.dirname(args.clusters), f"{safe_mode}_result.parquet")
    else:
        out = args.output

    result.write_parquet(out)
    print(f"[run_model] result saved → {out}")


if __name__ == "__main__":
    _cli()
