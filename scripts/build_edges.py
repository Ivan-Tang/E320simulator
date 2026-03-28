"""Build labeled edge table from simulated clusters and write to Parquet.

This is a standalone preprocessing step (no GPU needed) that can run as a
separate high-memory PBS job before the training benchmark.

Usage
-----
    python scripts/build_edges.py \
        --clusters /storage/agrp/yiwen/data_Run502/simulation/sim_clusters_train.parquet \
        --output   /storage/agrp/yiwen/data_Run502/simulation/edges_train.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import build_labeled_edges_from_sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled edge table from sim clusters")
    parser.add_argument("--clusters", required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--output",   required=True, help="Output path for edges.parquet")
    args = parser.parse_args()

    clusters_path = Path(args.clusters)
    output_path   = Path(args.output)

    if not clusters_path.exists():
        print(f"[build_edges] ERROR: clusters file not found: {clusters_path}", flush=True)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_edges] Loading clusters from {clusters_path}", flush=True)
    t0 = time.perf_counter()
    clusters_df = pl.read_parquet(clusters_path)
    print(f"[build_edges] Loaded {len(clusters_df):,} clusters "
          f"({clusters_df['event_id'].n_unique():,} events) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    print("[build_edges] Building labeled edges...", flush=True)
    t1 = time.perf_counter()
    edges_df = build_labeled_edges_from_sim(clusters_df)
    dt = time.perf_counter() - t1
    print(f"[build_edges] Built {len(edges_df):,} candidate edges in {dt:.1f}s", flush=True)

    n_pos = int(edges_df.filter(pl.col("edge_label") == 1).height)
    print(f"[build_edges] Positive edges: {n_pos:,}  "
          f"({100.0 * n_pos / len(edges_df):.3f}%)", flush=True)

    print(f"[build_edges] Writing to {output_path}...", flush=True)
    t2 = time.perf_counter()
    edges_df.write_parquet(output_path)
    print(f"[build_edges] Done in {time.perf_counter() - t2:.1f}s  "
          f"(total {time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
