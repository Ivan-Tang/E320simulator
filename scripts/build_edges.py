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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import build_labeled_edges_from_sim

# Number of events processed per batch before flushing to disk.
# Keeps peak RAM proportional to batch_size × edges_per_event rather than
# holding the entire dataset in memory at once.
_EVENTS_PER_BATCH = 200


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled edge table from sim clusters")
    parser.add_argument("--clusters",   required=True,            help="Path to sim_clusters.parquet")
    parser.add_argument("--output",     required=True,            help="Output path for edges.parquet")
    parser.add_argument("--batch-size", type=int, default=_EVENTS_PER_BATCH,
                        help=f"Events per batch (default {_EVENTS_PER_BATCH})")
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
    n_events = clusters_df["event_id"].n_unique()
    print(f"[build_edges] Loaded {len(clusters_df):,} clusters "
          f"({n_events:,} events) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    event_ids  = clusters_df["event_id"].unique().sort().to_list()
    batch_size = args.batch_size
    n_batches  = (len(event_ids) + batch_size - 1) // batch_size

    print(f"[build_edges] Building edges in {n_batches} batches "
          f"of ≤{batch_size} events each...", flush=True)

    writer: pq.ParquetWriter | None = None
    total_edges = 0
    total_pos   = 0
    t1 = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_eids    = event_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_clusters = clusters_df.filter(pl.col("event_id").is_in(batch_eids))

        batch_edges = build_labeled_edges_from_sim(batch_clusters)
        if len(batch_edges) == 0:
            print(f"[build_edges]   batch {batch_idx+1}/{n_batches}: 0 edges, skipping",
                  flush=True)
            continue

        total_edges += len(batch_edges)
        total_pos   += int(batch_edges.filter(pl.col("edge_label") == 1).height)

        table = batch_edges.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema)
        writer.write_table(table)

        # Free memory before next batch
        del batch_edges, table

        elapsed = time.perf_counter() - t1
        print(f"[build_edges]   batch {batch_idx+1}/{n_batches}: "
              f"{total_edges:,} edges total  ({elapsed:.0f}s)", flush=True)

    if writer is not None:
        writer.close()

    dt = time.perf_counter() - t1
    if total_edges == 0:
        print("[build_edges] WARNING: no edges produced — output file not written.", flush=True)
        sys.exit(1)

    pos_pct = 100.0 * total_pos / total_edges
    print(f"[build_edges] Total: {total_edges:,} edges, "
          f"{total_pos:,} positive ({pos_pct:.3f}%) in {dt:.1f}s", flush=True)
    print(f"[build_edges] Written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
