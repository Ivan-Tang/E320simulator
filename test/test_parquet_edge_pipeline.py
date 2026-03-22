"""Tests for the parquet-backed edge building and lazy loading pipeline.

Verifies that:
1. build_edges_to_parquet() produces the same edges as build_labeled_edges_from_sim()
2. ParquetEdgeSource iterates events correctly
3. train() works end-to-end with edges_dir (parquet path)
"""
import numpy as np
import polars as pl
import pytest

from src.utils import (
    build_labeled_edges_from_sim,
    build_edges_to_parquet,
    ParquetEdgeSource,
)
from src.train import train, TrainConfig


def _make_tiny_clusters(n_events: int = 6) -> pl.DataFrame:
    """Small synthetic cluster table: 2 tracks per event, 5 layers."""
    rows = []
    node_id = 0
    for event_id in range(n_events):
        for tid, x0 in [(10, -1.0), (20, 1.0)]:
            for lid in range(5):
                rows.append({
                    "event_id":   event_id,
                    "node_id":    node_id,
                    "layer_id":   lid,
                    "x_trk_mm":  float(x0 + 0.1 * lid),
                    "y_trk_mm":  float(0.2 * lid),
                    "z_trk_mm":  float(20.0 * lid),
                    "size_x":    1,
                    "size_y":    1,
                    "size":      1,
                    "track_id":  tid,
                    "is_signal": True,
                    "particle_type": 0,
                })
                node_id += 1
    return pl.from_dicts(rows)


def test_build_edges_to_parquet_matches_in_memory(tmp_path):
    """build_edges_to_parquet must produce the same edges as build_labeled_edges_from_sim."""
    clusters = _make_tiny_clusters(n_events=6)
    expected = build_labeled_edges_from_sim(clusters)

    out_dir = tmp_path / "edges"
    build_edges_to_parquet(clusters, out_dir, chunk_size=2)

    assert out_dir.exists()
    chunk_files = sorted(out_dir.glob("chunk_*.parquet"))
    assert len(chunk_files) == 3  # 6 events / chunk_size=2

    # Reconstruct full edge table from parquet chunks
    actual = pl.concat([pl.read_parquet(f) for f in chunk_files])

    # Sort both by event_id, src_node, dst_node for comparison
    sort_cols = ["event_id", "src_node", "dst_node"]
    expected_sorted = expected.sort(sort_cols)
    actual_sorted   = actual.sort(sort_cols)

    assert set(actual.columns) == set(expected.columns), \
        f"Column mismatch: {set(actual.columns) ^ set(expected.columns)}"
    assert len(actual) == len(expected), \
        f"Row count mismatch: actual={len(actual)}, expected={len(expected)}"

    for col in sort_cols + ["edge_label", "is_signal_edge"]:
        np.testing.assert_array_equal(
            actual_sorted[col].to_numpy(),
            expected_sorted[col].to_numpy(),
            err_msg=f"Column {col!r} differs",
        )


def test_parquet_edge_source_event_ids(tmp_path):
    """ParquetEdgeSource.event_ids must return all unique event IDs."""
    clusters = _make_tiny_clusters(n_events=4)
    out_dir = tmp_path / "edges"
    build_edges_to_parquet(clusters, out_dir, chunk_size=2)

    source = ParquetEdgeSource(out_dir)
    eids = source.event_ids
    assert set(eids.tolist()) == {0, 1, 2, 3}


def test_parquet_edge_source_iter_events(tmp_path):
    """iter_events must yield one DataFrame per event with correct event_id."""
    clusters = _make_tiny_clusters(n_events=4)
    out_dir = tmp_path / "edges"
    build_edges_to_parquet(clusters, out_dir, chunk_size=2)

    source = ParquetEdgeSource(out_dir)
    seen = {}
    for eid, ev_df in source.iter_events([0, 2, 1, 3]):
        assert eid not in seen, f"Duplicate event_id {eid}"
        seen[eid] = len(ev_df)
        assert (ev_df["event_id"] == eid).all(), "event_id mismatch in iter_events"

    assert set(seen.keys()) == {0, 1, 2, 3}


def test_train_with_edges_dir(tmp_path):
    """train() with edges_dir must complete without error and return model + history."""
    clusters = _make_tiny_clusters(n_events=6)
    out_dir = tmp_path / "edges"
    build_edges_to_parquet(clusters, out_dir, chunk_size=2)

    cfg = TrainConfig(
        model_type  = "mlp",
        n_epochs    = 2,
        hidden      = 16,
        device      = "cpu",
        val_fraction = 0.33,
        checkpoint_dir = None,
    )
    result = train(cfg=cfg, edges_dir=out_dir)

    assert result["model"] is not None
    assert len(result["history"]) == 2
    assert "train_eids" in result and "val_eids" in result
    assert len(result["train_eids"]) > 0
    assert len(result["val_eids"]) > 0
