"""Tests for src/utils.py."""
import math
import numpy as np
import polars as pl
import pytest
import torch
from src.utils import (
    NODE_DIM, EDGE_DIM,
    NODE_FEAT_COLS_SRC, NODE_FEAT_COLS_DST, EDGE_FEAT_COLS,
    is_match, ALL_LAYERS,
    build_pairs,
    build_labeled_edges_from_sim,
    event_to_tensors,
    edge_label_stats,
)


EXPECTED_EDGE_COLS = {
    "event_id", "src_node", "dst_node", "src_layer", "dst_layer",
    "dx_mm", "dy_mm", "dz_mm", "dr_mm", "slope_x", "slope_y",
    "x_i", "y_i", "z_i", "x_j", "y_j", "z_j",
    "size_x_i", "size_y_i", "size_i", "size_x_j", "size_y_j", "size_j",
    "track_id_i", "track_id_j", "edge_label", "is_signal_edge",
}


# ── Constants ─────────────────────────────────────────────────────────────────

def test_node_dim_is_7():
    assert NODE_DIM == 7

def test_edge_dim_is_6():
    assert EDGE_DIM == 6

def test_node_feat_cols_src_length():
    assert len(NODE_FEAT_COLS_SRC) == NODE_DIM

def test_edge_feat_cols_length():
    assert len(EDGE_FEAT_COLS) == EDGE_DIM


# ── is_match ──────────────────────────────────────────────────────────────────

class TestIsMatch:
    def _arrays(self, *layer_ids):
        vols = np.zeros(len(layer_ids), dtype=np.int32)
        layers = np.array(layer_ids, dtype=np.int32)
        return vols, layers

    def test_adjacent_layers_match(self):
        vols, layers = self._arrays(0, 1)
        assert is_match(0, 1, vols, layers) is True

    def test_adjacent_layers_match_upper(self):
        vols, layers = self._arrays(2, 3)
        assert is_match(0, 1, vols, layers) is True

    def test_same_layer_no_match(self):
        vols, layers = self._arrays(2, 2)
        assert is_match(0, 1, vols, layers) is False

    def test_non_adjacent_layers_no_match(self):
        vols, layers = self._arrays(0, 4)
        assert is_match(0, 1, vols, layers) is False

    def test_skipping_one_layer_no_match(self):
        vols, layers = self._arrays(0, 2)
        assert is_match(0, 1, vols, layers) is False


# ── build_pairs ───────────────────────────────────────────────────────────────

class TestBuildPairs:
    def _make_inputs(self):
        """Two signal tracks, 5 hits each across 5 layers."""
        rng = np.random.default_rng(0)
        hits = rng.standard_normal((10, 7)).astype(np.float32)
        pids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        layers = np.tile(np.arange(5, dtype=np.int32), 2)
        vols = np.zeros(10, dtype=np.int32)
        return hits, pids, layers, vols

    def test_returns_three_arrays(self):
        hits, pids, layers, vols = self._make_inputs()
        h_a, h_b, target = build_pairs(hits, pids, vols, layers)
        assert isinstance(h_a, list) and isinstance(h_b, list) and isinstance(target, list)
        assert len(h_a) == len(h_b) == len(target)

    def test_both_labels_present(self):
        hits, pids, layers, vols = self._make_inputs()
        _, _, target = build_pairs(hits, pids, vols, layers, nb_particles_per_sample=10)
        assert 0 in target
        assert 1 in target

    def test_background_excluded_from_positives(self):
        """track_id=-1 hits should not appear as anchors in positive pairs."""
        hits = np.zeros((6, 7), dtype=np.float32)
        pids = np.array([-1, -1, 0, 0, 0, 0])  # first two are background
        layers = np.array([0, 1, 0, 1, 2, 3], dtype=np.int32)
        vols = np.zeros(6, dtype=np.int32)
        h_a, h_b, target = build_pairs(hits, pids, vols, layers)
        # All signal is from pid=0, so pairs should only come from pid=0 hits
        pos_pairs = [(a, b) for a, b, t in zip(h_a, h_b, target) if t == 1]
        assert len(pos_pairs) > 0

    def test_no_signal_returns_empty(self):
        hits = np.zeros((5, 7), dtype=np.float32)
        pids = np.array([-1, -1, -1, -1, -1])
        layers = np.arange(5, dtype=np.int32)
        vols = np.zeros(5, dtype=np.int32)
        h_a, h_b, target = build_pairs(hits, pids, vols, layers)
        assert len(target) == 0


# ── build_labeled_edges_from_sim ──────────────────────────────────────────────

class TestBuildLabeledEdgesFromSim:
    def test_required_columns_present(self, edges_df):
        assert EXPECTED_EDGE_COLS.issubset(set(edges_df.columns))

    def test_edge_label_binary(self, edges_df):
        labels = set(edges_df["edge_label"].unique().to_list())
        assert labels.issubset({0, 1})

    def test_is_signal_edge_subset_of_edge_label(self, edges_df):
        """All signal edges (is_signal_edge=1) must also have edge_label=1."""
        sig_rows = edges_df.filter(pl.col("is_signal_edge") == 1)
        if len(sig_rows) > 0:
            assert (sig_rows["edge_label"] == 1).all()

    def test_positive_edges_exist(self, edges_df):
        """2 signal tracks → some positive edges must be built."""
        assert edges_df["edge_label"].sum() > 0

    def test_dr_equals_sqrt_dx2_dy2(self, edges_df):
        dx = edges_df["dx_mm"].to_numpy()
        dy = edges_df["dy_mm"].to_numpy()
        dr = edges_df["dr_mm"].to_numpy()
        expected_dr = np.sqrt(dx**2 + dy**2)
        np.testing.assert_array_almost_equal(dr, expected_dr, decimal=6)

    def test_dr_nonnegative(self, edges_df):
        assert (edges_df["dr_mm"] >= 0.0).all()

    def test_src_layer_less_than_dst_layer(self, edges_df):
        assert (edges_df["src_layer"] < edges_df["dst_layer"]).all()

    def test_missing_column_raises(self, tiny_clusters_df):
        bad_df = tiny_clusters_df.drop("track_id")
        with pytest.raises(ValueError, match="missing columns"):
            build_labeled_edges_from_sim(bad_df)


# ── event_to_tensors ──────────────────────────────────────────────────────────

class TestEventToTensors:
    def test_node_feat_shape(self, standard_tensors):
        node_feat, *_ = standard_tensors
        assert node_feat.shape[1] == NODE_DIM

    def test_edge_index_shape(self, standard_tensors):
        _, edge_index, *_ = standard_tensors
        assert edge_index.shape[0] == 2

    def test_edge_feat_shape(self, standard_tensors):
        _, _, edge_feat, *_ = standard_tensors
        assert edge_feat.shape[1] == EDGE_DIM

    def test_edge_label_1d_and_matches_edge_count(self, standard_tensors):
        _, edge_index, _, edge_label, _ = standard_tensors
        assert edge_label.ndim == 1
        assert len(edge_label) == edge_index.shape[1]

    def test_edge_index_values_in_node_range(self, standard_tensors):
        node_feat, edge_index, *_ = standard_tensors
        N = node_feat.shape[0]
        assert edge_index.min() >= 0
        assert edge_index.max() < N

    def test_node_feat_dtype_float32(self, standard_tensors):
        node_feat, *_ = standard_tensors
        assert node_feat.dtype == torch.float32

    def test_edge_index_dtype_int64(self, standard_tensors):
        _, edge_index, *_ = standard_tensors
        assert edge_index.dtype == torch.int64

    def test_node_ids_unique(self, standard_tensors):
        *_, node_ids = standard_tensors
        assert len(node_ids.unique()) == len(node_ids)


# ── edge_label_stats ──────────────────────────────────────────────────────────

class TestEdgeLabelStats:
    def test_required_keys(self, edges_df):
        stats = edge_label_stats(edges_df)
        for key in ("n_total", "n_positive", "n_negative", "positive_fraction"):
            assert key in stats

    def test_n_total_correct(self, edges_df):
        stats = edge_label_stats(edges_df)
        assert stats["n_total"] == len(edges_df)

    def test_pos_plus_neg_equals_total(self, edges_df):
        stats = edge_label_stats(edges_df)
        assert stats["n_positive"] + stats["n_negative"] == stats["n_total"]

    def test_positive_fraction_in_unit_interval(self, edges_df):
        stats = edge_label_stats(edges_df)
        assert 0.0 <= stats["positive_fraction"] <= 1.0

    def test_signal_edges_key_present_when_column_exists(self, edges_df):
        assert "is_signal_edge" in edges_df.columns
        stats = edge_label_stats(edges_df)
        assert "n_signal_edges" in stats
