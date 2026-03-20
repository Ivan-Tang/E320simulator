"""Tests for src/baseline.py internals."""
import numpy as np
import pytest
from src.baseline import (
    BaselineConfig,
    _build_edges,
    _build_chains,
    _fit_and_score,
    _shared_hit_rejection,
    _process_event,
)

Z_LAYERS = np.array([0.0, 20.0, 40.0, 60.0, 80.0])


def _straight_track_arrays(ax=0.005, ay=0.003, bx=0.0, by=0.0, noise=0.0, rng=None):
    """Return (x, y, z, layer, nid) for a perfect straight-line track."""
    if rng is None:
        rng = np.random.default_rng(0)
    z = Z_LAYERS.copy()
    x = ax * z + bx + rng.normal(0, noise, 5)
    y = ay * z + by + rng.normal(0, noise, 5)
    layer = np.arange(5, dtype=np.int8)
    nid = np.arange(5, dtype=np.int64)
    return x, y, z, layer, nid


# ── _build_edges ──────────────────────────────────────────────────────────────

class TestBuildEdges:
    def setup_method(self):
        self.cfg = BaselineConfig()

    def test_returns_six_arrays(self):
        x, y, z, layer, nid = _straight_track_arrays()
        result = _build_edges(x, y, z, layer, nid, self.cfg)
        assert len(result) == 6

    def test_valid_edge_kept(self):
        """Hits on layers 0 and 1 with zero slope → 1 edge returned."""
        x = np.array([0.0, 0.0], dtype=np.float64)
        y = np.array([0.0, 0.0], dtype=np.float64)
        z = np.array([0.0, 20.0], dtype=np.float64)
        layer = np.array([0, 1], dtype=np.int8)
        nid = np.array([0, 1], dtype=np.int64)
        src, dst, *_ = _build_edges(x, y, z, layer, nid, self.cfg)
        assert len(src) == 1

    def test_steep_slope_rejected(self):
        """dx=100mm / dz=20mm → slope=5.0, way beyond slope_x_max=0.2 → no edge."""
        x = np.array([0.0, 100.0], dtype=np.float64)
        y = np.array([0.0, 0.0], dtype=np.float64)
        z = np.array([0.0, 20.0], dtype=np.float64)
        layer = np.array([0, 1], dtype=np.int8)
        nid = np.array([0, 1], dtype=np.int64)
        src, *_ = _build_edges(x, y, z, layer, nid, self.cfg)
        assert len(src) == 0

    def test_single_layer_no_edges(self):
        """All hits on layer 0 → no adjacent-layer pairs → empty."""
        x = np.zeros(4, dtype=np.float64)
        y = np.zeros(4, dtype=np.float64)
        z = np.zeros(4, dtype=np.float64)
        layer = np.zeros(4, dtype=np.int8)
        nid = np.arange(4, dtype=np.int64)
        src, *_ = _build_edges(x, y, z, layer, nid, self.cfg)
        assert len(src) == 0

    def test_slope_values_correct(self):
        """dx=1mm, dz=20mm → slope_x == 0.05."""
        x = np.array([0.0, 1.0], dtype=np.float64)
        y = np.array([0.0, 0.0], dtype=np.float64)
        z = np.array([0.0, 20.0], dtype=np.float64)
        layer = np.array([0, 1], dtype=np.int8)
        nid = np.array([0, 1], dtype=np.int64)
        src, dst, sl, dl, sx, sy = _build_edges(x, y, z, layer, nid, self.cfg)
        assert len(sx) == 1
        assert sx[0] == pytest.approx(0.05, abs=1e-9)
        assert sy[0] == pytest.approx(0.0, abs=1e-9)

    def test_src_layer_less_than_dst_layer(self):
        x, y, z, layer, nid = _straight_track_arrays()
        src, dst, sl, dl, sx, sy = _build_edges(x, y, z, layer, nid, self.cfg)
        assert np.all(sl < dl)

    def test_knn_limits_edges_per_source(self):
        """5 hits on layer 1, 1 on layer 0, knn_k=2 → at most 2 edges from layer-0 hit."""
        cfg = BaselineConfig(knn_k=2)
        x = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.5], dtype=np.float64)
        y = np.zeros(6, dtype=np.float64)
        z = np.array([0.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float64)
        layer = np.array([0, 1, 1, 1, 1, 1], dtype=np.int8)
        nid = np.arange(6, dtype=np.int64)
        src, *_ = _build_edges(x, y, z, layer, nid, cfg)
        assert len(src) <= 2


# ── _build_chains ─────────────────────────────────────────────────────────────

class TestBuildChains:
    def setup_method(self):
        self.cfg = BaselineConfig()

    def _make_edges_for_track(self, ax=0.005, ay=0.003):
        x, y, z, layer, nid = _straight_track_arrays(ax=ax, ay=ay)
        return _build_edges(x, y, z, layer, nid, self.cfg) + (x, y, z, layer, nid)

    def test_empty_edges_return_empty_chains(self):
        empty = np.empty(0, dtype=np.int64)
        result = _build_chains(empty, empty, empty, empty, empty.astype(float), empty.astype(float), self.cfg)
        assert result == []

    def test_quintuplet_detected_for_perfect_track(self):
        src, dst, sl, dl, sx, sy, x, y, z, layer, nid = self._make_edges_for_track()
        chains = _build_chains(src, dst, sl, dl, sx, sy, self.cfg)
        max_len = max((len(nodes) for nodes, _ in chains), default=0)
        assert max_len >= 4, f"Expected chain of ≥4 but got max {max_len}"

    def test_inconsistent_slopes_rejected(self):
        """Two edges whose slope difference exceeds dslope_x_max cannot chain."""
        cfg = BaselineConfig(dslope_x_max=0.001)
        # Edge 0→1 with slope 0.01, edge 1→2 with slope 0.02 → diff=0.01 >> 0.001
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        sl  = np.array([0, 1], dtype=np.int8)
        dl  = np.array([1, 2], dtype=np.int8)
        sx  = np.array([0.01, 0.02], dtype=float)
        sy  = np.array([0.0, 0.0], dtype=float)
        chains = _build_chains(src, dst, sl, dl, sx, sy, cfg)
        # No chain should include both nodes 0 and 2
        for nodes, _ in chains:
            assert not (0 in nodes and 2 in nodes)


# ── _fit_and_score ────────────────────────────────────────────────────────────

class TestFitAndScore:
    def test_perfect_line_zero_chi2(self):
        ax, ay, bx, by = 0.01, 0.005, 0.5, 0.2
        z = Z_LAYERS
        x = ax * z + bx
        y = ay * z + by
        nodes = (0, 1, 2, 3, 4)
        chains = [(nodes, (0, 1, 2, 3, 4))]
        nid_to_local = {i: i for i in range(5)}
        results = _fit_and_score(chains, x, y, z, nid_to_local)
        assert len(results) == 1
        assert results[0]["chi2"] == pytest.approx(0.0, abs=1e-10)
        assert results[0]["rms"] == pytest.approx(0.0, abs=1e-10)

    def test_output_keys_present(self):
        z = Z_LAYERS
        x, y = np.zeros(5), np.zeros(5)
        results = _fit_and_score([((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))], x, y, z, {i: i for i in range(5)})
        expected = {"node_ids", "n_layers", "ax", "bx", "ay", "by", "chi2", "rms"}
        assert expected.issubset(set(results[0].keys()))

    def test_n_layers_matches_chain_length(self):
        z = Z_LAYERS[:4]
        x, y = np.zeros(4), np.zeros(4)
        results = _fit_and_score([((0, 1, 2, 3), (0, 1, 2, 3))], x, y, z, {i: i for i in range(4)})
        assert results[0]["n_layers"] == 4

    def test_noisy_track_positive_chi2(self):
        rng = np.random.default_rng(42)
        ax, ay = 0.005, 0.003
        z = Z_LAYERS
        x = ax * z + rng.normal(0, 0.05, 5)
        y = ay * z + rng.normal(0, 0.05, 5)
        results = _fit_and_score([((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))], x, y, z, {i: i for i in range(5)})
        assert results[0]["chi2"] > 0

    def test_all_float_fields_finite(self):
        z = Z_LAYERS
        x, y = 0.005 * z, 0.003 * z
        results = _fit_and_score([((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))], x, y, z, {i: i for i in range(5)})
        for key in ("ax", "bx", "ay", "by", "chi2", "rms"):
            assert np.isfinite(results[0][key])


# ── _shared_hit_rejection ─────────────────────────────────────────────────────

class TestSharedHitRejection:
    def test_best_chi2_kept_when_overlapping(self):
        candidates = [
            {"node_ids": [0, 1, 2, 3, 4], "chi2": 1.0},
            {"node_ids": [1, 2, 3, 4, 5], "chi2": 0.5},  # lower chi2
        ]
        result = _shared_hit_rejection(candidates)
        kept = {tuple(sorted(c["node_ids"])): c["is_kept"] for c in result}
        # (1,2,3,4,5) has chi2=0.5 → kept first; (0,1,2,3,4) shares nodes → rejected
        assert kept[(1, 2, 3, 4, 5)] is True
        assert kept[(0, 1, 2, 3, 4)] is False

    def test_non_overlapping_both_kept(self):
        candidates = [
            {"node_ids": [0, 1, 2, 3, 4], "chi2": 1.0},
            {"node_ids": [5, 6, 7, 8, 9], "chi2": 2.0},
        ]
        result = _shared_hit_rejection(candidates)
        assert all(c["is_kept"] for c in result)


# ── _process_event (end-to-end) ───────────────────────────────────────────────

class TestProcessEvent:
    def test_single_track_recovers_candidate(self):
        x, y, z, layer, nid = _straight_track_arrays()
        cfg = BaselineConfig()
        result = _process_event(0, x, y, z, layer, nid, cfg)
        assert len(result) > 0
        kept = [c for c in result if c["is_kept"]]
        assert len(kept) >= 1
        assert max(c["n_layers"] for c in kept) >= 4

    def test_empty_event_returns_empty(self):
        """Single hit on one layer only → no edges → empty result."""
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])
        layer = np.array([0], dtype=np.int8)
        nid = np.array([0], dtype=np.int64)
        result = _process_event(0, x, y, z, layer, nid, BaselineConfig())
        assert result == []

    def test_candidate_fields_complete(self, one_event_arrays):
        cfg = BaselineConfig()
        arr = one_event_arrays
        result = _process_event(0, arr["x"], arr["y"], arr["z"], arr["layer"], arr["nid"], cfg)
        if result:
            required = {"event_id", "candidate_id", "node_ids", "n_layers",
                        "ax", "bx", "ay", "by", "chi2", "rms", "is_kept"}
            assert required.issubset(set(result[0].keys()))
