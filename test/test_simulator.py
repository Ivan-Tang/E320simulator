"""Tests for src/simulator.py."""
import numpy as np
import polars as pl
import pytest
from src.simulator import SimConfig, simulate, SyntheticBackgroundPool


CLUSTERS_REQUIRED_COLS = {
    "event_id", "node_id", "layer_id",
    "x_trk_mm", "y_trk_mm", "z_trk_mm",
    "size_x", "size_y", "size",
    "track_id", "is_signal", "particle_type",
}
TRACKS_REQUIRED_COLS = {
    "event_id", "track_id", "is_signal",
    "x0_mm", "y0_mm", "z0_mm", "tx", "ty", "pz_GeV", "n_layers_hit",
}
Z_LAYERS = {0: 0.0, 1: 20.0, 2: 40.0, 3: 60.0, 4: 80.0}


@pytest.fixture(scope="module")
def sim_result():
    """Run a small simulation once for module-level tests."""
    cfg = SimConfig(n_events=10, seed=7, background_mode="none")
    return simulate(cfg)


class TestSimConfig:
    def test_default_n_events(self):
        assert SimConfig().n_events == 1000

    def test_default_seed(self):
        assert SimConfig().seed == 42

    def test_default_background_mode(self):
        assert SimConfig().background_mode == "none"

    def test_default_min_layers_hit(self):
        assert SimConfig().min_layers_hit == 4


class TestSimulateSchema:
    def test_clusters_has_required_columns(self, sim_result):
        clusters_df, _ = sim_result
        assert CLUSTERS_REQUIRED_COLS.issubset(set(clusters_df.columns))

    def test_tracks_has_required_columns(self, sim_result):
        _, tracks_df = sim_result
        assert TRACKS_REQUIRED_COLS.issubset(set(tracks_df.columns))

    def test_node_id_unique(self, sim_result):
        clusters_df, _ = sim_result
        assert clusters_df["node_id"].is_unique().all()

    def test_node_id_contiguous_from_zero(self, sim_result):
        clusters_df, _ = sim_result
        assert clusters_df["node_id"].to_list() == list(range(len(clusters_df)))

    def test_layer_ids_in_valid_range(self, sim_result):
        clusters_df, _ = sim_result
        assert clusters_df["layer_id"].min() >= 0
        assert clusters_df["layer_id"].max() <= 4


class TestSimulatePhysics:
    def test_z_trk_mm_matches_layer_id(self, sim_result):
        """z coordinate is deterministic (no smearing in z)."""
        clusters_df, _ = sim_result
        for lid, z_expected in Z_LAYERS.items():
            layer_rows = clusters_df.filter(pl.col("layer_id") == lid)
            z_vals = layer_rows["z_trk_mm"].to_numpy()
            assert np.all(np.abs(z_vals - z_expected) < 0.01), \
                f"Layer {lid}: z should be {z_expected}, got {z_vals}"

    def test_signal_clusters_have_nonneg_track_id(self, sim_result):
        clusters_df, _ = sim_result
        signal_rows = clusters_df.filter(pl.col("is_signal") == True)
        assert (signal_rows["track_id"] >= 0).all()

    def test_background_clusters_have_track_id_minus_one(self, sim_result):
        clusters_df, _ = sim_result
        bg_rows = clusters_df.filter(pl.col("is_signal") == False)
        if len(bg_rows) > 0:
            assert (bg_rows["track_id"] == -1).all()

    def test_min_layers_hit_enforced(self, sim_result):
        _, tracks_df = sim_result
        cfg = SimConfig()
        assert (tracks_df["n_layers_hit"] >= cfg.min_layers_hit).all()

    def test_particle_type_values(self, sim_result):
        clusters_df, _ = sim_result
        valid_types = {"signal_pos", "background"}
        found_types = set(clusters_df["particle_type"].unique().to_list())
        assert found_types.issubset(valid_types)

    def test_x_within_sensor_bounds(self, sim_result):
        """x must be within sensor half-width + 3σ smearing."""
        clusters_df, _ = sim_result
        from src.geometry import AlpideSpec
        spec = AlpideSpec()
        half_w = spec.width_mm / 2.0
        sigma = SimConfig().sigma_x_mm
        x_abs = clusters_df["x_trk_mm"].abs().max()
        assert x_abs <= half_w + 3 * sigma, f"|x|_max={x_abs} exceeds bound {half_w + 3*sigma}"


class TestSimulateReproducibility:
    def test_same_seed_same_result(self):
        cfg = SimConfig(n_events=5, seed=13)
        df1, _ = simulate(cfg)
        df2, _ = simulate(cfg)
        assert df1.equals(df2)

    def test_different_seeds_different_result(self):
        df1, _ = simulate(SimConfig(n_events=5, seed=1))
        df2, _ = simulate(SimConfig(n_events=5, seed=2))
        assert not df1.equals(df2)

    def test_zero_events_returns_empty_with_schema(self):
        df, tr = simulate(SimConfig(n_events=0, seed=0))
        assert len(df) == 0
        assert len(tr) == 0
        assert CLUSTERS_REQUIRED_COLS.issubset(set(df.columns))

    def test_high_signal_rate_produces_tracks(self):
        _, tracks_df = simulate(SimConfig(n_events=20, seed=0, mean_n_signal=5.0))
        assert len(tracks_df) > 0


class TestSyntheticBackgroundPool:
    def test_sample_returns_all_five_layers(self):
        pool = SyntheticBackgroundPool(n_per_layer=10)
        rng = np.random.default_rng(0)
        sample = pool.sample(rng)
        layer_ids = set(sample["layer_id"].tolist())
        assert layer_ids == {0, 1, 2, 3, 4}

    def test_sample_count(self):
        pool = SyntheticBackgroundPool(n_per_layer=7)
        rng = np.random.default_rng(0)
        sample = pool.sample(rng)
        assert len(sample["layer_id"]) == 35  # 7 * 5 layers
