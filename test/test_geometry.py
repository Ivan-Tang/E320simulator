"""Tests for src/geometry.py."""
import math
import numpy as np
import pytest
from src.geometry import (
    AlpideSpec, LayerGeometry, E320PrototypeGeometry, SENSOR_TO_LAYER
)
from src.config import HIT_LEVEL_PARQUET


# ── AlpideSpec ────────────────────────────────────────────────────────────────

class TestAlpideSpec:
    def test_default_pixels(self):
        s = AlpideSpec()
        assert s.n_rows == 512
        assert s.n_cols == 1024

    def test_pitch_values(self):
        s = AlpideSpec()
        assert s.pitch_row_mm == pytest.approx(27e-3)
        assert s.pitch_col_mm == pytest.approx(29e-3)

    def test_width_mm(self):
        s = AlpideSpec()
        assert s.width_mm == pytest.approx(1024 * 29e-3)

    def test_height_mm(self):
        s = AlpideSpec()
        assert s.height_mm == pytest.approx(512 * 27e-3)

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError
        s = AlpideSpec()
        with pytest.raises(FrozenInstanceError):
            s.n_rows = 256


# ── LayerGeometry ─────────────────────────────────────────────────────────────

class TestLayerGeometry:
    def test_defaults(self):
        layer = LayerGeometry(layer_id=0, z_trk_mm=0.0)
        assert layer.dx_mm == 0.0
        assert layer.dy_mm == 0.0
        assert layer.theta_z_rad == 0.0

    def test_rotation_matrix_identity_at_zero(self):
        R = LayerGeometry(layer_id=0, z_trk_mm=0.0).rotation_matrix_2d()
        np.testing.assert_array_almost_equal(R, np.eye(2))

    def test_rotation_matrix_orthogonal(self):
        """R @ R.T == I and det(R) == 1 for any angle."""
        R = LayerGeometry(layer_id=0, z_trk_mm=0.0, theta_z_rad=0.37).rotation_matrix_2d()
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(2), decimal=10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rotation_preserves_vector_length(self):
        R = LayerGeometry(layer_id=0, z_trk_mm=0.0, theta_z_rad=1.23).rotation_matrix_2d()
        v = np.array([3.0, 4.0])
        assert abs(np.linalg.norm(R @ v) - 5.0) < 1e-10

    def test_rotation_90deg(self):
        R = LayerGeometry(layer_id=0, z_trk_mm=0.0, theta_z_rad=math.pi / 2).rotation_matrix_2d()
        np.testing.assert_array_almost_equal(R, [[0, -1], [1, 0]])


# ── E320PrototypeGeometry ─────────────────────────────────────────────────────

class TestE320PrototypeGeometry:
    def setup_method(self):
        self.geom = E320PrototypeGeometry(use_alignment=False)
        self.geom_aln = E320PrototypeGeometry(use_alignment=True)
        self.spec = self.geom.spec

    def test_five_layers(self):
        assert len(self.geom.layers) == 5

    def test_layer_z_positions_no_alignment(self):
        for i in range(5):
            assert self.geom.layers[i].z_trk_mm == pytest.approx(i * 20.0)

    def test_layer0_no_alignment_with_alignment_on(self):
        l0 = self.geom_aln.layers[0]
        assert l0.dx_mm == 0.0 and l0.dy_mm == 0.0 and l0.theta_z_rad == 0.0

    def test_layer1_alignment_values(self):
        l1 = self.geom_aln.layers[1]
        assert l1.dx_mm == pytest.approx(-23.5e-3, abs=1e-9)
        assert l1.dy_mm == pytest.approx(39.3e-3, abs=1e-9)
        assert l1.theta_z_rad == pytest.approx(-1.8e-3, abs=1e-9)

    def test_pixel_to_chip_local_center_is_origin(self):
        cx = (self.spec.n_cols - 1) / 2.0
        cy = (self.spec.n_rows - 1) / 2.0
        result = self.geom.pixel_to_chip_local_mm(cx, cy)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_pixel_to_chip_local_linearity(self):
        """Column +1 shifts x by pitch_col_mm."""
        cx = 100.0
        cy = 100.0
        x0, _ = self.geom.pixel_to_chip_local_mm(cx, cy)
        x1, _ = self.geom.pixel_to_chip_local_mm(cx + 1, cy)
        assert abs(x1 - x0 - self.spec.pitch_col_mm) < 1e-9

    def test_pixel_to_trk_z_equals_layer_z(self):
        """z component of pixel_to_trk_mm must equal layer z regardless of pixel."""
        cx = (self.spec.n_cols - 1) / 2.0
        cy = (self.spec.n_rows - 1) / 2.0
        for lid in range(5):
            result = self.geom.pixel_to_trk_mm(lid, cx, cy)
            assert result[2] == pytest.approx(lid * 20.0)

    def test_pixel_to_trk_layer0_center_no_alignment(self):
        cx = (self.spec.n_cols - 1) / 2.0
        cy = (self.spec.n_rows - 1) / 2.0
        result = self.geom.pixel_to_trk_mm(0, cx, cy)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_pixel_to_trk_with_alignment_shifts_center(self):
        """Center pixel on layer 1 should be displaced by alignment dx_mm."""
        cx = (self.spec.n_cols - 1) / 2.0
        cy = (self.spec.n_rows - 1) / 2.0
        result = self.geom_aln.pixel_to_trk_mm(1, cx, cy)
        assert abs(result[0] - self.geom_aln.layers[1].dx_mm) < 1e-3
        assert abs(result[1] - self.geom_aln.layers[1].dy_mm) < 1e-3

    def test_trk_to_lab_output_shape_and_finite(self):
        result = self.geom.trk_to_lab_mm(5.0, 2.0, 40.0)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_full_chain_pixel_to_lab_finite(self):
        """Pixel → TRK → LAB all produce finite values."""
        cx, cy = 512.0, 256.0
        for lid in range(5):
            trk = self.geom.pixel_to_trk_mm(lid, cx, cy)
            lab = self.geom.trk_to_lab_mm(*trk)
            assert np.all(np.isfinite(trk))
            assert np.all(np.isfinite(lab))


# ── SENSOR_TO_LAYER ───────────────────────────────────────────────────────────

class TestSensorToLayer:
    def test_exact_mapping(self):
        assert SENSOR_TO_LAYER == {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}

    def test_only_five_entries(self):
        assert len(SENSOR_TO_LAYER) == 5


# ── Real-data integration (skipped if parquet absent) ─────────────────────────

class TestGeometryRealData:
    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        import polars as pl
        if not HIT_LEVEL_PARQUET.exists():
            pytest.skip(f"Real data not found: {HIT_LEVEL_PARQUET}")

    def test_pixel_to_trk_coords_finite_and_in_range(self):
        import polars as pl
        geom = E320PrototypeGeometry(use_alignment=True)
        df = pl.read_parquet(HIT_LEVEL_PARQUET).filter(pl.col("det_type") == "pixel")
        df = df.with_columns(pl.col("sensor_id").replace(SENSOR_TO_LAYER).alias("layer_id"))
        for row in df.head(50).iter_rows(named=True):
            trk = geom.pixel_to_trk_mm(int(row["layer_id"]), float(row["hit_x"]), float(row["hit_y"]))
            assert np.all(np.isfinite(trk))
            assert abs(trk[0]) < 20.0
            assert abs(trk[1]) < 15.0
