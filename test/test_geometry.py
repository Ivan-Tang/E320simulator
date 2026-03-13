import pytest
import polars as pl
import numpy as np
import math
from pathlib import Path
from src.geometry import (
    E320PrototypeGeometry, 
    SENSOR_TO_LAYER, 
    AlpideSpec,
    LayerGeometry
)


class TestAlpideSpec:
    """Test AlpideSpec dataclass"""
    
    def test_default_values(self):
        spec = AlpideSpec()
        assert spec.n_rows == 512
        assert spec.n_cols == 1024
        assert spec.pitch_row_mm == 27e-3
        assert spec.pitch_col_mm == 29e-3
    
    def test_width_mm(self):
        spec = AlpideSpec()
        expected_width = 1024 * 29e-3
        assert abs(spec.width_mm - expected_width) < 1e-6
    
    def test_height_mm(self):
        spec = AlpideSpec()
        expected_height = 512 * 27e-3
        assert abs(spec.height_mm - expected_height) < 1e-6


class TestLayerGeometry:
    """Test LayerGeometry dataclass"""
    
    def test_default_values(self):
        layer = LayerGeometry(layer_id=0, z_trk_mm=0.0)
        assert layer.layer_id == 0
        assert layer.z_trk_mm == 0.0
        assert layer.dx_mm == 0.0
        assert layer.dy_mm == 0.0
        assert layer.theta_z_rad == 0.0
    
    def test_rotation_matrix_identity(self):
        layer = LayerGeometry(layer_id=0, z_trk_mm=0.0)
        R = layer.rotation_matrix_2d()
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_matrix_90deg(self):
        layer = LayerGeometry(layer_id=0, z_trk_mm=0.0, theta_z_rad=np.pi/2)
        R = layer.rotation_matrix_2d()
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_matrix_small_angle(self):
        theta = 1e-3
        layer = LayerGeometry(layer_id=0, z_trk_mm=0.0, theta_z_rad=theta)
        R = layer.rotation_matrix_2d()
        c = math.cos(theta)
        s = math.sin(theta)
        expected = np.array([[c, -s], [s, c]])
        np.testing.assert_array_almost_equal(R, expected)


class TestE320PrototypeGeometry:
    """Test E320PrototypeGeometry class"""
    
    def test_initialization_without_alignment(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        assert len(geom.layers) == 5
        for i in range(5):
            layer = geom.layers[i]
            assert layer.layer_id == i
            assert layer.z_trk_mm == i * 20.0
            assert layer.dx_mm == 0.0
            assert layer.dy_mm == 0.0
            assert layer.theta_z_rad == 0.0
    
    def test_initialization_with_alignment(self):
        geom = E320PrototypeGeometry(use_alignment=True)
        assert len(geom.layers) == 5
        
        # Layer 0 should have no alignment
        assert geom.layers[0].dx_mm == 0.0
        assert geom.layers[0].dy_mm == 0.0
        assert geom.layers[0].theta_z_rad == 0.0
        
        # Layer 1 should have alignment values
        assert abs(geom.layers[1].dx_mm - (-23.5e-3)) < 1e-6
        assert abs(geom.layers[1].dy_mm - 39.3e-3) < 1e-6
        assert abs(geom.layers[1].theta_z_rad - (-1.8e-3)) < 1e-6
    
    def test_pixel_to_chip_local_center(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        # Center pixel should be at (0, 0)
        center_x = (geom.spec.n_cols - 1) / 2.0
        center_y = (geom.spec.n_rows - 1) / 2.0
        result = geom.pixel_to_chip_local_mm(center_x, center_y)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])
    
    def test_pixel_to_chip_local_corner(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        # Top-left corner pixel (0, 0)
        result = geom.pixel_to_chip_local_mm(0.0, 0.0)
        expected_x = -(geom.spec.n_cols - 1) / 2.0 * geom.spec.pitch_col_mm
        expected_y = -(geom.spec.n_rows - 1) / 2.0 * geom.spec.pitch_row_mm
        np.testing.assert_array_almost_equal(result, [expected_x, expected_y])
    
    def test_pixel_to_trk_layer0_no_alignment(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        # Center pixel on layer 0
        center_x = (geom.spec.n_cols - 1) / 2.0
        center_y = (geom.spec.n_rows - 1) / 2.0
        result = geom.pixel_to_trk_mm(0, center_x, center_y)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])
    
    def test_pixel_to_trk_layer1_with_alignment(self):
        geom = E320PrototypeGeometry(use_alignment=True)
        # Center pixel on layer 1 should be shifted by alignment
        center_x = (geom.spec.n_cols - 1) / 2.0
        center_y = (geom.spec.n_rows - 1) / 2.0
        result = geom.pixel_to_trk_mm(1, center_x, center_y)
        # Should be close to the alignment offsets (rotation is small)
        assert abs(result[0] - geom.layers[1].dx_mm) < 1e-3
        assert abs(result[1] - geom.layers[1].dy_mm) < 1e-3
        assert result[2] == 20.0
    
    def test_trk_to_lab_origin(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        result = geom.trk_to_lab_mm(0.0, 0.0, 0.0)
        # Origin in TRK frame should be shifted by global parameters
        assert abs(result[0] - geom.global_shift_x_lab_mm) < 1e-6
        # Y and Z are affected by rotation
        assert result[1] != 0.0  # Due to rotation
        assert result[2] != 0.0  # Due to z_offset and rotation
    
    def test_trk_to_lab_consistency(self):
        geom = E320PrototypeGeometry(use_alignment=False)
        # Test that transformation is consistent
        x, y, z = 10.0, 5.0, 20.0
        result = geom.trk_to_lab_mm(x, y, z)
        
        # Check basic properties
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


class TestSensorToLayer:
    """Test SENSOR_TO_LAYER mapping"""
    
    def test_sensor_to_layer_mapping(self):
        assert SENSOR_TO_LAYER[0] == 0
        assert SENSOR_TO_LAYER[2] == 1
        assert SENSOR_TO_LAYER[4] == 2
        assert SENSOR_TO_LAYER[6] == 3
        assert SENSOR_TO_LAYER[8] == 4
    
    def test_sensor_to_layer_length(self):
        assert len(SENSOR_TO_LAYER) == 5


class TestGeometryWithRealData:
    """Integration tests using real data"""
    
    @pytest.fixture
    def data_path(self):
        return Path('/Users/IvanTang/hep/data_Run502/hit_level.parquet')
    
    @pytest.fixture
    def geometry(self):
        return E320PrototypeGeometry(use_alignment=True)
    
    def test_pixel_to_trk_with_real_data(self, data_path, geometry):
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        
        # Read data with polars
        df = pl.read_parquet(data_path)
        df = df.filter(pl.col("det_type") == "pixel")
        
        # Map sensor_id to layer_id
        df = df.with_columns([
            pl.col("sensor_id").replace(SENSOR_TO_LAYER).alias("layer_id")
        ])
        
        # Apply transformation to first 100 hits
        df_sample = df.head(100)
        
        coords_list = []
        for row in df_sample.iter_rows(named=True):
            coords = geometry.pixel_to_trk_mm(
                int(row["layer_id"]),
                float(row["hit_x"]),
                float(row["hit_y"]),
            )
            coords_list.append(coords)
        
        # Check that all coordinates are finite
        coords_array = np.array(coords_list)
        assert np.all(np.isfinite(coords_array))
        
        # Check reasonable ranges (based on sensor dimensions and z positions)
        x_vals = coords_array[:, 0]
        y_vals = coords_array[:, 1]
        z_vals = coords_array[:, 2]
        
        # X and Y should be within sensor dimensions
        assert np.all(np.abs(x_vals) < 20.0)  # ~half of sensor width
        assert np.all(np.abs(y_vals) < 15.0)  # ~half of sensor height
        
        # Z should be at layer positions
        unique_z = np.unique(z_vals)
        expected_z = [0.0, 20.0, 40.0, 60.0, 80.0]
        for z in unique_z:
            assert any(abs(z - ez) < 0.1 for ez in expected_z)
    
    def test_full_transformation_chain(self, data_path, geometry):
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        
        # Read and process data with polars
        df = pl.read_parquet(data_path)
        df = df.filter(pl.col("det_type") == "pixel")
        df = df.with_columns([
            pl.col("sensor_id").replace(SENSOR_TO_LAYER).alias("layer_id")
        ])
        
        # Sample data
        df_sample = df.head(10)
        
        # Apply full transformation chain
        for row in df_sample.iter_rows(named=True):
            # Pixel -> TRK
            trk_coords = geometry.pixel_to_trk_mm(
                int(row["layer_id"]),
                float(row["hit_x"]),
                float(row["hit_y"]),
            )
            
            # TRK -> LAB
            lab_coords = geometry.trk_to_lab_mm(
                trk_coords[0],
                trk_coords[1],
                trk_coords[2]
            )
            
            # Verify all coordinates are finite
            assert np.all(np.isfinite(trk_coords))
            assert np.all(np.isfinite(lab_coords))


if __name__ == "__main__":
    # Run basic demonstration
    data_path = Path('/Users/IvanTang/hep/data_Run502/hit_level.parquet')
    
    if data_path.exists():
        print("Running basic data processing demo with polars...")
        geom = E320PrototypeGeometry(use_alignment=True)
        
        df = pl.read_parquet(data_path)
        df = df.filter(pl.col("det_type") == "pixel")
        df = df.with_columns([
            pl.col("sensor_id").replace(SENSOR_TO_LAYER).alias("layer_id")
        ])
        
        # Process first 10 rows
        coords_list = []
        for row in df.head(10).iter_rows(named=True):
            coords = geom.pixel_to_trk_mm(
                int(row["layer_id"]),
                float(row["hit_x"]),
                float(row["hit_y"]),
            )
            coords_list.append(coords.tolist())
        
        # Add coordinates as new columns
        coords_df = pl.DataFrame(
            coords_list,
            schema=["x_trk_mm", "y_trk_mm", "z_trk_mm"],
            orient='row'
        )
        
        result = pl.concat([df.head(10), coords_df], how="horizontal")
        print(result)
        
        print("\n✓ All transformations completed successfully!")
    else:
        print(f"Data file not found: {data_path}")
        print("Run tests with: pytest test_geometry.py")