from dataclasses import dataclass
import math
import numpy as np

SENSOR_TO_LAYER = {
    0:0,
    2:1,
    4:2,
    6:3,
    8:4,
}

@dataclass(frozen=True)
class AlpideSpec:
    n_rows: int = 512
    n_cols: int = 1024
    pitch_row_mm: float = 27e-3   # 27 um
    pitch_col_mm: float = 29e-3   # 29 um

    @property
    def width_mm(self) -> float:
        # long side, 1024 columns
        return self.n_cols * self.pitch_col_mm

    @property
    def height_mm(self) -> float:
        # short side, 512 rows
        return self.n_rows * self.pitch_row_mm
    

@dataclass
class LayerGeometry:
    layer_id: int
    z_trk_mm: float
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    theta_z_rad: float = 0.0

    def rotation_matrix_2d(self) -> np.ndarray:
        c = math.cos(self.theta_z_rad)
        s = math.sin(self.theta_z_rad)
        return np.array([[c, -s], [s, c]], dtype=float)
    


class E320PrototypeGeometry:
    def __init__(self, use_alignment: bool = True):
        self.spec = AlpideSpec()

        # TRK frame: origin at center of ALPIDE_0
        layers = [
            LayerGeometry(layer_id=0, z_trk_mm=0.0),
            LayerGeometry(layer_id=1, z_trk_mm=20.0),
            LayerGeometry(layer_id=2, z_trk_mm=40.0),
            LayerGeometry(layer_id=3, z_trk_mm=60.0),
            LayerGeometry(layer_id=4, z_trk_mm=80.0),
        ]

        if use_alignment:
            # Table 4 values from paper, converted to mm / rad
            layers[1].dx_mm = -23.5e-3
            layers[1].dy_mm = +39.3e-3
            layers[1].theta_z_rad = -1.8e-3

            layers[2].dx_mm = +28.8e-3
            layers[2].dy_mm = -53.9e-3
            layers[2].theta_z_rad = +1.6e-3

            layers[3].dx_mm = +64.9e-3
            layers[3].dy_mm = -64.2e-3
            layers[3].theta_z_rad = -1.5e-3

            layers[4].dx_mm = -24.6e-3
            layers[4].dy_mm = +27.5e-3
            layers[4].theta_z_rad = +0.5e-3

        self.layers = {layer.layer_id: layer for layer in layers}

        # Global relation to LAB frame
        self.z_lab_exit_to_first_chip_mm = 124.8
        self.global_shift_x_lab_mm = -9.1
        self.global_shift_y_lab_mm = +0.2
        self.global_tilt_x_rad = 4e-3  # rotation
        
    def pixel_to_chip_local_mm(self, hit_x: float, hit_y: float) -> np.ndarray:
        x_mm = (hit_x - (self.spec.n_cols - 1) / 2.0) * self.spec.pitch_col_mm
        y_mm = (hit_y - (self.spec.n_rows - 1) / 2.0) * self.spec.pitch_row_mm
        return np.array([x_mm, y_mm], dtype=float)

    def pixel_to_trk_mm(self, layer_id: int, hit_x: float, hit_y: float) -> np.ndarray:
        layer = self.layers[layer_id]
        xy = self.pixel_to_chip_local_mm(hit_x, hit_y)
        xy = layer.rotation_matrix_2d() @ xy + np.array([layer.dx_mm, layer.dy_mm])
        return np.array([xy[0], xy[1], layer.z_trk_mm], dtype=float)

    def trk_to_lab_mm(self, x_trk_mm: float, y_trk_mm: float, z_trk_mm: float) -> np.ndarray:
        z_lab = self.z_lab_exit_to_first_chip_mm + z_trk_mm
        x_lab = x_trk_mm + self.global_shift_x_lab_mm
        y_lab = y_trk_mm + self.global_shift_y_lab_mm

        c = math.cos(self.global_tilt_x_rad)
        s = math.sin(self.global_tilt_x_rad)
        y_lab2 = c * y_lab - s * z_lab
        z_lab2 = s * y_lab + c * z_lab

        return np.array([x_lab, y_lab2, z_lab2], dtype=float)