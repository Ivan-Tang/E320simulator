"""Quick smoke test for the Kalman filter pipeline."""
import numpy as np
from src.kalman_tracker import KalmanConfig, _process_event_kalman

cfg = KalmanConfig()

# Synthetic event: 1 track across 5 layers, x(z)=0.005*z+1, y(z)=-0.003*z+2
rng = np.random.default_rng(42)
z_layers = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
ax_true, bx_true = 0.005, 1.0
ay_true, by_true = -0.003, 2.0

x = ax_true * z_layers + bx_true + rng.normal(0, 0.01, 5)
y = ay_true * z_layers + by_true + rng.normal(0, 0.01, 5)
z = z_layers.copy()
layer = np.arange(5, dtype=np.int8)
nid = np.arange(5, dtype=np.int64)

# Add some noise hits
n_noise = 15
x_noise = rng.uniform(-5, 5, n_noise)
y_noise = rng.uniform(-5, 5, n_noise)
z_noise = rng.choice(z_layers, n_noise)
layer_noise = np.array(
    [int(np.where(z_layers == zz)[0][0]) for zz in z_noise], dtype=np.int8
)
nid_noise = np.arange(5, 5 + n_noise, dtype=np.int64)

x_all = np.concatenate([x, x_noise])
y_all = np.concatenate([y, y_noise])
z_all = np.concatenate([z, z_noise])
layer_all = np.concatenate([layer, layer_noise])
nid_all = np.concatenate([nid, nid_noise])

result = _process_event_kalman(0, x_all, y_all, z_all, layer_all, nid_all, cfg)
kept = [c for c in result if c["is_kept"]]
print(f"Total candidates: {len(result)}")
print(f"Kept tracks: {len(kept)}")
for c in kept:
    print(
        f"  nodes={c['node_ids']}, n_layers={c['n_layers']}, "
        f"ax={c['ax']:.5f}, ay={c['ay']:.5f}, "
        f"chi2={c['chi2']:.6f}, rms={c['rms']*1e3:.2f} um"
    )

truth_found = any(set(c["node_ids"]) == {0, 1, 2, 3, 4} for c in kept)
print(f"True track nodes [0,1,2,3,4] found: {truth_found}")

if not truth_found:
    print("ERROR: True track was NOT found!")
    exit(1)
else:
    print("SUCCESS: Kalman filter found the true track.")
