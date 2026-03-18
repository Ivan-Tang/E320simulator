"""
Kalman-filter track-finding for E320 prototype tracker.

Pipeline
--------
pixel hits → TRK coordinates → seed from layer 0+1 (slope window)
→ Kalman predict/update through layers 2 → 3 → 4
→ χ² gating at each layer → final scoring → greedy shared-hit rejection

State model
-----------
State vector  x = [x, slope_x, y, slope_y]ᵀ   (4 × 1)
Measurement   z = [x_meas, y_meas]ᵀ             (2 × 1)
Linear propagation: x(z) = x₀ + slope · Δz  (no magnetic field)

Output schema (identical to baseline.py / hough_baseline.py)
------------------------------------------------------------
event_id | candidate_id | node_ids | n_layers | ax | bx | ay | by | chi2 | rms | is_kept
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.geometry import E320PrototypeGeometry, SENSOR_TO_LAYER
from src.baseline import _vectorized_pixel_to_trk, _shared_hit_rejection


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KalmanConfig:
    """Tunable knobs for the Kalman filter tracker."""

    # ── seeding (layer 0 → 1) ────────────────────────────────────────────
    slope_x_max: float = 0.02          # |dx/dz| upper bound
    slope_y_max: float = 0.02          # |dy/dz| upper bound

    # ── measurement noise (pixel resolution) ─────────────────────────────
    #    σ ≈ pitch / √12:  col pitch 29 μm → ~8.4 μm,  row pitch 27 μm → ~7.8 μm
    sigma_x_mm: float = 8.4e-3
    sigma_y_mm: float = 7.8e-3

    # ── process noise (multiple scattering / alignment proxy) ────────────
    #    q_pos and q_slope are variance per mm of dz
    q_pos: float = 1e-6               # position process noise  [mm² / mm]
    q_slope: float = 1e-8             # slope process noise     [1 / mm]

    # ── initial covariance on seed slopes ────────────────────────────────
    seed_slope_sigma: float = 0.005    # σ on initial slope estimate

    # ── χ² gating per hit ────────────────────────────────────────────────
    chi2_gate: float = 15.0            # max χ² (2 DOF → ~99.5 % for 10.6)

    # ── track selection ──────────────────────────────────────────────────
    min_layers: int = 4                # minimum number of layers on track

    # ── parallelism ──────────────────────────────────────────────────────
    n_workers: int = 8


# ──────────────────────────────────────────────────────────────────────────────
# Kalman matrices (constant or dz-dependent)
# ──────────────────────────────────────────────────────────────────────────────
_H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
], dtype=np.float64)           # 2 × 4  measurement matrix


def _F(dz: float) -> np.ndarray:
    """4 × 4 state propagation matrix for step dz."""
    return np.array([
        [1.0,  dz, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0,  dz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _Q(dz: float, cfg: KalmanConfig) -> np.ndarray:
    """4 × 4 process-noise covariance for step dz.

    Uses a simple discrete noise model where position and slope
    noise scale linearly with |dz|.
    """
    adz = abs(dz)
    qp = cfg.q_pos * adz
    qs = cfg.q_slope * adz
    return np.diag([qp, qs, qp, qs])


def _R(cfg: KalmanConfig) -> np.ndarray:
    """2 × 2 measurement noise covariance."""
    return np.diag([cfg.sigma_x_mm ** 2, cfg.sigma_y_mm ** 2])


# ──────────────────────────────────────────────────────────────────────────────
# Kalman predict / update
# ──────────────────────────────────────────────────────────────────────────────
def _predict(state: np.ndarray, cov: np.ndarray, F: np.ndarray, Q: np.ndarray):
    """Kalman predict step.  Returns (state_pred, cov_pred)."""
    s = F @ state
    P = F @ cov @ F.T + Q
    return s, P


def _update(
    state: np.ndarray,
    cov: np.ndarray,
    meas: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kalman update step.

    Returns (state_upd, cov_upd, chi2_inc) where chi2_inc is the
    normalised innovation χ² for this measurement.
    """
    H = _H
    y = meas - H @ state                    # innovation (2 × 1)
    S = H @ cov @ H.T + R                   # innovation covariance (2 × 2)
    S_inv = np.linalg.inv(S)
    K = cov @ H.T @ S_inv                   # Kalman gain (4 × 2)
    state_upd = state + K @ y
    cov_upd = (np.eye(4) - K @ H) @ cov
    chi2_inc = float(y @ S_inv @ y)          # scalar χ²
    return state_upd, cov_upd, chi2_inc


# ──────────────────────────────────────────────────────────────────────────────
# Seeding: layer 0 → layer 1
# ──────────────────────────────────────────────────────────────────────────────
def _build_seeds(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    cfg: KalmanConfig,
) -> list[tuple[np.ndarray, np.ndarray, list[int], list[int], float]]:
    """Create Kalman seeds from hit pairs on layers 0 and 1.

    Returns a list of (state_4, cov_4x4, [nid0, nid1], [lid0, lid1], chi2_seed).
    """
    m0 = layer == 0
    m1 = layer == 1
    if not m0.any() or not m1.any():
        return []

    x0, y0, z0, n0 = x[m0], y[m0], z[m0], nid[m0]
    x1, y1, z1, n1 = x[m1], y[m1], z[m1], nid[m1]

    R_mat = _R(cfg)

    # Vectorised pair filtering — no Python double loop.
    dz_mat  = z1[None, :] - z0[:, None]          # (N0, N1)
    dx_mat  = x1[None, :] - x0[:, None]
    dy_mat  = y1[None, :] - y0[:, None]

    valid   = np.abs(dz_mat) >= 1e-9
    safe_dz = np.where(valid, dz_mat, 1.0)        # avoid division by zero
    sx_mat  = dx_mat / safe_dz
    sy_mat  = dy_mat / safe_dz

    mask = valid & (np.abs(sx_mat) <= cfg.slope_x_max) & (np.abs(sy_mat) <= cfg.slope_y_max)
    ii, jj = np.where(mask)
    if len(ii) == 0:
        return []

    sx_vals  = sx_mat[ii, jj]
    sy_vals  = sy_mat[ii, jj]
    dz_vals  = dz_mat[ii, jj]

    # χ² from layer-0 residual (vectorised)
    dx0      = x0[ii] - (x1[jj] - sx_vals * dz_vals)
    dy0      = y0[ii] - (y1[jj] - sy_vals * dz_vals)
    chi2_vals = dx0 ** 2 / R_mat[0, 0] + dy0 ** 2 / R_mat[1, 1]

    # Covariance is identical for every seed; copy once per seed.
    sig_s    = cfg.seed_slope_sigma
    base_cov = np.diag([cfg.sigma_x_mm ** 2, sig_s ** 2,
                        cfg.sigma_y_mm ** 2, sig_s ** 2])

    seeds: list[tuple[np.ndarray, np.ndarray, list[int], list[int], float]] = []
    for k in range(len(ii)):
        i, j  = int(ii[k]), int(jj[k])
        state = np.array([x1[j], sx_vals[k], y1[j], sy_vals[k]], dtype=np.float64)
        seeds.append((state, base_cov.copy(), [int(n0[i]), int(n1[j])], [0, 1],
                      float(chi2_vals[k])))

    return seeds


# ──────────────────────────────────────────────────────────────────────────────
# Track propagation through remaining layers
# ──────────────────────────────────────────────────────────────────────────────
def _propagate_seed(
    state: np.ndarray,
    cov: np.ndarray,
    seed_nids: list[int],
    seed_lids: list[int],
    chi2_acc: float,
    z_current: float,
    hits_by_layer: dict[int, list[tuple[float, float, int]]],
    layer_z: dict[int, float],
    cfg: KalmanConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], float] | None:
    """Propagate a seeded track through layers 2, 3, 4.

    At each layer, the hit with the lowest χ² that passes the gate is
    picked.  If no hit passes the gate at a layer, the layer is skipped
    (track can still survive if min_layers is met).

    Returns (final_state, final_cov, node_ids, layer_ids, total_chi2)
    or None if fewer than cfg.min_layers are collected.
    """
    R_mat = _R(cfg)
    node_ids = list(seed_nids)
    layer_ids = list(seed_lids)
    total_chi2 = chi2_acc

    s = state.copy()
    P = cov.copy()
    z_cur = z_current

    for lid in range(2, 5):
        hits = hits_by_layer.get(lid, [])
        if not hits:
            continue

        z_layer = layer_z[lid]
        dz = z_layer - z_cur

        # predict
        F_mat = _F(dz)
        Q_mat = _Q(dz, cfg)
        s_pred, P_pred = _predict(s, P, F_mat, Q_mat)

        # try all hits at this layer, pick best passing gate
        best_chi2 = cfg.chi2_gate
        best_state = None
        best_cov = None
        best_nid = -1

        for (hx, hy, hn) in hits:
            meas = np.array([hx, hy], dtype=np.float64)
            s_upd, P_upd, chi2_inc = _update(s_pred, P_pred, meas, R_mat)
            if chi2_inc < best_chi2:
                best_chi2 = chi2_inc
                best_state = s_upd
                best_cov = P_upd
                best_nid = hn

        if best_state is not None:
            s = best_state
            P = best_cov
            z_cur = z_layer
            node_ids.append(best_nid)
            layer_ids.append(lid)
            total_chi2 += best_chi2

    if len(node_ids) < cfg.min_layers:
        return None

    return s, P, node_ids, layer_ids, total_chi2


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────
def _score_track(
    node_ids: list[int],
    layer_ids: list[int],
    total_chi2: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nid_to_local: dict[int, int],
) -> dict:
    """Compute linear-fit parameters and RMS for a completed track."""
    idx = [nid_to_local[n] for n in node_ids]
    xs, ys, zs = x[idx], y[idx], z[idx]
    n_pts = len(idx)

    A = np.column_stack([zs, np.ones(n_pts)])
    (a_x, b_x), *_ = np.linalg.lstsq(A, xs, rcond=None)
    (a_y, b_y), *_ = np.linalg.lstsq(A, ys, rcond=None)

    dx = xs - (a_x * zs + b_x)
    dy = ys - (a_y * zs + b_y)
    r2 = dx ** 2 + dy ** 2

    dof = max(2 * n_pts - 4, 1)
    return {
        "node_ids": node_ids,
        "n_layers": n_pts,
        "ax": float(a_x),
        "bx": float(b_x),
        "ay": float(a_y),
        "by": float(b_y),
        "chi2": float(np.sum(r2)) / dof,
        "rms": float(np.sqrt(np.mean(r2))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-event orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def _process_event_kalman(
    event_id: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    cfg: KalmanConfig,
) -> list[dict]:
    """Run the full Kalman pipeline for a single event."""
    geom = E320PrototypeGeometry(use_alignment=True)
    layer_z = {lid: geom.layers[lid].z_trk_mm for lid in range(5)}

    # group hits by layer for fast lookup during propagation
    hits_by_layer: dict[int, list[tuple[float, float, int]]] = {lid: [] for lid in range(5)}
    for k in range(len(x)):
        lid = int(layer[k])
        hits_by_layer[lid].append((float(x[k]), float(y[k]), int(nid[k])))

    # 1. seed from layers 0 + 1
    seeds = _build_seeds(x, y, z, layer, nid, cfg)
    if not seeds:
        return []

    # 2. propagate each seed through layers 2 → 3 → 4
    completed: list[tuple[list[int], list[int], float]] = []
    for state, cov, seed_nids, seed_lids, chi2_seed in seeds:
        z_seed = layer_z[1]  # seed is initialised at layer 1
        result = _propagate_seed(
            state, cov, seed_nids, seed_lids, chi2_seed,
            z_seed, hits_by_layer, layer_z, cfg,
        )
        if result is not None:
            _, _, node_ids, layer_ids, total_chi2 = result
            completed.append((node_ids, layer_ids, total_chi2))

    if not completed:
        return []

    # 3. deduplicate identical node sets (keep lowest chi2)
    best_by_nodes: dict[frozenset[int], tuple[list[int], list[int], float]] = {}
    for node_ids, layer_ids, total_chi2 in completed:
        key = frozenset(node_ids)
        if key not in best_by_nodes or total_chi2 < best_by_nodes[key][2]:
            best_by_nodes[key] = (node_ids, layer_ids, total_chi2)

    # 4. score all unique candidates
    nid_to_local = {int(n): i for i, n in enumerate(nid)}
    candidates: list[dict] = []
    for node_ids, layer_ids, total_chi2 in best_by_nodes.values():
        cand = _score_track(node_ids, layer_ids, total_chi2, x, y, z, nid_to_local)
        candidates.append(cand)

    # 5. shared-hit rejection
    candidates = _shared_hit_rejection(candidates)

    for i, c in enumerate(candidates):
        c["event_id"] = event_id
        c["candidate_id"] = i
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def run_kalman(
    data_path: str,
    cfg: KalmanConfig | None = None,
) -> pl.DataFrame:
    """Execute the full Kalman filter tracker and return a polars DataFrame.

    Parameters
    ----------
    data_path : str
        Path to the ``hit_level.parquet`` file.
    cfg : KalmanConfig, optional
        Pipeline configuration.  Uses sensible defaults if *None*.

    Returns
    -------
    pl.DataFrame
        Columns: ``event_id, candidate_id, node_ids, n_layers,
        ax, bx, ay, by, chi2, rms, is_kept``
    """
    if cfg is None:
        cfg = KalmanConfig()

    geom = E320PrototypeGeometry(use_alignment=True)

    # ── 1. load & filter ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    df = (
        pl.read_parquet(data_path)
        .filter(pl.col("det_type") == "pixel")
        .with_columns(
            pl.col("sensor_id").replace(SENSOR_TO_LAYER).cast(pl.Int8).alias("layer_id"),
        )
    )

    # ── 2. vectorised pixel → TRK ────────────────────────────────────────
    x_trk, y_trk, z_trk = _vectorized_pixel_to_trk(
        df["hit_x"].to_numpy().astype(np.float64),
        df["hit_y"].to_numpy().astype(np.float64),
        df["layer_id"].to_numpy(),
        geom,
    )
    df = (
        df.with_columns(
            pl.Series("x_trk_mm", x_trk),
            pl.Series("y_trk_mm", y_trk),
            pl.Series("z_trk_mm", z_trk),
        )
        .sort("event_id")
        .with_row_index("node_id")
    )

    n_hits = df.height
    n_events = df["event_id"].n_unique()
    t1 = time.perf_counter()
    print(f"[kalman] {n_hits:,} hits, {n_events} events  (load {t1 - t0:.2f}s)")

    # ── 3. pre-split into numpy views per event ──────────────────────────
    eid_arr = df["event_id"].to_numpy()
    x_arr = df["x_trk_mm"].to_numpy()
    y_arr = df["y_trk_mm"].to_numpy()
    z_arr = df["z_trk_mm"].to_numpy()
    lid_arr = df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = df["node_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    event_slices = [
        (
            int(unique_events[i]),
            x_arr[starts[i] : starts[i] + counts[i]],
            y_arr[starts[i] : starts[i] + counts[i]],
            z_arr[starts[i] : starts[i] + counts[i]],
            lid_arr[starts[i] : starts[i] + counts[i]],
            nid_arr[starts[i] : starts[i] + counts[i]],
        )
        for i in range(len(unique_events))
    ]

    # ── 4. parallel per-event processing ─────────────────────────────────
    t2 = time.perf_counter()
    all_candidates: list[dict] = []

    with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
        futures = [
            pool.submit(_process_event_kalman, eid, xv, yv, zv, lv, nv, cfg)
            for eid, xv, yv, zv, lv, nv in event_slices
        ]
        for f in as_completed(futures):
            all_candidates.extend(f.result())

    t3 = time.perf_counter()

    # ── 5. assemble output ───────────────────────────────────────────────
    if not all_candidates:
        return pl.DataFrame(
            schema={
                "event_id": pl.Int64,
                "candidate_id": pl.Int32,
                "node_ids": pl.List(pl.UInt32),
                "n_layers": pl.Int8,
                "ax": pl.Float64,
                "bx": pl.Float64,
                "ay": pl.Float64,
                "by": pl.Float64,
                "chi2": pl.Float64,
                "rms": pl.Float64,
                "is_kept": pl.Boolean,
            }
        )

    result = pl.DataFrame(
        {
            "event_id": [c["event_id"] for c in all_candidates],
            "candidate_id": [c["candidate_id"] for c in all_candidates],
            "node_ids": [c["node_ids"] for c in all_candidates],
            "n_layers": [c["n_layers"] for c in all_candidates],
            "ax": [c["ax"] for c in all_candidates],
            "bx": [c["bx"] for c in all_candidates],
            "ay": [c["ay"] for c in all_candidates],
            "by": [c["by"] for c in all_candidates],
            "chi2": [c["chi2"] for c in all_candidates],
            "rms": [c["rms"] for c in all_candidates],
            "is_kept": [c["is_kept"] for c in all_candidates],
        }
    ).sort("event_id", "candidate_id")

    n_kept = result.filter(pl.col("is_kept")).height
    n5 = result.filter(pl.col("is_kept") & (pl.col("n_layers") == 5)).height
    n4 = result.filter(pl.col("is_kept") & (pl.col("n_layers") == 4)).height
    t4 = time.perf_counter()

    print(
        f"[kalman] {result.height} candidates, {n_kept} kept "
        f"(5-hit: {n5}, 4-hit: {n4})  "
        f"track-find {t3 - t2:.2f}s  total {t4 - t0:.2f}s"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
from src.config import DATA_ROOT, HIT_LEVEL_PARQUET
DATA_PATH = str(HIT_LEVEL_PARQUET)
RESULT_PATH = str(DATA_ROOT / "kalman.parquet")

if __name__ == "__main__":
    result = run_kalman(DATA_PATH)
    print(result)

    kept = result.filter(pl.col("is_kept"))
    print(f"\nKept tracks: {kept.height}")
    print(f"  5-hit: {kept.filter(pl.col('n_layers') == 5).height}")
    print(f"  4-hit: {kept.filter(pl.col('n_layers') == 4).height}")
    print(f"  Mean χ²:  {kept['chi2'].mean():.4f}")
    print(f"  Mean RMS: {kept['rms'].mean() * 1e3:.1f} μm")

    # save 5-hit result
    n5 = result.filter(pl.col("is_kept") & (pl.col("n_layers") == 5))
    n5.write_parquet(RESULT_PATH)
