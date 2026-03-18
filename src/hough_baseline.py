"""
Hough Transform track-finding baseline for E320 prototype tracker.

Pipeline
--------
pixel hits → TRK coordinates → slope voting (Hough accumulator)
→ peak finding → hit clustering per peak → 3-D line fit
→ χ²/RMS scoring → greedy shared-hit rejection

The 2-D accumulator is built in slope space ``(a_x, a_y)`` by computing
pair-wise slopes for all pairs of hits from *different* layers.
Peaks in the accumulator correspond to track slopes; hits from each peak
are then grouped by intercept ``(b_x, b_y)`` to separate tracks with
similar slopes but different positions.

Output schema (identical to baseline.py)
-----------------------------------------
event_id | candidate_id | node_ids | n_layers | ax | bx | ay | by | chi2 | rms | is_kept
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.geometry import E320PrototypeGeometry, SENSOR_TO_LAYER
from src.baseline import _vectorized_pixel_to_trk, _fit_and_score, _shared_hit_rejection


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HoughConfig:
    """Tunable knobs aligned with the paper description.

    Defaults follow the ranges quoted in Sec. 4 of the paper:
    θx∈[0.63, 2.51], ρx∈[−61.87, 61.86], θy∈[0.31, 2.83], ρy∈[−76.66, 76.65].
    """

    # Hough space bounds (TRK frame)
    theta_x_min: float = 0.63
    theta_x_max: float = 2.51
    rho_x_min: float = -61.87
    rho_x_max: float = 61.86
    theta_y_min: float = 0.31
    theta_y_max: float = 2.83
    rho_y_min: float = -76.66
    rho_y_max: float = 76.65

    # Bin divider: coarse (pre-alignment) vs fine (post-alignment)
    n_bins_coarse: int = 650
    n_bins_fine: int = 1700
    use_fine_binning: bool = True

    # Intersection thresholds (per paper: need 10 intersections; neighbours help if 5–9)
    min_intersections: int = 10
    neighbour_seed: int = 5

    # Lookup-table granularity for tunnel hit lookup (per layer)
    lut_nx: int = 2000
    lut_ny: int = 4000

    # Track requirements
    min_layers: int = 5  # tunnel must touch all 5 layers

    # Numerical safety
    denom_epsilon: float = 1e-9

    # parallelism
    n_workers: int = 8


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for paper-faithful Hough space
# ──────────────────────────────────────────────────────────────────────────────
def _binning(cfg: HoughConfig) -> tuple[int, int, int, int, float, float, float, float]:
    n_theta = cfg.n_bins_fine if cfg.use_fine_binning else cfg.n_bins_coarse
    n_rho = n_theta
    bin_theta_x = (cfg.theta_x_max - cfg.theta_x_min) / n_theta
    bin_theta_y = (cfg.theta_y_max - cfg.theta_y_min) / n_theta
    bin_rho_x = (cfg.rho_x_max - cfg.rho_x_min) / n_rho
    bin_rho_y = (cfg.rho_y_max - cfg.rho_y_min) / n_rho
    return n_theta, n_rho, n_theta, n_rho, bin_theta_x, bin_rho_x, bin_theta_y, bin_rho_y


def _encode_key(
    jtx: np.ndarray,
    jrx: np.ndarray,
    jty: np.ndarray,
    jry: np.ndarray,
    n_rho_x: int,
    n_theta_y: int,
    n_rho_y: int,
) -> np.ndarray:
    return (((jtx * n_rho_x + jrx) * n_theta_y + jty) * n_rho_y + jry).astype(np.int64)


def _decode_key(
    key: int,
    n_rho_x: int,
    n_theta_y: int,
    n_rho_y: int,
) -> tuple[int, int, int, int]:
    k = key
    jry = k % n_rho_y
    k //= n_rho_y
    jty = k % n_theta_y
    k //= n_theta_y
    jrx = k % n_rho_x
    jtx = k // n_rho_x
    return int(jtx), int(jrx), int(jty), int(jry)


def _theta_rho_from_hits(
    k1: np.ndarray,
    k2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    cfg: HoughConfig,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan2(z2 - z1, k1 - k2)
    theta = np.mod(theta, np.pi)  # map to [0, π)
    rho = k1 * np.sin(theta) + z1 * np.cos(theta)
    return theta, rho


def _fill_accumulator(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    cfg: HoughConfig,
) -> tuple[dict[int, int], tuple[int, int, int, int, float, float, float, float]]:
    """Analytical pairwise intersections in (θx, ρx, θy, ρy) space.

    Returns a zero-suppressed accumulator (dict: key → count) and binning metadata.
    """
    n_theta_x, n_rho_x, n_theta_y, n_rho_y, btx, brx, bty, bry = _binning(cfg)
    acc: dict[int, int] = defaultdict(int)

    # group indices by layer to avoid same-layer pairs
    layer_to_idx = {lid: np.where(layer == lid)[0] for lid in np.unique(layer)}
    lids = sorted(layer_to_idx.keys())
    if len(lids) < 2:
        return acc, (n_theta_x, n_rho_x, n_theta_y, n_rho_y, btx, brx, bty, bry)

    for i in range(len(lids)):
        for j in range(i + 1, len(lids)):
            ia = layer_to_idx[lids[i]]
            ib = layer_to_idx[lids[j]]
            if len(ia) == 0 or len(ib) == 0:
                continue

            xa, ya, za = x[ia], y[ia], z[ia]
            xb, yb, zb = x[ib], y[ib], z[ib]

            # Cartesian pairs
            k1x, k2x = np.meshgrid(xa, xb, indexing="ij")
            k1y, k2y = np.meshgrid(ya, yb, indexing="ij")
            z1, z2 = np.meshgrid(za, zb, indexing="ij")

            theta_x, rho_x = _theta_rho_from_hits(k1x.ravel(), k2x.ravel(), z1.ravel(), z2.ravel(), cfg)
            theta_y, rho_y = _theta_rho_from_hits(k1y.ravel(), k2y.ravel(), z1.ravel(), z2.ravel(), cfg)

            if len(theta_x) == 0 or len(theta_y) == 0:
                continue

            jtx = ((theta_x - cfg.theta_x_min) / btx).astype(np.int64)
            jrx = ((rho_x - cfg.rho_x_min) / brx).astype(np.int64)
            jty = ((theta_y - cfg.theta_y_min) / bty).astype(np.int64)
            jry = ((rho_y - cfg.rho_y_min) / bry).astype(np.int64)

            inside = (
                (jtx >= 0) & (jtx < n_theta_x)
                & (jrx >= 0) & (jrx < n_rho_x)
                & (jty >= 0) & (jty < n_theta_y)
                & (jry >= 0) & (jry < n_rho_y)
            )
            if not inside.any():
                continue

            keys = _encode_key(jtx[inside], jrx[inside], jty[inside], jry[inside], n_rho_x, n_theta_y, n_rho_y)
            uniq, counts = np.unique(keys, return_counts=True)
            for k, c in zip(uniq.tolist(), counts.tolist()):
                acc[int(k)] += int(c)

    return acc, (n_theta_x, n_rho_x, n_theta_y, n_rho_y, btx, brx, bty, bry)


def _select_cells_with_neighbours(
    acc: dict[int, int],
    cfg: HoughConfig,
    n_theta_x: int,
    n_rho_x: int,
    n_theta_y: int,
    n_rho_y: int,
) -> list[tuple[int, int, int, int, int]]:
    """Apply N(T) thresholds with neighbour compensation."""
    accepted: list[tuple[int, int, int, int, int]] = []
    acc_get = acc.get

    for key, cnt in acc.items():
        if cnt >= cfg.min_intersections:
            jtx, jrx, jty, jry = _decode_key(key, n_rho_x, n_theta_y, n_rho_y)
            accepted.append((jtx, jrx, jty, jry, cnt))
            continue

        if cnt < cfg.neighbour_seed:
            continue

        jtx0, jrx0, jty0, jry0 = _decode_key(key, n_rho_x, n_theta_y, n_rho_y)
        total = cnt
        for dtx in (-1, 0, 1):
            ntx = jtx0 + dtx
            if ntx < 0 or ntx >= n_theta_x:
                continue
            for drx in (-1, 0, 1):
                nrx = jrx0 + drx
                if nrx < 0 or nrx >= n_rho_x:
                    continue
                for dty in (-1, 0, 1):
                    nty = jty0 + dty
                    if nty < 0 or nty >= n_theta_y:
                        continue
                    for dry in (-1, 0, 1):
                        nry = jry0 + dry
                        if nry < 0 or nry >= n_rho_y:
                            continue
                        if dtx == drx == dty == dry == 0:
                            continue
                        nkey = _encode_key(
                            np.array([ntx]),
                            np.array([nrx]),
                            np.array([nty]),
                            np.array([nry]),
                            n_rho_x,
                            n_theta_y,
                            n_rho_y,
                        )[0]
                        total += acc_get(int(nkey), 0)
        if total >= cfg.min_intersections:
            accepted.append((jtx0, jrx0, jty0, jry0, total))

    return accepted


def _cell_param_bounds(
    jtx: int,
    jrx: int,
    jty: int,
    jry: int,
    cfg: HoughConfig,
    btx: float,
    brx: float,
    bty: float,
    bry: float,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    theta_x_min = cfg.theta_x_min + jtx * btx
    theta_x_max = cfg.theta_x_min + (jtx + 1) * btx
    theta_y_min = cfg.theta_y_min + jty * bty
    theta_y_max = cfg.theta_y_min + (jty + 1) * bty
    theta_x_c = theta_x_min + 0.5 * btx
    theta_y_c = theta_y_min + 0.5 * bty

    rho_x_min = cfg.rho_x_min + jrx * brx
    rho_x_max = cfg.rho_x_min + (jrx + 1) * brx
    rho_y_min = cfg.rho_y_min + jry * bry
    rho_y_max = cfg.rho_y_min + (jry + 1) * bry
    rho_x_c = rho_x_min + 0.5 * brx
    rho_y_c = rho_y_min + 0.5 * bry

    return (
        theta_x_c,
        theta_y_c,
        rho_x_c,
        rho_y_c,
        (theta_x_min, theta_x_max),
        (theta_y_min, theta_y_max),
        (rho_x_min, rho_x_max),
        (rho_y_min, rho_y_max),
    )


def _slope_intercept_ranges(theta_bounds: tuple[float, float], rho_bounds: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    theta_min, theta_max = theta_bounds
    rho_min, rho_max = rho_bounds
    a_min = -1.0 / np.tan(theta_max)
    a_max = -1.0 / np.tan(theta_min)
    b_min = rho_min / np.sin(theta_max)
    b_max = rho_max / np.sin(theta_min)
    a_low, a_high = (a_min, a_max) if a_min <= a_max else (a_max, a_min)
    b_low, b_high = (b_min, b_max) if b_min <= b_max else (b_max, b_min)
    return (a_low, a_high), (b_low, b_high)


def _rect_bounds_for_layer(z: float, a_rng: tuple[float, float], b_rng: tuple[float, float]) -> tuple[float, float]:
    a0, a1 = a_rng
    b0, b1 = b_rng
    combos = [a0 * z + b0, a0 * z + b1, a1 * z + b0, a1 * z + b1]
    return float(min(combos)), float(max(combos))


def _build_lut(
    x: np.ndarray,
    y: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    nx: int,
    ny: int,
    geom: E320PrototypeGeometry,
) -> tuple[dict[int, dict[int, list[int]]], tuple[float, float, float, float, float, float]]:
    """Layer-wise LUT for fast (x, y) window queries."""
    half_w = geom.spec.width_mm / 2.0
    half_h = geom.spec.height_mm / 2.0
    x_min, x_max = -half_w, half_w
    y_min, y_max = -half_h, half_h
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    lut: dict[int, dict[int, list[int]]] = {int(l): defaultdict(list) for l in range(5)}

    bx = ((x - x_min) / dx).astype(np.int32)
    by = ((y - y_min) / dy).astype(np.int32)

    valid = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
    for idx in np.where(valid)[0]:
        l = int(layer[idx])
        key = int(bx[idx] * ny + by[idx])
        lut[l][key].append(int(nid[idx]))

    return lut, (x_min, x_max, y_min, y_max, dx, dy)


def _query_lut(
    lut_layer: dict[int, list[int]],
    bx0: int,
    bx1: int,
    by0: int,
    by1: int,
    ny: int,
) -> list[int]:
    hits: list[int] = []
    for bx in range(bx0, bx1 + 1):
        for by in range(by0, by1 + 1):
            hits.extend(lut_layer.get(bx * ny + by, []))
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Per-event orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def _tunnel_candidates_for_cell(
    cell: tuple[int, int, int, int, int],
    z_by_layer: np.ndarray,
    lut: dict[int, dict[int, list[int]]],
    lut_meta: tuple[float, float, float, float, float, float],
    cfg: HoughConfig,
    bin_meta: tuple[int, int, int, int, float, float, float, float],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    jtx, jrx, jty, jry, _ = cell
    n_theta_x, n_rho_x, n_theta_y, n_rho_y, btx, brx, bty, bry = bin_meta
    x_min, x_max, y_min, y_max, dx, dy = lut_meta
    ny = cfg.lut_ny

    (
        theta_x_c,
        theta_y_c,
        rho_x_c,
        rho_y_c,
        theta_x_bounds,
        theta_y_bounds,
        rho_x_bounds,
        rho_y_bounds,
    ) = _cell_param_bounds(jtx, jrx, jty, jry, cfg, btx, brx, bty, bry)

    ax_rng, bx_rng = _slope_intercept_ranges(theta_x_bounds, rho_x_bounds)
    ay_rng, by_rng = _slope_intercept_ranges(theta_y_bounds, rho_y_bounds)

    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    hits_per_layer: list[list[int]] = []

    for lid, z in enumerate(z_by_layer):
        x0, x1 = _rect_bounds_for_layer(z, ax_rng, bx_rng)
        y0, y1 = _rect_bounds_for_layer(z, ay_rng, by_rng)

        bx0 = max(0, int(np.floor((min(x0, x1) - x_min) / dx)))
        bx1 = min(cfg.lut_nx - 1, int(np.floor((max(x0, x1) - x_min) / dx)))
        by0 = max(0, int(np.floor((min(y0, y1) - y_min) / dy)))
        by1 = min(cfg.lut_ny - 1, int(np.floor((max(y0, y1) - y_min) / dy)))

        layer_hits = _query_lut(lut.get(lid, {}), bx0, bx1, by0, by1, ny)
        hits_per_layer.append(layer_hits)

    if any(len(h) == 0 for h in hits_per_layer):
        return []

    for combo in itertools.product(*hits_per_layer):
        node_ids = tuple(int(h) for h in combo)
        layer_ids = tuple(range(len(z_by_layer)))
        candidates.append((node_ids, layer_ids))

    return candidates


def _process_event_hough(
    event_id: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    cfg: HoughConfig,
) -> list[dict]:
    """Run the Hough pipeline described in the paper for one event."""
    # 1. 4D accumulator (θx, ρx, θy, ρy)
    acc, bin_meta = _fill_accumulator(x, y, z, layer, cfg)
    n_theta_x, n_rho_x, n_theta_y, n_rho_y, btx, brx, bty, bry = bin_meta

    if not acc:
        return []

    # 2. Cell selection with neighbour rescue
    cells = _select_cells_with_neighbours(acc, cfg, n_theta_x, n_rho_x, n_theta_y, n_rho_y)
    if not cells:
        return []

    # 3. Build LUT once per event
    geom = E320PrototypeGeometry(use_alignment=True)
    lut, lut_meta = _build_lut(x, y, layer, nid, cfg.lut_nx, cfg.lut_ny, geom)
    z_by_layer = np.array([geom.layers[i].z_trk_mm for i in range(5)], dtype=float)

    # 4. For each accepted cell, form tunnels → seeds
    chains: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for cell in cells:
        chains.extend(_tunnel_candidates_for_cell(cell, z_by_layer, lut, lut_meta, cfg, bin_meta))

    if not chains:
        return []

    # 5. Deduplicate identical node sets
    uniq: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[frozenset[int]] = set()
    for nodes, layers in chains:
        key = frozenset(nodes)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((nodes, layers))

    # 6. Fit & score (reuse baseline utilities)
    nid_to_local = {int(n): i for i, n in enumerate(nid)}
    candidates = _fit_and_score(uniq, x, y, z, nid_to_local)
    candidates = _shared_hit_rejection(candidates)

    for i, c in enumerate(candidates):
        c["event_id"] = event_id
        c["candidate_id"] = i
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def run_hough(
    data_path: str,
    cfg: HoughConfig | None = None,
) -> pl.DataFrame:
    """Execute the full Hough baseline and return a polars DataFrame of track
    candidates.

    Parameters
    ----------
    data_path : str
        Path to the ``hit_level.parquet`` file.
    cfg : HoughConfig, optional
        Pipeline configuration.  Uses sensible defaults if *None*.

    Returns
    -------
    pl.DataFrame
        Columns: ``event_id, candidate_id, node_ids, n_layers,
        ax, bx, ay, by, chi2, rms, is_kept``
    """
    if pl is None:
        raise ImportError("polars is required to run the Hough pipeline; please install `polars`. ")

    if cfg is None:
        cfg = HoughConfig()

    geom = E320PrototypeGeometry(use_alignment=True)

    # ── 1. load & filter ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    df = (
        pl.read_parquet(data_path)
        .filter(pl.col("det_type") == "pixel")
        .with_columns(
            pl.col("sensor_id")
            .replace(SENSOR_TO_LAYER)
            .cast(pl.Int8)
            .alias("layer_id"),
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
    print(f"[hough] {n_hits:,} hits, {n_events} events  (load {t1 - t0:.2f}s)")

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
            pool.submit(_process_event_hough, eid, xv, yv, zv, lv, nv, cfg)
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
        f"[hough] {result.height} candidates, {n_kept} kept "
        f"(5-hit: {n5}, 4-hit: {n4})  "
        f"track-find {t3 - t2:.2f}s  total {t4 - t0:.2f}s"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
from src.config import DATA_ROOT, HIT_LEVEL_PARQUET
DATA_PATH = str(HIT_LEVEL_PARQUET)
RESULT_PATH = str(DATA_ROOT / "hough_baseline.parquet")

if __name__ == "__main__":
    result = run_hough(DATA_PATH)
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
