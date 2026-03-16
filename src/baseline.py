"""
Traditional seeding + fitting track-finding baseline for E320 prototype tracker.

Pipeline
--------
pixel hits → TRK coordinates → candidate edges (slope window + KNN)
→ chain seeding (triplets → quadruplets → quintuplets)
→ 3-D line fit → χ²/RMS scoring → greedy shared-hit rejection

Output schema
-------------
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


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BaselineConfig:
    """Tunable knobs for every pipeline stage."""
    # edge building
    slope_x_max: float = 0.2 # sqrt(15^2 + 7^2) / 80 mm / mm = 0.2
    slope_y_max: float = 0.2
    knn_k: int = 10
    # chain seeding
    dslope_x_max: float = 0.001
    dslope_y_max: float = 0.001
    # parallelism
    n_workers: int = 8


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised coordinate transform (no per-row Python)
# ──────────────────────────────────────────────────────────────────────────────
def _vectorized_pixel_to_trk(
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    layer_id: np.ndarray,
    geom: E320PrototypeGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(hit_x)
    x_trk = np.empty(n)
    y_trk = np.empty(n)
    z_trk = np.empty(n)

    cx = (geom.spec.n_cols - 1) / 2.0
    cy = (geom.spec.n_rows - 1) / 2.0
    px = geom.spec.pitch_col_mm
    py = geom.spec.pitch_row_mm

    for lid in range(5):
        mask = layer_id == lid
        if not mask.any():
            continue
        layer = geom.layers[lid]
        c = math.cos(layer.theta_z_rad)
        s = math.sin(layer.theta_z_rad)
        xc = (hit_x[mask] - cx) * px
        yc = (hit_y[mask] - cy) * py
        x_trk[mask] = c * xc - s * yc + layer.dx_mm
        y_trk[mask] = s * xc + c * yc + layer.dy_mm
        z_trk[mask] = layer.z_trk_mm

    return x_trk, y_trk, z_trk


# ──────────────────────────────────────────────────────────────────────────────
# Per-event building blocks
# ──────────────────────────────────────────────────────────────────────────────
def _build_edges(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    cfg: BaselineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slope-window + KNN edge building between adjacent layers."""
    src_a: list[np.ndarray] = []
    dst_a: list[np.ndarray] = []
    sl_a: list[np.ndarray] = []
    dl_a: list[np.ndarray] = []
    sx_a: list[np.ndarray] = []
    sy_a: list[np.ndarray] = []

    for lid in range(4):
        ma = layer == lid
        mb = layer == lid + 1
        if not ma.any() or not mb.any():
            continue

        ax_, ay_, az_, ai = x[ma], y[ma], z[ma], nid[ma]
        bx_, by_, bz_, bi = x[mb], y[mb], z[mb], nid[mb]

        dx = bx_[None, :] - ax_[:, None]
        dy = by_[None, :] - ay_[:, None]
        dz = bz_[None, :] - az_[:, None]
        s_x = dx / dz
        s_y = dy / dz

        ok = (np.abs(s_x) < cfg.slope_x_max) & (np.abs(s_y) < cfg.slope_y_max)
        ia, ib = np.where(ok)
        if len(ia) == 0:
            continue

        e_sx = s_x[ia, ib]
        e_sy = s_y[ia, ib]
        e_src = ai[ia]
        e_dst = bi[ib]

        # KNN: keep at most k nearest per source node
        if cfg.knn_k > 0:
            e_dr = np.sqrt(dx[ia, ib] ** 2 + dy[ia, ib] ** 2)
            keep = np.ones(len(e_src), dtype=bool)
            for s in np.unique(e_src):
                m = e_src == s
                if m.sum() <= cfg.knn_k:
                    continue
                idx = np.where(m)[0]
                order = np.argsort(e_dr[idx])
                keep[idx[order[cfg.knn_k :]]] = False
            e_src, e_dst = e_src[keep], e_dst[keep]
            e_sx, e_sy = e_sx[keep], e_sy[keep]

        src_a.append(e_src)
        dst_a.append(e_dst)
        sl_a.append(np.full(len(e_src), lid, dtype=np.int8))
        dl_a.append(np.full(len(e_src), lid + 1, dtype=np.int8))
        sx_a.append(e_sx)
        sy_a.append(e_sy)

    if not src_a:
        z0 = np.empty(0, dtype=np.int64)
        return z0, z0, z0, z0, z0.astype(float), z0.astype(float)

    return (
        np.concatenate(src_a),
        np.concatenate(dst_a),
        np.concatenate(sl_a),
        np.concatenate(dl_a),
        np.concatenate(sx_a),
        np.concatenate(sy_a),
    )


def _build_chains(
    src: np.ndarray,
    dst: np.ndarray,
    sl: np.ndarray,
    dl: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    cfg: BaselineConfig,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Slope-consistent chain extension: triplets → quads → quints.

    Returns list of ``(node_ids, layer_ids)`` for 4- and 5-hit chains.
    """
    ne = len(src)
    if ne == 0:
        return []

    dsx = cfg.dslope_x_max
    dsy = cfg.dslope_y_max

    # outgoing-edge index: src_node → [edge_idx, …]
    out: dict[int, list[int]] = {}
    for i in range(ne):
        out.setdefault(int(src[i]), []).append(i)

    # ---- triplets (3-hit) ----
    Triplet = tuple[tuple[int, ...], tuple[int, ...], float, float]
    triplets: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = [] # list[Triplet]
    for mid in range(1, 4):
        for i in np.where(dl == mid)[0]:
            mid_node = int(dst[i])
            for j in out.get(mid_node, []):
                if sl[j] != mid:
                    continue
                if abs(sx[j] - sx[i]) < dsx and abs(sy[j] - sy[i]) < dsy:
                    triplets.append((
                        (int(src[i]), mid_node, int(dst[j])),
                        (mid - 1, mid, mid + 1),
                        float(sx[j]),
                        float(sy[j]),
                    ))

    # ---- extend → quadruplets (4-hit) ----
    quads: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = []
    for nodes, layers, lsx, lsy in triplets:
        ll = layers[-1]
        if ll >= 4:
            continue
        for j in out.get(nodes[-1], []):
            if int(dl[j]) != ll + 1:
                continue
            if abs(sx[j] - lsx) < dsx and abs(sy[j] - lsy) < dsy:
                quads.append((
                    nodes + (int(dst[j]),),
                    layers + (ll + 1,),
                    float(sx[j]),
                    float(sy[j]),
                ))

    # ---- extend → quintuplets (5-hit) ----
    quints: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = []
    for nodes, layers, lsx, lsy in quads:
        ll = layers[-1]
        if ll >= 4:
            continue
        for j in out.get(nodes[-1], []):
            if int(dl[j]) != ll + 1:
                continue
            if abs(sx[j] - lsx) < dsx and abs(sy[j] - lsy) < dsy:
                quints.append((
                    nodes + (int(dst[j]),),
                    layers + (ll + 1,),
                    float(sx[j]),
                    float(sy[j]),
                ))

    # quintuplets first (higher priority), then quadruplets
    return [(n, l) for n, l, _, _ in quints] + [(n, l) for n, l, _, _ in quads]


def _fit_and_score(
    chains: list[tuple[tuple[int, ...], tuple[int, ...]]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nid_to_local: dict[int, int],
) -> list[dict]:
    """Linear fit ``x(z), y(z)`` for every candidate chain."""
    results: list[dict] = []
    for nodes, _layers in chains:
        idx = [nid_to_local[n] for n in nodes]
        xs, ys, zs = x[idx], y[idx], z[idx]
        n_pts = len(idx)

        A = np.column_stack([zs, np.ones(n_pts)])
        (a_x, b_x), *_ = np.linalg.lstsq(A, xs, rcond=None)
        (a_y, b_y), *_ = np.linalg.lstsq(A, ys, rcond=None)

        dx = xs - (a_x * zs + b_x)
        dy = ys - (a_y * zs + b_y)
        r2 = dx ** 2 + dy ** 2

        dof = max(2 * n_pts - 4, 1)
        results.append({
            "node_ids": list(nodes),
            "n_layers": n_pts,
            "ax": float(a_x),
            "bx": float(b_x),
            "ay": float(a_y),
            "by": float(b_y),
            "chi2": float(np.sum(r2)) / dof,
            "rms": float(np.sqrt(np.mean(r2))),
        })
    return results


def _shared_hit_rejection(candidates: list[dict]) -> list[dict]:
    """Greedy: iterate by χ² (best first), reject if any hit already used."""
    candidates.sort(key=lambda c: c["chi2"])
    used: set[int] = set()
    for c in candidates:
        nids = set(c["node_ids"])
        c["is_kept"] = not bool(nids & used)
        if c["is_kept"]:
            used |= nids
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Per-event orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def _process_event(
    event_id: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    layer: np.ndarray,
    nid: np.ndarray,
    cfg: BaselineConfig,
) -> list[dict]:
    """Run the full pipeline for a single event."""
    src, dst, sl, dl, sx, sy = _build_edges(x, y, z, layer, nid, cfg)
    if len(src) == 0:
        return []

    chains = _build_chains(src, dst, sl, dl, sx, sy, cfg)
    if not chains:
        return []

    nid_to_local = {int(n): i for i, n in enumerate(nid)}
    candidates = _fit_and_score(chains, x, y, z, nid_to_local)
    candidates = _shared_hit_rejection(candidates)

    for i, c in enumerate(candidates):
        c["event_id"] = event_id
        c["candidate_id"] = i
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def run_baseline(
    data_path: str,
    cfg: BaselineConfig | None = None,
) -> pl.DataFrame:
    """Execute the full baseline and return a polars DataFrame of track candidates.

    Parameters
    ----------
    data_path : str
        Path to the ``hit_level.parquet`` file.
    cfg : BaselineConfig, optional
        Pipeline configuration.  Uses sensible defaults if *None*.

    Returns
    -------
    pl.DataFrame
        Columns: ``event_id, candidate_id, node_ids, n_layers,
        ax, bx, ay, by, chi2, rms, is_kept``
    """
    if cfg is None:
        cfg = BaselineConfig()

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
    print(f"[baseline] {n_hits:,} hits, {n_events} events  (load {t1 - t0:.2f}s)")

    # ── 3. pre-split into numpy views per event (zero-copy) ──────────────
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
            pool.submit(_process_event, eid, xv, yv, zv, lv, nv, cfg)
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
        f"[baseline] {result.height} candidates, {n_kept} kept "
        f"(5-hit: {n5}, 4-hit: {n4})  "
        f"track-find {t3 - t2:.2f}s  total {t4 - t0:.2f}s"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/IvanTang/hep/data_Run502/hit_level.parquet"
RESULT_PATH = '/Users/IvanTang/hep/data_Run502/baseline.parquet'

if __name__ == "__main__":
    result = run_baseline(DATA_PATH)
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
