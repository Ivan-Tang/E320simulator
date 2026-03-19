"""
Cluster-level fast simulator for E320 prototype tracker.

Generates cluster tables with truth labels for GNN seeding training.
Works entirely in TRK frame (origin at centre of ALPIDE layer 0).

Levels
------
Level 1 – Toy simulator:
    Straight-line truth tracks → layer intersections → Gaussian smearing → clusters.
Level 2 – Toy + real background:
    Same signal generation, but overlays background clusters sampled from Run 502 data.

Output tables
-------------
clusters : polars.DataFrame
    event_id, node_id, layer_id, x_trk_mm, y_trk_mm, z_trk_mm,
    size_x, size_y, size, track_id, is_signal, particle_type
tracks : polars.DataFrame
    event_id, track_id, is_signal, x0_mm, y0_mm, z0_mm, tx, ty, pz_GeV, n_layers_hit
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from src.config import HIT_LEVEL_PROCESSED


# ──────────────────────────────────────────────────────────────────────────────
# Constants (TRK frame, matching geometry.py)
# ──────────────────────────────────────────────────────────────────────────────
Z_LAYERS = np.array([0.0, 20.0, 40.0, 60.0, 80.0])  # mm
N_LAYERS = len(Z_LAYERS)

# ALPIDE active area half-widths
X_HALF = 1024 * 29e-3 / 2  # 14.848 mm
Y_HALF = 512 * 27e-3 / 2   # 6.912 mm

# ALPIDE spatial resolution (σ ≈ 5 µm)
SIGMA_X_MM = 0.005
SIGMA_Y_MM = 0.005

# size distribution
SIZE_NUM = {1: 1024416,
            2: 1975504,
            3: 1032871,
            4: 1327485}

SIZE_X_NUM = {
    1: 2148147,
    2: 3180951,
    3: 30964,
    4: 214,
}

SIZE_Y_NUM = {
    1: 1882453,
    2: 3419185,
    3: 58245,
    4: 393
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """All tuneable knobs for the simulator."""
    # -- signal generation --
    mean_n_signal: float = 0.12          # Poisson mean for signal tracks / BX
    min_layers_hit: int = 4              # discard tracks hitting < this many layers
    # truth track parameter ranges (at z = 0 reference plane)
    x0_range: tuple[float, float] = (-10.0, 10.0)    # mm
    y0_range: tuple[float, float] = (-5.0, 5.0)      # mm
    tx_range: tuple[float, float] = (-0.015, 0.015)   # dx/dz
    ty_range: tuple[float, float] = (-0.015, 0.015)   # dy/dz
    pz_range: tuple[float, float] = (1.5, 4.0)        # GeV, placeholder

    # -- multiple scattering --
    multiple_scattering_mrad: float = 0.2

    # -- measurement smearing --
    sigma_x_mm: float = SIGMA_X_MM
    sigma_y_mm: float = SIGMA_Y_MM

    # -- cluster size --
    cluster_size_mode: Literal["fixed", "empirical"] = "fixed"

    # -- background --
    background_mode: Literal["none", "data", "synthetic"] = "none"
    background_data_path: str | None = None
    # when background_mode == "data", one Run 502 event is sampled
    # per simulated event and overlaid
    # when background_mode == "synthetic", clusters are generated
    # uniformly on each layer's active area
    synthetic_bg_n_per_layer: int = 700  # clusters per layer per event

    # -- event generation --
    n_events: int = 1000
    seed: int = 42

    # -- empirical cluster size table (populated from data if needed) --
    _cluster_size_table: np.ndarray | None = field(default=None, repr=False)

    # -- dataset mode --
    mode: str = 'train'
    train_test_split: float = 0.2


# ──────────────────────────────────────────────────────────────────────────────
# Cluster-size sampler (empirical from Run 502)
# ──────────────────────────────────────────────────────────────────────────────
def _load_cluster_size_table(path: str) -> np.ndarray:
    """Load joint (size_x, size_y, size) tuples from real data for resampling."""
    from src.geometry import SENSOR_TO_LAYER

    df = (
        pl.read_parquet(path)
        .filter(pl.col("det_type") == "pixel")
        .filter(pl.col("size") > 0)
        .select("size_x", "size_y", "size")
    )
    return df.to_numpy().astype(np.int32)


def _sample_cluster_size(
    rng: np.random.Generator,
    n: int,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (size_x, size_y, size) arrays of length *n*."""
    if cfg.cluster_size_mode == "fixed":
        ones = np.ones(n, dtype=np.int32)
        return ones.copy(), ones.copy(), ones.copy()

    # empirical: resample from real distribution
    table = cfg._cluster_size_table
    if table is None:
        raise ValueError(
            "cluster_size_mode='empirical' requires background_data_path to be set."
        )
    idx = rng.integers(0, len(table), size=n)
    return table[idx, 0].copy(), table[idx, 1].copy(), table[idx, 2].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Background loader
# ──────────────────────────────────────────────────────────────────────────────
class SyntheticBackgroundPool:
    """Generates background clusters uniformly on each layer's active area."""

    def __init__(self, n_per_layer: int = 50, use_alignment: bool = True,
                 cluster_size_mode: str = "fixed",
                 background_data_path: str | None = None):
        from src.geometry import E320PrototypeGeometry

        self.n_per_layer = n_per_layer
        self.cluster_size_mode = cluster_size_mode
        self.geom = E320PrototypeGeometry(use_alignment=use_alignment)

        self._cluster_size_table = None
        if cluster_size_mode == "empirical":
            path = background_data_path or str(HIT_LEVEL_PROCESSED)
            self._cluster_size_table = _load_cluster_size_table(path)

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Generate one synthetic background event."""
        n = self.n_per_layer
        total = n * N_LAYERS

        layer_ids = np.empty(total, dtype=np.int8)
        x_trk = np.empty(total)
        y_trk = np.empty(total)
        z_trk = np.empty(total)

        for lid in range(N_LAYERS):
            sl = slice(lid * n, (lid + 1) * n)
            layer = self.geom.layers[lid]

            # uniform in chip-local frame, then apply alignment
            x_local = rng.uniform(-X_HALF, X_HALF, size=n)
            y_local = rng.uniform(-Y_HALF, Y_HALF, size=n)

            c = math.cos(layer.theta_z_rad)
            s = math.sin(layer.theta_z_rad)
            x_trk[sl] = c * x_local - s * y_local + layer.dx_mm
            y_trk[sl] = s * x_local + c * y_local + layer.dy_mm
            z_trk[sl] = layer.z_trk_mm
            layer_ids[sl] = lid

        # build a minimal config proxy for _sample_cluster_size
        _cfg_proxy = SimConfig(cluster_size_mode=self.cluster_size_mode,
                               _cluster_size_table=self._cluster_size_table)
        sx, sy, sz = _sample_cluster_size(rng, total, _cfg_proxy)
        size_x = sx.astype(np.int32)
        size_y = sy.astype(np.int32)
        size = sz.astype(np.int32)

        return {
            "layer_id": layer_ids,
            "x_trk_mm": x_trk,
            "y_trk_mm": y_trk,
            "z_trk_mm": z_trk,
            "size_x": size_x,
            "size_y": size_y,
            "size": size,
        }


class BackgroundPool:
    """Pre-loads all Run 502 events into TRK-frame arrays for fast overlay."""

    def __init__(self, data_path: str):
        from src.geometry import E320PrototypeGeometry, SENSOR_TO_LAYER

        geom = E320PrototypeGeometry(use_alignment=True)
    
        raw = (
            pl.read_parquet(data_path)
            .filter(pl.col("det_type") == "pixel")
            .with_columns(
                pl.col("sensor_id")
                .replace(SENSOR_TO_LAYER)
                .cast(pl.Int8)
                .alias("layer_id"),
            )
        )

        # Vectorised pixel → TRK conversion
        hit_x = raw["hit_x"].to_numpy().astype(np.float64)
        hit_y = raw["hit_y"].to_numpy().astype(np.float64)
        layer_id = raw["layer_id"].to_numpy()

        cx = (geom.spec.n_cols - 1) / 2.0
        cy = (geom.spec.n_rows - 1) / 2.0
        px = geom.spec.pitch_col_mm
        py = geom.spec.pitch_row_mm

        n = len(hit_x)
        x_trk = np.empty(n)
        y_trk = np.empty(n)
        z_trk = np.empty(n)

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

        raw = raw.with_columns(
            pl.Series("x_trk_mm", x_trk),
            pl.Series("y_trk_mm", y_trk),
            pl.Series("z_trk_mm", z_trk),
        )

        # Pre-split by event into list of dicts
        self.events: list[dict[str, np.ndarray]] = []
        eid_arr = raw["event_id"].to_numpy()
        unique_events, starts = np.unique(eid_arr, return_index=True)
        counts = np.diff(np.append(starts, len(eid_arr)))

        size_x = raw["size_x"].to_numpy().astype(np.int32)
        size_y = raw["size_y"].to_numpy().astype(np.int32)
        size = raw["size"].to_numpy().astype(np.int32)
        layer_id_arr = raw["layer_id"].to_numpy().astype(np.int8)

        for i in range(len(unique_events)):
            s, c_ = int(starts[i]), int(counts[i])
            self.events.append({
                "layer_id": layer_id_arr[s : s + c_],
                "x_trk_mm": x_trk[s : s + c_],
                "y_trk_mm": y_trk[s : s + c_],
                "z_trk_mm": z_trk[s : s + c_],
                "size_x": size_x[s : s + c_],
                "size_y": size_y[s : s + c_],
                "size": size[s : s + c_],
            })

        self.n_events = len(self.events)
        total_hits = sum(len(ev["layer_id"]) for ev in self.events)
        print(f"[BackgroundPool] loaded {self.n_events} events ({total_hits:,} total hits)")

    def split(
        self,
        train_test_split: float = 0.2,
        seed: int = 42,
    ) -> tuple[BackgroundPool, BackgroundPool]:
        """Return (train_pool, test_pool) with non-overlapping background events."""
        n_total = len(self.events)
        split_rng = np.random.default_rng(seed)
        perm = split_rng.permutation(n_total)
        n_test = max(1, int(round(n_total * train_test_split)))

        train_pool = BackgroundPool.__new__(BackgroundPool)
        train_pool.events = [self.events[i] for i in sorted(perm[n_test:])]
        train_pool.n_events = len(train_pool.events)

        test_pool = BackgroundPool.__new__(BackgroundPool)
        test_pool.events = [self.events[i] for i in sorted(perm[:n_test])]
        test_pool.n_events = len(test_pool.events)

        print(
            f"[BackgroundPool] split: train={train_pool.n_events}, "
            f"test={test_pool.n_events} (total={n_total})"
        )
        return train_pool, test_pool

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return one random background event."""
        idx = rng.integers(0, self.n_events)
        return self.events[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Signal truth-track generation (Level 1)
# ──────────────────────────────────────────────────────────────────────────────
def _generate_truth_track(rng: np.random.Generator, cfg: SimConfig) -> dict:
    """Sample a single truth track at the z=0 reference plane."""
    return {
        "x0": rng.uniform(*cfg.x0_range),
        "y0": rng.uniform(*cfg.y0_range),
        "tx": rng.uniform(*cfg.tx_range),
        "ty": rng.uniform(*cfg.ty_range),
        "pz": rng.uniform(*cfg.pz_range),
    }


def _intersect_layers(track: dict, rng: np.random.Generator, cfg: SimConfig) -> list[tuple[int, float, float, float]]:
    """Propagate straight-line track to each layer with optional multiple scattering; return hits inside active area."""
    hits: list[tuple[int, float, float, float]] = []
    
    x = track["x0"]
    y = track["y0"]
    tx = track["tx"]
    ty = track["ty"]
    current_z = 0.0

    for lid, next_z in enumerate(Z_LAYERS):
        dz = next_z - current_z
        x += tx * dz
        y += ty * dz
        current_z = next_z
        
        if -X_HALF <= x <= X_HALF and -Y_HALF <= y <= Y_HALF:
            hits.append((lid, x, y, current_z))
            
        # Apply multiple scattering after passing through the layer
        if cfg.multiple_scattering_mrad > 0:
            ms_rad = cfg.multiple_scattering_mrad * 1e-3
            tx += rng.normal(0, ms_rad)
            ty += rng.normal(0, ms_rad)

    return hits


def _smear_hit(
    x: float, y: float, rng: np.random.Generator, cfg: SimConfig,
) -> tuple[float, float]:
    """Apply Gaussian measurement smearing to truth position."""
    return (
        x + rng.normal(0, cfg.sigma_x_mm),
        y + rng.normal(0, cfg.sigma_y_mm),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-event simulation
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_event(
    event_id: int,
    rng: np.random.Generator,
    cfg: SimConfig,
    bg_pool: BackgroundPool | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate one event: signal clusters + optional background overlay.

    Returns (tracks_list, clusters_list) where each element is a dict row.
    """
    tracks: list[dict] = []
    clusters: list[dict] = []
    next_track_id = 0

    # ── A. Signal tracks ────────────────────────────────────────────
    n_sig = rng.poisson(cfg.mean_n_signal)

    for _ in range(n_sig):
        tr = _generate_truth_track(rng, cfg)
        truth_hits = _intersect_layers(tr, rng, cfg)

        if len(truth_hits) < cfg.min_layers_hit:
            continue

        track_id = next_track_id
        next_track_id += 1

        tracks.append({
            "event_id": event_id,
            "track_id": track_id,
            "is_signal": True,
            "x0_mm": tr["x0"],
            "y0_mm": tr["y0"],
            "z0_mm": 0.0,
            "tx": tr["tx"],
            "ty": tr["ty"],
            "pz_GeV": tr["pz"],
            "n_layers_hit": len(truth_hits),
        })

        # cluster sizes for this track's hits
        sx, sy, sz = _sample_cluster_size(rng, len(truth_hits), cfg)

        for i, (layer_id, x, y, z) in enumerate(truth_hits):
            xm, ym = _smear_hit(x, y, rng, cfg)
            clusters.append({
                "event_id": event_id,
                "layer_id": layer_id,
                "x_trk_mm": xm,
                "y_trk_mm": ym,
                "z_trk_mm": z,
                "size_x": int(sx[i]),
                "size_y": int(sy[i]),
                "size": int(sz[i]),
                "track_id": track_id,
                "is_signal": True,
                "particle_type": "signal_pos",
            })

    # ── B. Background overlay ───────────────────────────────────────
    if bg_pool is not None:
        bg = bg_pool.sample(rng)
        n_bg = len(bg["layer_id"])
        for j in range(n_bg):
            clusters.append({
                "event_id": event_id,
                "layer_id": int(bg["layer_id"][j]),
                "x_trk_mm": float(bg["x_trk_mm"][j]),
                "y_trk_mm": float(bg["y_trk_mm"][j]),
                "z_trk_mm": float(bg["z_trk_mm"][j]),
                "size_x": int(bg["size_x"][j]),
                "size_y": int(bg["size_y"][j]),
                "size": int(bg["size"][j]),
                "track_id": -1,
                "is_signal": False,
                "particle_type": "background",
            })

    return tracks, clusters


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
_CLUSTERS_SCHEMA = {
    "event_id": pl.Int64,
    "node_id": pl.UInt32,
    "layer_id": pl.Int8,
    "x_trk_mm": pl.Float64,
    "y_trk_mm": pl.Float64,
    "z_trk_mm": pl.Float64,
    "size_x": pl.Int32,
    "size_y": pl.Int32,
    "size": pl.Int32,
    "track_id": pl.Int64,
    "is_signal": pl.Boolean,
    "particle_type": pl.String,
}

_TRACKS_SCHEMA = {
    "event_id": pl.Int64,
    "track_id": pl.Int64,
    "is_signal": pl.Boolean,
    "x0_mm": pl.Float64,
    "y0_mm": pl.Float64,
    "z0_mm": pl.Float64,
    "tx": pl.Float64,
    "ty": pl.Float64,
    "pz_GeV": pl.Float64,
    "n_layers_hit": pl.Int32,
}


def simulate(
    cfg: SimConfig | None = None,
    _bg_pool: BackgroundPool | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full simulation and return (clusters_df, tracks_df).

    Parameters
    ----------
    cfg : SimConfig, optional
        Simulation configuration.  Uses sensible defaults if *None*.
    _bg_pool : BackgroundPool, optional
        Pre-built background pool (used internally by ``simulate_train_test``).

    Returns
    -------
    clusters_df : pl.DataFrame
        One row per cluster. Contains ``node_id`` (global unique),
        ``track_id`` (≥0 for signal, -1 for background), and truth labels.
    tracks_df : pl.DataFrame
        One row per truth track.
    """
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.default_rng(cfg.seed)

    # Optionally prepare background pool
    bg_pool: BackgroundPool | SyntheticBackgroundPool | None = _bg_pool
    if bg_pool is None and cfg.background_mode == "data":
        if cfg.background_data_path is None:
            cfg.background_data_path = str(HIT_LEVEL_PROCESSED)
        full_pool = BackgroundPool(cfg.background_data_path)
        train_pool, test_pool = full_pool.split(cfg.train_test_split, cfg.seed)
        bg_pool = test_pool if cfg.mode == "test" else train_pool
    elif bg_pool is None and cfg.background_mode == "synthetic":
        bg_pool = SyntheticBackgroundPool(
            n_per_layer=cfg.synthetic_bg_n_per_layer,
            cluster_size_mode=cfg.cluster_size_mode,
            background_data_path=cfg.background_data_path,
        )

    # Optionally load cluster-size table
    if cfg.cluster_size_mode == "empirical":
        if cfg.background_data_path is None:
            cfg.background_data_path = str(HIT_LEVEL_PROCESSED)
        cfg._cluster_size_table = _load_cluster_size_table(cfg.background_data_path)

    # ── Generate events ──────────────────────────────────────────────
    all_tracks: list[dict] = []
    all_clusters: list[dict] = []
    n_sig_total = 0

    for eid in range(cfg.n_events):
        trk, cls = _simulate_event(eid, rng, cfg, bg_pool)
        all_tracks.extend(trk)
        all_clusters.extend(cls)
        n_sig_total += len(trk)

        if (eid + 1) % 500 == 0 or eid == cfg.n_events - 1:
            print(
                f"[simulator] {eid + 1}/{cfg.n_events} events  "
                f"({n_sig_total} signal tracks, {len(all_clusters):,} clusters)"
            )

    # ── Assemble DataFrames ──────────────────────────────────────────
    if not all_clusters:
        clusters_df = pl.DataFrame(schema=_CLUSTERS_SCHEMA)
    else:
        clusters_df = (
            pl.DataFrame(all_clusters)
            .sort("event_id", "layer_id")
            .with_row_index("node_id")
            .cast({"node_id": pl.UInt32})
        )

    if not all_tracks:
        tracks_df = pl.DataFrame(schema=_TRACKS_SCHEMA)
    else:
        tracks_df = pl.DataFrame(all_tracks).sort("event_id", "track_id")

    # ── Summary ──────────────────────────────────────────────────────
    n_signal_clusters = clusters_df.filter(pl.col("is_signal")).height
    n_bg_clusters = clusters_df.filter(~pl.col("is_signal")).height
    n_tracks_5 = tracks_df.filter(pl.col("n_layers_hit") == 5).height
    n_tracks_4 = tracks_df.filter(pl.col("n_layers_hit") == 4).height

    print(f"\n[simulator] Summary")
    print(f"  events:           {cfg.n_events}")
    print(f"  signal tracks:    {tracks_df.height}  (5-hit: {n_tracks_5}, 4-hit: {n_tracks_4})")
    print(f"  signal clusters:  {n_signal_clusters}")
    print(f"  background clust: {n_bg_clusters}")
    print(f"  total clusters:   {clusters_df.height}")
    if clusters_df.height > 0:
        signal_frac = n_signal_clusters / clusters_df.height * 100
        print(f"  signal fraction:  {signal_frac:.2f}%")

    return clusters_df, tracks_df


def simulate_train_test(
    cfg: SimConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run simulation for both train and test splits in one call.

    Loads background data once, deterministically splits events into
    non-overlapping train / test pools, then generates both datasets.

    Returns
    -------
    train_clusters, train_tracks, test_clusters, test_tracks
    """
    if cfg is None:
        cfg = SimConfig()

    # ── Load background once and split ─────────────────────────────
    train_pool: BackgroundPool | SyntheticBackgroundPool | None = None
    test_pool: BackgroundPool | SyntheticBackgroundPool | None = None
    if cfg.background_mode == "data":
        if cfg.background_data_path is None:
            cfg.background_data_path = str(HIT_LEVEL_PROCESSED)
        full_pool = BackgroundPool(cfg.background_data_path)
        train_pool, test_pool = full_pool.split(cfg.train_test_split, cfg.seed)
    elif cfg.background_mode == "synthetic":
        # SyntheticBackgroundPool is stateless; safe to share the same instance
        train_pool = SyntheticBackgroundPool(
            n_per_layer=cfg.synthetic_bg_n_per_layer,
            cluster_size_mode=cfg.cluster_size_mode,
            background_data_path=cfg.background_data_path,
        )
        test_pool = SyntheticBackgroundPool(
            n_per_layer=cfg.synthetic_bg_n_per_layer,
            cluster_size_mode=cfg.cluster_size_mode,
            background_data_path=cfg.background_data_path,
        )

    # ── Load cluster-size table once ────────────────────────────────
    if cfg.cluster_size_mode == "empirical":
        if cfg.background_data_path is None:
            cfg.background_data_path = str(HIT_LEVEL_PROCESSED)
        cfg._cluster_size_table = _load_cluster_size_table(cfg.background_data_path)

    # ── Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating TRAIN dataset")
    print("=" * 60)
    train_clusters, train_tracks = simulate(cfg, _bg_pool=train_pool)

    # ── Test (use a different seed so signal tracks differ) ────────
    from dataclasses import replace as _dc_replace

    cfg_test = _dc_replace(cfg, mode="test", seed=cfg.seed + 1)
    print("\n" + "=" * 60)
    print("  Generating TEST dataset")
    print("=" * 60)
    test_clusters, test_tracks = simulate(cfg_test, _bg_pool=test_pool)

    return train_clusters, train_tracks, test_clusters, test_tracks


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.config import HIT_LEVEL_PROCESSED, SIM_DIR

    parser = argparse.ArgumentParser(description="E320 cluster-level fast simulator")
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--mean-n-signal", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ms-mrad", type=float, default=0.2, help="Multiple scattering standard deviation in mrad")
    parser.add_argument("--background", choices=["none", "data", "synthetic"], default="synthetic")
    parser.add_argument(
        "--bg-data-path",
        type=str,
        default=str(HIT_LEVEL_PROCESSED),
    )
    parser.add_argument("--cluster-size", choices=["fixed", "empirical"], default="fixed")
    parser.add_argument(
        "--synthetic-bg-n-per-layer", type=int, default=700,
        help="Number of background clusters per layer per event (synthetic mode)",
    )
    parser.add_argument("--output-dir", type=str, default=str(SIM_DIR))
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    cfg = SimConfig(
        n_events=args.n_events,
        mean_n_signal=args.mean_n_signal,
        seed=args.seed,
        background_mode=args.background,
        background_data_path=args.bg_data_path if args.background == "data" else None,
        cluster_size_mode=args.cluster_size,
        synthetic_bg_n_per_layer=args.synthetic_bg_n_per_layer,
        mode=args.mode,
        multiple_scattering_mrad=args.ms_mrad,
    )
    if cfg.cluster_size_mode == "empirical":
        cfg.background_data_path = args.bg_data_path

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "both":
        train_cl, train_tr, test_cl, test_tr = simulate_train_test(cfg)
        for tag, cl, tr in [("train", train_cl, train_tr), ("test", test_cl, test_tr)]:
            cp = os.path.join(args.output_dir, f"sim_clusters_{tag}.parquet")
            tp = os.path.join(args.output_dir, f"sim_tracks_{tag}.parquet")
            cl.write_parquet(cp)
            tr.write_parquet(tp)
            print(f"[simulator] Saved {tag}: {cp}")
            print(f"[simulator] Saved {tag}: {tp}")
    else:
        clusters_df, tracks_df = simulate(cfg)
        cp = os.path.join(args.output_dir, f"sim_clusters_{args.mode}.parquet")
        tp = os.path.join(args.output_dir, f"sim_tracks_{args.mode}.parquet")
        clusters_df.write_parquet(cp)
        tracks_df.write_parquet(tp)
        print(f"\n[simulator] Saved to {cp}")
        print(f"[simulator] Saved to {tp}")