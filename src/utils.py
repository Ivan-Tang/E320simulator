"""
Utility functions for GNN training data generation.

Key entry points
----------------
build_labeled_edges_from_sim(clusters_df, cfg)
    → polars.DataFrame of truth-labeled candidate edges,
      ready to feed into an edge-classification GNN.

build_edges_to_parquet(clusters_df, output_dir, cfg, chunk_size)
    → builds edges in chunks and writes each chunk to a parquet file,
      avoiding OOM for large datasets (>2k events).

ParquetEdgeSource(edges_dir)
    → lazy, event-at-a-time iterator over a directory of parquet chunks.
"""
from __future__ import annotations

import math
import numpy as np
import polars as pl
import torch
from functools import cached_property
from pathlib import Path
from typing import Iterator
from torch import Tensor
from src.baseline import BaselineConfig, _build_edges


# Default layer adjacency table for E320-style data.
# Row format: [volume_id, layer_id].
# For E320 simulator hits, volume_id is treated as 0 and layer_id in [0..4].
ALL_LAYERS = np.array([
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
], dtype=np.int32)


def is_match(
    hit_id_a: int,
    hit_id_b: int,
    vols: np.ndarray,
    layers: np.ndarray,
    all_layers: np.ndarray = ALL_LAYERS,
) -> bool:
    """Check whether two hits lie on adjacent detector layers.

    Adjacency is defined by consecutive entries in ``all_layers``.
    """
    va, la = int(vols[hit_id_a]), int(layers[hit_id_a])
    vb, lb = int(vols[hit_id_b]), int(layers[hit_id_b])

    matches = np.where((all_layers[:, 0] == va) & (all_layers[:, 1] == la))[0]
    if len(matches) == 0:
        return False

    i = int(matches[0])
    lower_ok = i > 0 and np.array_equal(np.array([vb, lb]), all_layers[i - 1])
    upper_ok = (i + 1) < len(all_layers) and np.array_equal(np.array([vb, lb]), all_layers[i + 1])
    return bool(lower_ok or upper_ok)


def get_true_pairs_layerwise(
    hits: list | np.ndarray,
    where_track: list[int],
    vols: np.ndarray,
    layers: np.ndarray,
    all_layers: np.ndarray = ALL_LAYERS,
) -> tuple[list, list]:
    """Build positive pairs (target=1) from adjacent layers of one particle."""
    hits_a: list = []
    hits_b: list = []
    n = len(where_track)

    for i in range(n):
        for j in range(i + 1, n):
            ha = where_track[i]
            hb = where_track[j]
            if is_match(ha, hb, vols, layers, all_layers=all_layers):
                # keep both directions
                hits_a.append(hits[ha])
                hits_b.append(hits[hb])
                hits_a.append(hits[hb])
                hits_b.append(hits[ha])
    return hits_a, hits_b


def get_false_pairs(
    hits: list | np.ndarray,
    where_track: list[int],
    particle_ids: np.ndarray,
    pid: int,
    nb_false_pairs: int,
    rng: np.random.Generator,
) -> tuple[list, list]:
    """Build negative pairs (target=0) by pairing current-track hits with other particles."""
    if nb_false_pairs <= 0:
        return [], []

    where_not_track = np.where(particle_ids != pid)[0]
    if len(where_not_track) == 0 or len(where_track) == 0:
        return [], []

    neg_idx = rng.choice(where_not_track, size=nb_false_pairs, replace=(len(where_not_track) < nb_false_pairs))
    pos_idx = rng.choice(np.array(where_track), size=nb_false_pairs, replace=True)

    h_a = [hits[int(i)] for i in pos_idx]
    h_b = [hits[int(j)] for j in neg_idx]
    return h_a, h_b


def get_pairs_one_pid(
    hits: list | np.ndarray,
    particle_ids: np.ndarray,
    pid: int,
    z: np.ndarray,
    vols: np.ndarray,
    layers: np.ndarray,
    rng: np.random.Generator,
    all_layers: np.ndarray = ALL_LAYERS,
) -> tuple[list, list, list[int]]:
    """Build balanced (positive + negative) pairs for one particle id."""
    del z  # kept for compatibility with prior API shape

    where_track = list(np.where(particle_ids == pid)[0])
    if len(where_track) < 2:
        return [], [], []

    h_true_a, h_true_b = get_true_pairs_layerwise(
        hits, where_track, vols, layers, all_layers=all_layers
    )
    target_true = [1] * len(h_true_a)

    if len(h_true_a) == 0:
        return [], [], []

    h_false_a, h_false_b = get_false_pairs(
        hits, where_track, particle_ids, pid, len(h_true_a), rng
    )
    target_false = [0] * len(h_false_a)

    return h_true_a + h_false_a, h_true_b + h_false_b, target_true + target_false


def build_pairs(
    hits: list | np.ndarray,
    particle_ids: list | np.ndarray,
    vols: list | np.ndarray,
    layers: list | np.ndarray,
    nb_particles_per_sample: int = 2000,
    *,
    rng: np.random.Generator | None = None,
    all_layers: np.ndarray = ALL_LAYERS,
) -> tuple[list, list, list[int]]:
    """Construct hit-pair dataset for embedding training.

    Strategy
    --------
    - Positive pairs: same particle, adjacent detector layers (via ``all_layers``)
    - Negative pairs: current particle hit paired with another particle hit
      while keeping class counts balanced per particle.
    """
    if rng is None:
        rng = np.random.default_rng()

    hits_arr = list(hits)
    pids = np.asarray(particle_ids)
    vols_arr = np.asarray(vols)
    layers_arr = np.asarray(layers)

    if len(hits_arr) != len(pids) or len(hits_arr) != len(vols_arr) or len(hits_arr) != len(layers_arr):
        raise ValueError("hits, particle_ids, vols, layers must have the same length")

    # In this repository signal track ids are >= 0 and background is -1.
    unique_pids = [int(p) for p in np.unique(pids) if int(p) >= 0]
    if not unique_pids:
        return [], [], []

    unique_pids = list(rng.permutation(unique_pids))
    n_to_sample = min(int(nb_particles_per_sample), len(unique_pids))

    hits_a: list = []
    hits_b: list = []
    target: list[int] = []

    hits_np = np.asarray(hits)
    z = hits_np[:, 2] if hits_np.ndim == 2 and hits_np.shape[1] > 2 else np.zeros(len(hits_arr), dtype=np.float32)

    for i in range(n_to_sample):
        pid = unique_pids[i]
        h_a, h_b, t = get_pairs_one_pid(
            hits_arr,
            pids,
            pid,
            z,
            vols_arr,
            layers_arr,
            rng,
            all_layers=all_layers,
        )
        hits_a.extend(h_a)
        hits_b.extend(h_b)
        target.extend(t)

    return hits_a, hits_b, target


def build_paris(*args, **kwargs):
    """Backward-compatible alias (typo kept for compatibility)."""
    return build_pairs(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: build edges for one event (vectorized, no Python dict loop)
# ──────────────────────────────────────────────────────────────────────────────

def _build_event_edges_df(
    ev: pl.DataFrame,
    event_id_val: int,
    cfg: BaselineConfig | None,
) -> pl.DataFrame | None:
    """Build a truth-labeled edge DataFrame for a single event.

    Returns ``None`` if the event has no candidate edges.
    All array operations are vectorized (no Python-level per-edge loop).
    """
    if cfg is None:
        cfg = BaselineConfig()

    ev = ev.sort("node_id")

    x   = ev["x_trk_mm"].to_numpy()
    y   = ev["y_trk_mm"].to_numpy()
    z   = ev["z_trk_mm"].to_numpy()
    lid = ev["layer_id"].to_numpy()
    nid = ev["node_id"].to_numpy()

    src, dst, sl, dl, sx, sy = _build_edges(x, y, z, lid, nid, cfg)
    if len(src) == 0:
        return None

    # node_id → local-index lookup for this event
    nid_to_local: dict[int, int] = {int(n): i for i, n in enumerate(nid)}
    li = np.array([nid_to_local[int(s)] for s in src], dtype=np.int64)
    lj = np.array([nid_to_local[int(d)] for d in dst], dtype=np.int64)

    track_id   = ev["track_id"].to_numpy()
    is_signal  = ev["is_signal"].to_numpy().astype(bool)
    size_x_arr = ev["size_x"].to_numpy()
    size_y_arr = ev["size_y"].to_numpy()
    size_arr   = ev["size"].to_numpy()

    xi, yi, zi = x[li], y[li], z[li]
    xj, yj, zj = x[lj], y[lj], z[lj]
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    dr = np.sqrt(dx**2 + dy**2)

    tid_i = track_id[li]
    tid_j = track_id[lj]
    sig_i = is_signal[li]
    sig_j = is_signal[lj]

    edge_label     = ((tid_i == tid_j) & (tid_i >= 0)).astype(np.int8)
    is_signal_edge = ((edge_label == 1) & sig_i & sig_j).astype(np.int8)

    return pl.DataFrame({
        "event_id":       np.full(len(src), event_id_val, dtype=np.int32),
        "src_node":       src.astype(np.int64),
        "dst_node":       dst.astype(np.int64),
        "src_layer":      sl.astype(np.int32),
        "dst_layer":      dl.astype(np.int32),
        "dx_mm":          dx.astype(np.float32),
        "dy_mm":          dy.astype(np.float32),
        "dz_mm":          dz.astype(np.float32),
        "dr_mm":          dr.astype(np.float32),
        "slope_x":        sx.astype(np.float32),
        "slope_y":        sy.astype(np.float32),
        "x_i":            xi.astype(np.float32),
        "y_i":            yi.astype(np.float32),
        "z_i":            zi.astype(np.float32),
        "size_x_i":       size_x_arr[li].astype(np.int32),
        "size_y_i":       size_y_arr[li].astype(np.int32),
        "size_i":         size_arr[li].astype(np.int32),
        "x_j":            xj.astype(np.float32),
        "y_j":            yj.astype(np.float32),
        "z_j":            zj.astype(np.float32),
        "size_x_j":       size_x_arr[lj].astype(np.int32),
        "size_y_j":       size_y_arr[lj].astype(np.int32),
        "size_j":         size_arr[lj].astype(np.int32),
        "track_id_i":     tid_i.astype(np.int32),
        "track_id_j":     tid_j.astype(np.int32),
        "edge_label":     edge_label,
        "is_signal_edge": is_signal_edge,
    })


# ──────────────────────────────────────────────────────────────────────────────
# dataset builder
def build_labeled_edges_from_sim(
    clusters_df: pl.DataFrame,
    cfg: BaselineConfig | None = None,
) -> pl.DataFrame:
    """Build a truth-labeled edge table from simulated clusters.

    Parameters
    ----------
    clusters_df:
        Simulator output with columns:
        ``event_id, node_id, layer_id, x_trk_mm, y_trk_mm, z_trk_mm,
          size_x, size_y, size, track_id, is_signal, particle_type``
    cfg:
        Baseline configuration (slope window + KNN).  Defaults to
        ``BaselineConfig()`` if not given.

    Returns
    -------
    polars.DataFrame with one row per candidate edge and columns::

        event_id, src_node, dst_node, src_layer, dst_layer,
        dx_mm, dy_mm, dz_mm, dr_mm, slope_x, slope_y,
        x_i, y_i, z_i, x_j, y_j, z_j,
        size_i, size_j, size_x_i, size_y_i, size_x_j, size_y_j,
        track_id_i, track_id_j,
        edge_label,       # 1 if same truth track (track_id >= 0), else 0
        is_signal_edge    # 1 if edge_label==1 AND both endpoints are signal
    """
    if cfg is None:
        cfg = BaselineConfig()

    needed = {
        "event_id", "node_id", "layer_id",
        "x_trk_mm", "y_trk_mm", "z_trk_mm",
        "size_x", "size_y", "size",
        "track_id", "is_signal",
    }
    missing = needed - set(clusters_df.columns)
    if missing:
        raise ValueError(f"clusters_df is missing columns: {missing}")

    frames: list[pl.DataFrame] = []

    for eid, ev in clusters_df.group_by("event_id"):
        event_id_val = int(eid[0]) if isinstance(eid, (list, tuple)) else int(eid)
        ev_df = _build_event_edges_df(ev, event_id_val, cfg)
        if ev_df is not None:
            frames.append(ev_df)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames)

def edge_label_stats(edges_df: pl.DataFrame) -> dict:
    """Return class-balance statistics for an edge table.

    Example
    -------
    >>> stats = edge_label_stats(edges_df)
    >>> print(stats)
    {'n_total': 45230, 'n_positive': 312, 'positive_fraction': 0.0069, ...}
    """
    n_total = len(edges_df)
    n_pos   = int(edges_df["edge_label"].sum())
    n_neg   = n_total - n_pos

    stats: dict = {
        "n_total":           n_total,
        "n_positive":        n_pos,
        "n_negative":        n_neg,
        "positive_fraction": n_pos / n_total if n_total else float("nan"),
    }
    if "is_signal_edge" in edges_df.columns:
        n_sig = int(edges_df["is_signal_edge"].sum())
        stats["n_signal_edges"]          = n_sig
        stats["signal_fraction_of_pos"]  = n_sig / n_pos if n_pos else float("nan")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Chunked edge building with parquet persistence (avoids OOM for large datasets)
# ──────────────────────────────────────────────────────────────────────────────

def build_edges_to_parquet(
    clusters_df: pl.DataFrame,
    output_dir: str | Path,
    cfg: BaselineConfig | None = None,
    chunk_size: int = 200,
) -> Path:
    """Build truth-labeled edges in chunks and persist each chunk to parquet.

    This avoids the OOM that occurs when building all edges for 10k events into
    a single in-memory DataFrame.  Each chunk of ``chunk_size`` events is built
    and immediately written to disk, then freed.

    Parameters
    ----------
    clusters_df:
        Full simulator cluster table (all events).
    output_dir:
        Directory where chunk parquet files are written.
        Created if it does not exist.
    cfg:
        Baseline configuration (slope window + KNN).
    chunk_size:
        Number of events per chunk file.  200 events ≈ 1 GB peak RAM per chunk.

    Returns
    -------
    Path to ``output_dir``.
    """
    if cfg is None:
        cfg = BaselineConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_eids = clusters_df["event_id"].unique().sort().to_numpy()
    n_chunks  = math.ceil(len(all_eids) / chunk_size)
    print(f"[build_edges_to_parquet] {len(all_eids)} events → {n_chunks} chunks "
          f"(chunk_size={chunk_size})  output={output_dir}")

    for chunk_idx in range(n_chunks):
        chunk_eids = all_eids[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        chunk_clusters = clusters_df.filter(pl.col("event_id").is_in(chunk_eids.tolist()))

        frames: list[pl.DataFrame] = []
        for eid, ev in chunk_clusters.group_by("event_id"):
            event_id_val = int(eid[0]) if isinstance(eid, (list, tuple)) else int(eid)
            ev_df = _build_event_edges_df(ev, event_id_val, cfg)
            if ev_df is not None:
                frames.append(ev_df)

        if not frames:
            continue

        chunk_df = pl.concat(frames)
        out_path = output_dir / f"chunk_{chunk_idx:04d}.parquet"
        chunk_df.write_parquet(out_path)
        n_edges = len(chunk_df)
        del chunk_df, frames
        print(f"  chunk {chunk_idx:04d}/{n_chunks-1:04d}  events={len(chunk_eids)}  "
              f"edges={n_edges:,}  → {out_path.name}")

    print(f"[build_edges_to_parquet] done. {n_chunks} parquet files in {output_dir}")
    return output_dir


# ──────────────────────────────────────────────────────────────────────────────
# Lazy event-at-a-time data source backed by parquet chunk files
# ──────────────────────────────────────────────────────────────────────────────

class ParquetEdgeSource:
    """Lazy, event-at-a-time data source over a directory of parquet edge chunks.

    Designed as a drop-in complement to a materialised ``edges_df`` DataFrame.
    Only one chunk file is loaded into RAM at a time, keeping peak memory at
    roughly ``chunk_size * edges_per_event * bytes_per_edge``.

    Parameters
    ----------
    edges_dir:
        Directory produced by :func:`build_edges_to_parquet`.
    """

    def __init__(self, edges_dir: str | Path) -> None:
        self.edges_dir = Path(edges_dir)
        self._files = sorted(self.edges_dir.glob("chunk_*.parquet"))
        if not self._files:
            raise FileNotFoundError(
                f"No chunk_*.parquet files found in {self.edges_dir}"
            )
        # Build {event_id: file_index} index lazily once requested
        self._eid_to_file: dict[int, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @cached_property
    def event_ids(self) -> np.ndarray:
        """Sorted array of all unique event IDs across all chunks."""
        lf = pl.scan_parquet(self.edges_dir / "chunk_*.parquet")
        return (
            lf.select("event_id")
            .unique()
            .sort("event_id")
            .collect()["event_id"]
            .to_numpy()
        )

    def iter_events(
        self,
        event_ids: list[int] | None = None,
    ) -> Iterator[tuple[int, pl.DataFrame]]:
        """Yield ``(event_id, ev_df)`` tuples, one event at a time.

        Parameters
        ----------
        event_ids:
            Ordered list of event IDs to iterate.  Pass ``None`` to iterate
            all events in file order.
        """
        self._build_index()
        assert self._eid_to_file is not None

        if event_ids is None:
            # iterate all events in natural file order
            for f in self._files:
                chunk = pl.read_parquet(f)
                for eid, ev_df in chunk.group_by("event_id"):
                    eid_val = int(eid[0]) if isinstance(eid, (list, tuple)) else int(eid)
                    yield eid_val, ev_df
            return

        # Group requested event IDs by their chunk file to minimise re-reads
        file_to_eids: dict[int, list[int]] = {}
        for eid in event_ids:
            fidx = self._eid_to_file.get(eid)
            if fidx is None:
                continue
            file_to_eids.setdefault(fidx, []).append(eid)

        # Yield in the order requested, loading each file at most once
        loaded_chunk: dict[int, pl.DataFrame] = {}  # file_idx → chunk df
        for eid in event_ids:
            fidx = self._eid_to_file.get(eid)
            if fidx is None:
                continue
            if fidx not in loaded_chunk:
                # Evict previously loaded chunk to free memory
                loaded_chunk.clear()
                loaded_chunk[fidx] = pl.read_parquet(self._files[fidx])
            chunk = loaded_chunk[fidx]
            ev_df = chunk.filter(pl.col("event_id") == eid)
            if ev_df.is_empty():
                continue
            yield eid, ev_df

    def compute_edge_label_stats(self, event_ids: set[int] | None = None) -> dict:
        """Streaming computation of edge label statistics over selected events."""
        lf = pl.scan_parquet(self.edges_dir / "chunk_*.parquet")
        if event_ids is not None:
            lf = lf.filter(pl.col("event_id").is_in(list(event_ids)))
        result = lf.select(
            pl.len().alias("n_total"),
            pl.col("edge_label").sum().alias("n_positive"),
        ).collect()
        n_total = int(result["n_total"][0])
        n_pos   = int(result["n_positive"][0])
        n_neg   = n_total - n_pos
        return {
            "n_total":           n_total,
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "positive_fraction": n_pos / n_total if n_total else float("nan"),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build the {event_id → file_index} mapping (once)."""
        if self._eid_to_file is not None:
            return
        self._eid_to_file = {}
        for fidx, f in enumerate(self._files):
            eids = pl.read_parquet(f, columns=["event_id"])["event_id"].unique().to_list()
            for eid in eids:
                self._eid_to_file[int(eid)] = fidx


# ──────────────────────────────────────────────────────────────────────────────
# dataset converter

NODE_FEAT_COLS_SRC = ["src_layer", "x_i", "y_i", "z_i", "size_x_i", "size_y_i", "size_i"]
NODE_FEAT_COLS_DST = ["dst_layer", "x_j", "y_j", "z_j", "size_x_j", "size_y_j", "size_j"]
EDGE_FEAT_COLS     = ["dx_mm", "dy_mm", "dz_mm", "dr_mm", "slope_x", "slope_y"]

NODE_DIM = len(NODE_FEAT_COLS_SRC)   # 7
EDGE_DIM = len(EDGE_FEAT_COLS)       # 6

def event_to_tensors(
    ev: pl.DataFrame,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convert one event's edge rows to PyTorch tensors.

    Parameters
    ----------
    ev:
        Rows from ``edges_df`` for a single ``event_id``.

    Returns
    -------
    node_feat  : (N, 7)  float32   — deduplicated node features
    edge_index : (2, E)  int64     — local node indices [src; dst]
    edge_feat  : (E, 6)  float32   — edge geometry features
    edge_label : (E,)    float32   — binary truth label
    node_ids   : (N,)    int64     — global node_id (for bookkeeping)
    """
    # --- Build deduplicated node table ---
    src_nodes = ev.select([
        pl.col("src_node").alias("node_id"),
        *[pl.col(c) for c in NODE_FEAT_COLS_SRC],
    ]).rename(dict(zip(NODE_FEAT_COLS_SRC, ["layer", "x", "y", "z", "sx", "sy", "s"])))

    dst_nodes = ev.select([
        pl.col("dst_node").alias("node_id"),
        *[pl.col(c) for c in NODE_FEAT_COLS_DST],
    ]).rename(dict(zip(NODE_FEAT_COLS_DST, ["layer", "x", "y", "z", "sx", "sy", "s"])))

    nodes = (
        pl.concat([src_nodes, dst_nodes])
        .unique(subset=["node_id"])
        .sort("node_id")
    )

    nid_arr = nodes["node_id"].to_numpy()
    nid_to_local = {int(n): i for i, n in enumerate(nid_arr)}

    node_feat_np = nodes.select(["layer", "x", "y", "z", "sx", "sy", "s"]).to_numpy(allow_copy=True).astype(np.float32)

    # --- Edge index (local) ---
    src_local = np.array([nid_to_local[n] for n in ev["src_node"].to_numpy()], dtype=np.int64)
    dst_local = np.array([nid_to_local[n] for n in ev["dst_node"].to_numpy()], dtype=np.int64)

    # --- Edge features & labels ---
    edge_feat_np  = ev.select(EDGE_FEAT_COLS).to_numpy(allow_copy=True).astype(np.float32)
    edge_label_np = ev["edge_label"].to_numpy().astype(np.float32)

    return (
        torch.from_numpy(node_feat_np.copy()),                         # (N, 7)
        torch.from_numpy(np.stack([src_local, dst_local]).copy()),     # (2, E)
        torch.from_numpy(edge_feat_np.copy()),                         # (E, 6)
        torch.from_numpy(edge_label_np.copy()),                        # (E,)
        torch.from_numpy(nid_arr.copy()),                              # (N,)
    )
