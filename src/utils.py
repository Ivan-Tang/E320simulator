"""
Utility functions for GNN training data generation.

Key entry point
---------------
build_labeled_edges_from_sim(clusters_df, cfg)
    → polars.DataFrame of truth-labeled candidate edges,
      ready to feed into an edge-classification GNN.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
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

    # Accumulate per-event DataFrames; batch-concat every _BATCH events to
    # avoid holding millions of Python dicts in memory (the original approach
    # caused OOM / segfault on large datasets).
    _BATCH = 500
    frames: list[pl.DataFrame] = []
    batched: list[pl.DataFrame] = []

    for eid, ev in clusters_df.group_by("event_id"):
        ev = ev.sort("node_id")

        x   = ev["x_trk_mm"].to_numpy()
        y   = ev["y_trk_mm"].to_numpy()
        z   = ev["z_trk_mm"].to_numpy()
        lid = ev["layer_id"].to_numpy()
        nid = ev["node_id"].to_numpy()

        src, dst, sl, dl, sx, sy = _build_edges(x, y, z, lid, nid, cfg)
        if len(src) == 0:
            continue

        # node_id → local-index lookup for this event
        nid_to_local: dict[int, int] = {int(n): i for i, n in enumerate(nid)}

        track_id  = ev["track_id"].to_numpy()
        is_signal = ev["is_signal"].to_numpy().astype(bool)
        size_x_arr = ev["size_x"].to_numpy()
        size_y_arr = ev["size_y"].to_numpy()
        size_arr   = ev["size"].to_numpy()

        event_id_val = int(eid[0]) if isinstance(eid, (list, tuple)) else int(eid)

        li = np.array([nid_to_local[int(s)] for s in src], dtype=np.int64)
        lj = np.array([nid_to_local[int(d)] for d in dst], dtype=np.int64)

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

        n = len(src)
        frames.append(pl.DataFrame({
            "event_id":       np.full(n, event_id_val, dtype=np.int64),
            "src_node":       src.astype(np.int64),
            "dst_node":       dst.astype(np.int64),
            "src_layer":      sl.astype(np.int64),
            "dst_layer":      dl.astype(np.int64),
            "dx_mm":          dx.astype(np.float64),
            "dy_mm":          dy.astype(np.float64),
            "dz_mm":          dz.astype(np.float64),
            "dr_mm":          dr.astype(np.float64),
            "slope_x":        sx.astype(np.float64),
            "slope_y":        sy.astype(np.float64),
            "x_i":            xi.astype(np.float64),
            "y_i":            yi.astype(np.float64),
            "z_i":            zi.astype(np.float64),
            "size_x_i":       size_x_arr[li].astype(np.int64),
            "size_y_i":       size_y_arr[li].astype(np.int64),
            "size_i":         size_arr[li].astype(np.int64),
            "x_j":            xj.astype(np.float64),
            "y_j":            yj.astype(np.float64),
            "z_j":            zj.astype(np.float64),
            "size_x_j":       size_x_arr[lj].astype(np.int64),
            "size_y_j":       size_y_arr[lj].astype(np.int64),
            "size_j":         size_arr[lj].astype(np.int64),
            "track_id_i":     tid_i.astype(np.int64),
            "track_id_j":     tid_j.astype(np.int64),
            "edge_label":     edge_label.astype(np.int64),
            "is_signal_edge": is_signal_edge.astype(np.int64),
        }))

        if len(frames) >= _BATCH:
            batched.append(pl.concat(frames))
            frames.clear()

    if frames:
        batched.append(pl.concat(frames))
        frames.clear()

    if not batched:
        return pl.DataFrame()

    return pl.concat(batched)

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
