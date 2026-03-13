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

    records: list[dict] = []

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

        for k in range(len(src)):
            records.append({
                "event_id":       event_id_val,
                "src_node":       int(src[k]),
                "dst_node":       int(dst[k]),
                "src_layer":      int(sl[k]),
                "dst_layer":      int(dl[k]),
                "dx_mm":          float(dx[k]),
                "dy_mm":          float(dy[k]),
                "dz_mm":          float(dz[k]),
                "dr_mm":          float(dr[k]),
                "slope_x":        float(sx[k]),
                "slope_y":        float(sy[k]),
                "x_i":            float(xi[k]),
                "y_i":            float(yi[k]),
                "z_i":            float(zi[k]),
                "size_x_i":       int(size_x_arr[li[k]]),
                "size_y_i":       int(size_y_arr[li[k]]),
                "size_i":         int(size_arr[li[k]]),
                "x_j":            float(xj[k]),
                "y_j":            float(yj[k]),
                "z_j":            float(zj[k]),
                "size_x_j":       int(size_x_arr[lj[k]]),
                "size_y_j":       int(size_y_arr[lj[k]]),
                "size_j":         int(size_arr[lj[k]]),
                "track_id_i":     int(tid_i[k]),
                "track_id_j":     int(tid_j[k]),
                "edge_label":     int(edge_label[k]),
                "is_signal_edge": int(is_signal_edge[k]),
            })

    if not records:
        return pl.DataFrame()

    return pl.from_dicts(records)

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
