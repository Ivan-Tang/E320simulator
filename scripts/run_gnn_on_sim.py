"""
GNN-seeded track reconstruction on simulated data.

Loads a trained edge-classification checkpoint, scores every candidate edge
per event, keeps edges above a score threshold, then feeds the surviving
edges into the *same* chain building / line-fit / shared-hit-rejection
pipeline used by the baseline.  Output is therefore directly comparable to
``run_baseline_on_sim.py``.

Usage
-----
    python -m scripts.run_gnn_on_sim \\
        --clusters  /path/to/sim_clusters.parquet \\
        --tracks    /path/to/sim_tracks.parquet   \\
        --checkpoint runs/exp_gnn_v1/best_model.pt \\
        --threshold 0.5 \\
        --output    /path/to/sim_gnn_reco.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

# ── path setup (works both as a script and as a module) ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import (
    BaselineConfig,
    _build_chains,
    _build_edges,
    _fit_and_score,
    _shared_hit_rejection,
)
from src.train import TrainConfig, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Per-event tensor builder (no polars intermediary, no truth required)
# ──────────────────────────────────────────────────────────────────────────────

def _event_to_tensors(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lid: np.ndarray,
    nid: np.ndarray,
    size_x: np.ndarray,
    size_y: np.ndarray,
    size: np.ndarray,
    e_src: np.ndarray,   # global node IDs
    e_dst: np.ndarray,
    e_sl: np.ndarray,
    e_sx: np.ndarray,
    e_sy: np.ndarray,
    nid_to_local: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (node_feat, edge_index, edge_feat) tensors for one event.

    Mirrors the logic of ``src.utils.event_to_tensors`` but takes raw
    numpy arrays so it works without truth labels and avoids the polars
    round-trip.

    Node features (per unique node)  : layer, x, y, z, size_x, size_y, size  (7)
    Edge features (per candidate edge): dx, dy, dz, dr, slope_x, slope_y     (6)
    """
    ne = len(e_src)

    # ── deduplicated node table ───────────────────────────────────────────────
    all_nids = np.unique(np.concatenate([e_src, e_dst]))
    n_nodes  = len(all_nids)
    nid_to_row = {int(n): i for i, n in enumerate(all_nids)}

    node_feat = np.empty((n_nodes, 7), dtype=np.float32)
    for row_i, gid in enumerate(all_nids):
        li = nid_to_local[int(gid)]
        node_feat[row_i] = [lid[li], x[li], y[li], z[li],
                            size_x[li], size_y[li], size[li]]

    # ── edge index (local to this deduplicated node table) ────────────────────
    src_local = np.array([nid_to_row[int(s)] for s in e_src], dtype=np.int64)
    dst_local = np.array([nid_to_row[int(d)] for d in e_dst], dtype=np.int64)

    # ── edge features ─────────────────────────────────────────────────────────
    li = np.array([nid_to_local[int(s)] for s in e_src], dtype=np.int64)
    lj = np.array([nid_to_local[int(d)] for d in e_dst], dtype=np.int64)
    dx = x[lj] - x[li]
    dy = y[lj] - y[li]
    dz = z[lj] - z[li]
    dr = np.sqrt(dx ** 2 + dy ** 2)

    edge_feat = np.stack([dx, dy, dz, dr, e_sx, e_sy], axis=1).astype(np.float32)

    return (
        torch.from_numpy(node_feat),
        torch.from_numpy(np.stack([src_local, dst_local])),
        torch.from_numpy(edge_feat),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main inference + reconstruction function
# ──────────────────────────────────────────────────────────────────────────────

def run_gnn_reco(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame | None,
    checkpoint_path: str,
    threshold: float = 0.5,
    baseline_cfg: BaselineConfig | None = None,
    device: str = "cpu",
) -> pl.DataFrame:
    """GNN-seeded reconstruction.  Returns DataFrame matching the schema of
    ``evaluate_baseline_on_sim``.

    Parameters
    ----------
    clusters_df:
        Simulator clusters (must have node_id, layer_id, x/y/z_trk_mm,
        size_x, size_y, size).  ``track_id`` / ``is_signal`` are optional
        and used only for matched-truth evaluation.
    tracks_df:
        Truth track table for efficiency evaluation.  Pass ``None`` to skip.
    checkpoint_path:
        Path to ``best_model.pt`` from ``src.train``.
    threshold:
        GNN score cut.  Edges with ``score >= threshold`` are kept.
    baseline_cfg:
        Edge-building and chain-building knobs.  Defaults to
        ``BaselineConfig()``.
    """
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = load_checkpoint(checkpoint_path, device=device)
    model     = ckpt["model"].to(device)
    node_mean = ckpt["node_mean"]
    node_std  = ckpt["node_std"]
    edge_mean = ckpt["edge_mean"]
    edge_std  = ckpt["edge_std"]
    model.eval()
    print(f"[gnn] loaded checkpoint  epoch={ckpt['epoch']}  best_AP={ckpt['best_ap']:.4f}")
    print(f"[gnn] device={device}  threshold={threshold}")

    # ── per-cluster truth lookup (optional) ──────────────────────────────────
    has_truth = "track_id" in clusters_df.columns

    eid_arr   = clusters_df["event_id"].to_numpy()
    x_arr     = clusters_df["x_trk_mm"].to_numpy()
    y_arr     = clusters_df["y_trk_mm"].to_numpy()
    z_arr     = clusters_df["z_trk_mm"].to_numpy()
    lid_arr   = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr   = clusters_df["node_id"].to_numpy()
    sx_arr    = clusters_df["size_x"].to_numpy()
    sy_arr    = clusters_df["size_y"].to_numpy()
    s_arr     = clusters_df["size"].to_numpy()
    tid_arr   = clusters_df["track_id"].to_numpy() if has_truth else None

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    all_candidates: list[dict] = []
    n_edges_total   = 0
    n_edges_kept    = 0

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid   = int(unique_events[i])

        xv  = x_arr [s: s + c_]
        yv  = y_arr [s: s + c_]
        zv  = z_arr [s: s + c_]
        lv  = lid_arr[s: s + c_]
        nv  = nid_arr[s: s + c_]
        sxv = sx_arr [s: s + c_]
        syv = sy_arr [s: s + c_]
        sv  = s_arr  [s: s + c_]
        tv  = tid_arr[s: s + c_] if has_truth else None

        nid_to_local: dict[int, int] = {int(n): j for j, n in enumerate(nv)}

        # ── 1. candidate edges (physics pruning) ─────────────────────────────
        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(
            xv, yv, zv, lv, nv, baseline_cfg
        )
        if len(e_src) == 0:
            continue
        n_edges_total += len(e_src)

        # ── 2. GNN scoring ────────────────────────────────────────────────────
        nf, ei, ef = _event_to_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sl, e_sx, e_sy,
            nid_to_local,
        )
        # ensure normalization stats are tensors on the target device
        node_mean = torch.as_tensor(node_mean, device=device)
        node_std  = torch.as_tensor(node_std,  device=device)
        edge_mean = torch.as_tensor(edge_mean, device=device)
        edge_std  = torch.as_tensor(edge_std,  device=device)

        nf = nf.to(device)
        ef = ef.to(device)

        nf = (nf - node_mean) / node_std
        ef = (ef - edge_mean) / edge_std

        with torch.no_grad():
            scores = model(
                nf.to(device),
                ei.to(device),
                ef.to(device),
            ).cpu().numpy()

        # ── 3. edge filter ────────────────────────────────────────────────────
        mask = scores >= threshold
        if not mask.any():
            continue

        f_src = e_src[mask]
        f_dst = e_dst[mask]
        f_sl  = e_sl [mask]
        f_dl  = e_dl [mask]   # e_dl was returned as the 4th element
        f_sx  = e_sx [mask]
        f_sy  = e_sy [mask]
        n_edges_kept += mask.sum()

        # ── 4. chain building → fit → shared-hit rejection ───────────────────
        chains = _build_chains(f_src, f_dst, f_sl, f_dl, f_sx, f_sy, baseline_cfg)
        if not chains:
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        # ── 5. truth matching (optional) ──────────────────────────────────────
        for ci, cand in enumerate(candidates):
            cand["event_id"]    = eid
            cand["candidate_id"] = ci
            if has_truth:
                node_tids = [
                    int(tv[nid_to_local[n]])
                    for n in cand["node_ids"]
                ]
                counter = Counter(t for t in node_tids if t >= 0)
                if counter:
                    best_tid, best_count = counter.most_common(1)[0]
                    cand["matched_track_id"] = best_tid if best_count >= 4 else -1
                    cand["n_matched"]        = best_count
                else:
                    cand["matched_track_id"] = -1
                    cand["n_matched"]        = 0

        all_candidates.extend(candidates)

    # ── summary ───────────────────────────────────────────────────────────────
    keep_rate = n_edges_kept / max(n_edges_total, 1) * 100
    print(f"[gnn] edges total={n_edges_total:,}  "
          f"kept={n_edges_kept:,}  ({keep_rate:.2f}% pass threshold)")

    if not all_candidates:
        print("[gnn] no candidates found")
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")

    # ── efficiency / fake-rate ────────────────────────────────────────────────
    if has_truth and tracks_df is not None:
        kept          = result.filter(pl.col("is_kept"))
        n_kept        = kept.height
        matched_kept  = kept.filter(pl.col("matched_track_id") >= 0)
        n_matched_cands = matched_kept.height      # candidate-level (may double-count truth)
        n_fake        = n_kept - n_matched_cands

        # NOTE: track_id is per-event, so use (event_id, matched_track_id) as the unique key
        n_unique_truth_matched = matched_kept.select("event_id", "matched_track_id").unique().height
        n_truth    = tracks_df.height
        eff_truth  = n_unique_truth_matched / n_truth * 100 if n_truth else 0.0
        fake_rate  = n_fake / n_kept * 100 if n_kept else 0.0
        print(f"\n[gnn reco eval]")
        print(f"  truth tracks:          {n_truth}")
        print(f"  kept reco candidates:  {n_kept}")
        print(f"  matched candidates:    {n_matched_cands}")
        print(f"  unique truth matched:  {n_unique_truth_matched}  (track eff = {eff_truth:.1f}%)")
        print(f"  fakes:                 {n_fake}  (fake rate  = {fake_rate:.1f}%)")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="GNN-seeded track reconstruction on simulated data"
    )
    parser.add_argument("--clusters",   required=True,
                        help="Path to sim_clusters.parquet")
    parser.add_argument("--tracks",     default=None,
                        help="Path to sim_tracks.parquet (optional, for efficiency eval)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from src.train")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="GNN score threshold for edge filtering (default: 0.5)")
    parser.add_argument("--output",     default=None,
                        help="Output parquet path (default: <clusters_dir>/sim_gnn_reco.parquet)")
    parser.add_argument("--device",     default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu)")
    args = parser.parse_args()

    clusters_df = pl.read_parquet(args.clusters)
    tracks_df   = pl.read_parquet(args.tracks) if args.tracks else None
    print(f"[cli] clusters={len(clusters_df):,}  "
          f"events={clusters_df['event_id'].n_unique()}")

    result = run_gnn_reco(
        clusters_df    = clusters_df,
        tracks_df      = tracks_df,
        checkpoint_path = args.checkpoint,
        threshold      = args.threshold,
        device         = args.device,
    )

    out = args.output or os.path.join(
        os.path.dirname(args.clusters), "sim_gnn_reco.parquet"
    )
    result.write_parquet(out)
    print(f"[cli] result saved → {out}")


if __name__ == "__main__":
    _cli()
