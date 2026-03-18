"""Unified inference entry for comparing multiple ML models.

Supported modes
---------------
- edge             : edge-classifier only (GNN/MLP checkpoint from src.train)
- edge+embedder    : embedder pre-filter + edge-classifier scoring
- embedder         : embedder-only pre-filtered reconstruction
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import (
    BaselineConfig,
    _build_chains,
    _build_edges,
    _fit_and_score,
    _shared_hit_rejection,
)
from src.train import load_checkpoint, _augment_with_embedder
from src.train_embedder import infer_embedding_neighbors


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_gnn_tensors(
    xv, yv, zv, lv, nv, sxv, syv, sv,
    e_src, e_dst, e_sl, e_sx, e_sy,
    nid_to_local: dict[int, int],
):
    """Build node/edge feature tensors for one event from pre-built edges."""
    all_nids = np.unique(np.concatenate([e_src, e_dst]))
    nid_to_row = {int(n): i for i, n in enumerate(all_nids)}

    node_feat = np.empty((len(all_nids), 7), dtype=np.float32)
    for ri, gid in enumerate(all_nids):
        li = nid_to_local[int(gid)]
        node_feat[ri] = [lv[li], xv[li], yv[li], zv[li], sxv[li], syv[li], sv[li]]

    src_l = np.array([nid_to_row[int(s)] for s in e_src], dtype=np.int64)
    dst_l = np.array([nid_to_row[int(d)] for d in e_dst], dtype=np.int64)

    li_idx = np.array([nid_to_local[int(s)] for s in e_src], dtype=np.int64)
    lj_idx = np.array([nid_to_local[int(d)] for d in e_dst], dtype=np.int64)
    dx = xv[lj_idx] - xv[li_idx]
    dy = yv[lj_idx] - yv[li_idx]
    dz = zv[lj_idx] - zv[li_idx]
    dr = np.sqrt(dx**2 + dy**2)
    edge_feat = np.stack([dx, dy, dz, dr, e_sx, e_sy], axis=1).astype(np.float32)

    return (
        torch.from_numpy(node_feat),
        torch.from_numpy(np.stack([src_l, dst_l])),
        torch.from_numpy(edge_feat),
    )


def _apply_embedder_filter(
    e_src, e_dst, e_sl, e_dl, e_sx, e_sy,
    nid_to_local: dict[int, int],
    nbrs,  # array of neighbor-index arrays (local indices per hit)
):
    """Filter edges to only those connecting embedding-neighbours."""
    keep = np.array([
        nid_to_local[int(e_dst[k])] in set(nbrs[nid_to_local[int(e_src[k])]])
        for k in range(len(e_src))
    ])
    return (
        e_src[keep], e_dst[keep],
        e_sl[keep],  e_dl[keep],
        e_sx[keep],  e_sy[keep],
    )


def _match_candidates(candidates, nid_to_local, tv, has_truth: bool) -> None:
    """Annotate each candidate dict with matched_track_id / n_matched in-place."""
    for ci, cand in enumerate(candidates):
        cand["candidate_id"] = ci
        if has_truth and tv is not None:
            node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
            counter = Counter(t for t in node_tids if t >= 0)
            if counter:
                best_tid, best_count = counter.most_common(1)[0]
                cand["matched_track_id"] = best_tid if best_count >= 4 else -1
                cand["n_matched"] = best_count
            else:
                cand["matched_track_id"] = -1
                cand["n_matched"] = 0
        else:
            cand["matched_track_id"] = -1
            cand["n_matched"] = 0


def _extract_event_arrays(clusters_df: pl.DataFrame):
    """Extract numpy arrays from clusters DataFrame, return a namedtuple-like dict."""
    has_truth = "track_id" in clusters_df.columns
    return {
        "eid_arr": clusters_df["event_id"].to_numpy(),
        "x_arr":   clusters_df["x_trk_mm"].to_numpy(),
        "y_arr":   clusters_df["y_trk_mm"].to_numpy(),
        "z_arr":   clusters_df["z_trk_mm"].to_numpy(),
        "lid_arr": clusters_df["layer_id"].to_numpy().astype(np.int8),
        "nid_arr": clusters_df["node_id"].to_numpy(),
        "sx_arr":  clusters_df["size_x"].to_numpy(),
        "sy_arr":  clusters_df["size_y"].to_numpy(),
        "s_arr":   clusters_df["size"].to_numpy(),
        "tid_arr": clusters_df["track_id"].to_numpy() if has_truth else None,
        "has_truth": has_truth,
    }


def _print_reco_summary(label: str, result: pl.DataFrame, tracks_df: pl.DataFrame | None) -> None:
    kept = result.filter(pl.col("is_kept"))
    n_kept = kept.height
    n_truth = tracks_df.height if tracks_df is not None else 0
    n_matched = kept.filter(pl.col("matched_track_id") >= 0).height
    n_fake = n_kept - n_matched
    eff = n_matched / n_truth * 100 if n_truth else 0.0
    fake_rate = n_fake / n_kept * 100 if n_kept else 0.0
    print(f"\n[{label}]")
    print(f"  truth tracks:    {n_truth}")
    print(f"  kept candidates: {n_kept}")
    print(f"  matched: {n_matched}  (eff={eff:.1f}%)  fakes: {n_fake}  (fake_rate={fake_rate:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Public reco functions
# ─────────────────────────────────────────────────────────────────────────────

def run_edge_classifier_reco(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame | None,
    checkpoint_path: str,
    threshold: float = 0.5,
    baseline_cfg: BaselineConfig | None = None,
    device: str = "cpu",
    embedder_checkpoint_path: str | None = None,
    embedder_radius: float = 1.0,
    embedder_device: str | None = None,
) -> pl.DataFrame:
    """Run edge-classifier reconstruction on simulated clusters.

    Works with any edge-classifier model type (mlp, gnn, transformer,
    resgnn, mpnn, agnn, eggnet, hgnn) — the model architecture is
    auto-detected from the checkpoint's sibling ``config.json``.

    Optionally pre-filters edges with a metric-learning embedder
    (``embedder_checkpoint_path`` / ``embedder_radius``).

    Returns a candidates DataFrame in the same format as
    ``evaluate_baseline_on_sim``.
    """
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    # Optional embedder pre-filter
    embedder_neighbors: dict | None = None
    if embedder_checkpoint_path is not None:
        embedder_neighbors = infer_embedding_neighbors(
            clusters_df,
            checkpoint_path=embedder_checkpoint_path,
            radius=embedder_radius,
            device=(embedder_device or device),
        )

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model = ckpt["model"]
    embedder_info = ckpt.get("embedder_info")
    device_t = torch.device(device)
    node_mean = ckpt["node_mean"].to(device_t)
    node_std  = ckpt["node_std"].to(device_t)
    edge_mean = ckpt["edge_mean"].to(device_t)
    edge_std  = ckpt["edge_std"].to(device_t)
    model.to(device_t).eval()

    arrs = _extract_event_arrays(clusters_df)
    unique_events, starts = np.unique(arrs["eid_arr"], return_index=True)
    counts = np.diff(np.append(starts, len(arrs["eid_arr"])))

    all_candidates: list[dict] = []
    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv  = arrs["x_arr"][s:s+c_];  yv  = arrs["y_arr"][s:s+c_]
        zv  = arrs["z_arr"][s:s+c_];  lv  = arrs["lid_arr"][s:s+c_]
        nv  = arrs["nid_arr"][s:s+c_]
        sxv = arrs["sx_arr"][s:s+c_];  syv = arrs["sy_arr"][s:s+c_]
        sv  = arrs["s_arr"][s:s+c_]
        tv  = arrs["tid_arr"][s:s+c_] if arrs["has_truth"] else None

        nid_to_local = {int(n): j for j, n in enumerate(nv)}

        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(xv, yv, zv, lv, nv, baseline_cfg)
        if len(e_src) == 0:
            continue

        if embedder_neighbors is not None:
            nbrs = embedder_neighbors.get(eid)
            if nbrs is not None:
                e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _apply_embedder_filter(
                    e_src, e_dst, e_sl, e_dl, e_sx, e_sy, nid_to_local, nbrs
                )
                if len(e_src) == 0:
                    continue

        nf, ei, ef = _build_gnn_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sl, e_sx, e_sy,
            nid_to_local,
        )
        # Augment node features with pretrained embedder output (two-stage pipeline)
        if embedder_info is not None:
            nf = _augment_with_embedder(nf, embedder_info)
        # Normalise only the base node features; embedder output is kept as-is
        base_dim = node_mean.shape[0]
        if nf.shape[1] > base_dim:
            nf = torch.cat([(nf[:, :base_dim].to(device_t) - node_mean) / node_std,
                            nf[:, base_dim:].to(device_t)], dim=-1)
        else:
            nf = (nf.to(device_t) - node_mean) / node_std
        ef = (ef.to(device_t) - edge_mean) / edge_std

        with torch.no_grad():
            scores = model(nf, ei.to(device_t), ef).cpu().numpy()

        mask = scores >= threshold
        if not mask.any():
            continue

        chains = _build_chains(
            e_src[mask], e_dst[mask], e_sl[mask], e_dl[mask],
            e_sx[mask],  e_sy[mask],  baseline_cfg,
        )
        if not chains:
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        for cand in candidates:
            cand["event_id"] = eid
        _match_candidates(candidates, nid_to_local, tv, arrs["has_truth"])
        all_candidates.extend(candidates)

    if not all_candidates:
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")
    _print_reco_summary("edge-classifier reco", result, tracks_df)
    return result


def run_embedder_reco(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame | None,
    embedder_checkpoint_path: str,
    embedder_radius: float = 1.0,
    baseline_cfg: BaselineConfig | None = None,
    device: str = "cpu",
) -> pl.DataFrame:
    """Run embedder-only reconstruction: use radius-neighbour edges as candidates.

    Edges are formed between hits that are mutual embedding-neighbours on
    adjacent layers, then passed through the baseline chain-builder and fitter.
    """
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    embedder_neighbors = infer_embedding_neighbors(
        clusters_df,
        checkpoint_path=embedder_checkpoint_path,
        radius=embedder_radius,
        device=device,
    )

    arrs = _extract_event_arrays(clusters_df)
    unique_events, starts = np.unique(arrs["eid_arr"], return_index=True)
    counts = np.diff(np.append(starts, len(arrs["eid_arr"])))

    all_candidates: list[dict] = []
    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv  = arrs["x_arr"][s:s+c_];  yv = arrs["y_arr"][s:s+c_]
        zv  = arrs["z_arr"][s:s+c_];  lv = arrs["lid_arr"][s:s+c_]
        nv  = arrs["nid_arr"][s:s+c_]
        tv  = arrs["tid_arr"][s:s+c_] if arrs["has_truth"] else None

        nid_to_local = {int(n): j for j, n in enumerate(nv)}

        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(xv, yv, zv, lv, nv, baseline_cfg)
        if len(e_src) == 0:
            continue

        nbrs = embedder_neighbors.get(eid)
        if nbrs is not None:
            e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _apply_embedder_filter(
                e_src, e_dst, e_sl, e_dl, e_sx, e_sy, nid_to_local, nbrs
            )
            if len(e_src) == 0:
                continue

        chains = _build_chains(e_src, e_dst, e_sl, e_dl, e_sx, e_sy, baseline_cfg)
        if not chains:
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        for cand in candidates:
            cand["event_id"] = eid
        _match_candidates(candidates, nid_to_local, tv, arrs["has_truth"])
        all_candidates.extend(candidates)

    if not all_candidates:
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")
    _print_reco_summary("Embedder reco", result, tracks_df)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_model_reco(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame | None,
    mode: str,
    *,
    edge_checkpoint: str | None = None,
    edge_threshold: float = 0.5,
    embedder_checkpoint: str | None = None,
    embedder_radius: float = 1.0,
    device: str = "cpu",
    embedder_device: str | None = None,
    baseline_cfg: BaselineConfig | None = None,
) -> pl.DataFrame:
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    if mode == "edge":
        if edge_checkpoint is None:
            raise ValueError("mode='edge' requires edge_checkpoint")
        return run_edge_classifier_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            checkpoint_path=edge_checkpoint,
            threshold=edge_threshold,
            baseline_cfg=baseline_cfg,
            device=device,
        )

    if mode == "edge+embedder":
        if edge_checkpoint is None or embedder_checkpoint is None:
            raise ValueError("mode='edge+embedder' requires both edge_checkpoint and embedder_checkpoint")
        return run_edge_classifier_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            checkpoint_path=edge_checkpoint,
            threshold=edge_threshold,
            baseline_cfg=baseline_cfg,
            device=device,
            embedder_checkpoint_path=embedder_checkpoint,
            embedder_radius=embedder_radius,
            embedder_device=embedder_device,
        )

    if mode == "embedder":
        if embedder_checkpoint is None:
            raise ValueError("mode='embedder' requires embedder_checkpoint")
        return run_embedder_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            embedder_checkpoint_path=embedder_checkpoint,
            embedder_radius=embedder_radius,
            baseline_cfg=baseline_cfg,
            device=(embedder_device or device),
        )

    raise ValueError(f"Unsupported mode: {mode}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unified model inference for E320simulator")
    parser.add_argument("--mode", default="edge", choices=["edge", "edge+embedder", "embedder"])
    parser.add_argument("--clusters", required=True, help="Path to sim_clusters.parquet")
    parser.add_argument("--tracks", default=None, help="Path to sim_tracks.parquet (optional)")

    parser.add_argument("--edge-checkpoint", default=None,
                        help="Path to best_model.pt from src.train")
    parser.add_argument("--edge-threshold", type=float, default=0.5)

    parser.add_argument("--embedder-checkpoint", default=None,
                        help="Path to best_embedder.pt from src.train_embedder")
    parser.add_argument("--embedder-radius", type=float, default=1.0)

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--embedder-device", default=None, choices=["cpu", "cuda", "mps"])

    parser.add_argument("--output", default=None,
                        help="Output parquet path. Default: <clusters_dir>/<mode>_result.parquet")
    args = parser.parse_args()

    clusters_df = pl.read_parquet(args.clusters)
    tracks_df = pl.read_parquet(args.tracks) if args.tracks else None

    print(f"[run_model] mode={args.mode}  clusters={len(clusters_df):,}")

    result = run_model_reco(
        clusters_df=clusters_df,
        tracks_df=tracks_df,
        mode=args.mode,
        edge_checkpoint=args.edge_checkpoint,
        edge_threshold=args.edge_threshold,
        embedder_checkpoint=args.embedder_checkpoint,
        embedder_radius=args.embedder_radius,
        device=args.device,
        embedder_device=args.embedder_device,
    )

    if args.output is None:
        safe_mode = args.mode.replace("+", "_plus_")
        out = os.path.join(os.path.dirname(args.clusters), f"{safe_mode}_result.parquet")
    else:
        out = args.output

    result.write_parquet(out)
    print(f"[run_model] result saved → {out}")


if __name__ == "__main__":
    _cli()
