"""
Horizontal benchmark: compare non-ML and ML track-finding algorithms on E320 simulation.

Non-ML (run on test only):
  - Baseline  (slope-window + chain seeding)
  - Hough Transform
  - Kalman Filter

ML (train on train, evaluate on test):
  - MLP, GNN, ResGNN, MPNN, AGNN, EggNet

Usage:
    cd /Users/IvanTang/hep/E320simulator
    python scripts/run_benchmark.py [--device mps] [--epochs 50] [--force-retrain]
"""
from __future__ import annotations

import argparse
import sys
import time
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
from src.kalman_tracker import KalmanConfig, _process_event_kalman
from src.train import TrainConfig, load_checkpoint, train
from src.utils import build_labeled_edges_from_sim
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.run_hough import evaluate_hough_on_sim
from scripts.compare_reco import compare

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path.home() / "hep/data_Run502"
SIM_DIR        = _BASE / "simulation"
RUNS_DIR       = _BASE / "runs"
OUT_DIR        = _BASE / "outputs"

TRAIN_CLUSTERS = SIM_DIR / "sim_clusters_train.parquet"
TEST_CLUSTERS  = SIM_DIR / "sim_clusters_test.parquet"
TEST_TRACKS    = SIM_DIR / "sim_tracks_test.parquet"

# ML model types to benchmark
ML_MODELS = ["mlp", "gnn", "resgnn", "mpnn", "agnn", "eggnet"]


# ── Kalman wrapper for sim_clusters format ────────────────────────────────────

def evaluate_kalman_on_sim(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run the Kalman filter tracker on sim_clusters and evaluate against truth."""
    cfg = KalmanConfig()
    all_candidates: list[dict] = []

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr   = clusters_df["x_trk_mm"].to_numpy()
    y_arr   = clusters_df["y_trk_mm"].to_numpy()
    z_arr   = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    tid_arr = clusters_df["track_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid   = int(unique_events[i])
        xv = x_arr[s : s + c_]
        yv = y_arr[s : s + c_]
        zv = z_arr[s : s + c_]
        lv = lid_arr[s : s + c_]
        nv = nid_arr[s : s + c_]
        tv = tid_arr[s : s + c_]

        candidates = _process_event_kalman(eid, xv, yv, zv, lv, nv, cfg)
        if not candidates:
            continue

        nid_to_local = {int(n): j for j, n in enumerate(nv)}
        for cand in candidates:
            node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
            counter = Counter(t for t in node_tids if t >= 0)
            if counter:
                best_tid, best_count = counter.most_common(1)[0]
                cand["matched_track_id"] = best_tid if best_count >= 4 else -1
                cand["n_matched"] = best_count
            else:
                cand["matched_track_id"] = -1
                cand["n_matched"] = 0
        all_candidates.extend(candidates)

    if not all_candidates:
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")

    kept = result.filter(pl.col("is_kept"))
    n_kept = kept.height
    n_matched = kept.filter(pl.col("matched_track_id") >= 0).height
    n_truth = tracks_df.height
    eff = n_matched / n_truth * 100 if n_truth > 0 else 0.0
    fake = (n_kept - n_matched) / n_kept * 100 if n_kept > 0 else 0.0
    print(f"\n[kalman eval]")
    print(f"  truth tracks:          {n_truth}")
    print(f"  kept reco candidates:  {n_kept}")
    print(f"  matched candidates:    {n_matched} (track eff = {eff:.1f}%)")
    print(f"  fakes:                 {n_kept - n_matched}  (fake rate = {fake:.1f}%)")

    return result


# ── GNN inference on sim_clusters format ─────────────────────────────────────

def _build_gnn_tensors(
    xv, yv, zv, lv, nv, sxv, syv, sv,
    e_src, e_dst, e_sx, e_sy,
    nid_to_local: dict,
):
    """Build (node_feat, edge_index, edge_feat) tensors for a single event."""
    all_nids   = np.unique(np.concatenate([e_src, e_dst]))
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


def evaluate_gnn_on_sim(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
    checkpoint_path: str,
    threshold: float = 0.5,
    device: str = "mps",
) -> pl.DataFrame:
    """Run a GNN edge-classifier on sim_clusters and produce a reco DataFrame."""
    baseline_cfg = BaselineConfig()
    device_t = torch.device(device)

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model     = ckpt["model"]
    node_mean = ckpt["node_mean"].to(device_t)
    node_std  = ckpt["node_std"].to(device_t)
    edge_mean = ckpt["edge_mean"].to(device_t)
    edge_std  = ckpt["edge_std"].to(device_t)
    model.eval()

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr   = clusters_df["x_trk_mm"].to_numpy()
    y_arr   = clusters_df["y_trk_mm"].to_numpy()
    z_arr   = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    sx_arr  = clusters_df["size_x"].to_numpy().astype(np.float32)
    sy_arr  = clusters_df["size_y"].to_numpy().astype(np.float32)
    s_arr   = clusters_df["size"].to_numpy().astype(np.float32)
    tid_arr = clusters_df["track_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    all_candidates: list[dict] = []

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv  = x_arr[s : s + c_]
        yv  = y_arr[s : s + c_]
        zv  = z_arr[s : s + c_]
        lv  = lid_arr[s : s + c_]
        nv  = nid_arr[s : s + c_]
        sxv = sx_arr[s : s + c_]
        syv = sy_arr[s : s + c_]
        sv  = s_arr[s : s + c_]
        tv  = tid_arr[s : s + c_]

        nid_to_local = {int(n): j for j, n in enumerate(nv)}

        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(
            xv, yv, zv, lv, nv, baseline_cfg
        )
        if len(e_src) == 0:
            continue

        nf, ei, ef = _build_gnn_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sx, e_sy, nid_to_local,
        )
        nf = (nf.to(device_t) - node_mean) / node_std
        ef = (ef.to(device_t) - edge_mean) / edge_std

        with torch.no_grad():
            scores = model(nf, ei.to(device_t), ef).cpu().numpy()

        mask = scores >= threshold
        if not mask.any():
            continue

        chains = _build_chains(
            e_src[mask], e_dst[mask], e_sl[mask], e_dl[mask],
            e_sx[mask], e_sy[mask], baseline_cfg,
        )
        if not chains:
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        for ci, cand in enumerate(candidates):
            cand["event_id"]    = eid
            cand["candidate_id"] = ci
            node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
            counter = Counter(t for t in node_tids if t >= 0)
            if counter:
                best_tid, best_count = counter.most_common(1)[0]
                cand["matched_track_id"] = best_tid if best_count >= 4 else -1
                cand["n_matched"] = best_count
            else:
                cand["matched_track_id"] = -1
                cand["n_matched"] = 0
        all_candidates.extend(candidates)

    if not all_candidates:
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")

    kept = result.filter(pl.col("is_kept"))
    n_kept    = kept.height
    n_matched = kept.filter(pl.col("matched_track_id") >= 0).height
    n_truth   = tracks_df.height
    eff  = n_matched / n_truth * 100 if n_truth > 0 else 0.0
    fake = (n_kept - n_matched) / n_kept * 100 if n_kept > 0 else 0.0
    print(f"  kept={n_kept}  matched={n_matched}  "
          f"eff={eff:.1f}%  fake={fake:.1f}%")

    return result


# ── Training helper ───────────────────────────────────────────────────────────

def train_ml_model(
    model_type: str,
    clusters_df: pl.DataFrame,
    checkpoint_dir: Path,
    device: str = "mps",
    n_epochs: int = 50,
    force: bool = False,
) -> str:
    """Build edges, train model, and save checkpoint.  Returns checkpoint path."""
    ckpt_path = checkpoint_dir / "best_model.pt"
    if ckpt_path.exists() and not force:
        print(f"  checkpoint exists → skipping training: {ckpt_path}")
        return str(ckpt_path)

    print(f"  Building labeled edges from train clusters...")
    edges_df = build_labeled_edges_from_sim(clusters_df)
    print(f"  Built {len(edges_df):,} candidate edges")

    cfg = TrainConfig(
        model_type     = model_type,
        n_epochs       = n_epochs,
        hidden         = 64,
        n_mp           = 2,
        lr             = 3e-4,
        device         = device,
        checkpoint_dir = str(checkpoint_dir),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train(edges_df, cfg)
    return str(ckpt_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(device: str = "mps", force_retrain: bool = False, epochs: int = 50) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Loading data...")
    clusters_train = pl.read_parquet(TRAIN_CLUSTERS)
    clusters_test  = pl.read_parquet(TEST_CLUSTERS)
    tracks_test    = pl.read_parquet(TEST_TRACKS)
    print(f"  Train clusters : {len(clusters_train):,}")
    print(f"  Test  clusters : {len(clusters_test):,}")
    print(f"  Test  tracks   : {len(tracks_test):,}")

    reco_paths: dict[str, str] = {}

    # ── Non-ML algorithms ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Non-ML algorithms (test set only)")

    for name, fn in [
        ("Baseline", lambda: evaluate_baseline_on_sim(clusters_test, tracks_test)),
        ("Hough",    lambda: evaluate_hough_on_sim(clusters_test, tracks_test)),
        ("Kalman",   lambda: evaluate_kalman_on_sim(clusters_test, tracks_test)),
    ]:
        print(f"\n[{name}]")
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        out = OUT_DIR / f"{name.lower()}_test.parquet"
        result.write_parquet(out)
        reco_paths[name] = str(out)
        print(f"  → {out}  ({dt:.1f}s)")

    # ── ML algorithms ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ML algorithms  (train={epochs} epochs on {device}, then test)")

    for model_type in ML_MODELS:
        print(f"\n[{model_type.upper()}]")
        ckpt_dir  = RUNS_DIR / model_type
        ckpt_path = str(ckpt_dir / "best_model.pt")

        # Train
        t0 = time.perf_counter()
        try:
            ckpt_path = train_ml_model(
                model_type, clusters_train, ckpt_dir,
                device=device, n_epochs=epochs, force=force_retrain,
            )
        except Exception as e:
            print(f"  [ERROR] training failed: {e}")
            continue
        train_dt = time.perf_counter() - t0

        # Inference
        t0 = time.perf_counter()
        try:
            result = evaluate_gnn_on_sim(
                clusters_test, tracks_test, ckpt_path,
                threshold=0.5, device=device,
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            continue
        infer_dt = time.perf_counter() - t0

        out = OUT_DIR / f"{model_type}_test.parquet"
        result.write_parquet(out)
        reco_paths[model_type.upper()] = str(out)
        print(f"  → {out}  (train {train_dt:.0f}s  infer {infer_dt:.1f}s)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    compare(str(TEST_TRACKS), reco_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark E320 tracking algorithms")
    parser.add_argument("--device",        default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train even if checkpoint already exists")
    args = parser.parse_args()
    main(device=args.device, force_retrain=args.force_retrain, epochs=args.epochs)
