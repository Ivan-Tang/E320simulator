"""
Grid search over ML edge-classifier score threshold on simulated data.

Inference is run *once* per event; the resulting edge scores are cached in
memory.  The threshold sweep is then just repeated numpy masking + chain
building — no GPU round-trip per threshold value.

A reference line from the default baseline configuration is drawn on every
plot for direct comparison.

Outputs (saved to SIM_DIR):
    grid_search_<model_name>_threshold.png  — efficiency & fake rate vs threshold
    grid_search_<model_name>_edge_keep.png  — edge keep rate & reco track count

Usage
-----
    python scripts/grid_search_ml.py \\
        --checkpoint runs/exp_gnn_v1/best_model.pt

    python scripts/grid_search_ml.py \\
        --checkpoint runs/exp_gnn_v1/best_model.pt \\
        --model-name gnn_v1 \\
        --device mps

    # or edit SIM_DIR / CHECKPOINT_PATH at the top of the file
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.train import load_checkpoint
from src.config import SIM_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Paths (override via CLI or edit here)
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "runs/exp_gnn_v1/best_model.pt"
DEVICE          = "cpu"   # "cpu" | "mps" | "cuda"

# ──────────────────────────────────────────────────────────────────────────────
# CLI (optional overrides)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search ML threshold")
    p.add_argument("--checkpoint",  default=CHECKPOINT_PATH)
    p.add_argument("--device",      default=DEVICE, choices=["cpu", "cuda", "mps"])
    p.add_argument("--sim_dir",     default=SIM_DIR)
    p.add_argument("--model-name",  default=None,
                   help="Display name for the model (derived from checkpoint if omitted)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model name derivation
# ──────────────────────────────────────────────────────────────────────────────

def _derive_model_name(checkpoint_path: str, ckpt: dict) -> str:
    # 1. Try cfg stored in checkpoint
    cfg = ckpt.get("cfg") or ckpt.get("config")
    if cfg and hasattr(cfg, "model"):
        return cfg.model
    # 2. Fall back to parent directory name
    return Path(checkpoint_path).parent.name


# ──────────────────────────────────────────────────────────────────────────────
# Per-event tensor builder  (same logic as run_model._build_gnn_tensors)
# ──────────────────────────────────────────────────────────────────────────────

def _build_tensors(
    xv, yv, zv, lv, nv, sxv, syv, sv,
    e_src, e_dst, e_sl, e_sx, e_sy,
    nid_to_local: dict[int, int],
):
    all_nids   = np.unique(np.concatenate([e_src, e_dst]))
    nid_to_row = {int(n): i for i, n in enumerate(all_nids)}

    node_feat = np.empty((len(all_nids), 7), dtype=np.float32)
    for ri, gid in enumerate(all_nids):
        li = nid_to_local[int(gid)]
        node_feat[ri] = [lv[li], xv[li], yv[li], zv[li], sxv[li], syv[li], sv[li]]

    src_l = np.array([nid_to_row[int(s)] for s in e_src], dtype=np.int64)
    dst_l = np.array([nid_to_row[int(d)] for d in e_dst], dtype=np.int64)

    li = np.array([nid_to_local[int(s)] for s in e_src], dtype=np.int64)
    lj = np.array([nid_to_local[int(d)] for d in e_dst], dtype=np.int64)
    dx = xv[lj] - xv[li]; dy = yv[lj] - yv[li]; dz = zv[lj] - zv[li]
    dr = np.sqrt(dx**2 + dy**2)

    edge_feat = np.stack([dx, dy, dz, dr, e_sx, e_sy], axis=1).astype(np.float32)

    return (
        torch.from_numpy(node_feat),
        torch.from_numpy(np.stack([src_l, dst_l])),
        torch.from_numpy(edge_feat),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – run inference once, cache results per event
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    event_slices,
    model,
    node_mean, node_std, edge_mean, edge_std,
    baseline_cfg: BaselineConfig,
    device: str,
    embedder_info=None,
) -> list[dict]:
    """Return a list of per-event dicts with pre-computed edge scores."""
    from src.train import _augment_with_embedder
    device_t   = torch.device(device)
    node_mean  = node_mean.to(device_t)
    node_std   = node_std .to(device_t)
    edge_mean  = edge_mean.to(device_t)
    edge_std   = edge_std .to(device_t)

    model.to(device_t).eval()

    cache = []
    for eid, xv, yv, zv, lv, nv, sxv, syv, sv, tv in event_slices:
        nid_to_local = {int(n): j for j, n in enumerate(nv)}

        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(
            xv, yv, zv, lv, nv, baseline_cfg
        )
        if len(e_src) == 0:
            continue

        nf, ei, ef = _build_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sl, e_sx, e_sy,
            nid_to_local,
        )
        nf = nf.to(device_t)
        if embedder_info is not None:
            nf = _augment_with_embedder(nf, embedder_info)
        if nf.shape[1] == node_mean.shape[0]:   # skip when embedder changes dim
            nf = (nf - node_mean) / node_std
        ef = ((ef.to(device_t) - edge_mean) / edge_std)

        with torch.no_grad():
            scores = model(nf, ei.to(device_t), ef).cpu().numpy()

        cache.append({
            "eid":          eid,
            "xv": xv, "yv": yv, "zv": zv,
            "nid_to_local": nid_to_local,
            "e_src": e_src, "e_dst": e_dst,
            "e_sl":  e_sl,  "e_dl":  e_dl,
            "e_sx":  e_sx,  "e_sy":  e_sy,
            "scores": scores,
            "tv": tv,
        })

    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – evaluate one threshold value using cached scores
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_threshold(
    cache: list[dict],
    threshold: float,
    baseline_cfg: BaselineConfig,
    n_truth: int,
) -> dict:
    n_matched = n_kept = n_fake = n_edges_kept = n_edges_total = 0

    for ev in cache:
        scores       = ev["scores"]
        e_src, e_dst = ev["e_src"], ev["e_dst"]
        e_sl,  e_dl  = ev["e_sl"],  ev["e_dl"]
        e_sx,  e_sy  = ev["e_sx"],  ev["e_sy"]
        xv, yv, zv   = ev["xv"],    ev["yv"],  ev["zv"]
        nid_to_local = ev["nid_to_local"]
        tv           = ev["tv"]

        n_edges_total += len(scores)
        mask = scores >= threshold
        if not mask.any():
            continue
        n_edges_kept += int(mask.sum())

        chains = _build_chains(
            e_src[mask], e_dst[mask], e_sl[mask], e_dl[mask],
            e_sx[mask],  e_sy[mask],  baseline_cfg,
        )
        if not chains:
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        for cand in candidates:
            if not cand["is_kept"]:
                continue
            n_kept += 1
            if tv is not None:
                node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
                counter   = Counter(t for t in node_tids if t >= 0)
                if counter:
                    _, best_count = counter.most_common(1)[0]
                    if best_count >= 4:
                        n_matched += 1
                    else:
                        n_fake += 1
                else:
                    n_fake += 1

    eff       = n_matched / n_truth * 100 if n_truth  else 0.0
    fake_rate = n_fake    / n_kept  * 100 if n_kept   else 0.0
    keep_rate = n_edges_kept / max(n_edges_total, 1) * 100
    return {
        "threshold":   threshold,
        "efficiency":  eff,
        "fake_rate":   fake_rate,
        "n_kept":      n_kept,
        "n_matched":   n_matched,
        "n_fake":      n_fake,
        "edge_keep_%": keep_rate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline reference (single run at default config)
# ──────────────────────────────────────────────────────────────────────────────

def baseline_reference(event_slices, n_truth: int) -> dict:
    cfg = BaselineConfig()
    n_matched = n_kept = n_fake = 0
    for eid, xv, yv, zv, lv, nv, sxv, syv, sv, tv in event_slices:
        nid_to_local = {int(n): j for j, n in enumerate(nv)}
        src, dst, sl, dl, sx, sy = _build_edges(xv, yv, zv, lv, nv, cfg)
        if len(src) == 0:
            continue
        chains = _build_chains(src, dst, sl, dl, sx, sy, cfg)
        if not chains:
            continue
        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)
        for cand in candidates:
            if not cand["is_kept"]:
                continue
            n_kept += 1
            if tv is not None:
                node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
                counter   = Counter(t for t in node_tids if t >= 0)
                if counter:
                    _, best_count = counter.most_common(1)[0]
                    if best_count >= 4:
                        n_matched += 1
                    else:
                        n_fake += 1
                else:
                    n_fake += 1
    return {
        "efficiency": n_matched / n_truth * 100 if n_truth else 0.0,
        "fake_rate":  n_fake    / n_kept  * 100 if n_kept  else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_threshold_sweep(
    results: list[dict], bl_ref: dict, save_path: str, model_name: str
) -> None:
    thresholds   = [r["threshold"]   for r in results]
    efficiencies = [r["efficiency"]  for r in results]
    fake_rates   = [r["fake_rate"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"{model_name} threshold sweep  –  efficiency & fake rate",
        fontsize=13, fontweight="bold",
    )

    col_eff  = "#4878CF"
    col_fake = "#D65F5F"
    col_edge = "#6ACC65"
    col_bl   = "#888888"

    # ── (1) efficiency ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(thresholds, efficiencies, "o-", color=col_eff, lw=2, ms=5, label=model_name)
    ax.axhline(bl_ref["efficiency"], ls="--", lw=1.5, color=col_bl,
               label=f"Baseline ({bl_ref['efficiency']:.1f}%)")
    ax.set_xlabel("Score threshold"); ax.set_ylabel("Efficiency (%)")
    ax.set_title("Track efficiency vs threshold")
    ax.set_ylim(0, 115); ax.legend(); ax.grid(alpha=0.3)

    # ── (2) fake rate ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(thresholds, fake_rates, "s-", color=col_fake, lw=2, ms=5, label=model_name)
    ax.axhline(bl_ref["fake_rate"], ls="--", lw=1.5, color=col_bl,
               label=f"Baseline ({bl_ref['fake_rate']:.1f}%)")
    ax.set_xlabel("Score threshold"); ax.set_ylabel("Fake rate (%)")
    ax.set_title("Fake rate vs threshold")
    ax.set_ylim(0, max(fake_rates) * 1.25 + 5); ax.legend(); ax.grid(alpha=0.3)

    # ── (3) efficiency vs fake rate (operating-point curve) ───────────────────
    ax = axes[2]
    sc = ax.scatter(fake_rates, efficiencies, c=thresholds, cmap="viridis",
                    s=50, zorder=3, label=f"{model_name} (threshold →)")
    plt.colorbar(sc, ax=ax, label="threshold")
    for r in results[::max(1, len(results)//6)]:   # annotate every ~6th point
        ax.annotate(f"{r['threshold']:.2f}",
                    (r["fake_rate"], r["efficiency"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axvline(bl_ref["fake_rate"],   ls="--", lw=1.2, color=col_bl, alpha=0.8)
    ax.axhline(bl_ref["efficiency"],  ls="--", lw=1.2, color=col_bl, alpha=0.8,
               label=f"Baseline ({bl_ref['efficiency']:.1f}% eff  {bl_ref['fake_rate']:.1f}% fake)")
    ax.set_xlabel("Fake rate (%)"); ax.set_ylabel("Efficiency (%)")
    ax.set_title("Efficiency vs Fake rate  (operating curve)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_edge_keep_rate(results: list[dict], save_path: str, model_name: str) -> None:
    thresholds = [r["threshold"]   for r in results]
    keeps      = [r["edge_keep_%"] for r in results]
    n_kepts    = [r["n_kept"]      for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    col_edge = "#6ACC65"; col_kept = "#4878CF"

    ax1.plot(thresholds, keeps, "o-", color=col_edge, lw=2, ms=5, label="Edge keep rate (%)")
    ax1.set_xlabel("Score threshold"); ax1.set_ylabel("Edges kept (%)", color=col_edge)
    ax1.tick_params(axis="y", labelcolor=col_edge)
    ax1.set_ylim(0, 110); ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, n_kepts, "s--", color=col_kept, lw=2, ms=5, label="Reco tracks kept")
    ax2.set_ylabel("# kept reco tracks", color=col_kept)
    ax2.tick_params(axis="y", labelcolor=col_kept)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"Edge keep rate & reco track count vs threshold ({model_name})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(checkpoint_path: str, device: str, sim_dir: str, model_name: str | None) -> None:
    # ── load data ─────────────────────────────────────────────────────────────
    clusters_path = f"{sim_dir}/sim_clusters_test.parquet"
    tracks_path   = f"{sim_dir}/sim_tracks_test.parquet"
    # fall back to non-split files if test split doesn't exist
    if not Path(clusters_path).exists():
        clusters_path = f"{sim_dir}/sim_clusters.parquet"
        tracks_path   = f"{sim_dir}/sim_tracks.parquet"

    clusters_df = pl.read_parquet(clusters_path)
    tracks_df   = pl.read_parquet(tracks_path)
    n_truth     = tracks_df.height
    has_truth   = "track_id" in clusters_df.columns

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr   = clusters_df["x_trk_mm"].to_numpy()
    y_arr   = clusters_df["y_trk_mm"].to_numpy()
    z_arr   = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    sx_arr  = clusters_df["size_x"].to_numpy()
    sy_arr  = clusters_df["size_y"].to_numpy()
    s_arr   = clusters_df["size"].to_numpy()
    tid_arr = clusters_df["track_id"].to_numpy() if has_truth else None

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    event_slices = []
    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        event_slices.append((
            int(unique_events[i]),
            x_arr [s:s+c_], y_arr[s:s+c_], z_arr[s:s+c_],
            lid_arr[s:s+c_], nid_arr[s:s+c_],
            sx_arr[s:s+c_], sy_arr[s:s+c_], s_arr[s:s+c_],
            tid_arr[s:s+c_] if has_truth else None,
        ))

    print(f"Loaded {len(event_slices)} events  {clusters_df.height:,} clusters  {n_truth} truth tracks")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"Checkpoint: epoch={ckpt['epoch']}  best_AP={ckpt['best_ap']:.4f}")

    # ── resolve model name ────────────────────────────────────────────────────
    if model_name is None:
        model_name = _derive_model_name(checkpoint_path, ckpt)
    safe_name = model_name.replace(" ", "_")

    baseline_cfg = BaselineConfig()

    # ── two-stage embedder (optional) ─────────────────────────────────────────
    embedder_info = ckpt.get("embedder_info", None)
    if embedder_info is not None:
        print(f"  Embedder detected: node_dim 7 → {embedder_info.get('embedding_dim', '?')}")

    # ── inference pass (once) ─────────────────────────────────────────────────
    print(f"\nRunning {model_name} inference on all events (once) …")
    t0 = time.perf_counter()
    cache = run_inference(
        event_slices,
        ckpt["model"], ckpt["node_mean"], ckpt["node_std"],
        ckpt["edge_mean"], ckpt["edge_std"],
        baseline_cfg, device,
        embedder_info=embedder_info,
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s  ({len(cache)} events with edges)")

    # ── baseline reference (once) ─────────────────────────────────────────────
    print("\nRunning baseline (default config) for reference …")
    t0 = time.perf_counter()
    bl_ref = baseline_reference(event_slices, n_truth)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    print(f"  Baseline: eff={bl_ref['efficiency']:.1f}%  fake={bl_ref['fake_rate']:.1f}%")

    # ── threshold sweep ───────────────────────────────────────────────────────
    thresholds = np.concatenate([
        np.linspace(0.01, 0.09, 5),
        np.linspace(0.10, 0.90, 17),
        np.linspace(0.91, 0.99, 5),
    ])
    thresholds = np.unique(thresholds.round(3))

    print(f"\n{'='*60}")
    print(f"Threshold sweep  ({len(thresholds)} values)  default baseline for reference")
    print(f"{'='*60}")
    print(f"  {'threshold':>9}  {'eff%':>7}  {'fake%':>7}  {'kept':>6}  {'edge_keep%':>10}")

    results = []
    for t in thresholds:
        t0  = time.perf_counter()
        res = evaluate_threshold(cache, t, baseline_cfg, n_truth)
        dt  = time.perf_counter() - t0
        results.append(res)
        print(f"  {t:9.3f}  {res['efficiency']:7.1f}  {res['fake_rate']:7.1f}  "
              f"{res['n_kept']:6d}  {res['edge_keep_%']:10.2f}  ({dt:.2f}s)")

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating plots …")
    plot_threshold_sweep(
        results, bl_ref,
        save_path=f"{sim_dir}/grid_search_{safe_name}_threshold.png",
        model_name=model_name,
    )
    plot_edge_keep_rate(
        results,
        save_path=f"{sim_dir}/grid_search_{safe_name}_edge_keep.png",
        model_name=model_name,
    )
    print("Done!")


if __name__ == "__main__":
    args = _parse_args()
    main(args.checkpoint, args.device, args.sim_dir, args.model_name)
