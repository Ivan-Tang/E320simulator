"""
Grid search over baseline parameters on simulated data.

Sweeps three parameter groups independently (keeping others at default):
  1. slope_max  (slope_x_max = slope_y_max)
  2. knn_k
  3. dslope_max (dslope_x_max = dslope_y_max)

Metrics: efficiency (matched / truth) and fake rate (fakes / kept).
Outputs three plots to data_Run502/simulation/.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/IvanTang/hep/E320simulator")
from src.baseline import BaselineConfig, _build_edges, _build_chains, _fit_and_score, _shared_hit_rejection


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
SIM_DIR = "/Users/IvanTang/hep/data_Run502/simulation"

clusters_df = pl.read_parquet(f"{SIM_DIR}/sim_clusters_train.parquet")
tracks_df = pl.read_parquet(f"{SIM_DIR}/sim_tracks_train.parquet")

# Pre-extract numpy arrays (shared across all parameter combos)
eid_arr = clusters_df["event_id"].to_numpy()
x_arr = clusters_df["x_trk_mm"].to_numpy()
y_arr = clusters_df["y_trk_mm"].to_numpy()
z_arr = clusters_df["z_trk_mm"].to_numpy()
lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
nid_arr = clusters_df["node_id"].to_numpy()
tid_arr = clusters_df["track_id"].to_numpy()

unique_events, starts = np.unique(eid_arr, return_index=True)
counts = np.diff(np.append(starts, len(eid_arr)))
n_events = len(unique_events)

# Build per-event slices once
event_slices = []
for i in range(n_events):
    s, c_ = int(starts[i]), int(counts[i])
    event_slices.append((
        int(unique_events[i]),
        x_arr[s:s+c_], y_arr[s:s+c_], z_arr[s:s+c_],
        lid_arr[s:s+c_], nid_arr[s:s+c_], tid_arr[s:s+c_],
    ))

n_truth = tracks_df.height
print(f"Loaded {n_events} events, {clusters_df.height:,} clusters, {n_truth} truth tracks")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation function
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(cfg: BaselineConfig) -> dict:
    """Run baseline with given config, return efficiency & fake rate."""
    n_matched = 0
    n_kept = 0
    n_fake = 0

    for eid, xv, yv, zv, lv, nv, tv in event_slices:
        src, dst, sl, dl, sx, sy = _build_edges(xv, yv, zv, lv, nv, cfg)
        if len(src) == 0:
            continue
        chains = _build_chains(src, dst, sl, dl, sx, sy, cfg)
        if not chains:
            continue
        nid_to_local = {int(n): j for j, n in enumerate(nv)}
        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        for cand in candidates:
            if not cand["is_kept"]:
                continue
            n_kept += 1
            node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
            counter = Counter(t for t in node_tids if t >= 0)
            if counter:
                best_tid, best_count = counter.most_common(1)[0]
                if best_count >= 4:
                    n_matched += 1
                else:
                    n_fake += 1
            else:
                n_fake += 1

    eff = n_matched / n_truth * 100 if n_truth > 0 else 0.0
    fake = n_fake / n_kept * 100 if n_kept > 0 else 0.0
    return {"efficiency": eff, "fake_rate": fake, "n_kept": n_kept, "n_matched": n_matched}


# ──────────────────────────────────────────────────────────────────────────────
# Default config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT = BaselineConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Sweep 1: slope_max
# ──────────────────────────────────────────────────────────────────────────────
slope_values = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]
print(f"\n{'='*60}")
print(f"Sweep 1: slope_max  (default knn_k={DEFAULT.knn_k}, dslope={DEFAULT.dslope_x_max})")
print(f"{'='*60}")

slope_results = []
for sv in slope_values:
    cfg = BaselineConfig(
        slope_x_max=sv, slope_y_max=sv,
        knn_k=DEFAULT.knn_k,
        dslope_x_max=DEFAULT.dslope_x_max,
        dslope_y_max=DEFAULT.dslope_y_max,
    )
    t0 = time.perf_counter()
    res = evaluate(cfg)
    dt = time.perf_counter() - t0
    res["param"] = sv
    slope_results.append(res)
    print(f"  slope_max={sv:.3f}  eff={res['efficiency']:.1f}%  fake={res['fake_rate']:.1f}%  "
          f"kept={res['n_kept']}  ({dt:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Sweep 2: knn_k
# ──────────────────────────────────────────────────────────────────────────────
knn_values = [3, 5, 8, 10, 15, 20, 30, 50, 0]  # 0 = no KNN
print(f"\n{'='*60}")
print(f"Sweep 2: knn_k  (default slope={DEFAULT.slope_x_max}, dslope={DEFAULT.dslope_x_max})")
print(f"{'='*60}")

knn_results = []
for kv in knn_values:
    cfg = BaselineConfig(
        slope_x_max=DEFAULT.slope_x_max, slope_y_max=DEFAULT.slope_y_max,
        knn_k=kv,
        dslope_x_max=DEFAULT.dslope_x_max,
        dslope_y_max=DEFAULT.dslope_y_max,
    )
    t0 = time.perf_counter()
    res = evaluate(cfg)
    dt = time.perf_counter() - t0
    res["param"] = kv
    knn_results.append(res)
    label = "none" if kv == 0 else str(kv)
    print(f"  knn_k={label:>4s}  eff={res['efficiency']:.1f}%  fake={res['fake_rate']:.1f}%  "
          f"kept={res['n_kept']}  ({dt:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Sweep 3: dslope_max
# ──────────────────────────────────────────────────────────────────────────────
dslope_values = [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015]
print(f"\n{'='*60}")
print(f"Sweep 3: dslope_max  (default slope={DEFAULT.slope_x_max}, knn_k={DEFAULT.knn_k})")
print(f"{'='*60}")

dslope_results = []
for dv in dslope_values:
    cfg = BaselineConfig(
        slope_x_max=DEFAULT.slope_x_max, slope_y_max=DEFAULT.slope_y_max,
        knn_k=DEFAULT.knn_k,
        dslope_x_max=dv, dslope_y_max=dv,
    )
    t0 = time.perf_counter()
    res = evaluate(cfg)
    dt = time.perf_counter() - t0
    res["param"] = dv
    dslope_results.append(res)
    print(f"  dslope_max={dv:.4f}  eff={res['efficiency']:.1f}%  fake={res['fake_rate']:.1f}%  "
          f"kept={res['n_kept']}  ({dt:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_sweep(results, param_name, xlabel, logx=False, save_path=None):
    params = [r["param"] for r in results]
    effs = [r["efficiency"] for r in results]
    fakes = [r["fake_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_eff = "tab:blue"
    color_fake = "tab:red"

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Efficiency [%]", color=color_eff, fontsize=12)
    l1 = ax1.plot(params, effs, "o-", color=color_eff, linewidth=2, markersize=7, label="Efficiency")
    ax1.tick_params(axis="y", labelcolor=color_eff)
    ax1.set_ylim(0, max(105, max(effs) + 5))

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fake Rate [%]", color=color_fake, fontsize=12)
    l2 = ax2.plot(params, fakes, "s--", color=color_fake, linewidth=2, markersize=7, label="Fake Rate")
    ax2.tick_params(axis="y", labelcolor=color_fake)
    ax2.set_ylim(0, max(105, max(fakes) + 5))

    if logx:
        ax1.set_xscale("log")

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11)

    ax1.set_title(f"Baseline performance vs {param_name}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


print(f"\n{'='*60}")
print("Generating plots...")
print(f"{'='*60}")

plot_sweep(
    slope_results, "slope_max",
    "slope_max (= slope_x_max = slope_y_max)",
    logx=True,
    save_path=f"{SIM_DIR}/grid_search_slope_max.png",
)

# For knn, filter out 0 (no KNN) for cleaner plot, or handle specially
knn_results_plot = [r for r in knn_results if r["param"] > 0]
plot_sweep(
    knn_results_plot, "knn_k",
    "knn_k (neighbours per source node)",
    logx=True,
    save_path=f"{SIM_DIR}/grid_search_knn_k.png",
)
# Print the no-KNN case separately
no_knn = [r for r in knn_results if r["param"] == 0]
if no_knn:
    r = no_knn[0]
    print(f"  [knn_k=0 (disabled)]: eff={r['efficiency']:.1f}%, fake={r['fake_rate']:.1f}%, kept={r['n_kept']}")

plot_sweep(
    dslope_results, "dslope_max",
    "dslope_max (= dslope_x_max = dslope_y_max)",
    logx=True,
    save_path=f"{SIM_DIR}/grid_search_dslope_max.png",
)

print("\nDone!")
