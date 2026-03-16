import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from collections import Counter
import time
import matplotlib.pyplot as plt

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hough_baseline import HoughConfig, _process_event_hough
from src.baseline import (
    BaselineConfig,
    _build_edges,
    _build_chains,
    _fit_and_score,
    _shared_hit_rejection,
)
from src.kalman_tracker import KalmanConfig, _process_event_kalman
from src.train import TrainConfig, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build GNN tensors for one event  (from run_gnn_on_sim.py)
# ──────────────────────────────────────────────────────────────────────────────

def _event_to_tensors(
    x, y, z, lid, nid, size_x, size_y, size,
    e_src, e_dst, e_sl, e_sx, e_sy,
    nid_to_local,
):
    """Build (node_feat, edge_index, edge_feat) tensors for one event."""
    all_nids = np.unique(np.concatenate([e_src, e_dst]))
    n_nodes = len(all_nids)
    nid_to_row = {int(n): i for i, n in enumerate(all_nids)}

    node_feat = np.empty((n_nodes, 7), dtype=np.float32)
    for row_i, gid in enumerate(all_nids):
        li = nid_to_local[int(gid)]
        node_feat[row_i] = [lid[li], x[li], y[li], z[li],
                            size_x[li], size_y[li], size[li]]

    src_local = np.array([nid_to_row[int(s)] for s in e_src], dtype=np.int64)
    dst_local = np.array([nid_to_row[int(d)] for d in e_dst], dtype=np.int64)

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
# Per-event timing:  Baseline
# ──────────────────────────────────────────────────────────────────────────────

def time_baseline_per_event(
    clusters_df: pl.DataFrame,
    cfg: BaselineConfig,
) -> list[float]:
    """Measure per-event timing for baseline method."""
    
    event_times = []
    
    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr = clusters_df["x_trk_mm"].to_numpy()
    y_arr = clusters_df["y_trk_mm"].to_numpy()
    z_arr = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv = x_arr[s : s + c_]
        yv = y_arr[s : s + c_]
        zv = z_arr[s : s + c_]
        lv = lid_arr[s : s + c_]
        nv = nid_arr[s : s + c_]

        t_start = time.perf_counter()
        
        src, dst, sl, dl, sx, sy = _build_edges(xv, yv, zv, lv, nv, cfg)
        if len(src) == 0:
            t_elapsed = time.perf_counter() - t_start
            event_times.append(t_elapsed * 1000)  # convert to ms
            continue
            
        chains = _build_chains(src, dst, sl, dl, sx, sy, cfg)
        if not chains:
            t_elapsed = time.perf_counter() - t_start
            event_times.append(t_elapsed * 1000)
            continue
            
        nid_to_local = {int(n): j for j, n in enumerate(nv)}
        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)
        
        t_elapsed = time.perf_counter() - t_start
        event_times.append(t_elapsed * 1000)  # convert to ms

    return event_times


def time_hough_per_event(
    clusters_df: pl.DataFrame,
    cfg: HoughConfig,
) -> list[float]:
    """Measure per-event timing for Hough method."""
    
    event_times = []
    
    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr = clusters_df["x_trk_mm"].to_numpy()
    y_arr = clusters_df["y_trk_mm"].to_numpy()
    z_arr = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv = x_arr[s : s + c_]
        yv = y_arr[s : s + c_]
        zv = z_arr[s : s + c_]
        lv = lid_arr[s : s + c_]
        nv = nid_arr[s : s + c_]

        t_start = time.perf_counter()
        candidates = _process_event_hough(eid, xv, yv, zv, lv, nv, cfg)
        t_elapsed = time.perf_counter() - t_start
        
        event_times.append(t_elapsed * 1000)  # convert to ms

    return event_times


# ──────────────────────────────────────────────────────────────────────────────
# Per-event timing:  Kalman
# ──────────────────────────────────────────────────────────────────────────────

def time_kalman_per_event(
    clusters_df: pl.DataFrame,
    cfg: KalmanConfig,
) -> list[float]:
    """Measure per-event timing for Kalman filter method."""

    event_times = []

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr = clusters_df["x_trk_mm"].to_numpy()
    y_arr = clusters_df["y_trk_mm"].to_numpy()
    z_arr = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        eid = int(unique_events[i])
        xv = x_arr[s : s + c_]
        yv = y_arr[s : s + c_]
        zv = z_arr[s : s + c_]
        lv = lid_arr[s : s + c_]
        nv = nid_arr[s : s + c_]

        t_start = time.perf_counter()
        candidates = _process_event_kalman(eid, xv, yv, zv, lv, nv, cfg)
        t_elapsed = time.perf_counter() - t_start

        event_times.append(t_elapsed * 1000)  # convert to ms

    return event_times


# ──────────────────────────────────────────────────────────────────────────────
# Per-event timing:  GNN
# ──────────────────────────────────────────────────────────────────────────────

def time_gnn_per_event(
    clusters_df: pl.DataFrame,
    checkpoint_path: str,
    baseline_cfg: BaselineConfig,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[float]:
    """Measure per-event timing for GNN-seeded method.

    Timing covers: edge building → GNN scoring → edge filtering →
    chain building → fit → shared-hit rejection  (the full GNN pipeline).
    Model loading is NOT included in per-event time.
    """

    # ── load model (outside timing loop) ─────────────────────────────────────
    ckpt = load_checkpoint(checkpoint_path, device=device)
    model     = ckpt["model"].to(device)
    node_mean = torch.as_tensor(ckpt["node_mean"], device=device)
    node_std  = torch.as_tensor(ckpt["node_std"],  device=device)
    edge_mean = torch.as_tensor(ckpt["edge_mean"], device=device)
    edge_std  = torch.as_tensor(ckpt["edge_std"],  device=device)
    model.eval()
    print(f"  [gnn] checkpoint loaded  epoch={ckpt['epoch']}  best_AP={ckpt['best_ap']:.4f}")

    # ── arrays ────────────────────────────────────────────────────────────────
    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr   = clusters_df["x_trk_mm"].to_numpy()
    y_arr   = clusters_df["y_trk_mm"].to_numpy()
    z_arr   = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    sx_arr  = clusters_df["size_x"].to_numpy()
    sy_arr  = clusters_df["size_y"].to_numpy()
    s_arr   = clusters_df["size"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    event_times: list[float] = []

    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])

        xv  = x_arr [s: s + c_]
        yv  = y_arr [s: s + c_]
        zv  = z_arr [s: s + c_]
        lv  = lid_arr[s: s + c_]
        nv  = nid_arr[s: s + c_]
        sxv = sx_arr [s: s + c_]
        syv = sy_arr [s: s + c_]
        sv  = s_arr  [s: s + c_]

        nid_to_local: dict[int, int] = {int(n): j for j, n in enumerate(nv)}

        t_start = time.perf_counter()

        # 1. candidate edges (physics pruning)
        e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(
            xv, yv, zv, lv, nv, baseline_cfg
        )
        if len(e_src) == 0:
            event_times.append((time.perf_counter() - t_start) * 1000)
            continue

        # 2. GNN scoring
        nf, ei, ef = _event_to_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sl, e_sx, e_sy,
            nid_to_local,
        )
        nf = (nf.to(device) - node_mean) / node_std
        ef = (ef.to(device) - edge_mean) / edge_std

        with torch.no_grad():
            scores = model(nf, ei.to(device), ef).cpu().numpy()

        # 3. edge filter
        mask = scores >= threshold
        if not mask.any():
            event_times.append((time.perf_counter() - t_start) * 1000)
            continue

        f_src = e_src[mask]
        f_dst = e_dst[mask]
        f_sl  = e_sl [mask]
        f_dl  = e_dl [mask]
        f_sx  = e_sx [mask]
        f_sy  = e_sy [mask]

        # 4. chain building → fit → shared-hit rejection
        chains = _build_chains(f_src, f_dst, f_sl, f_dl, f_sx, f_sy, baseline_cfg)
        if not chains:
            event_times.append((time.perf_counter() - t_start) * 1000)
            continue

        candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
        candidates = _shared_hit_rejection(candidates)

        t_elapsed = time.perf_counter() - t_start
        event_times.append(t_elapsed * 1000)

    return event_times


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_timing(
    data_dir: str,
    suffix: str = "test",
    checkpoint_path: str | None = None,
    gnn_threshold: float = 0.5,
    gnn_device: str = "cpu",
):
    """Compare timing between baseline, Hough, Kalman, and GNN methods."""
    
    print(f"\n{'='*70}")
    print(f"  Timing comparison on {suffix} set")
    print(f"{'='*70}\n")
    
    # Load data
    clusters_df = pl.read_parquet(
        os.path.join(data_dir, f"sim_clusters_{suffix}.parquet")
    )
    
    n_events = clusters_df["event_id"].n_unique()
    print(f"Loading {n_events} events...")
    
    # Time baseline method
    print("\nTiming baseline method...")
    baseline_cfg = BaselineConfig()
    baseline_times = time_baseline_per_event(clusters_df, baseline_cfg)
    
    # Time Hough method
    print("Timing Hough method...")
    hough_cfg = HoughConfig()
    hough_times = time_hough_per_event(clusters_df, hough_cfg)

    # Time Kalman method
    #print("Timing Kalman method...")
    #kalman_cfg = KalmanConfig()
    #kalman_times = time_kalman_per_event(clusters_df, kalman_cfg)

    # Time GNN method
    run_gnn = checkpoint_path is not None
    if run_gnn:
        print("Timing GNN method...")
        gnn_times = time_gnn_per_event(
            clusters_df, checkpoint_path, baseline_cfg,
            threshold=gnn_threshold, device=gnn_device,
        )
    
    # ── Compute statistics ────────────────────────────────────────────────────
    baseline_times = np.array(baseline_times)
    hough_times = np.array(hough_times)
    #kalman_times = np.array(kalman_times)
    if run_gnn:
        gnn_times = np.array(gnn_times)
    
    def _print_stats(name: str, times: np.ndarray):
        print(f"\n{name}:")
        print(f"  Mean time per event:   {times.mean():.3f} ms")
        print(f"  Median time per event: {np.median(times):.3f} ms")
        print(f"  Std dev:               {times.std():.3f} ms")
        print(f"  Min:                   {times.min():.3f} ms")
        print(f"  Max:                   {times.max():.3f} ms")

    print(f"\n{'='*70}")
    print(f"  Results Summary")
    print(f"{'='*70}")

    _print_stats("Baseline method", baseline_times)
    _print_stats("Hough method", hough_times)
    #_print_stats("Kalman method", kalman_times)
    if run_gnn:
        _print_stats("GNN method", gnn_times)

    # ── speedup table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Speedup table (mean time)")
    print(f"{'='*70}")
    bl_mean = baseline_times.mean()
    hg_mean = hough_times.mean()
    #km_mean = kalman_times.mean()
    print(f"\n  Hough  vs Baseline : {bl_mean / hg_mean:.2f}x")
    #print(f"  Kalman vs Baseline : {bl_mean / km_mean:.2f}x")
    #print(f"  Kalman vs Hough    : {hg_mean / km_mean:.2f}x")
    if run_gnn:
        gn_mean = gnn_times.mean()
        print(f"  GNN    vs Baseline : {bl_mean / gn_mean:.2f}x")
        print(f"  GNN    vs Hough    : {hg_mean / gn_mean:.2f}x")
        #print(f"  GNN    vs Kalman   : {km_mean / gn_mean:.2f}x")

    # ── build list of all methods for plotting ────────────────────────────────
    method_names  = ["Baseline", "Hough"]
    method_times  = [baseline_times, hough_times]
    method_colors = ["#4C72B0", "#DD8452", "#8172B2"]
    method_colors_light = ["#A6C8E0", "#F4C7A3", "#C4B8E0"]
    if run_gnn:
        method_names.append("GNN")
        method_times.append(gnn_times)
        method_colors.append("#55A868")
        method_colors_light.append("#A8D8B0")

    n_methods = len(method_names)

    # ── Create comparison plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histograms
    ax = axes[0, 0]
    all_vals = np.concatenate(method_times)
    bins = np.linspace(all_vals.min(), all_vals.max(), 50)
    for name, t, c in zip(method_names, method_times, method_colors):
        ax.hist(t, bins=bins, alpha=0.55, label=name, color=c, edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12)
    ax.set_title('Distribution of per-event computation time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log-scale histograms
    ax = axes[0, 1]
    for name, t, c in zip(method_names, method_times, method_colors):
        ax.hist(t, bins=bins, alpha=0.55, label=name, color=c, edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Distribution (log scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    for name, t, c in zip(method_names, method_times, method_colors):
        s_ = np.sort(t)
        cdf = np.arange(1, len(s_) + 1) / len(s_)
        ax.plot(s_, cdf, label=name, color=c, linewidth=2)
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)
    ax.set_title('Cumulative distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plot comparison
    ax = axes[1, 1]
    bp = ax.boxplot(
        method_times,
        tick_labels=method_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='green', markersize=8),
    )
    for box, c in zip(bp['boxes'], method_colors_light):
        box.set_facecolor(c)
    ax.set_ylabel('Time per event (ms)', fontsize=12)
    ax.set_title('Box plot comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(data_dir, f'timing_comparison_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    #plt.show()
    
    # Save timing data
    timing_data = {
        'event_index': range(len(baseline_times)),
        'baseline_time_ms': baseline_times,
        'hough_time_ms': hough_times,
        #'kalman_time_ms': kalman_times,
    }
    if run_gnn:
        timing_data['gnn_time_ms'] = gnn_times
    timing_df = pl.DataFrame(timing_data)
    timing_path = os.path.join(data_dir, f'timing_data_{suffix}.parquet')
    timing_df.write_parquet(timing_path)
    print(f"Timing data saved to: {timing_path}")


if __name__ == "__main__":
    data_dir        = "/Users/IvanTang/hep/data_Run502/simulation/"
    checkpoint_path = "/Users/IvanTang/hep/data_Run502/runs/gnn/best_model.pt"
    
    # Compare on test set (can also run on 'train')
    compare_timing(data_dir, suffix="test", checkpoint_path=checkpoint_path)
