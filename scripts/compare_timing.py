import os
import numpy as np
import polars as pl
from collections import Counter
import time
import matplotlib.pyplot as plt

from src.hough_baseline import HoughConfig, _process_event_hough
from src.baseline import (
    BaselineConfig,
    _build_edges,
    _build_chains,
    _fit_and_score,
    _shared_hit_rejection,
)


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


def compare_timing(data_dir: str, suffix: str = "test"):
    """Compare timing between baseline and Hough methods."""
    
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
    
    # Compute statistics
    baseline_times = np.array(baseline_times)
    hough_times = np.array(hough_times)
    
    print(f"\n{'='*70}")
    print(f"  Results Summary")
    print(f"{'='*70}")
    print(f"\nBaseline method:")
    print(f"  Mean time per event:   {baseline_times.mean():.3f} ms")
    print(f"  Median time per event: {np.median(baseline_times):.3f} ms")
    print(f"  Std dev:               {baseline_times.std():.3f} ms")
    print(f"  Min:                   {baseline_times.min():.3f} ms")
    print(f"  Max:                   {baseline_times.max():.3f} ms")
    
    print(f"\nHough method:")
    print(f"  Mean time per event:   {hough_times.mean():.3f} ms")
    print(f"  Median time per event: {np.median(hough_times):.3f} ms")
    print(f"  Std dev:               {hough_times.std():.3f} ms")
    print(f"  Min:                   {hough_times.min():.3f} ms")
    print(f"  Max:                   {hough_times.max():.3f} ms")
    
    speedup = baseline_times.mean() / hough_times.mean()
    print(f"\nSpeedup factor: {speedup:.2f}x")
    if speedup > 1:
        print(f"  → Hough is {speedup:.2f}x faster than baseline")
    else:
        print(f"  → Baseline is {1/speedup:.2f}x faster than Hough")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histograms
    ax = axes[0, 0]
    bins = np.linspace(
        min(baseline_times.min(), hough_times.min()),
        max(baseline_times.max(), hough_times.max()),
        50
    )
    ax.hist(baseline_times, bins=bins, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
    ax.hist(hough_times, bins=bins, alpha=0.6, label='Hough', color='red', edgecolor='black')
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12)
    ax.set_title('Distribution of per-event computation time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-scale histograms for better visualization
    ax = axes[0, 1]
    ax.hist(baseline_times, bins=bins, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
    ax.hist(hough_times, bins=bins, alpha=0.6, label='Hough', color='red', edgecolor='black')
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Distribution (log scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    baseline_sorted = np.sort(baseline_times)
    hough_sorted = np.sort(hough_times)
    baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
    hough_cdf = np.arange(1, len(hough_sorted) + 1) / len(hough_sorted)
    ax.plot(baseline_sorted, baseline_cdf, label='Baseline', color='blue', linewidth=2)
    ax.plot(hough_sorted, hough_cdf, label='Hough', color='red', linewidth=2)
    ax.set_xlabel('Time per event (ms)', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)
    ax.set_title('Cumulative distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Box plot comparison
    ax = axes[1, 1]
    bp = ax.boxplot(
        [baseline_times, hough_times],
        tick_labels=['Baseline', 'Hough'],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='green', markersize=8)
    )
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Time per event (ms)', fontsize=12)
    ax.set_title('Box plot comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(data_dir, f'timing_comparison_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()
    
    # Save timing data
    timing_df = pl.DataFrame({
        'event_index': range(len(baseline_times)),
        'baseline_time_ms': baseline_times,
        'hough_time_ms': hough_times,
    })
    timing_path = os.path.join(data_dir, f'timing_data_{suffix}.parquet')
    timing_df.write_parquet(timing_path)
    print(f"Timing data saved to: {timing_path}")


if __name__ == "__main__":
    data_dir = "/Users/IvanTang/hep/data_Run502/simulation/"
    
    # Compare on test set (can also run on 'train')
    compare_timing(data_dir, suffix="test")
