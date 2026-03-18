import os
import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Ensure we can import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import SimConfig, simulate, SyntheticBackgroundPool
from src.config import DATA_ROOT, RUNS_DIR, OUTPUTS_DIR
from E320simulator.scripts.run_baseline import evaluate_baseline_on_sim
from E320simulator.scripts.run_hough import evaluate_hough_on_sim
from E320simulator.scripts.run_model import run_model_reco
from E320simulator.scripts.compare_reco import compute_metrics


def run_experiment(
    n_events: int,
    mean_n_signal: float,
    synthetic_bg_n_per_layer: int,
    checkpoint_path: str,
    device: str = "cpu"
) -> tuple[dict, dict, dict]:
    print(f"\n{'='*70}")
    print(f"  Experiment: sig={mean_n_signal}, bg={synthetic_bg_n_per_layer}, n_events={n_events}")
    print(f"{'='*70}")
    
    cfg = SimConfig(
        n_events=n_events,
        mean_n_signal=mean_n_signal,
        synthetic_bg_n_per_layer=synthetic_bg_n_per_layer,
        background_mode="synthetic",
        cluster_size_mode="fixed", 
        seed=42
    )
    
    # 1. Simulate data
    print("[1] Simulating data...")
    t0 = time.time()
    clusters_df, tracks_df = simulate(cfg)
    print(f"Simulation took {time.time() - t0:.2f} s")
    print(f"Clusters: {len(clusters_df)}, Tracks: {len(tracks_df)}")
    
    if len(tracks_df) == 0:
        print("Warning: No tracks generated. Skipping evaluation.")
        return {}, {}, {}

    # 2. Run Baseline
    print("\n[2] Running Baseline reco...")
    t0 = time.time()
    base_res = evaluate_baseline_on_sim(clusters_df, tracks_df)
    base_time = time.time() - t0
    base_metrics = compute_metrics(base_res, tracks_df) if len(base_res) > 0 else {}
    base_metrics["time_s"] = base_time

    # 3. Run Hough
    print("\n[3] Running Hough reco...")
    t0 = time.time()
    hough_res = evaluate_hough_on_sim(clusters_df, tracks_df)
    hough_time = time.time() - t0
    hough_metrics = compute_metrics(hough_res, tracks_df) if len(hough_res) > 0 else {}
    hough_metrics["time_s"] = hough_time

    # 4. Run GNN 
    print("\n[4] Running GNN reco...")
    t0 = time.time()
    if checkpoint_path and os.path.exists(checkpoint_path):
        gnn_res = run_model_reco(
            clusters_df=clusters_df,
            tracks_df=tracks_df,
            mode="edge",
            edge_checkpoint=checkpoint_path,
            edge_threshold=0.5,
            device=device,
        )
        gnn_time = time.time() - t0
        gnn_metrics = compute_metrics(gnn_res, tracks_df) if len(gnn_res) > 0 else {}
        gnn_metrics["time_s"] = gnn_time
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping GNN.")
        gnn_metrics = {}
    
    return base_metrics, hough_metrics, gnn_metrics


def plot_metrics(
    x_values: list, 
    results_list: list[tuple[dict, dict, dict]], 
    x_label: str, 
    output_dir: str, 
    prefix: str
):
    """Generate and save plots for given metrics against the x_values."""
    
    metrics_to_plot = {
        "efficiency_%": "Efficiency (%)",
        "signal_eff_%": "Signal Efficiency (%)",
        "fake_rate_%": "Fake Rate (%)",
        "chi2_median": "Median χ²",
        "rms_median_um": "Median RMS (µm)",
        "time_s": "Reconstruction Time (s)"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    alg_names = ["Baseline", "Hough", "GNN"]
    alg_colors = ["#4C72B0", "#DD8452", "#55A868"]
    alg_markers = ["o", "s", "^"]
    
    for metric_key, metric_name in metrics_to_plot.items():
        plt.figure(figsize=(8, 6))
        
        for alg_idx, (name, color, marker) in enumerate(zip(alg_names, alg_colors, alg_markers)):
            y_values = []
            valid_x = []
            
            for i, res_tuple in enumerate(results_list):
                metrics = res_tuple[alg_idx]
                if metrics and metric_key in metrics:
                    y_values.append(metrics[metric_key])
                    valid_x.append(x_values[i])
            
            if valid_x:
                plt.plot(valid_x, y_values, marker=marker, color=color, linewidth=2, label=name)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} vs {x_label}', fontsize=14, fontweight='bold')
        
        if "eff" in metric_key.lower() or "rate" in metric_key.lower():
            plt.ylim(0, 105)
            
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        safe_metric = metric_key.replace("%", "pct").replace("/", "_")
        out_path = os.path.join(output_dir, f"{prefix}_{safe_metric}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Baseline vs Hough vs GNN scaling")
    parser.add_argument("--events", type=int, default=500, help="Number of events per simulation run")
    parser.add_argument("--device", type=str, default="cpu", help="Device for GNN inference")
    parser.add_argument("--checkpoint", type=str, default=str(RUNS_DIR / "gnn/best_model.pt"), help="Path to GNN model checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR / "plots/"), help="Output directory for plots")
    args = parser.parse_args()
    
    # ── Sweep 1: Background scaling ──────────────────────────────────────────
    bg_values = [0, 200, 400, 600, 800, 1000]
    default_sig = 0.12 # fixed mean_n_signal for bg sweep
    
    print("\n" + "="*70)
    print("  Starting Background Scaling Sweep")
    print("="*70)
    
    bg_results = []
    for bg in bg_values:
        res = run_experiment(
            n_events=args.events, 
            mean_n_signal=default_sig, 
            synthetic_bg_n_per_layer=bg, 
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        bg_results.append(res)
        
    print("\nGenerating Background Scaling plots...")
    plot_metrics(
        x_values=bg_values, 
        results_list=bg_results, 
        x_label="Background Clusters per Layer per Event", 
        output_dir=args.output_dir, 
        prefix="sweep_bg"
    )

    # ── Sweep 2: Signal / NTracks scaling ────────────────────────────────────
    sig_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    default_bg = 700 # fixed bg for sig sweep
    
    print("\n" + "="*70)
    print("  Starting Signal Tracks Scaling Sweep")
    print("="*70)
    
    sig_results = []
    for sig in sig_values:
        res = run_experiment(
            n_events=args.events, 
            mean_n_signal=sig, 
            synthetic_bg_n_per_layer=default_bg, 
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        sig_results.append(res)
        
    print("\nGenerating Track Scaling plots...")
    plot_metrics(
        x_values=sig_values, 
        results_list=sig_results, 
        x_label="Mean Signal Tracks per Event", 
        output_dir=args.output_dir, 
        prefix="sweep_ntracks"
    )
    
    print("\nAll benchmarking complete. Plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
