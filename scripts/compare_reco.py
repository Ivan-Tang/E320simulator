"""
Compare multiple reconstruction methods on the same simulated dataset.

Inputs  (all parquet files)
------
  --tracks      sim_tracks.parquet       (truth)
  --reco        Name=/path/to/reco.parquet Name2=/path/to/reco2.parquet ...

Outputs
-------
  stdout  : summary table

Usage
-----
    python -m scripts.compare_reco \
        --tracks   /data/sim_tracks.parquet   \
        --reco Baseline=/data/sim_baseline.parquet Kalman=/data/sim_kalman_reco.parquet GNN=/data/sim_gnn_reco.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(reco: pl.DataFrame, tracks: pl.DataFrame) -> dict:
    """Compute efficiency, fake rate and quality stats.

    Parameters
    ----------
    reco:
        Output of evaluate_baseline_on_sim / run_gnn_reco.
        Required columns: is_kept, matched_track_id, n_layers, chi2, rms.
    tracks:
        Truth track table with columns: track_id, event_id, is_signal.
    """
    kept = reco.filter(pl.col("is_kept"))

    n_kept   = kept.height
    n_truth  = tracks.height
    n_signal = tracks.filter(pl.col("is_signal")).height

    matched    = kept.filter(pl.col("matched_track_id") >= 0)
    n_matched  = matched.height
    n_fake     = n_kept - n_matched

    # Track-level efficiency per event
    # — for each truth track: was it matched by at least one kept candidate?
    n_unique_matched = matched["matched_track_id"].n_unique()

    efficiency    = n_unique_matched / n_truth * 100 if n_truth  else 0.0
    fake_rate     = n_fake           / n_kept  * 100 if n_kept   else 0.0

    # Signal-only efficiency
    signal_tracks = tracks.filter(pl.col("is_signal"))
    # we need to check which signal track_ids appear in matched
    # join matched with signal tracks on matched_track_id == track_id AND event_id
    if n_signal > 0 and n_matched > 0:
        sig_matched = (
            matched
            .join(
                signal_tracks.select(["event_id", "track_id"]),
                left_on  = ["event_id", "matched_track_id"],
                right_on = ["event_id", "track_id"],
                how = "inner",
            )
        )
        n_sig_matched = sig_matched["matched_track_id"].n_unique()
        signal_eff = n_sig_matched / n_signal * 100
    else:
        n_sig_matched = 0
        signal_eff    = 0.0

    return {
        "n_truth":          n_truth,
        "n_signal":         n_signal,
        "n_kept":           n_kept,
        "n_matched":        n_matched,
        "n_unique_matched": n_unique_matched,
        "n_fake":           n_fake,
        "efficiency_%":     round(efficiency,  2),
        "signal_eff_%":     round(signal_eff,  2),
        "fake_rate_%":      round(fake_rate,   2),
        # quality
        "chi2_median":  float(np.median(kept["chi2"].to_numpy())),
        "chi2_p95":     float(np.percentile(kept["chi2"].to_numpy(), 95)),
        "rms_median_um": float(np.median(kept["rms"].to_numpy()) * 1e3),
        "n_layers_mean": float(kept["n_layers"].mean()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Summary table printer  (N-method)
# ──────────────────────────────────────────────────────────────────────────────

def print_table(method_metrics: dict[str, dict]) -> None:
    names = list(method_metrics.keys())
    rows = [
        ("Truth tracks",        "n_truth",          ""),
        ("Signal tracks",       "n_signal",         ""),
        ("Kept candidates",     "n_kept",            ""),
        ("─" * 22,              None,                ""),
        ("Efficiency",          "efficiency_%",      "%"),
        ("Signal efficiency",   "signal_eff_%",      "%"),
        ("Fake rate",           "fake_rate_%",       "%"),
        ("─" * 22,              None,                ""),
        ("Median χ²",           "chi2_median",       ""),
        ("95th-pct χ²",         "chi2_p95",          ""),
        ("Median RMS",          "rms_median_um",     "µm"),
        ("Mean n_layers",       "n_layers_mean",     ""),
    ]

    w = 24
    col_w = 12
    hdr = f"{'Metric':<{w}}" + "".join(f"  {n:>{col_w}}" for n in names)
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for label, key, unit in rows:
        if key is None:
            print(label)
            continue
        parts = f"{label:<{w}}"
        for n in names:
            v = method_metrics[n][key]
            if isinstance(v, float):
                parts += f"  {v:>{col_w - 2}.2f}{unit:>2}"
            else:
                parts += f"  {v:>{col_w - 2}d}{unit:>2}"
        print(parts)
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def compare(
    tracks_path: str,
    reco_paths: dict[str, str],
) -> dict[str, dict]:
    print("[compare] loading data …")
    tracks = pl.read_parquet(tracks_path)
    print(f"  tracks    : {len(tracks):,} truth tracks")

    # Build ordered dict of method_name → reco DataFrame
    reco_dfs: dict[str, pl.DataFrame] = {}

    for name, path in reco_paths.items():
        reco = pl.read_parquet(path)
        reco_dfs[name] = reco
        print(f"  {name:10}: {len(reco):,} candidates")

    print("[compare] computing metrics …")
    method_metrics: dict[str, dict] = {}
    for name, rdf in reco_dfs.items():
        method_metrics[name] = compute_metrics(rdf, tracks)

    print_table(method_metrics)

    return method_metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reconstruction methods."
    )
    parser.add_argument("--tracks", required=True, help="sim_tracks.parquet")
    parser.add_argument("--reco", nargs='+', required=True, 
                        help="Reconstruction files in format Name=/path/to/file.parquet Name2=/path/to/file2.parquet")
    args = parser.parse_args()

    reco_paths = {}
    for item in args.reco:
        if "=" not in item:
            print(f"Error: --reco argument '{item}' must be in Name=Path format.")
            sys.exit(1)
        name, path = item.split("=", 1)
        reco_paths[name] = path

    compare(args.tracks, reco_paths)


if __name__ == "__main__":
    _cli()
