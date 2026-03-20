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
        --reco Baseline=/data/sim_baseline.parquet Hough=/data/sim_hough_reco.parquet GNN=/data/sim_gnn_reco.parquet
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
        Output of evaluate_baseline_on_sim / run_edge_classifier_reco.
        Required columns: is_kept, matched_track_id, n_layers, chi2, rms.
    tracks:
        Truth track table with columns: track_id, event_id, is_signal.
    """
    n_truth  = tracks.height

    if reco.is_empty() or "is_kept" not in reco.columns:
        return {
            "n_truth": n_truth, "n_kept": 0,
            "match_5": 0, "match_4": 0, "match_3": 0,
            "match_2": 0, "match_1": 0, "match_0": 0,
            "n_matched": 0, "n_fake": 0,
            "efficiency_%": 0.0, "fake_rate_%": 0.0, "f1_score_%": 0.0,
            "chi2_mean": float("nan"), "rms_mean_um": float("nan"),
        }

    kept = reco.filter(pl.col("is_kept"))
    n_kept = kept.height

    matched   = kept.filter(pl.col("matched_track_id") >= 0)
    n_matched = matched.height
    n_fake    = n_kept - n_matched

    # Track-level efficiency: how many unique truth tracks were matched
    n_unique_matched = matched["matched_track_id"].n_unique()
    efficiency = n_unique_matched / n_truth * 100 if n_truth else 0.0
    fake_rate  = n_fake / n_kept * 100 if n_kept else 0.0

    precision = n_matched / n_kept if n_kept else 0.0
    recall    = n_unique_matched / n_truth if n_truth else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) * 100 if (precision + recall) else 0.0

    # Matching degree: count kept candidates by n_matched hits
    nm_arr = kept["n_matched"].to_numpy() if "n_matched" in kept.columns else np.zeros(n_kept, dtype=int)
    match_counts = {k: int(np.sum(nm_arr == k)) for k in range(6)}

    return {
        "n_truth":      n_truth,
        "n_kept":       n_kept,
        "match_5":      match_counts[5],
        "match_4":      match_counts[4],
        "match_3":      match_counts[3],
        "match_2":      match_counts[2],
        "match_1":      match_counts[1],
        "match_0":      match_counts[0],
        "n_matched":    n_matched,
        "n_fake":       n_fake,
        "efficiency_%": round(efficiency, 2),
        "fake_rate_%":  round(fake_rate,  2),
        "f1_score_%":   round(f1_score,   2),
        # fit quality
        "chi2_mean":   float(np.mean(kept["chi2"].to_numpy())),
        "rms_mean_um": float(np.mean(kept["rms"].to_numpy()) * 1e3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Summary table printer  (N-method)
# ──────────────────────────────────────────────────────────────────────────────

def print_table(method_metrics: dict[str, dict]) -> None:
    names = list(method_metrics.keys())
    rows = [
        ("Truth tracks",        "n_truth",          ""),
        ("Kept candidates",     "n_kept",           ""),
        ("─" * 22,              None,               ""),
        ("  5-hit match",       "match_5",          ""),
        ("  4-hit match",       "match_4",          ""),
        ("  3-hit match",       "match_3",          ""),
        ("  2-hit match",       "match_2",          ""),
        ("  1-hit match",       "match_1",          ""),
        ("  0-hit match",       "match_0",          ""),
        ("─" * 22,              None,               ""),
        ("Efficiency",          "efficiency_%",     "%"),
        ("Fake rate",           "fake_rate_%",      "%"),
        ("F1 score",            "f1_score_%",       "%"),
        ("─" * 22,              None,               ""),
        ("Mean χ²",             "chi2_mean",        ""),
        ("Mean RMS",            "rms_mean_um",      "µm"),
    ]

    # Keys that need higher precision display
    HIGH_PREC_KEYS = {"chi2_mean"}

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
                fmt = ".4f" if key in HIGH_PREC_KEYS else ".2f"
                parts += f"  {v:>{col_w - 2}{fmt}}{unit:>2}"
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
