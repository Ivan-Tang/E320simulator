"""
Compare baseline, Kalman, and GNN reconstruction on the same simulated dataset.

Inputs  (all parquet files)
------
  --baseline    sim_baseline.parquet     (from run_baseline_on_sim.py)
  --kalman      sim_kalman_reco.parquet  (from run_kalman_on_sim.py)
  --gnn         sim_gnn_reco.parquet     (from run_gnn_on_sim.py)
  --tracks      sim_tracks.parquet       (truth)

Outputs
-------
  stdout  : summary table
  figures : saved to --outdir (default: same dir as --baseline)
    comparison_summary.png
    chi2_rms_distributions.png
    per_event_efficiency.png

Usage
-----
    python -m scripts.compare_reco \\
        --baseline /data/sim_baseline.parquet \\
        --kalman   /data/sim_kalman_reco.parquet \\
        --gnn      /data/sim_gnn_reco.parquet \\
        --tracks   /data/sim_tracks.parquet   \\
        --outdir   /data/plots/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(reco: pl.DataFrame, tracks: pl.DataFrame) -> dict:
    """Compute efficiency, fake rate, clone rate and quality stats.

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

    # Duplicate / clone count: same matched_track_id appears > 1 time
    if n_matched > 0:
        clone_counts = (
            matched
            .group_by("matched_track_id")
            .agg(pl.len().alias("cnt"))
            .filter(pl.col("cnt") > 1)
        )
        n_clones = int((clone_counts["cnt"] - 1).sum())
    else:
        n_clones = 0

    # Track-level efficiency per event
    # — for each truth track: was it matched by at least one kept candidate?
    matched_tids = set(matched["matched_track_id"].to_list())
    truth_tids   = set(
        tracks.select(["event_id", "track_id"])
              .with_columns((pl.col("event_id").cast(str) + "_" + pl.col("track_id").cast(str)).alias("key"))
              ["key"].to_list()
    )
    # Simpler: use unique track_ids in matched
    n_unique_matched = matched["matched_track_id"].n_unique()

    efficiency    = n_unique_matched / n_truth * 100 if n_truth  else 0.0
    fake_rate     = n_fake           / n_kept  * 100 if n_kept   else 0.0
    clone_rate    = n_clones         / n_kept  * 100 if n_kept   else 0.0

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
        "n_clones":         n_clones,
        "efficiency_%":     round(efficiency,  2),
        "signal_eff_%":     round(signal_eff,  2),
        "fake_rate_%":      round(fake_rate,   2),
        "clone_rate_%":     round(clone_rate,  2),
        # quality
        "chi2_median":  float(np.median(kept["chi2"].to_numpy())),
        "chi2_p95":     float(np.percentile(kept["chi2"].to_numpy(), 95)),
        "rms_median_um": float(np.median(kept["rms"].to_numpy()) * 1e3),
        "n_layers_mean": float(kept["n_layers"].mean()),
    }


def per_event_efficiency(reco: pl.DataFrame, tracks: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame with per-event efficiency and fake counts."""
    kept    = reco.filter(pl.col("is_kept"))
    matched = kept.filter(pl.col("matched_track_id") >= 0)

    n_truth_per_event = (
        tracks.group_by("event_id")
              .agg(pl.len().alias("n_truth"))
    )
    n_matched_per_event = (
        matched.group_by("event_id")
               .agg(pl.col("matched_track_id").n_unique().alias("n_matched"))
    )
    n_fake_per_event = (
        kept.group_by("event_id")
            .agg(
                pl.len().alias("n_kept"),
                (pl.col("matched_track_id") < 0).sum().alias("n_fake"),
            )
    )

    summary = (
        n_truth_per_event
        .join(n_matched_per_event, on="event_id", how="left")
        .join(n_fake_per_event,    on="event_id", how="left")
        .with_columns([
            pl.col("n_matched").fill_null(0),
            pl.col("n_kept").fill_null(0),
            pl.col("n_fake").fill_null(0),
        ])
        .with_columns(
            (pl.col("n_matched") / pl.col("n_truth") * 100).alias("eff_%")
        )
        .sort("event_id")
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (N-method comparison)
# ──────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "Baseline": "#4878CF",
    "Kalman":   "#8172B2",
    "GNN":      "#D65F5F",
}
_MARKERS = {"Baseline": "o", "Kalman": "D", "GNN": "s"}


def _bar_comparison(ax, names, values, title, ylabel, fmt=".1f"):
    n = len(names)
    w = min(0.6, 0.8 / n)
    bars = ax.bar(names, values, color=[_COLORS.get(n_, "#999999") for n_ in names], width=w)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)


def plot_summary(method_metrics: dict[str, dict], outpath: str) -> None:
    names = list(method_metrics.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Reconstruction comparison: " + " vs ".join(names),
                 fontsize=14, fontweight="bold")

    pairs = [
        ("efficiency_%",   "Track efficiency (%)",  "Efficiency (%)"),
        ("signal_eff_%",   "Signal efficiency (%)", "Efficiency (%)"),
        ("fake_rate_%",    "Fake rate (%)",          "Fake rate (%)"),
        ("clone_rate_%",   "Clone rate (%)",         "Clone rate (%)"),
        ("chi2_median",    "Median χ²",              "χ²"),
        ("rms_median_um",  "Median RMS (µm)",        "µm"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, pairs):
        vals = [method_metrics[n][key] for n in names]
        _bar_comparison(ax, names, vals, title, ylabel)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


def plot_distributions(kept_by_method: dict[str, pl.DataFrame], outpath: str) -> None:
    names = list(kept_by_method.keys())

    chi2_data = {n: kept_by_method[n]["chi2"].to_numpy() for n in names}
    rms_data  = {n: kept_by_method[n]["rms"].to_numpy() * 1e3 for n in names}
    nl_data   = {n: kept_by_method[n]["n_layers"].to_numpy() for n in names}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Track quality distributions", fontsize=13)

    kw = dict(bins=50, density=True, alpha=0.55, histtype="stepfilled")
    all_chi2 = np.concatenate(list(chi2_data.values()))
    chi2_max = min(np.percentile(all_chi2, 99), 20)

    for n in names:
        axes[0].hist(chi2_data[n], **kw, color=_COLORS.get(n, "#999999"),
                     label=n, range=(0, chi2_max))
    axes[0].set_xlabel("χ² / dof"); axes[0].set_ylabel("Density")
    axes[0].set_title("χ² distribution"); axes[0].legend()

    all_rms = np.concatenate(list(rms_data.values()))
    rms_max = min(np.percentile(all_rms, 99), 200)
    for n in names:
        axes[1].hist(rms_data[n], **kw, color=_COLORS.get(n, "#999999"),
                     label=n, range=(0, rms_max))
    axes[1].set_xlabel("RMS (µm)"); axes[1].set_ylabel("Density")
    axes[1].set_title("Residual RMS distribution"); axes[1].legend()

    all_nl = sorted(set(v for arr in nl_data.values() for v in arr.tolist()))
    x_pos = np.arange(len(all_nl))
    n_m = len(names)
    w = min(0.8 / n_m, 0.35)
    for mi, n in enumerate(names):
        cnt = np.array([(nl_data[n] == v).sum() for v in all_nl])
        frac = cnt / cnt.sum() * 100 if cnt.sum() > 0 else cnt
        offset = (mi - (n_m - 1) / 2) * w
        axes[2].bar(x_pos + offset, frac, w, color=_COLORS.get(n, "#999999"), label=n)
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(all_nl)
    axes[2].set_xlabel("n_layers"); axes[2].set_ylabel("Fraction (%)")
    axes[2].set_title("Hits per track"); axes[2].legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


def plot_per_event(ev_by_method: dict[str, pl.DataFrame], outpath: str) -> None:
    names = list(ev_by_method.keys())

    # Start from the first method and successively join the rest
    base_name = names[0]
    joined = (
        ev_by_method[base_name]
        .select(["event_id", "eff_%", "n_fake"])
        .rename({"eff_%": f"eff_{base_name}", "n_fake": f"fake_{base_name}"})
    )
    for n in names[1:]:
        joined = joined.join(
            ev_by_method[n]
            .select(["event_id", "eff_%", "n_fake"])
            .rename({"eff_%": f"eff_{n}", "n_fake": f"fake_{n}"}),
            on="event_id", how="full",
        ).fill_null(0)
    joined = joined.sort("event_id")

    eids = joined["event_id"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Per-event comparison", fontsize=13)

    for n in names:
        c = _COLORS.get(n, "#999999")
        m = _MARKERS.get(n, "o")
        eff = joined[f"eff_{n}"].to_numpy()
        fk  = joined[f"fake_{n}"].to_numpy()
        axes[0].plot(eids, eff, f"{m}-", ms=3, lw=0.8, color=c, label=n)
        axes[0].axhline(np.mean(eff), ls="--", lw=1, color=c, alpha=0.7)
        axes[1].plot(eids, fk,  f"{m}-", ms=3, lw=0.8, color=c, label=n)

    axes[0].set_ylabel("Efficiency (%)"); axes[0].set_ylim(-5, 115)
    axes[0].legend(); axes[0].set_title("Track efficiency per event")
    axes[1].set_xlabel("event_id"); axes[1].set_ylabel("# fake tracks")
    axes[1].legend(); axes[1].set_title("Fake tracks per event")

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


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
        ("Clone rate",          "clone_rate_%",      "%"),
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
    baseline_path: str,
    tracks_path:   str,
    kalman_path:   str | None = None,
    gnn_path:      str | None = None,
    outdir:        str | None = None,
) -> dict[str, dict]:
    print("[compare] loading data …")
    tracks = pl.read_parquet(tracks_path)
    print(f"  tracks    : {len(tracks):,} truth tracks")

    # Build ordered dict of method_name → reco DataFrame
    reco_dfs: dict[str, pl.DataFrame] = {}

    bl = pl.read_parquet(baseline_path)
    reco_dfs["Baseline"] = bl
    print(f"  baseline  : {len(bl):,} candidates")

    if kalman_path is not None:
        km = pl.read_parquet(kalman_path)
        reco_dfs["Kalman"] = km
        print(f"  kalman    : {len(km):,} candidates")

    if gnn_path is not None:
        gnn = pl.read_parquet(gnn_path)
        reco_dfs["GNN"] = gnn
        print(f"  gnn       : {len(gnn):,} candidates")

    print("[compare] computing metrics …")
    method_metrics: dict[str, dict] = {}
    for name, rdf in reco_dfs.items():
        method_metrics[name] = compute_metrics(rdf, tracks)

    print_table(method_metrics)

    if outdir is None:
        outdir = str(Path(baseline_path).parent)
    os.makedirs(outdir, exist_ok=True)
    print(f"[compare] saving figures to {outdir}/")

    plot_summary(method_metrics,
                 os.path.join(outdir, "comparison_summary.png"))

    kept_by_method = {n: rdf.filter(pl.col("is_kept")) for n, rdf in reco_dfs.items()}
    plot_distributions(
        kept_by_method,
        os.path.join(outdir, "chi2_rms_distributions.png"),
    )

    ev_by_method = {n: per_event_efficiency(rdf, tracks) for n, rdf in reco_dfs.items()}
    plot_per_event(ev_by_method,
                   os.path.join(outdir, "per_event_efficiency.png"))

    return method_metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reconstruction methods (Baseline, Kalman, GNN)"
    )
    parser.add_argument("--baseline", required=True, help="sim_baseline.parquet")
    parser.add_argument("--kalman",   default=None,  help="sim_kalman_reco.parquet")
    parser.add_argument("--gnn",      default=None,  help="sim_gnn_reco.parquet")
    parser.add_argument("--tracks",   required=True, help="sim_tracks.parquet")
    parser.add_argument("--outdir",   default=None,  help="Directory for figures")
    args = parser.parse_args()

    compare(
        args.baseline, args.tracks,
        kalman_path=args.kalman, gnn_path=args.gnn,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    _cli()
