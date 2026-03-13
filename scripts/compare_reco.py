"""
Compare baseline vs GNN reconstruction on the same simulated dataset.

Inputs  (all parquet files)
------
  --baseline    sim_baseline.parquet     (from run_baseline_on_sim.py)
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
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_COLORS = {"Baseline": "#4878CF", "GNN": "#D65F5F"}


def _bar_comparison(ax, names, values, title, ylabel, fmt=".1f"):
    bars = ax.bar(names, values, color=[_COLORS[n] for n in names], width=0.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.25)


def plot_summary(metrics_bl: dict, metrics_gnn: dict, outpath: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Baseline vs GNN reconstruction comparison", fontsize=14, fontweight="bold")

    pairs = [
        ("efficiency_%",   "Track efficiency (%)",  "Efficiency (%)"),
        ("signal_eff_%",   "Signal efficiency (%)", "Efficiency (%)"),
        ("fake_rate_%",    "Fake rate (%)",          "Fake rate (%)"),
        ("clone_rate_%",   "Clone rate (%)",         "Clone rate (%)"),
        ("chi2_median",    "Median χ²",              "χ²"),
        ("rms_median_um",  "Median RMS (µm)",        "µm"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, pairs):
        vals = [metrics_bl[key], metrics_gnn[key]]
        _bar_comparison(ax, ["Baseline", "GNN"], vals, title, ylabel)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


def plot_distributions(kept_bl: pl.DataFrame, kept_gnn: pl.DataFrame, outpath: str) -> None:
    chi2_bl  = kept_bl["chi2"].to_numpy()
    chi2_gnn = kept_gnn["chi2"].to_numpy()
    rms_bl   = kept_bl["rms"].to_numpy() * 1e3    # µm
    rms_gnn  = kept_gnn["rms"].to_numpy() * 1e3

    nl_bl  = kept_bl["n_layers"].to_numpy()
    nl_gnn = kept_gnn["n_layers"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Track quality distributions", fontsize=13)

    kw = dict(bins=50, density=True, alpha=0.65, histtype="stepfilled")
    chi2_max = min(np.percentile(np.concatenate([chi2_bl, chi2_gnn]), 99), 20)

    axes[0].hist(chi2_bl,  **kw, color=_COLORS["Baseline"], label="Baseline",
                 range=(0, chi2_max))
    axes[0].hist(chi2_gnn, **kw, color=_COLORS["GNN"],      label="GNN",
                 range=(0, chi2_max))
    axes[0].set_xlabel("χ² / dof"); axes[0].set_ylabel("Density")
    axes[0].set_title("χ² distribution"); axes[0].legend()

    rms_max = min(np.percentile(np.concatenate([rms_bl, rms_gnn]), 99), 200)
    axes[1].hist(rms_bl,  **kw, color=_COLORS["Baseline"], label="Baseline",
                 range=(0, rms_max))
    axes[1].hist(rms_gnn, **kw, color=_COLORS["GNN"],      label="GNN",
                 range=(0, rms_max))
    axes[1].set_xlabel("RMS (µm)"); axes[1].set_ylabel("Density")
    axes[1].set_title("Residual RMS distribution"); axes[1].legend()

    nl_vals = sorted(set(nl_bl.tolist() + nl_gnn.tolist()))
    x = np.arange(len(nl_vals))
    w = 0.35
    cnt_bl  = np.array([(nl_bl  == v).sum() for v in nl_vals])
    cnt_gnn = np.array([(nl_gnn == v).sum() for v in nl_vals])
    axes[2].bar(x - w/2, cnt_bl  / cnt_bl.sum()  * 100, w, color=_COLORS["Baseline"], label="Baseline")
    axes[2].bar(x + w/2, cnt_gnn / cnt_gnn.sum() * 100, w, color=_COLORS["GNN"],      label="GNN")
    axes[2].set_xticks(x); axes[2].set_xticklabels(nl_vals)
    axes[2].set_xlabel("n_layers"); axes[2].set_ylabel("Fraction (%)")
    axes[2].set_title("Hits per track"); axes[2].legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


def plot_per_event(ev_bl: pl.DataFrame, ev_gnn: pl.DataFrame, outpath: str) -> None:
    # align on common events
    joined = (
        ev_bl.select(["event_id", "eff_%", "n_fake"])
             .rename({"eff_%": "eff_bl", "n_fake": "fake_bl"})
             .join(
                 ev_gnn.select(["event_id", "eff_%", "n_fake"])
                       .rename({"eff_%": "eff_gnn", "n_fake": "fake_gnn"}),
                 on="event_id", how="full"
             )
             .fill_null(0)
             .sort("event_id")
    )

    eids    = joined["event_id"].to_numpy()
    eff_bl  = joined["eff_bl"].to_numpy()
    eff_gnn = joined["eff_gnn"].to_numpy()
    fk_bl   = joined["fake_bl"].to_numpy()
    fk_gnn  = joined["fake_gnn"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Per-event comparison", fontsize=13)

    axes[0].plot(eids, eff_bl,  "o-", ms=3, lw=0.8, color=_COLORS["Baseline"], label="Baseline")
    axes[0].plot(eids, eff_gnn, "s-", ms=3, lw=0.8, color=_COLORS["GNN"],      label="GNN")
    axes[0].axhline(np.mean(eff_bl),  ls="--", lw=1, color=_COLORS["Baseline"], alpha=0.7)
    axes[0].axhline(np.mean(eff_gnn), ls="--", lw=1, color=_COLORS["GNN"],      alpha=0.7)
    axes[0].set_ylabel("Efficiency (%)"); axes[0].set_ylim(-5, 115)
    axes[0].legend(); axes[0].set_title("Track efficiency per event")

    axes[1].plot(eids, fk_bl,  "o-", ms=3, lw=0.8, color=_COLORS["Baseline"], label="Baseline")
    axes[1].plot(eids, fk_gnn, "s-", ms=3, lw=0.8, color=_COLORS["GNN"],      label="GNN")
    axes[1].set_xlabel("event_id"); axes[1].set_ylabel("# fake tracks")
    axes[1].legend(); axes[1].set_title("Fake tracks per event")

    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    print(f"  → {outpath}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ──────────────────────────────────────────────────────────────────────────────

def print_table(metrics_bl: dict, metrics_gnn: dict) -> None:
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
    hdr = f"{'Metric':<{w}}  {'Baseline':>12}  {'GNN':>12}"
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for label, key, unit in rows:
        if key is None:
            print(label)
            continue
        vb  = metrics_bl[key]
        vg  = metrics_gnn[key]
        fmt = ".2f" if isinstance(vb, float) else "d"
        delta = vg - vb if isinstance(vb, float) else vg - vb
        sign  = "+" if delta >= 0 else ""
        print(
            f"{label:<{w}}  {vb:>10.{2 if isinstance(vb,float) else 0}f}{unit:>2}  "
            f"{vg:>10.{2 if isinstance(vg,float) else 0}f}{unit:>2}  "
            f"({sign}{delta:.2f})"
        )
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def compare(
    baseline_path: str,
    gnn_path:      str,
    tracks_path:   str,
    outdir:        str | None = None,
) -> tuple[dict, dict]:
    print("[compare] loading data …")
    bl     = pl.read_parquet(baseline_path)
    gnn    = pl.read_parquet(gnn_path)
    tracks = pl.read_parquet(tracks_path)
    print(f"  baseline  : {len(bl):,} candidates")
    print(f"  gnn       : {len(gnn):,} candidates")
    print(f"  tracks    : {len(tracks):,} truth tracks")

    print("[compare] computing metrics …")
    m_bl  = compute_metrics(bl,  tracks)
    m_gnn = compute_metrics(gnn, tracks)

    print_table(m_bl, m_gnn)

    if outdir is None:
        outdir = str(Path(baseline_path).parent)
    os.makedirs(outdir, exist_ok=True)
    print(f"[compare] saving figures to {outdir}/")

    plot_summary(m_bl, m_gnn,
                 os.path.join(outdir, "comparison_summary.png"))

    plot_distributions(
        bl.filter(pl.col("is_kept")),
        gnn.filter(pl.col("is_kept")),
        os.path.join(outdir, "chi2_rms_distributions.png"),
    )

    ev_bl  = per_event_efficiency(bl,  tracks)
    ev_gnn = per_event_efficiency(gnn, tracks)
    plot_per_event(ev_bl, ev_gnn,
                   os.path.join(outdir, "per_event_efficiency.png"))

    return m_bl, m_gnn


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs GNN reconstruction")
    parser.add_argument("--baseline", required=True, help="sim_baseline.parquet")
    parser.add_argument("--gnn",      required=True, help="sim_gnn_reco.parquet")
    parser.add_argument("--tracks",   required=True, help="sim_tracks.parquet")
    parser.add_argument("--outdir",   default=None,  help="Directory for figures")
    args = parser.parse_args()

    compare(args.baseline, args.gnn, args.tracks, args.outdir)


if __name__ == "__main__":
    _cli()
