"""Scaling study: compare all benchmark models as background / signal varies.

Two sweeps
----------
1. Background scaling : synthetic_bg_n_per_layer varies, mean_n_signal fixed
2. Signal scaling     : mean_n_signal varies, background fixed

All ML models use pre-trained checkpoints from RUNS_DIR (no retraining).
Transformer uses the two-stage hit-filter + MaskFormer pipeline.

Metric
------
F1 = 2 · precision · recall / (precision + recall)
  recall    = efficiency  = n_unique_matched / n_truth
  precision = 1 - fake_rate = n_matched / n_kept

Usage
-----
    cd /Users/IvanTang/hep/E320simulator
    python scripts/compare_scaling.py [--device mps] [--events 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import SimConfig, simulate
from src.baseline import BaselineConfig
from src.hough_baseline import HoughConfig
from src.config import RUNS_DIR, OUTPUTS_DIR
from scripts.run_baseline import evaluate_baseline_on_sim
from scripts.run_hough import evaluate_hough_on_sim
from scripts.run_model import run_edge_classifier_reco, run_trackformer_reco
from scripts.compare_reco import compute_metrics


# ── Model catalogue ───────────────────────────────────────────────────────────

# Non-ML algorithms (no checkpoint needed)
NON_ML_MODELS = ["baseline", "hough"]

# Edge-classification models (one checkpoint each)
EDGE_MODELS = ["mlp", "gnn", "interaction_net", "eggnet", "hgnn"]

# Matches run_benchmark.py ML_MODELS list
ALL_MODELS = NON_ML_MODELS + EDGE_MODELS + ["transformer"]

# Colour palette — one colour per model, consistent across plots
_PALETTE = [
    "#4C72B0",  # baseline
    "#DD8452",  # hough
    "#55A868",  # mlp
    "#C44E52",  # gnn
    "#8172B2",  # interaction_net
    "#8C8C8C",  # eggnet
    "#CCB974",  # hgnn
    "#64B5CD",  # transformer
]
MODEL_STYLE: dict[str, dict] = {
    m: {"color": _PALETTE[i], "marker": ["o","s","^","D","v","P","X","*"][i]}
    for i, m in enumerate(ALL_MODELS)
}


# ── F1 helper ─────────────────────────────────────────────────────────────────

def _f1(metrics: dict) -> float:
    """F1 from efficiency (recall) and fake_rate."""
    recall    = metrics.get("efficiency_%", 0.0) / 100.0
    fake_rate = metrics.get("fake_rate_%",  0.0) / 100.0
    precision = 1.0 - fake_rate
    denom = precision + recall
    if denom < 1e-9:
        return 0.0
    return 2.0 * precision * recall / denom


# ── Inference dispatcher ──────────────────────────────────────────────────────

def _run_one_model(
    model_name: str,
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
    device: str,
    baseline_cfg: BaselineConfig,
    hough_cfg: HoughConfig,
) -> dict:
    """Run inference for one model; return metrics dict (or {} on failure)."""
    try:
        if model_name == "baseline":
            reco = evaluate_baseline_on_sim(clusters_df, tracks_df, baseline_cfg)

        elif model_name == "hough":
            reco = evaluate_hough_on_sim(clusters_df, tracks_df, hough_cfg)

        elif model_name == "transformer":
            hf_ckpt = RUNS_DIR / "transformer" / "hit_filter.pt"
            tf_ckpt = RUNS_DIR / "transformer" / "best_model.pt"
            if not hf_ckpt.exists() or not tf_ckpt.exists():
                return {}
            reco = run_trackformer_reco(
                clusters_df, tracks_df,
                checkpoint_path=str(tf_ckpt),
                hit_filter_checkpoint=str(hf_ckpt),
                hit_filter_threshold=0.1,
                conf_threshold=0.5,
                mask_threshold=0.5,
                min_layers=4,
                device=device,
            )

        else:  # edge-classification models
            ckpt = RUNS_DIR / model_name / "best_model.pt"
            if not ckpt.exists():
                return {}
            reco = run_edge_classifier_reco(
                clusters_df, tracks_df,
                checkpoint_path=str(ckpt),
                threshold=0.1,
                device=device,
            )

        if reco.is_empty():
            return {}
        return compute_metrics(reco, tracks_df)

    except Exception as exc:
        print(f"    [{model_name}] ERROR: {exc}")
        return {}


# ── Single sweep point ────────────────────────────────────────────────────────

def _run_point(
    n_events: int,
    mean_n_signal: float,
    synthetic_bg_n_per_layer: int,
    device: str,
    seed: int = 42,
) -> dict[str, dict]:
    """Simulate one (bg, sig) configuration; run all models; return metrics."""
    cfg = SimConfig(
        n_events=n_events,
        mean_n_signal=mean_n_signal,
        synthetic_bg_n_per_layer=synthetic_bg_n_per_layer,
        background_mode="synthetic",
        cluster_size_mode="fixed",
        seed=seed,
    )
    clusters_df, tracks_df = simulate(cfg)

    if tracks_df.is_empty():
        return {m: {} for m in ALL_MODELS}

    baseline_cfg = BaselineConfig()
    hough_cfg    = HoughConfig()

    results: dict[str, dict] = {}
    for model in ALL_MODELS:
        t0 = time.perf_counter()
        m  = _run_one_model(model, clusters_df, tracks_df,
                            device, baseline_cfg, hough_cfg)
        dt = time.perf_counter() - t0
        f1 = _f1(m) if m else float("nan")
        eff = m.get("efficiency_%", float("nan")) if m else float("nan")
        fr  = m.get("fake_rate_%",  float("nan")) if m else float("nan")
        results[model] = m
        print(f"    {model:12s}  F1={f1:.3f}  eff={eff:.1f}%  fake={fr:.1f}%  ({dt:.1f}s)")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_sweep(
    x_values: list[float | int],
    sweep_results: list[dict[str, dict]],
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric_key, metric_label, ylim) in zip(
        axes,
        [
            ("f1",           "F1 score",      (0, 1)),
            ("efficiency_%", "Efficiency (%)", (0, 105)),
        ],
    ):
        for model in ALL_MODELS:
            ys, xs = [], []
            for xi, point in zip(x_values, sweep_results):
                m = point.get(model, {})
                if not m:
                    continue
                val = _f1(m) if metric_key == "f1" else m.get(metric_key, float("nan"))
                if not np.isnan(val):
                    ys.append(val)
                    xs.append(xi)
            if not xs:
                continue
            style = MODEL_STYLE[model]
            ax.plot(xs, ys,
                    color=style["color"], marker=style["marker"],
                    linewidth=2, markersize=6, label=model.upper())

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_title(f"{metric_label} vs {x_label}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _plot_fake_rate(
    x_values: list[float | int],
    sweep_results: list[dict[str, dict]],
    x_label: str,
    out_path: Path,
) -> None:
    """Separate fake-rate plot (log-scale friendly)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ALL_MODELS:
        ys, xs = [], []
        for xi, point in zip(x_values, sweep_results):
            m = point.get(model, {})
            if not m:
                continue
            val = m.get("fake_rate_%", float("nan"))
            if not np.isnan(val):
                ys.append(val)
                xs.append(xi)
        if not xs:
            continue
        style = MODEL_STYLE[model]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=6, label=model.upper())

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Fake Rate (%)", fontsize=11)
    ax.set_title(f"Fake Rate vs {x_label}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling study: all benchmark models vs bg / signal"
    )
    parser.add_argument("--events",    type=int,   default=200,
                        help="Events per simulation point (default 200)")
    parser.add_argument("--device",    default="mps",
                        choices=["cpu", "cuda", "mps"],
                        help="Device for ML inference (default mps)")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "scaling"),
                        help="Directory for output plots")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    n_events = args.events
    device   = args.device

    # ── Print which checkpoints are available ─────────────────────────────────
    print("Checkpoint availability:")
    for m in EDGE_MODELS:
        p = RUNS_DIR / m / "best_model.pt"
        print(f"  {m:12s}  {'OK' if p.exists() else 'MISSING'}")
    for f in ["hit_filter.pt", "best_model.pt"]:
        p = RUNS_DIR / "transformer" / f
        print(f"  transformer/{f}  {'OK' if p.exists() else 'MISSING'}")

    # ── Sweep 1: Background scaling ───────────────────────────────────────────
    # Total hits/event ≈ 5 layers × bg_per_layer
    # E320 real data ≈ 700/layer → 3500 total
    bg_values    = [0, 100, 300, 500, 700, 1000]
    fixed_signal = 0.5   # ~0.5 tracks/event mean (matches E320 low-signal regime)

    print(f"\n{'='*65}")
    print(f"Sweep 1: Background scaling  (mean_n_signal={fixed_signal}, "
          f"events={n_events})")
    print(f"{'='*65}")

    bg_results: list[dict[str, dict]] = []
    for bg in bg_values:
        total_hits_approx = 5 * bg
        print(f"\n  bg_per_layer={bg}  (~{total_hits_approx} hits/event)")
        res = _run_point(n_events, fixed_signal, bg, device)
        bg_results.append(res)

    _plot_sweep(
        x_values=[5 * b for b in bg_values],   # convert to total hits/event
        sweep_results=bg_results,
        x_label="Background hits per event",
        title=f"Scaling with background  (mean_n_signal={fixed_signal})",
        out_path=out_dir / "bg_scaling_f1_eff.png",
    )
    _plot_fake_rate(
        x_values=[5 * b for b in bg_values],
        sweep_results=bg_results,
        x_label="Background hits per event",
        out_path=out_dir / "bg_scaling_fake_rate.png",
    )

    # ── Sweep 2: Signal scaling ───────────────────────────────────────────────
    sig_values = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
    fixed_bg   = 700   # per layer ≈ 3500 total, matching real E320 background

    print(f"\n{'='*65}")
    print(f"Sweep 2: Signal scaling  (bg_per_layer={fixed_bg}, "
          f"events={n_events})")
    print(f"{'='*65}")

    sig_results: list[dict[str, dict]] = []
    for sig in sig_values:
        print(f"\n  mean_n_signal={sig}")
        res = _run_point(n_events, sig, fixed_bg, device)
        sig_results.append(res)

    _plot_sweep(
        x_values=sig_values,
        sweep_results=sig_results,
        x_label="Mean signal tracks per event",
        title=f"Scaling with signal tracks  (bg_per_layer={fixed_bg})",
        out_path=out_dir / "sig_scaling_f1_eff.png",
    )
    _plot_fake_rate(
        x_values=sig_values,
        sweep_results=sig_results,
        x_label="Mean signal tracks per event",
        out_path=out_dir / "sig_scaling_fake_rate.png",
    )

    print(f"\nAll done. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
