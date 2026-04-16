"""scripts/analyze_diagnosis.py

读取 diagnose_failures.py 的 per-track parquet 输出，深度分析
  ② model_miss  — 模型对哪类真实边打分低？
  ③ postproc_loss — 后处理在什么条件下丢径迹？

还支持多模型对比（e.g. TransformerEdge vs GNN at 0.1/0.5）。

用法：
  python scripts/analyze_diagnosis.py \\
    --inputs \\
        /storage/.../diagnose_transformer_t01.parquet \\
        /storage/.../diagnose_gnn_t05.parquet \\
        /storage/.../diagnose_gnn_t01.parquet \\
    --labels "TransformerEdge@0.1" "GNN@0.5" "GNN@0.1" \\
    --clusters /storage/.../sim_clusters_test.parquet

单文件模式（快速检查）：
  python scripts/analyze_diagnosis.py \\
    --inputs /storage/.../diagnose_transformer_t01.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_scores_from_reason(reason: str) -> list[float]:
    """Extract edge scores from model_miss reason string.

    Format: "model_miss: 2/4 in-graph true edges scored below threshold; scores=[0.023, 0.067]"
    """
    m = re.search(r"scores=\[([^\]]*)\]", reason)
    if not m:
        return []
    parts = m.group(1).split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                out.append(float(p))
            except ValueError:
                pass
    return out


def _parse_src_layers_from_reason(reason: str) -> list[int]:
    """Extract src_layer indices from graph_miss reason string.

    Format: "graph_miss: 1/4 true edges missing from kNN graph at src_layers [2]"
    """
    m = re.search(r"src_layers\s+\[([^\]]*)\]", reason)
    if not m:
        return []
    parts = m.group(1).split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                out.append(int(p))
            except ValueError:
                pass
    return out


def _percentile_str(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "n/a"
    return (f"min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
            f"p50={np.percentile(arr,50):.3f}  p75={np.percentile(arr,75):.3f}  "
            f"max={arr.max():.3f}")


def _hist_str(arr: np.ndarray, bins: list[float], fmt: str = ".0f") -> str:
    """Simple ASCII histogram using given bin edges."""
    if len(arr) == 0:
        return "  (empty)"
    lines = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = int(np.sum((arr >= lo) & (arr < hi)))
        bar = "█" * min(count, 40)
        lines.append(f"  [{lo:{fmt}}, {hi:{fmt}}): {count:4d}  {bar}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_one(df: pl.DataFrame, label: str, clusters_df: pl.DataFrame | None) -> dict:
    """Analyse one diagnosis parquet.  Returns a summary dict."""
    print(f"\n{'='*72}")
    print(f"  分析：{label}")
    print(f"{'='*72}")

    total      = len(df)
    recon_df   = df.filter(pl.col("is_reconstructible"))
    n_recon_b  = len(recon_df)
    n_success  = int(recon_df.filter(pl.col("failure_category") == 0).height)
    n_graph    = int(recon_df.filter(pl.col("failure_category") == 1).height)
    n_model    = int(recon_df.filter(pl.col("failure_category") == 2).height)
    n_post     = int(recon_df.filter(pl.col("failure_category") == 3).height)
    n_noloss   = int(recon_df.filter(pl.col("failure_category") == -2).height)
    n_loss     = n_recon_b - n_success

    eff = n_success / n_recon_b * 100 if n_recon_b else 0.0

    print(f"\n  可重建径迹: {n_recon_b}  已重建: {n_success}  效率: {eff:.1f}%")
    print(f"  失效 {n_loss} 条  →  "
          f"①graph_miss={n_graph}({n_graph/n_loss*100:.1f}%)  "
          f"②model_miss={n_model}({n_model/n_loss*100:.1f}%)  "
          f"③postproc={n_post}({n_post/n_loss*100:.1f}%)"
          if n_loss else "  无失效")

    # ── ② model_miss 深度分析 ──────────────────────────────────────────────
    model_miss_df = recon_df.filter(pl.col("failure_category") == 2)
    if n_model > 0:
        print(f"\n  ── ② model_miss 深度分析 ({n_model} 条径迹) ──")

        # 解析 missed edge scores
        all_missed_scores: list[float] = []
        layer_miss_counter: Counter = Counter()
        n_layers_counter: Counter = Counter()

        for row in model_miss_df.iter_rows(named=True):
            scores = _parse_scores_from_reason(row["reason"])
            all_missed_scores.extend(scores)
            n_layers_counter[row["n_layers"]] += 1
            # layer miss from src_layers_from_reason not always available in model_miss
            # use n_true_edges vs n_scored_ok as proxy
            n_miss = row["n_true_edges"] - (row["n_scored_ok"] or 0)
            layer_miss_counter[f"n_miss={n_miss}"] += 1

        # score distribution of missed edges
        if all_missed_scores:
            arr = np.array(all_missed_scores)
            print(f"  missed edge 分数分布 (n={len(arr)}):")
            print(f"    {_percentile_str(arr)}")
            bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.01]
            print(_hist_str(arr, bins, fmt=".2f"))

        # n_layers distribution of model_miss tracks
        print(f"  model_miss 按径迹层数分布:")
        for nl in sorted(n_layers_counter):
            print(f"    {nl}-layer: {n_layers_counter[nl]}")

        # fraction of true edges missed per track
        print(f"  每条径迹缺失边数分布:")
        for key in sorted(layer_miss_counter):
            print(f"    {key}: {layer_miss_counter[key]}")

        # n_scored_ok distribution
        scored_ok = model_miss_df["n_scored_ok"].to_numpy()
        n_true_e  = model_miss_df["n_true_edges"].to_numpy()
        frac_ok   = scored_ok / np.maximum(n_true_e, 1)
        print(f"  在图中通过阈值的真实边比例: {_percentile_str(frac_ok)}")
        # Tracks where ALL true edges fail threshold (hardest to fix)
        all_fail = int((scored_ok == 0).sum())
        print(f"  完全失效（所有真实边打分低）: {all_fail} / {n_model}  ({all_fail/n_model*100:.1f}%)")

    # ── ③ postproc_loss 深度分析 ──────────────────────────────────────────
    post_loss_df = recon_df.filter(pl.col("failure_category") == 3)
    if n_post > 0:
        print(f"\n  ── ③ postproc_loss 深度分析 ({n_post} 条径迹) ──")

        # event-level: which events have postproc losses?
        post_eids = post_loss_df["event_id"].to_numpy()
        post_per_event = Counter(post_eids.tolist())
        post_event_counts = np.array(list(post_per_event.values()))
        print(f"  发生 postproc_loss 的事件数: {len(post_per_event)}")
        print(f"  每事件 postproc_loss 径迹数: {_percentile_str(post_event_counts)}")

        # n_layers distribution
        nl_cnt = Counter(post_loss_df["n_layers"].to_list())
        print(f"  postproc_loss 按径迹层数分布:")
        for nl in sorted(nl_cnt):
            print(f"    {nl}-layer: {nl_cnt[nl]}")

        # Compare with success events: is postproc worse in busy events?
        if clusters_df is not None:
            hits_per_event = (
                clusters_df.group_by("event_id").len()
                .rename({"len": "n_hits"})
            )
            success_df = recon_df.filter(pl.col("failure_category") == 0)
            for cat_name, cat_df in [("success", success_df), ("postproc_loss", post_loss_df)]:
                merged = cat_df.join(hits_per_event, on="event_id", how="left")
                h = merged["n_hits"].drop_nulls().to_numpy()
                if len(h) > 0:
                    print(f"  {cat_name} 事件击中数: mean={h.mean():.1f}  {_percentile_str(h)}")

    # ── 边级统计：in-graph TPR ─────────────────────────────────────────────
    n_true_total = int(recon_df["n_true_edges"].sum())
    n_in_graph   = int(recon_df["n_in_graph"].sum())
    n_scored_ok  = recon_df["n_scored_ok"].drop_nulls()
    if len(n_scored_ok) > 0:
        n_scored_ok_total = int(n_scored_ok.sum())
        in_graph_tpr = n_scored_ok_total / n_in_graph * 100 if n_in_graph > 0 else 0.0
        print(f"\n  ── 边级汇总 ──")
        print(f"  真实邻层边总数:     {n_true_total}")
        print(f"  在图中:             {n_in_graph}  ({n_in_graph/n_true_total*100:.1f}%)")
        print(f"  在图中且打分通过:   {n_scored_ok_total}  (in-graph TPR={in_graph_tpr:.1f}%)")

    return {
        "label":        label,
        "efficiency":   round(eff, 2),
        "n_recon_b":    n_recon_b,
        "n_success":    n_success,
        "n_graph_miss": n_graph,
        "n_model_miss": n_model,
        "n_postproc":   n_post,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-run comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_runs(summaries: list[dict]) -> None:
    if len(summaries) < 2:
        return
    print(f"\n{'='*72}")
    print("  多模型对比汇总")
    print(f"{'='*72}")
    header = f"  {'标签':<28} {'效率':>6}  {'①graph':>7}  {'②model':>7}  {'③post':>7}"
    print(header)
    print("  " + "-" * 60)
    for s in summaries:
        n_loss = s["n_recon_b"] - s["n_success"]
        pct = lambda n: f"{n}({n/n_loss*100:.0f}%)" if n_loss > 0 else str(n)
        print(f"  {s['label']:<28} {s['efficiency']:>5.1f}%  "
              f"{pct(s['n_graph_miss']):>8}  "
              f"{pct(s['n_model_miss']):>8}  "
              f"{pct(s['n_postproc']):>8}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Analyse diagnose_failures.py output parquets")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more per-track parquet files from diagnose_failures.py")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Display label for each input (default: filename stem)")
    p.add_argument("--clusters", default=None,
                   help="Path to sim_clusters_test.parquet for event-level features")
    p.add_argument("--output", default=None,
                   help="Optional path to save comparison JSON")
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    labels = args.labels or [p.stem for p in inputs]
    if len(labels) < len(inputs):
        labels += [p.stem for p in inputs[len(labels):]]

    # Load clusters (optional, for event multiplicity analysis)
    clusters_df = None
    if args.clusters:
        print(f"Loading clusters: {args.clusters}")
        clusters_df = pl.read_parquet(args.clusters)

    summaries = []
    for path, label in zip(inputs, labels):
        if not path.exists():
            print(f"[warn] file not found: {path}")
            continue
        print(f"\nLoading {path}")
        df = pl.read_parquet(path)
        summary = analyze_one(df, label, clusters_df)
        summaries.append(summary)

    if len(summaries) > 1:
        compare_runs(summaries)

    if args.output:
        Path(args.output).write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved summary → {args.output}")


if __name__ == "__main__":
    main()
