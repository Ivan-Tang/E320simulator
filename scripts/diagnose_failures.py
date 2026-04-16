"""scripts/diagnose_failures.py

诊断径迹重建失效案例，将效率损失拆解为三类：

  ① 真实边不在图里 (graph_miss)   → 根因：kNN 图构建
  ② 真实边在图但模型打分低 (model_miss) → 根因：模型
  ③ 边打分正确但后处理丢失 (postproc_loss) → 根因：chi2/NMS

分类优先级（互斥）：
  ① > ② > ③ > success > not_reconstructible

用法（交互）:
  conda run -n e320root python scripts/diagnose_failures.py \\
    --clusters /storage/agrp/yiwen/data_Run502/simulation/sim_clusters_test.parquet \\
    --tracks   /storage/agrp/yiwen/data_Run502/simulation/sim_tracks_test.parquet \\
    --checkpoint /storage/agrp/yiwen/runs/loop5_pos_weight_fix/best_model.pt \\
    --threshold 0.5 \\
    --n-events 500 \\
    --device cuda \\
    --output /storage/agrp/yiwen/data_Run502/outputs/diagnose_results.json

用法（无模型，只诊断图构建覆盖率）:
  conda run -n e320root python scripts/diagnose_failures.py \\
    --clusters ... --tracks ... --n-events 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import (
    BaselineConfig,
    _build_chains,
    _build_edges,
    _fit_and_score,
    _shared_hit_rejection,
)
from src.train import load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Feature tensor builder (mirrors run_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tensors(xv, yv, zv, lv, nv, sxv, syv, sv,
                   e_src, e_dst, e_sl, e_sx, e_sy, nid_to_local):
    """Build node/edge feature tensors for one event."""
    all_nids = np.unique(np.concatenate([e_src, e_dst]))
    nid_to_row = {int(n): i for i, n in enumerate(all_nids)}

    node_feat = np.empty((len(all_nids), 7), dtype=np.float32)
    for ri, gid in enumerate(all_nids):
        li = nid_to_local[int(gid)]
        node_feat[ri] = [lv[li], xv[li], yv[li], zv[li], sxv[li], syv[li], sv[li]]

    src_l = np.array([nid_to_row[int(s)] for s in e_src], dtype=np.int64)
    dst_l = np.array([nid_to_row[int(d)] for d in e_dst], dtype=np.int64)

    li_idx = np.array([nid_to_local[int(s)] for s in e_src], dtype=np.int64)
    lj_idx = np.array([nid_to_local[int(d)] for d in e_dst], dtype=np.int64)
    dx = xv[lj_idx] - xv[li_idx]
    dy = yv[lj_idx] - yv[li_idx]
    dz = zv[lj_idx] - zv[li_idx]
    dr = np.sqrt(dx**2 + dy**2)
    edge_feat = np.stack([dx, dy, dz, dr, e_sx, e_sy], axis=1).astype(np.float32)

    return (
        torch.from_numpy(node_feat),
        torch.from_numpy(np.stack([src_l, dst_l])),
        torch.from_numpy(edge_feat),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-event diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose_event(
    eid: int,
    xv, yv, zv, lv, nv, tv, sxv, syv, sv,
    model, node_mean, node_std, edge_mean, edge_std,
    threshold: float,
    baseline_cfg: BaselineConfig,
    device_t: torch.device,
    has_model: bool,
    embedder_info=None,
) -> list[dict]:
    """Run full diagnosis pipeline for one event.

    Returns a list of per-track dicts with failure classification.
    """
    nid_to_local = {int(n): j for j, n in enumerate(nv)}

    # ── Step 1: 找出所有信号径迹及其真实相邻层边 ────────────────────────────
    # reconstructible 条件：信号径迹在 ≥4 个不同层有击中
    track_info: dict[int, dict] = {}
    for j in range(len(nv)):
        tid = int(tv[j])
        if tid < 0:
            continue
        if tid not in track_info:
            track_info[tid] = {"nids": [], "lids": []}
        track_info[tid]["nids"].append(int(nv[j]))
        track_info[tid]["lids"].append(int(lv[j]))

    # 为每条径迹建立真实邻层边列表：(src_nid, dst_nid, src_layer)
    track_edges: dict[int, list[tuple[int, int, int]]] = {}
    track_n_layers: dict[int, int] = {}
    for tid, info in track_info.items():
        nids_arr = np.array(info["nids"])
        lids_arr = np.array(info["lids"])
        track_n_layers[tid] = int(np.unique(lids_arr).size)
        edges: list[tuple[int, int, int]] = []
        for l in range(4):
            src_nids = nids_arr[lids_arr == l]
            dst_nids = nids_arr[lids_arr == l + 1]
            if len(src_nids) > 0 and len(dst_nids) > 0:
                edges.append((int(src_nids[0]), int(dst_nids[0]), l))
        track_edges[tid] = edges

    # ── Step 2: 建图 ──────────────────────────────────────────────────────
    e_src, e_dst, e_sl, e_dl, e_sx, e_sy = _build_edges(xv, yv, zv, lv, nv, baseline_cfg)
    graph_set = set(zip(e_src.tolist(), e_dst.tolist()))

    # ── Step 3: ML 打分（可选）────────────────────────────────────────────
    score_lookup: dict[tuple[int, int], float] | None = None
    scores_arr: np.ndarray | None = None
    if has_model and len(e_src) > 0:
        nf, ei, ef = _build_tensors(
            xv, yv, zv, lv, nv, sxv, syv, sv,
            e_src, e_dst, e_sl, e_sx, e_sy,
            nid_to_local,
        )
        nf = nf.to(device_t)
        # Two-stage pipeline: apply pretrained embedder if present
        if embedder_info is not None:
            from src.train import _augment_with_embedder
            nf = _augment_with_embedder(nf, embedder_info)
        # Node normalisation: skip if embedder was applied (embedder handles its own norm)
        if nf.shape[1] == node_mean.shape[0]:
            nf = (nf - node_mean) / node_std
        ef = (ef.to(device_t) - edge_mean) / edge_std
        with torch.no_grad():
            scores_arr = model(nf, ei.to(device_t), ef).cpu().numpy()
        score_lookup = {
            (int(e_src[k]), int(e_dst[k])): float(scores_arr[k])
            for k in range(len(e_src))
        }

    # ── Step 4: 在 ML 过滤后的边上重建径迹 ──────────────────────────────
    reconstructed_tids: set[int] = set()
    postproc_n_chains = 0       # threshold-passing edges 后 chain 建立数量
    postproc_n_kept   = 0       # shared-hit rejection 后 kept 数量

    if has_model and scores_arr is not None:
        mask = scores_arr >= threshold
        if mask.any():
            chains = _build_chains(
                e_src[mask], e_dst[mask], e_sl[mask], e_dl[mask],
                e_sx[mask], e_sy[mask], baseline_cfg,
            )
            postproc_n_chains = len(chains)
            if chains:
                candidates = _fit_and_score(chains, xv, yv, zv, nid_to_local)
                candidates = _shared_hit_rejection(candidates)
                for cand in candidates:
                    if not cand["is_kept"]:
                        continue
                    postproc_n_kept += 1
                    node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
                    counter = Counter(t for t in node_tids if t >= 0)
                    if counter:
                        best_tid, best_count = counter.most_common(1)[0]
                        if best_count >= 4:
                            reconstructed_tids.add(best_tid)

    # ── Step 5: 为每条径迹生成诊断结论 ──────────────────────────────────
    results: list[dict] = []
    for tid, true_edges in track_edges.items():
        n_layers   = track_n_layers[tid]
        n_true     = len(true_edges)
        is_reco    = tid in reconstructed_tids
        is_recon_b = n_layers >= 4  # reconstructible 标准：≥4 层有击中

        in_graph_flags = [e[:2] in graph_set for e in true_edges]
        n_in_graph = sum(in_graph_flags)

        n_scored_ok: int | None = None
        scored_ok_flags: list[bool] | None = None
        if score_lookup is not None:
            scored_ok_flags = [
                e[:2] in score_lookup and score_lookup[e[:2]] >= threshold
                for e in true_edges
            ]
            n_scored_ok = sum(scored_ok_flags)

        # 失效分类（互斥，优先级 ① > ② > ③）
        if is_reco:
            cat    = 0
            reason = "reconstructed"
        elif not is_recon_b:
            cat    = -1
            reason = f"not_reconstructible (only {n_layers} layers)"
        elif n_in_graph < n_true:
            missing = [e[2] for e, ig in zip(true_edges, in_graph_flags) if not ig]
            cat    = 1
            reason = (f"graph_miss: {n_true - n_in_graph}/{n_true} true edges "
                      f"missing from kNN graph at src_layers {missing}")
        elif not has_model:
            cat    = -2
            reason = f"all {n_true} edges in graph (no model → can't diagnose further)"
        elif n_scored_ok is not None and n_scored_ok < n_true:
            low_scores = [
                round(score_lookup.get(e[:2], -1.0), 3)
                for e, ok in zip(true_edges, scored_ok_flags or [])
                if not ok
            ]
            cat    = 2
            reason = (f"model_miss: {n_true - n_scored_ok}/{n_true} in-graph true edges "
                      f"scored below threshold; scores={low_scores}")
        else:
            cat    = 3
            reason = (f"postproc_loss: all {n_true} true edges in graph and "
                      f"scored ≥ threshold; lost in chain-building / chi2-cut / NMS "
                      f"(chains_built={postproc_n_chains}, kept={postproc_n_kept})")

        results.append({
            "event_id":         eid,
            "track_id":         tid,
            "n_layers":         n_layers,
            "n_true_edges":     n_true,
            "n_in_graph":       n_in_graph,
            "n_scored_ok":      n_scored_ok,
            "is_reconstructible": is_recon_b,
            "is_reconstructed": is_reco,
            "failure_category": cat,
            "reason":           reason,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnosis(
    clusters_path: str,
    tracks_path: str,
    checkpoint_path: str | None,
    threshold: float,
    n_events: int | None,
    device: str,
    output_json: str | None,
    output_parquet: str | None,
    baseline_cfg: BaselineConfig | None = None,
) -> dict:
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig()

    # ── 加载数据 ──────────────────────────────────────────────────────────
    print(f"[diagnose] loading clusters: {clusters_path}")
    clusters_df = pl.read_parquet(clusters_path)
    print(f"[diagnose] loading tracks:   {tracks_path}")
    tracks_df = pl.read_parquet(tracks_path)

    if n_events is not None:
        eids = clusters_df["event_id"].unique().sort()[:n_events].to_list()
        clusters_df = clusters_df.filter(pl.col("event_id").is_in(eids))
        tracks_df   = tracks_df.filter(pl.col("event_id").is_in(eids))
        print(f"[diagnose] limited to {n_events} events")

    n_total_events = clusters_df["event_id"].n_unique()
    print(f"[diagnose] events={n_total_events}  "
          f"hits={len(clusters_df):,}  truth_tracks={len(tracks_df):,}")

    # ── 加载模型（可选）─────────────────────────────────────────────────
    model = node_mean = node_std = edge_mean = edge_std = None
    has_model = False
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    embedder_info = None
    if checkpoint_path is not None:
        print(f"[diagnose] loading checkpoint: {checkpoint_path}  device={device_t}")
        ckpt      = load_checkpoint(checkpoint_path, device=str(device_t))
        model     = ckpt["model"].to(device_t).eval()
        node_mean = ckpt["node_mean"].to(device_t)
        node_std  = ckpt["node_std"].to(device_t)
        edge_mean = ckpt["edge_mean"].to(device_t)
        edge_std  = ckpt["edge_std"].to(device_t)
        # Two-stage pipeline: embedder replaces raw node features before classifier
        embedder_info = ckpt.get("embedder_info", None)
        has_model = True
        extra = f"  embedder=True" if embedder_info is not None else ""
        print(f"[diagnose] model loaded: {model.__class__.__name__}  threshold={threshold}{extra}")
    else:
        print("[diagnose] no checkpoint → only graph-coverage analysis (stage ①)")

    # ── 提取 numpy arrays ─────────────────────────────────────────────────
    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr   = clusters_df["x_trk_mm"].to_numpy()
    y_arr   = clusters_df["y_trk_mm"].to_numpy()
    z_arr   = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    tid_arr = clusters_df["track_id"].to_numpy()
    sx_arr  = clusters_df["size_x"].to_numpy()
    sy_arr  = clusters_df["size_y"].to_numpy()
    s_arr   = clusters_df["size"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    # ── 逐事件诊断 ────────────────────────────────────────────────────────
    all_records: list[dict] = []

    for i, eid in enumerate(unique_events):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique_events)}] event {eid} ...")
        s, c_ = int(starts[i]), int(counts[i])
        record = _diagnose_event(
            eid=int(eid),
            xv=x_arr[s:s+c_], yv=y_arr[s:s+c_], zv=z_arr[s:s+c_],
            lv=lid_arr[s:s+c_], nv=nid_arr[s:s+c_], tv=tid_arr[s:s+c_],
            sxv=sx_arr[s:s+c_], syv=sy_arr[s:s+c_], sv=s_arr[s:s+c_],
            model=model,
            node_mean=node_mean, node_std=node_std,
            edge_mean=edge_mean, edge_std=edge_std,
            threshold=threshold,
            baseline_cfg=baseline_cfg,
            device_t=device_t,
            has_model=has_model,
            embedder_info=embedder_info,
        )
        all_records.extend(record)

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    df = pl.DataFrame(all_records)

    n_total      = len(df)
    n_recon_b    = int(df["is_reconstructible"].sum())
    n_recon      = int(df["is_reconstructed"].sum())
    n_recon_recon = int(df.filter(pl.col("is_reconstructible"))["is_reconstructed"].sum())
    efficiency   = n_recon_recon / n_recon_b * 100 if n_recon_b > 0 else 0.0

    cat_counts = df.filter(pl.col("is_reconstructible")).group_by("failure_category").len().sort("failure_category")

    # 边级统计
    n_true_edges_total = int(df["n_true_edges"].sum())
    n_in_graph_total   = int(df["n_in_graph"].sum())
    edge_cov_rate      = n_in_graph_total / n_true_edges_total if n_true_edges_total else float("nan")

    n_scored_ok_total = None
    model_tpr = None
    if has_model and "n_scored_ok" in df.columns:
        scored_ok_col = df["n_scored_ok"].drop_nulls()
        if len(scored_ok_col) > 0:
            n_scored_ok_total = int(scored_ok_col.sum())
            model_tpr = n_scored_ok_total / n_in_graph_total if n_in_graph_total > 0 else float("nan")

    # 每类在 reconstructible 径迹中的数量
    recon_df = df.filter(pl.col("is_reconstructible"))
    cat_map: dict[int, int] = {}
    for row in cat_counts.iter_rows(named=True):
        cat_map[row["failure_category"]] = row["len"]

    n_graph_miss  = cat_map.get(1, 0)
    n_model_miss  = cat_map.get(2, 0)
    n_post_miss   = cat_map.get(3, 0)
    n_no_model_ig = cat_map.get(-2, 0)
    n_loss_total  = n_recon_b - n_recon_recon

    # ── 打印报告 ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  E320 失效诊断报告")
    print("=" * 70)
    print(f"  事件数:              {n_total_events}")
    print(f"  总信号径迹:           {n_total}")
    print(f"  可重建径迹 (≥4 层):   {n_recon_b}")
    print(f"  已重建:               {n_recon_recon}")
    print(f"  效率:                 {efficiency:.1f}%")
    print()
    print(f"  ─── 失效 {n_loss_total} 条可重建径迹的根因分解 ───")
    if n_loss_total > 0:
        pct = lambda n: n / n_loss_total * 100
        print(f"  ① 真实边不在图里 (graph_miss):      {n_graph_miss:5d}  ({pct(n_graph_miss):.1f}%)")
        if has_model:
            print(f"  ② 模型打分低    (model_miss):        {n_model_miss:5d}  ({pct(n_model_miss):.1f}%)")
            print(f"  ③ 后处理丢失    (postproc_loss):     {n_post_miss:5d}  ({pct(n_post_miss):.1f}%)")
        else:
            print(f"  ② 在图中但原因未知 (需要模型):      {n_no_model_ig:5d}  ({pct(n_no_model_ig):.1f}%)")
    print()
    print(f"  ─── 边级覆盖率 ───")
    print(f"  真实邻层边总数:           {n_true_edges_total}")
    print(f"  在图中的真实边:           {n_in_graph_total}  ({edge_cov_rate*100:.2f}%)")
    if has_model and n_scored_ok_total is not None:
        print(f"  在图中且打分 ≥ {threshold}: {n_scored_ok_total}  "
              f"(in-graph TPR={model_tpr*100:.2f}%)")
    print("=" * 70)
    print()

    # ── 构建结果 dict ─────────────────────────────────────────────────────
    result = {
        "config": {
            "clusters_path":   clusters_path,
            "tracks_path":     tracks_path,
            "checkpoint_path": checkpoint_path,
            "threshold":       threshold,
            "n_events":        n_total_events,
            "device":          str(device_t),
            "knn_k":           baseline_cfg.knn_k,
            "slope_x_max":     baseline_cfg.slope_x_max,
            "slope_y_max":     baseline_cfg.slope_y_max,
        },
        "summary": {
            "n_events":            n_total_events,
            "n_total_tracks":      n_total,
            "n_reconstructible":   n_recon_b,
            "n_reconstructed":     n_recon_recon,
            "efficiency_pct":      round(efficiency, 2),
            "n_losses":            n_loss_total,
        },
        "failure_breakdown": {
            "graph_miss":    {"count": n_graph_miss,
                              "pct_of_loss": round(n_graph_miss / n_loss_total * 100, 1) if n_loss_total else 0},
            "model_miss":    {"count": n_model_miss,
                              "pct_of_loss": round(n_model_miss / n_loss_total * 100, 1) if n_loss_total else 0},
            "postproc_loss": {"count": n_post_miss,
                              "pct_of_loss": round(n_post_miss / n_loss_total * 100, 1) if n_loss_total else 0},
        },
        "edge_coverage": {
            "n_true_edges":        n_true_edges_total,
            "n_in_graph":          n_in_graph_total,
            "graph_edge_cov_pct":  round(edge_cov_rate * 100, 2),
            "n_scored_ok":         n_scored_ok_total,
            "in_graph_tpr_pct":    round(model_tpr * 100, 2) if model_tpr is not None else None,
        },
    }

    # ── 保存 JSON ─────────────────────────────────────────────────────────
    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[diagnose] JSON saved → {out_path}")

    # ── 保存 Parquet（逐条径迹明细）─────────────────────────────────────
    if output_parquet is not None:
        out_path = Path(output_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 将 n_scored_ok (可能是 None) 转为 Int32
        df_out = df.with_columns(
            pl.col("n_scored_ok").cast(pl.Int32),
        )
        df_out.write_parquet(out_path)
        print(f"[diagnose] per-track parquet saved → {out_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose E320 track-reco failures into three root causes"
    )
    p.add_argument("--clusters",    required=True,
                   help="Path to sim_clusters_test.parquet")
    p.add_argument("--tracks",      required=True,
                   help="Path to sim_tracks_test.parquet")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to best_model.pt (edge-classifier checkpoint). "
                        "Omit to only diagnose graph coverage.")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Edge score threshold for kept edges (default 0.5)")
    p.add_argument("--n-events",    type=int, default=None,
                   help="Limit to first N events (default: all)")
    p.add_argument("--device",      default="cuda",
                   choices=["cuda", "cpu"],
                   help="PyTorch device (default cuda)")
    p.add_argument("--output",      default=None,
                   help="Save aggregate results to this JSON path")
    p.add_argument("--output-parquet", default=None,
                   help="Save per-track detail to this parquet path")
    p.add_argument("--knn-k",       type=int,   default=10)
    p.add_argument("--slope-x-max", type=float, default=0.2)
    p.add_argument("--slope-y-max", type=float, default=0.2)
    args = p.parse_args()

    cfg = BaselineConfig(
        knn_k=args.knn_k,
        slope_x_max=args.slope_x_max,
        slope_y_max=args.slope_y_max,
    )

    run_diagnosis(
        clusters_path=args.clusters,
        tracks_path=args.tracks,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        n_events=args.n_events,
        device=args.device,
        output_json=args.output,
        output_parquet=args.output_parquet,
        baseline_cfg=cfg,
    )


if __name__ == "__main__":
    _cli()
