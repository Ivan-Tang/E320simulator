import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from src.hough_baseline import HoughConfig, _process_event_hough


def _process_and_match_event_hough(
    eid: int, xv, yv, zv, lv, nv, tv, cfg: HoughConfig,
) -> list[dict]:
    """Run Hough reco for one event and match candidates to truth tracks."""
    candidates = _process_event_hough(eid, xv, yv, zv, lv, nv, cfg)
    if not candidates:
        return []

    nid_to_local = {int(n): j for j, n in enumerate(nv)}
    for ci, cand in enumerate(candidates):
        node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
        counter = Counter(t for t in node_tids if t >= 0)
        if counter:
            best_tid, best_count = counter.most_common(1)[0]
            cand["matched_track_id"] = best_tid if best_count >= 4 else -1
            cand["n_matched"] = best_count
        else:
            cand["matched_track_id"] = -1
            cand["n_matched"] = 0
    return candidates


def evaluate_hough_on_sim(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
    cfg: HoughConfig | None = None,
) -> pl.DataFrame:
    """Run the Hough track finder on simulated clusters and evaluate.

    Returns the result DataFrame with an extra ``matched_track_id``
    column indicating which truth track (if any) was reconstructed.
    """
    if cfg is None:
        cfg = HoughConfig()

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr = clusters_df["x_trk_mm"].to_numpy()
    y_arr = clusters_df["y_trk_mm"].to_numpy()
    z_arr = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    tid_arr = clusters_df["track_id"].to_numpy()

    unique_events, starts = np.unique(eid_arr, return_index=True)
    counts = np.diff(np.append(starts, len(eid_arr)))

    # Build per-event argument tuples (copy slices for process safety)
    event_args = []
    for i in range(len(unique_events)):
        s, c_ = int(starts[i]), int(counts[i])
        event_args.append((
            int(unique_events[i]),
            x_arr[s:s+c_].copy(),
            y_arr[s:s+c_].copy(),
            z_arr[s:s+c_].copy(),
            lid_arr[s:s+c_].copy(),
            nid_arr[s:s+c_].copy(),
            tid_arr[s:s+c_].copy(),
            cfg,
        ))

    all_candidates: list[dict] = []
    with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
        futures = [pool.submit(_process_and_match_event_hough, *args) for args in event_args]
        for f in as_completed(futures):
            all_candidates.extend(f.result())

    if not all_candidates:
        return pl.DataFrame()

    result = pl.DataFrame(all_candidates).sort("event_id", "candidate_id")

    # Compute efficiency metrics
    kept = result.filter(pl.col("is_kept"))
    n_kept = kept.height
    matched_kept = kept.filter(pl.col("matched_track_id") >= 0)
    n_matched_cands = matched_kept.height
    n_fake = n_kept - n_matched_cands

    n_truth = tracks_df.height
    eff_truth = n_matched_cands / n_truth * 100 if n_truth > 0 else 0.0
    fake_rate = n_fake / n_kept * 100 if n_kept > 0 else 0.0

    print(f"\n[hough eval]")
    print(f"  truth tracks:          {n_truth}")
    print(f"  kept reco candidates:  {n_kept}")
    print(f"  matched candidates:    {n_matched_cands} (track eff = {eff_truth:.1f}%)")
    print(f"  fakes:                 {n_fake}  (fake rate = {fake_rate:.1f}%)")

    return result


if __name__ == "__main__":
    data_dir = "/Users/IvanTang/hep/data_Run502/simulation/"
    suffixs = ["train", "test"]

    for suffix in suffixs:
        clusters_df = pl.read_parquet(
            os.path.join(data_dir, f"sim_clusters_{suffix}.parquet")
        )
        tracks_df = pl.read_parquet(
            os.path.join(data_dir, f"sim_tracks_{suffix}.parquet")
        )
        print(f"\n{'='*60}")
        print(f"  Hough evaluation on {suffix} set")
        print(f"{'='*60}")
        hough_result = evaluate_hough_on_sim(clusters_df, tracks_df)
        hough_path = os.path.join(data_dir, f"hough_result_{suffix}.parquet")
        hough_result.write_parquet(hough_path)
        print(f"[hough] Saved Hough result to {hough_path}")
