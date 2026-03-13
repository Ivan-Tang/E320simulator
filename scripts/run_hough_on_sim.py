import os
import numpy as np
import polars as pl
from collections import Counter

from src.hough_baseline import (
    HoughConfig,
    _hough_voting,
    _find_peaks,
    _collect_candidates_for_peak,
    _process_event_hough,
)
from src.baseline import _fit_and_score, _shared_hit_rejection


def evaluate_hough_on_sim(
    clusters_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
) -> pl.DataFrame:
    """Run the Hough track finder on simulated clusters and evaluate.

    Returns the result DataFrame with an extra ``matched_track_id``
    column indicating which truth track (if any) was reconstructed.
    """

    cfg = HoughConfig()
    all_candidates: list[dict] = []

    eid_arr = clusters_df["event_id"].to_numpy()
    x_arr = clusters_df["x_trk_mm"].to_numpy()
    y_arr = clusters_df["y_trk_mm"].to_numpy()
    z_arr = clusters_df["z_trk_mm"].to_numpy()
    lid_arr = clusters_df["layer_id"].to_numpy().astype(np.int8)
    nid_arr = clusters_df["node_id"].to_numpy()
    tid_arr = clusters_df["track_id"].to_numpy()

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
        tv = tid_arr[s : s + c_]

        candidates = _process_event_hough(eid, xv, yv, zv, lv, nv, cfg)
        if not candidates:
            continue

        nid_to_local = {int(n): j for j, n in enumerate(nv)}

        # Match each kept candidate to truth tracks
        for ci, cand in enumerate(candidates):
            # find majority truth track_id among nodes
            node_tids = [int(tv[nid_to_local[n]]) for n in cand["node_ids"]]
            counter = Counter(t for t in node_tids if t >= 0)
            if counter:
                best_tid, best_count = counter.most_common(1)[0]
                cand["matched_track_id"] = best_tid if best_count >= 4 else -1
                cand["n_matched"] = best_count
            else:
                cand["matched_track_id"] = -1
                cand["n_matched"] = 0
        all_candidates.extend(candidates)

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
