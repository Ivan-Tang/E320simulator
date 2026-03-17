import numpy as np
import polars as pl

from src.train_embedder import EmbedderTrainConfig, train_embedder, embed_hits


def _make_tiny_clusters() -> pl.DataFrame:
    rows = []
    node_id = 0
    # Two tiny events, each with two signal tracks on 5 layers
    for event_id in [0, 1]:
        for tid, x0 in [(10, -1.0), (20, 1.0)]:
            for lid in range(5):
                rows.append(
                    {
                        "event_id": event_id,
                        "node_id": node_id,
                        "layer_id": lid,
                        "x_trk_mm": x0 + 0.1 * lid,
                        "y_trk_mm": 0.2 * lid,
                        "z_trk_mm": 20.0 * lid,
                        "size_x": 1,
                        "size_y": 1,
                        "size": 1,
                        "track_id": tid,
                    }
                )
                node_id += 1
    return pl.from_dicts(rows)


def test_train_embedder_and_infer_smoke(tmp_path):
    clusters_df = _make_tiny_clusters()
    ckpt_dir = tmp_path / "embed_ckpt"

    cfg = EmbedderTrainConfig(
        n_epochs=1,
        batch_size=16,
        emb_dim=4,
        hidden_dim=16,
        n_layers=2,
        checkpoint_dir=str(ckpt_dir),
        max_pairs=2000,
        nb_particles_per_sample=4,
        device="cpu",
    )

    out = train_embedder(clusters_df, cfg)
    assert out["model"] is not None
    assert len(out["history"]) == 1

    hits = clusters_df.select(cfg.hit_feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    emb = embed_hits(
        model=out["model"],
        hits=hits,
        mean=out["mean"],
        std=out["std"],
        device="cpu",
    )
    assert emb.shape[0] == hits.shape[0]
    assert emb.shape[1] == 4
