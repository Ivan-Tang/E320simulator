"""Tests for src/train_embedder.py."""
import numpy as np
import pytest
from src.train_embedder import (
    EmbedderTrainConfig,
    build_pair_dataset_from_clusters,
    embed_hits,
    train_embedder,
)
from src.models import Embedder


# ── build_pair_dataset_from_clusters ──────────────────────────────────────────

class TestBuildPairDataset:
    def test_returns_three_arrays(self, tiny_clusters_df):
        cfg = EmbedderTrainConfig(nb_particles_per_sample=10, max_pairs=200)
        h_a, h_b, target = build_pair_dataset_from_clusters(tiny_clusters_df, cfg)
        assert len(h_a) == len(h_b) == len(target)

    def test_correct_feature_dim(self, tiny_clusters_df):
        cfg = EmbedderTrainConfig(nb_particles_per_sample=10, max_pairs=200)
        h_a, h_b, _ = build_pair_dataset_from_clusters(tiny_clusters_df, cfg)
        if len(h_a) > 0:
            assert h_a.shape[1] == len(cfg.hit_feature_cols)

    def test_both_labels_present(self, tiny_clusters_df):
        cfg = EmbedderTrainConfig(nb_particles_per_sample=10, max_pairs=500)
        _, _, target = build_pair_dataset_from_clusters(tiny_clusters_df, cfg)
        assert 0.0 in target
        assert 1.0 in target

    def test_empty_when_no_signal(self):
        """All background clusters → no positive pairs → empty output."""
        import polars as pl
        cfg = EmbedderTrainConfig(max_pairs=100)
        rows = []
        for eid in range(2):
            for lid in range(5):
                rows.append({
                    "event_id": eid, "node_id": eid*5+lid, "layer_id": lid,
                    "x_trk_mm": 0.0, "y_trk_mm": 0.0, "z_trk_mm": lid*20.0,
                    "size_x": 1, "size_y": 1, "size": 1,
                    "track_id": -1, "is_signal": False, "particle_type": "background",
                })
        bg_df = pl.from_dicts(rows)
        h_a, h_b, target = build_pair_dataset_from_clusters(bg_df, cfg)
        assert len(target) == 0


# ── embed_hits ────────────────────────────────────────────────────────────────

class TestEmbedHits:
    @pytest.fixture(scope="class")
    def embed_setup(self):
        import numpy as np
        model = Embedder(in_dim=7, out_dim=4)
        model.eval()
        hits = np.random.randn(30, 7).astype(np.float32)
        mean = np.zeros(7, dtype=np.float32)
        std = np.ones(7, dtype=np.float32)
        return model, hits, mean, std

    def test_output_shape(self, embed_setup):
        model, hits, mean, std = embed_setup
        emb = embed_hits(model, hits, mean, std, device="cpu")
        assert emb.shape == (30, 4)

    def test_output_dtype(self, embed_setup):
        model, hits, mean, std = embed_setup
        emb = embed_hits(model, hits, mean, std, device="cpu")
        assert emb.dtype == np.float32

    def test_no_nan(self, embed_setup):
        model, hits, mean, std = embed_setup
        emb = embed_hits(model, hits, mean, std, device="cpu")
        assert np.all(np.isfinite(emb))


# ── train_embedder (marked slow) ──────────────────────────────────────────────

def _tiny_cfg(**kwargs):
    defaults = dict(
        n_epochs=1, batch_size=16, emb_dim=4, hidden_dim=16, n_layers=2,
        max_pairs=300, nb_particles_per_sample=10, device="cpu",
        val_fraction=0.2, seed=0,
    )
    defaults.update(kwargs)
    return EmbedderTrainConfig(**defaults)


@pytest.mark.slow
def test_train_embedder_required_keys(tiny_clusters_df):
    result = train_embedder(tiny_clusters_df, _tiny_cfg())
    for key in ("model", "history", "best_val_loss", "mean", "std"):
        assert key in result


@pytest.mark.slow
def test_train_embedder_history_length(tiny_clusters_df):
    result = train_embedder(tiny_clusters_df, _tiny_cfg())
    assert len(result["history"]) == 1


@pytest.mark.slow
def test_train_embedder_loss_is_finite(tiny_clusters_df):
    result = train_embedder(tiny_clusters_df, _tiny_cfg())
    loss = result["history"][0]["train_loss"]
    assert np.isfinite(loss) and loss > 0


@pytest.mark.slow
def test_train_embedder_checkpoint_saved(tiny_clusters_df, tmp_path):
    result = train_embedder(tiny_clusters_df, _tiny_cfg(checkpoint_dir=str(tmp_path)))
    ckpt = tmp_path / "best_embedder.pt"
    assert ckpt.exists(), f"Expected checkpoint at {ckpt}"


@pytest.mark.slow
def test_train_embedder_embed_hits_shape(tiny_clusters_df):
    cfg = _tiny_cfg()
    result = train_embedder(tiny_clusters_df, cfg)
    hits = tiny_clusters_df.select(cfg.hit_feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    emb = embed_hits(result["model"], hits, result["mean"], result["std"], device="cpu")
    assert emb.shape == (len(hits), cfg.emb_dim)
