"""Tests for src/models.py and src/layers.py."""
import pytest
import torch
import torch.nn as nn
from src.utils import NODE_DIM, EDGE_DIM
from src.models import (
    Embedder,
    EdgeMLP,
    InteractionNet,
    ResGNN,
    EggNet,
    HierarchicalGNN,
    TransformerEmbedder,
    TransformerEdgeClassifier,
    E320HitFilter,
    E320TrackFormer,
    MaskFormerDecoderLayer,
)
from src.layers import MLP, PositionalEncoding3D


# ── Helpers ───────────────────────────────────────────────────────────────────

def no_nan(tensor):
    return not torch.isnan(tensor).any()

def in_unit_interval(tensor):
    return (tensor >= 0.0).all() and (tensor <= 1.0).all()


# ── MLP (layers.py) ───────────────────────────────────────────────────────────

class TestMLP:
    def test_output_shape(self):
        mlp = MLP(10, 32, 4, n_layers=2)
        out = mlp(torch.randn(5, 10))
        assert out.shape == (5, 4)

    def test_no_nan(self):
        mlp = MLP(7, 64, 1, n_layers=3)
        assert no_nan(mlp(torch.randn(20, 7)))


# ── PositionalEncoding3D ──────────────────────────────────────────────────────

class TestPositionalEncoding3D:
    def test_output_shape_unbatched(self):
        pe = PositionalEncoding3D(d_model=32)
        N = 10
        # forward(x, y, z) takes coordinate tensors, returns (N, d_model)
        x_coords = torch.rand(N)
        y_coords = torch.rand(N)
        z_coords = torch.rand(N)
        out = pe(x_coords, y_coords, z_coords)
        assert out.shape == (N, 32)

    def test_no_nan(self):
        pe = PositionalEncoding3D(d_model=32)
        N = 10
        out = pe(torch.rand(N), torch.rand(N), torch.rand(N))
        assert no_nan(out)


# ── Embedder ──────────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_output_shape(self):
        model = Embedder(in_dim=NODE_DIM, out_dim=4)
        out = model(torch.randn(20, NODE_DIM))
        assert out.shape == (20, 4)

    def test_no_nan(self):
        model = Embedder(in_dim=NODE_DIM, out_dim=4)
        assert no_nan(model(torch.randn(20, NODE_DIM)))


# ── EdgeMLP ───────────────────────────────────────────────────────────────────

class TestEdgeMLP:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = EdgeMLP()
        out = model(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = EdgeMLP()
        out = model(nf, ei, ef)
        assert in_unit_interval(out)

    def test_no_nan(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        assert no_nan(EdgeMLP()(nf, ei, ef))


# ── InteractionNet ────────────────────────────────────────────────────────────

class TestInteractionNet:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = InteractionNet(hidden=32, n_mp=1, emb_dim=4)
        out = model(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = InteractionNet(hidden=32, n_mp=1, emb_dim=4)(nf, ei, ef)
        assert in_unit_interval(out)

    def test_last_embeddings_set(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = InteractionNet(hidden=32, n_mp=1, emb_dim=4)
        model(nf, ei, ef)
        assert model.last_embeddings is not None
        assert model.last_embeddings.shape == (nf.shape[0], 4)

    def test_last_embeddings_unit_norm(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = InteractionNet(hidden=32, n_mp=1, emb_dim=4)
        model(nf, ei, ef)
        norms = model.last_embeddings.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


# ── ResGNN ────────────────────────────────────────────────────────────────────

class TestResGNN:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = ResGNN(hidden=32, n_graph_iters=2)(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = ResGNN(hidden=32, n_graph_iters=2)(nf, ei, ef)
        assert in_unit_interval(out)

    def test_no_nan(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        assert no_nan(ResGNN(hidden=32)(nf, ei, ef))


# ── EggNet ────────────────────────────────────────────────────────────────────

class TestEggNet:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = EggNet(hidden=32, n_iters=2, n_gnns_per_iter=1)(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = EggNet(hidden=32, n_iters=2, n_gnns_per_iter=1)(nf, ei, ef)
        assert in_unit_interval(out)

    def test_no_nan(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        assert no_nan(EggNet(hidden=32, n_iters=1, n_gnns_per_iter=1)(nf, ei, ef))


# ── HierarchicalGNN ───────────────────────────────────────────────────────────

class TestHierarchicalGNN:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = HierarchicalGNN(hidden_dim=32, n_interaction_iters=1, n_hierarchical_iters=1)(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        out = HierarchicalGNN(hidden_dim=32, n_interaction_iters=1, n_hierarchical_iters=1)(nf, ei, ef)
        assert in_unit_interval(out)

    def test_last_embeddings_set(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = HierarchicalGNN(hidden_dim=32, n_interaction_iters=1, n_hierarchical_iters=1)
        model(nf, ei, ef)
        assert model.last_embeddings is not None


# ── TransformerEmbedder ───────────────────────────────────────────────────────

class TestTransformerEmbedder:
    def test_output_shape_unbatched(self):
        model = TransformerEmbedder(in_dim=NODE_DIM, out_dim=4, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(20, NODE_DIM)
        out = model(x)
        assert out.shape == (20, 4)

    def test_output_shape_batched(self):
        model = TransformerEmbedder(in_dim=NODE_DIM, out_dim=4, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(3, 20, NODE_DIM)
        out = model(x)
        assert out.shape == (3, 20, 4)

    def test_no_nan_unbatched(self):
        model = TransformerEmbedder(in_dim=NODE_DIM, out_dim=4, d_model=32, n_heads=2, n_layers=1)
        assert no_nan(model(torch.randn(20, NODE_DIM)))


# ── TransformerEdgeClassifier ─────────────────────────────────────────────────

class TestTransformerEdgeClassifier:
    def test_output_shape(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = TransformerEdgeClassifier(d_model=32, n_heads=2, n_encoder_layers=1)
        out = model(nf, ei, ef)
        assert out.shape == (ei.shape[1],)

    def test_output_in_unit_interval(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = TransformerEdgeClassifier(d_model=32, n_heads=2, n_encoder_layers=1)
        out = model(nf, ei, ef)
        assert in_unit_interval(out)

    def test_no_nan(self, gnn_inputs):
        nf, ei, ef = gnn_inputs
        model = TransformerEdgeClassifier(d_model=32, n_heads=2, n_encoder_layers=1)
        assert no_nan(model(nf, ei, ef))


# ── E320HitFilter ─────────────────────────────────────────────────────────────

class TestE320HitFilter:
    def test_output_shape(self):
        model = E320HitFilter(d_model=16, n_heads=2, n_layers=1, window_size=32)
        x = torch.randn(30, NODE_DIM)
        out = model(x)
        assert out.shape == (30,)

    def test_no_nan(self):
        model = E320HitFilter(d_model=16, n_heads=2, n_layers=1, window_size=32)
        assert no_nan(model(torch.randn(30, NODE_DIM)))

    def test_sort_invariance(self):
        """Shuffling input rows should permute the output consistently."""
        model = E320HitFilter(d_model=16, n_heads=2, n_layers=1, window_size=64)
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(30, NODE_DIM)
        with torch.no_grad():
            out_orig = model(x)
            perm = torch.randperm(30)
            out_perm = model(x[perm])
        torch.testing.assert_close(out_orig[perm], out_perm, atol=1e-4, rtol=1e-4)


# ── MaskFormerDecoderLayer ────────────────────────────────────────────────────

class TestMaskFormerDecoderLayer:
    def test_output_shape_without_mask(self):
        Q, N, d = 10, 30, 32
        layer = MaskFormerDecoderLayer(d_model=d, n_heads=2)
        queries = torch.randn(Q, d)
        memory = torch.randn(N, d)
        out = layer(queries, memory)
        assert out.shape == (Q, d)

    def test_output_shape_with_mask(self):
        Q, N, d = 10, 30, 32
        layer = MaskFormerDecoderLayer(d_model=d, n_heads=2)
        queries = torch.randn(Q, d)
        memory = torch.randn(N, d)
        mask_logits = torch.randn(Q, N)
        out = layer(queries, memory, mask_logits=mask_logits)
        assert out.shape == (Q, d)

    def test_no_nan(self):
        Q, N, d = 10, 30, 32
        layer = MaskFormerDecoderLayer(d_model=d, n_heads=2)
        out = layer(torch.randn(Q, d), torch.randn(N, d), mask_logits=torch.randn(Q, N))
        assert no_nan(out)


# ── E320TrackFormer ───────────────────────────────────────────────────────────

class TestE320TrackFormer:
    @pytest.fixture(scope="class")
    def model_and_output(self):
        torch.manual_seed(0)
        Q, N = 5, 20
        model = E320TrackFormer(
            d_model=32, n_heads=2, n_encoder_layers=1, n_decoder_layers=2, max_queries=Q
        )
        nf = torch.randn(N, NODE_DIM)
        nf[:, 0] = torch.arange(N, dtype=torch.float32) % 5  # valid layer IDs
        out = model(nf)
        return model, out, Q, N

    def test_output_keys(self, model_and_output):
        _, out, *_ = model_and_output
        assert set(out.keys()) == {"track_logits", "mask_logits", "aux_mask_logits", "hit_memory"}

    def test_track_logits_shape(self, model_and_output):
        _, out, Q, N = model_and_output
        assert out["track_logits"].shape == (Q,)

    def test_mask_logits_shape(self, model_and_output):
        _, out, Q, N = model_and_output
        assert out["mask_logits"].shape == (Q, N)

    def test_aux_mask_logits_count(self, model_and_output):
        _, out, Q, N = model_and_output
        # n_decoder_layers=2 → aux_mask_logits has n_decoder_layers-1 = 1 element
        assert len(out["aux_mask_logits"]) == 1

    def test_hit_memory_shape(self, model_and_output):
        model, out, Q, N = model_and_output
        assert out["hit_memory"].shape == (N, model.d_model)

    def test_no_nan(self, model_and_output):
        _, out, *_ = model_and_output
        assert no_nan(out["track_logits"])
        assert no_nan(out["mask_logits"])
        assert no_nan(out["hit_memory"])
