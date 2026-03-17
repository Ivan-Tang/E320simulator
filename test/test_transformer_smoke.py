"""Smoke tests for transformer-based models."""
import torch
import numpy as np

from src.models import TransformerEmbedder, TransformerEdgeClassifier, TrackFormerSeed
from src.utils import NODE_DIM, EDGE_DIM


def test_transformer_embedder():
    """Test TransformerEmbedder instantiation and forward pass."""
    model = TransformerEmbedder(
        in_dim=NODE_DIM,
        out_dim=3,
        d_model=128,  # smaller for testing
        n_heads=4,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    )

    # Test with single event (unbatched)
    batch_size = 1
    seq_len = 50
    x = torch.randn(seq_len, NODE_DIM)
    output = model(x)
    assert output.shape == (seq_len, 3)
    assert not torch.isnan(output).any()

    # Test with batched input
    batch_size = 4
    x = torch.randn(batch_size, seq_len, NODE_DIM)
    output = model(x)
    assert output.shape == (batch_size, seq_len, 3)
    assert not torch.isnan(output).any()

    print("✓ TransformerEmbedder smoke test passed")


def test_transformer_edge_classifier():
    """Test TransformerEdgeClassifier instantiation and forward pass."""
    model = TransformerEdgeClassifier(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    )

    # Create dummy data
    num_nodes = 100
    num_edges = 200

    node_feat = torch.randn(num_nodes, NODE_DIM)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_feat = torch.randn(num_edges, EDGE_DIM)

    output = model(node_feat, edge_index, edge_feat)
    assert output.shape == (num_edges,)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid output
    assert not torch.isnan(output).any()

    print("✓ TransformerEdgeClassifier smoke test passed")


def test_trackformer_seed():
    """Test TrackFormerSeed instantiation and forward pass."""
    model = TrackFormerSeed(
        node_dim=NODE_DIM,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=512,
        max_seeds=50,
        dropout=0.1,
    )

    # Test with single event
    num_nodes = 100
    node_feat = torch.randn(num_nodes, NODE_DIM)

    output = model(node_feat)

    assert isinstance(output, dict)
    assert 'seed_confidence' in output
    assert 'seed_parameters' in output
    assert 'hit_assignments' in output
    assert 'seed_embeddings' in output

    assert output['seed_confidence'].shape == (50,)  # max_seeds
    assert output['seed_parameters'].shape == (50, 5)  # param_dim=5
    assert output['hit_assignments'].shape == (50, num_nodes)
    assert output['seed_embeddings'].shape == (50, 128)  # d_model

    # Check confidence is in [0, 1]
    conf = output['seed_confidence']
    assert torch.all(conf >= 0) and torch.all(conf <= 1)

    # Check assignments sum to 1 per seed
    assignments = output['hit_assignments']
    assert torch.allclose(assignments.sum(dim=1), torch.ones(50), atol=1e-5)

    # Test with batched input
    batch_size = 4
    node_feat = torch.randn(batch_size, num_nodes, NODE_DIM)
    output = model(node_feat)

    assert output['seed_confidence'].shape == (batch_size, 50)
    assert output['seed_parameters'].shape == (batch_size, 50, 5)
    assert output['hit_assignments'].shape == (batch_size, 50, num_nodes)
    assert output['seed_embeddings'].shape == (batch_size, 50, 128)

    print("✓ TrackFormerSeed smoke test passed")


def test_model_parameter_counts():
    """Print parameter counts for comparison."""
    gnn = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.Linear(64, 1))
    gnn_params = sum(p.numel() for p in gnn.parameters())

    transformer_small = TransformerEdgeClassifier(
        d_model=128, n_encoder_layers=2, n_heads=4
    )
    transformer_params = sum(p.numel() for p in transformer_small.parameters())

    transformer_medium = TransformerEdgeClassifier(
        d_model=256, n_encoder_layers=6, n_heads=8
    )
    transformer_medium_params = sum(p.numel() for p in transformer_medium.parameters())

    trackformer = TrackFormerSeed(
        d_model=256, n_encoder_layers=6, n_decoder_layers=6, n_heads=8, max_seeds=100
    )
    trackformer_params = sum(p.numel() for p in trackformer.parameters())

    print(f"\nModel parameter counts:")
    print(f"  Simple GNN (10→64→1): {gnn_params:,}")
    print(f"  TransformerEdgeClassifier (small): {transformer_params:,}")
    print(f"  TransformerEdgeClassifier (medium): {transformer_medium_params:,}")
    print(f"  TrackFormerSeed (full): {trackformer_params:,}")


if __name__ == "__main__":
    test_transformer_embedder()
    test_transformer_edge_classifier()
    test_trackformer_seed()
    test_model_parameter_counts()
    print("\n✅ All transformer model smoke tests passed!")