"""Smoke tests for transformer-based models."""
import torch
import numpy as np

from src.models import TransformerEmbedder, TransformerEdgeClassifier
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

    print(f"\nModel parameter counts:")
    print(f"  Simple GNN (10→64→1): {gnn_params:,}")
    print(f"  TransformerEdgeClassifier (small): {transformer_params:,}")
    print(f"  TransformerEdgeClassifier (medium): {transformer_medium_params:,}")


if __name__ == "__main__":
    test_transformer_embedder()
    test_transformer_edge_classifier()
    test_model_parameter_counts()
    print("\n✅ All transformer model smoke tests passed!")