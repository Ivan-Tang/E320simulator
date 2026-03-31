"""Shared fixtures for the E320simulator test suite."""
import numpy as np
import polars as pl
import pytest
import torch


Z_LAYERS = [0.0, 20.0, 40.0, 60.0, 80.0]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (>5 s)")
    config.addinivalue_line("markers", "gpu: mark test as requiring CUDA GPU")


@pytest.fixture(scope="session")
def tiny_clusters_df():
    """3 events, 2 signal tracks/event across 5 layers + 1 bg cluster/layer/event.

    Track 1: x = 0.005*z,       y = 0.003*z
    Track 2: x = 2.0 - 0.004*z, y = 1.0 + 0.002*z
    Background: x ≈ 5+lid, y ≈ 3+lid  (far from signal)
    """
    rows = []
    node_id = 0
    for event_id in range(3):
        tid_base = event_id * 10

        # Signal track 1
        for lid, z in enumerate(Z_LAYERS):
            rows.append({
                "event_id": event_id, "node_id": node_id, "layer_id": lid,
                "x_trk_mm": 0.005 * z, "y_trk_mm": 0.003 * z, "z_trk_mm": z,
                "size_x": 1, "size_y": 1, "size": 1,
                "track_id": tid_base, "is_signal": True, "particle_type": "signal_pos",
            })
            node_id += 1

        # Signal track 2
        for lid, z in enumerate(Z_LAYERS):
            rows.append({
                "event_id": event_id, "node_id": node_id, "layer_id": lid,
                "x_trk_mm": 2.0 - 0.004 * z, "y_trk_mm": 1.0 + 0.002 * z, "z_trk_mm": z,
                "size_x": 1, "size_y": 1, "size": 1,
                "track_id": tid_base + 1, "is_signal": True, "particle_type": "signal_pos",
            })
            node_id += 1

        # Background
        for lid, z in enumerate(Z_LAYERS):
            rows.append({
                "event_id": event_id, "node_id": node_id, "layer_id": lid,
                "x_trk_mm": 5.0 + lid * 0.5, "y_trk_mm": 3.0 + lid * 0.3, "z_trk_mm": z,
                "size_x": 2, "size_y": 2, "size": 4,
                "track_id": -1, "is_signal": False, "particle_type": "background",
            })
            node_id += 1

    return pl.from_dicts(rows)


@pytest.fixture(scope="session")
def tiny_tracks_df():
    """Truth tracks matching tiny_clusters_df."""
    rows = []
    for event_id in range(3):
        tid_base = event_id * 10
        rows.append({
            "event_id": event_id, "track_id": tid_base, "is_signal": True,
            "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0,
            "tx": 0.005, "ty": 0.003, "pz_GeV": 2.5, "n_layers_hit": 5,
        })
        rows.append({
            "event_id": event_id, "track_id": tid_base + 1, "is_signal": True,
            "x0_mm": 2.0, "y0_mm": 1.0, "z0_mm": 0.0,
            "tx": -0.004, "ty": 0.002, "pz_GeV": 2.5, "n_layers_hit": 5,
        })
    return pl.from_dicts(rows)


@pytest.fixture(scope="session")
def one_event_arrays(tiny_clusters_df):
    """Numpy arrays for event 0 from tiny_clusters_df."""
    ev = tiny_clusters_df.filter(pl.col("event_id") == 0).sort("node_id")
    return {
        "x":     ev["x_trk_mm"].to_numpy().astype(np.float64),
        "y":     ev["y_trk_mm"].to_numpy().astype(np.float64),
        "z":     ev["z_trk_mm"].to_numpy().astype(np.float64),
        "layer": ev["layer_id"].to_numpy().astype(np.int8),
        "nid":   ev["node_id"].to_numpy().astype(np.int64),
    }


@pytest.fixture(scope="session")
def edges_df(tiny_clusters_df):
    """Pre-built labeled edge table from tiny_clusters_df."""
    from src.utils import build_labeled_edges_from_sim
    return build_labeled_edges_from_sim(tiny_clusters_df)


@pytest.fixture(scope="session")
def standard_tensors(edges_df):
    """PyTorch tensors for event 0 from edges_df."""
    from src.utils import event_to_tensors
    ev = edges_df.filter(pl.col("event_id") == 0)
    return event_to_tensors(ev)


@pytest.fixture(scope="session")
def gnn_inputs():
    """Minimal (node_feat, edge_index, edge_feat) for model smoke tests.
    node_feat[:, 0] contains valid layer IDs (0-4) for HierarchicalGNN.
    """
    from src.utils import NODE_DIM, EDGE_DIM
    N, E = 50, 100
    torch.manual_seed(0)
    nf = torch.randn(N, NODE_DIM)
    nf[:, 0] = torch.arange(N, dtype=torch.float32) % 5  # layer IDs 0-4
    ei = torch.randint(0, N, (2, E))
    ef = torch.randn(E, EDGE_DIM)
    return nf, ei, ef
