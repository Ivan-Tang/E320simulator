"""
Edge-classification models for GNN-based track seeding.

Inputs
------
Node features (7-dim):
    layer_id, x_trk_mm, y_trk_mm, z_trk_mm, size_x, size_y, size

Edge features (6-dim):
    dx_mm, dy_mm, dz_mm, dr_mm, slope_x, slope_y

Target
------
edge_label : {0, 1}   — same truth track on both endpoints

Models
------
EdgeMLP          Pure edge-feature MLP.  No message passing.  Fast baseline.
InteractionNet   Node encoder + 2-round message passing + edge decoder.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils import NODE_FEAT_COLS_SRC, NODE_FEAT_COLS_DST, EDGE_FEAT_COLS

NODE_DIM = len(NODE_FEAT_COLS_SRC)   # 7
EDGE_DIM = len(EDGE_FEAT_COLS)       # 6


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class EdgeMLP(nn.Module):
    """Baseline edge classifier without graph structure.

    Input  : edge_feat (E, 6)  +  concatenated endpoint node features (E, 14)
    Output : edge_score (E,)  ∈ [0, 1]
    """

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM, hidden: int = 64):
        super().__init__()
        in_dim = 2 * node_dim + edge_dim   # 2*7 + 6 = 20
        self.mlp = _mlp(in_dim, hidden, 1, n_layers=3)

    def forward(
        self,
        node_feat: Tensor,   # (N, 7)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, 6)
    ) -> Tensor:             # (E,)
        src, dst = edge_index[0], edge_index[1]
        x = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        return self.mlp(x).squeeze(-1).sigmoid()

class InteractionNet(nn.Module):
    """Edge-classification GNN using Interaction Network message passing.

    Architecture
    ------------
    1. NodeEncoder  : node_dim → hidden
    2. EdgeEncoder  : edge_dim → hidden
    3. MP rounds × n_mp:
       a. edge update : MLP([h_i, h_j, h_edge]) → h_edge'
       b. node update : MLP([h_node, mean(h_edge' from incident edges)]) → h_node'
    4. EdgeDecoder  : [h_i, h_j, h_edge] → sigmoid score
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_mp: int = 2,
    ):
        super().__init__()
        self.hidden = hidden

        self.node_enc = _mlp(node_dim, hidden, hidden, n_layers=2)
        self.edge_enc = _mlp(edge_dim, hidden, hidden, n_layers=2)

        # Interaction layers
        self.edge_mlps = nn.ModuleList([
            _mlp(3 * hidden, hidden, hidden, n_layers=2) for _ in range(n_mp)
        ])
        self.node_mlps = nn.ModuleList([
            _mlp(2 * hidden, hidden, hidden, n_layers=2) for _ in range(n_mp)
        ])

        # Decoder
        self.decoder = _mlp(3 * hidden, hidden, 1, n_layers=2)

    def forward(
        self,
        node_feat: Tensor,   # (N, node_dim)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, edge_dim)
    ) -> Tensor:             # (E,)
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]

        h_n = self.node_enc(node_feat)   # (N, H)
        h_e = self.edge_enc(edge_feat)   # (E, H)

        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            # --- edge update ---
            h_e = edge_mlp(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))  # (E, H)

            # --- node update: aggregate incoming edge messages ---
            agg = torch.zeros(N, self.hidden, device=h_n.device, dtype=h_n.dtype)
            cnt = torch.zeros(N, 1, device=h_n.device, dtype=h_n.dtype)
            agg.index_add_(0, dst, h_e)
            cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h_n.device, dtype=h_n.dtype))
            cnt.clamp_(min=1.0)
            agg = agg / cnt   # mean aggregation

            h_n = node_mlp(torch.cat([h_n, agg], dim=-1))  # (N, H)

        score = self.decoder(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))  # (E, 1)
        return score.squeeze(-1).sigmoid()

