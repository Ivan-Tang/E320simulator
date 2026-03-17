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

import math
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils import NODE_FEAT_COLS_SRC, NODE_FEAT_COLS_DST, EDGE_FEAT_COLS
from src.layers import MLP, PositionalEncoding3D, TransformerEncoderLayer, TransformerDecoderLayer

NODE_DIM = len(NODE_FEAT_COLS_SRC)   # 7
EDGE_DIM = len(EDGE_FEAT_COLS)       # 6


class Embedder(nn.Module):
    """Metric-learning hit embedder.

    Maps raw hit features to an embedding space where same-track hits are
    close and different-track hits are far.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.mlp = MLP(in_dim=in_dim, hidden=hidden_dim, out_dim=out_dim, n_layers=n_layers)

    def forward(self, input_feat: Tensor) -> Tensor:
        return self.mlp(input_feat)

        

class EdgeMLP(nn.Module):
    """Baseline edge classifier without graph structure.

    Input  : edge_feat (E, 6)  +  concatenated endpoint node features (E, 14)
    Output : edge_score (E,)  ∈ [0, 1]
    """

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM, hidden: int = 64):
        super().__init__()
        in_dim = 2 * node_dim + edge_dim   # 2*7 + 6 = 20
        self.mlp = MLP(in_dim, hidden, 1, n_layers=3)

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

        self.node_enc = MLP(node_dim, hidden, hidden, n_layers=2)
        self.edge_enc = MLP(edge_dim, hidden, hidden, n_layers=2)

        # Interaction layers
        self.edgeMLPs = nn.ModuleList([
            MLP(3 * hidden, hidden, hidden, n_layers=2) for _ in range(n_mp)
        ])
        self.nodeMLPs = nn.ModuleList([
            MLP(2 * hidden, hidden, hidden, n_layers=2) for _ in range(n_mp)
        ])

        # Decoder
        self.decoder = MLP(3 * hidden, hidden, 1, n_layers=2)

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

        for edgeMLP, nodeMLP in zip(self.edgeMLPs, self.nodeMLPs):
            # --- edge update ---
            h_e = edgeMLP(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))  # (E, H)

            # --- node update: aggregate incoming edge messages ---
            agg = torch.zeros(N, self.hidden, device=h_n.device, dtype=h_n.dtype)
            cnt = torch.zeros(N, 1, device=h_n.device, dtype=h_n.dtype)
            agg.index_add_(0, dst, h_e)
            cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h_n.device, dtype=h_n.dtype))
            cnt.clamp_(min=1.0)
            agg = agg / cnt   # mean aggregation

            h_n = nodeMLP(torch.cat([h_n, agg], dim=-1))  # (N, H)

        score = self.decoder(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))  # (E, 1)
        return score.squeeze(-1).sigmoid()


class ResGNN(nn.Module):
    """ResGNN model from exatrkx-ctd2020.
    
    Sparse message-passing graph neural network for segment classification.
    Uses edge and node networks with residual connections.
    """
    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_graph_iters: int = 3,
    ):
        super().__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden
        
        self.node_encoder = MLP(node_dim, hidden, hidden, n_layers=2)
        self.edge_network = MLP(2 * hidden, hidden, 1, n_layers=4)
        self.node_network = MLP(3 * hidden, hidden, hidden, n_layers=4)

    def forward(
        self,
        node_feat: Tensor,   # (N, node_dim)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, edge_dim)
    ) -> Tensor:             # (E,)
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]

        x = self.node_encoder(node_feat)
        
        for i in range(self.n_graph_iters):
            x0 = x
            edge_inputs = torch.cat([x[src], x[dst]], dim=-1)
            e = torch.sigmoid(self.edge_network(edge_inputs).squeeze(-1))
            
            mi = torch.zeros((N, self.hidden_dim), device=x.device, dtype=x.dtype)
            mo = torch.zeros((N, self.hidden_dim), device=x.device, dtype=x.dtype)
            
            mi.index_add_(0, dst, e.unsqueeze(-1) * x[src])
            mo.index_add_(0, src, e.unsqueeze(-1) * x[dst])
            
            node_inputs = torch.cat([mi, mo, x], dim=-1)
            x = self.node_network(node_inputs)
            
            x = x + x0
            
        edge_inputs = torch.cat([x[src], x[dst]], dim=-1)
        return self.edge_network(edge_inputs).squeeze(-1).sigmoid()


class MPNN(nn.Module):
    """DeepMind's InteractionNetwork with Residual connections (from exatrkx-ctd2020)"""
    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_graph_iters: int = 1,
    ):
        super().__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden
        
        self.node_encoder = MLP(node_dim, hidden, hidden, n_layers=2)
        self.edge_network = MLP(2 * hidden, hidden, hidden, n_layers=4)
        self.node_network = MLP(2 * hidden, hidden, hidden, n_layers=4)
        self.edge_classifier = MLP(2 * hidden, hidden, 1, n_layers=2)
        
    def forward(
        self,
        node_feat: Tensor,   # (N, node_dim)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, edge_dim)
    ) -> Tensor:             # (E,)
        send_idx = torch.cat([edge_index[0], edge_index[1]], dim=0)
        recv_idx = torch.cat([edge_index[1], edge_index[0]], dim=0)
        
        x = self.node_encoder(node_feat)
        N = node_feat.shape[0]
        
        for i in range(self.n_graph_iters):
            x0 = x
            edge_inputs = torch.cat([x[send_idx], x[recv_idx]], dim=-1)
            e = self.edge_network(edge_inputs)
            
            aggr_messages = torch.zeros((N, self.hidden_dim), device=x.device, dtype=x.dtype)
            aggr_messages.index_add_(0, recv_idx, e)
            
            node_inputs = torch.cat([x, aggr_messages], dim=-1)
            x = self.node_network(node_inputs)
            x = x + x0
            
        start_idx, end_idx = edge_index
        clf_inputs = torch.cat([x[start_idx], x[end_idx]], dim=-1)
        return self.edge_classifier(clf_inputs).squeeze(-1).sigmoid()


class AGNN(nn.Module):
    """AGNN model from exatrkx-ctd2020.
    
    Like ResGNN but shortcut connects inputs to the hidden representation.
    """
    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_graph_iters: int = 3,
    ):
        super().__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden
        
        self.input_network = MLP(node_dim, hidden, hidden, n_layers=2)
        
        in_dim = node_dim + hidden
        self.edge_network = MLP(in_dim * 2, hidden, 1, n_layers=4)
        self.node_network = MLP(in_dim * 3, hidden, hidden, n_layers=4)
        
    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]
        
        h = self.input_network(node_feat)
        x = torch.cat([h, node_feat], dim=-1)
        
        for i in range(self.n_graph_iters):
            edge_inputs = torch.cat([x[src], x[dst]], dim=-1)
            e = torch.sigmoid(self.edge_network(edge_inputs).squeeze(-1))
            
            dim_x = x.shape[-1]
            mi = torch.zeros((N, dim_x), device=x.device, dtype=x.dtype)
            mo = torch.zeros((N, dim_x), device=x.device, dtype=x.dtype)
            
            mi.index_add_(0, dst, e.unsqueeze(-1) * x[src])
            mo.index_add_(0, src, e.unsqueeze(-1) * x[dst])
            
            node_inputs = torch.cat([mi, mo, x], dim=-1)
            h = self.node_network(node_inputs)
            
            x = torch.cat([h, node_feat], dim=-1)
            
        edge_inputs = torch.cat([x[src], x[dst]], dim=-1)
        return self.edge_network(edge_inputs).squeeze(-1).sigmoid()


class TransformerEmbedder(nn.Module):
    """Transformer-based hit embedder.

    Architecture based on Stroud et al. (2024) "Transformers for Charged Particle
    Track Reconstruction in High Energy Physics".

    Maps raw hit features to an embedding space using Transformer encoder.
    """

    def __init__(
        self,
        in_dim: int = NODE_DIM,
        out_dim: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_model = d_model

        # Feature projection to d_model
        self.input_proj = nn.Linear(in_dim, d_model)

        # Positional encoding for 3D coordinates
        self.pos_encoding = PositionalEncoding3D(d_model, max_len=max_len)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_feat: (batch_size, seq_len, in_dim) or (seq_len, in_dim)

        Returns:
            embeddings: (batch_size, seq_len, out_dim) or (seq_len, out_dim)
        """
        # Handle both batched and unbatched inputs
        if input_feat.dim() == 2:
            input_feat = input_feat.unsqueeze(0)  # (1, seq_len, in_dim)
            batched = False
        else:
            batched = True

        batch_size, seq_len, _ = input_feat.shape

        # Project input to d_model
        x = self.input_proj(input_feat)  # (batch_size, seq_len, d_model)

        # Extract position coordinates (assumes x,y,z are at indices 1,2,3)
        # Normalize coordinates to [0, 1] range for positional encoding
        x_coords = (input_feat[:, :, 1] + 50) / 100  # assuming x in [-50, 50] mm
        y_coords = (input_feat[:, :, 2] + 50) / 100  # assuming y in [-50, 50] mm
        z_coords = input_feat[:, :, 3] / 200  # assuming z in [0, 200] mm

        # Add positional encoding
        pos_enc = self.pos_encoding(x_coords, y_coords, z_coords)
        x = x + pos_enc

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Layer normalization
        x = self.norm(x)

        # Project to output dimension
        embeddings = self.output_proj(x)

        if not batched:
            embeddings = embeddings.squeeze(0)

        return embeddings


class TransformerEdgeClassifier(nn.Module):
    """Transformer-based edge classifier for track seeding.

    Combines Transformer encoder for hit feature extraction with edge classification head.
    Inspired by Stroud et al. (2024) but adapted for edge classification task.
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.d_model = d_model

        # Node feature encoder
        self.node_encoder = TransformerEmbedder(
            in_dim=node_dim,
            out_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        # Edge feature encoder (MLP)
        self.edge_encoder = MLP(edge_dim, d_model // 2, d_model, n_layers=2)

        # Edge classification head
        self.classifier = MLP(3 * d_model, d_model, 1, n_layers=2)

    def forward(
        self,
        node_feat: torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_feat: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:             # (E,)
        # Encode node features with Transformer
        # TransformerEmbedder expects (seq_len, in_dim) or (batch, seq_len, in_dim)
        # Here we have single event, so shape is (N, node_dim)
        node_emb = self.node_encoder(node_feat)  # (N, d_model)

        # Encode edge features
        edge_emb = self.edge_encoder(edge_feat)  # (E, d_model)

        # Get source and destination node embeddings
        src, dst = edge_index[0], edge_index[1]
        src_emb = node_emb[src]  # (E, d_model)
        dst_emb = node_emb[dst]  # (E, d_model)

        # Concatenate and classify
        combined = torch.cat([src_emb, dst_emb, edge_emb], dim=-1)  # (E, 3*d_model)
        logits = self.classifier(combined)  # (E, 1)

        return logits.squeeze(-1).sigmoid()


class TrackFormerSeed(nn.Module):
    """TrackFormer-Seed: Complete transformer-based seeding model.

    Implements the full TrackFormer-Seed architecture from the proposal:
    1. Hit embedding and encoding with Transformer encoder
    2. Seed proposal generation with learnable seed queries and Transformer decoder
    3. Seed prediction heads for confidence, parameters, and hit assignment

    This is a more advanced model following the full Stroud et al. (2024) approach.
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seeds: int = 100,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.d_model = d_model
        self.max_seeds = max_seeds

        # Hit encoder (shared with TransformerEmbedder)
        self.hit_encoder = TransformerEmbedder(
            in_dim=node_dim,
            out_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        # Learnable seed queries
        self.seed_queries = nn.Parameter(torch.randn(max_seeds, d_model))

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Layer normalization for decoder
        self.decoder_norm = nn.LayerNorm(d_model)

        # Seed prediction heads
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.parameter_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5)  # e.g., position (3), direction (2), or more
        )

        self.assignment_head = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_feat: torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor | None = None,  # Optional, not used in this model
        edge_feat: torch.Tensor | None = None,   # Optional, not used in this model
    ) -> dict:
        """
        Args:
            node_feat: Hit features (N, node_dim)
            edge_index: Optional, for compatibility with edge classification API
            edge_feat: Optional, for compatibility with edge classification API

        Returns:
            dict with keys:
                seed_confidence: (max_seeds,) confidence scores for each seed
                seed_parameters: (max_seeds, param_dim) seed parameters
                hit_assignments: (max_seeds, N) attention weights for hit assignment
                seed_embeddings: (max_seeds, d_model) seed embeddings
        """
        # Encode hits
        hit_emb = self.hit_encoder(node_feat)  # (N, d_model)

        # Prepare seed queries (repeat for batch if needed)
        if hit_emb.dim() == 3:  # batched input (batch_size, N, d_model)
            batch_size = hit_emb.shape[0]
            seed_queries = self.seed_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, max_seeds, d_model)
        else:  # single event (N, d_model)
            hit_emb = hit_emb.unsqueeze(0)  # (1, N, d_model)
            seed_queries = self.seed_queries.unsqueeze(0)  # (1, max_seeds, d_model)
            batch_size = 1

        # Apply transformer decoder layers
        seed_features = seed_queries
        for layer in self.decoder_layers:
            seed_features = layer(seed_features, hit_emb)

        seed_features = self.decoder_norm(seed_features)

        # Apply prediction heads
        seed_confidence = self.confidence_head(seed_features).squeeze(-1)  # (batch_size, max_seeds)

        seed_parameters = self.parameter_head(seed_features)  # (batch_size, max_seeds, param_dim)

        # Hit assignment via attention
        seed_features_proj = self.assignment_head(seed_features)  # (batch_size, max_seeds, d_model)
        hit_assignments = torch.matmul(seed_features_proj, hit_emb.transpose(-2, -1))  # (batch_size, max_seeds, N)
        hit_assignments = F.softmax(hit_assignments / math.sqrt(self.d_model), dim=-1)

        # Remove batch dimension if single event
        if batch_size == 1 and node_feat.dim() == 2:
            seed_confidence = seed_confidence.squeeze(0)
            seed_parameters = seed_parameters.squeeze(0)
            hit_assignments = hit_assignments.squeeze(0)
            seed_features = seed_features.squeeze(0)

        return {
            'seed_confidence': seed_confidence,
            'seed_parameters': seed_parameters,
            'hit_assignments': hit_assignments,
            'seed_embeddings': seed_features,
        }


class EggNet(nn.Module):
    """EggNet attention GNN for edge classification.

    Faithful adaptation of the EggNet architecture (Calafiura et al. 2024,
    https://github.com/GNN4ITkTeam/Eggnet) for edge scoring on the E320
    detector graph.

    Key differences from the original (which outputs node embeddings for
    clustering via dynamic KNN): this version operates on a pre-built
    edge_index and produces per-edge sigmoid scores for direct edge
    classification.

    Architecture
    ------------
    1. Node encoder     : node_dim → hidden
    2. node_network_0   : hidden → hidden  (initial node processing, as in original)
    3. Edge encoder     : edge_dim → hidden
    4. n_iters × n_gnns_per_iter rounds of attention message passing:
       j = 0 : edge_net([h_i, h_j])       → (hidden+1)  — no edge features first pass
       j > 0 : edge_net([h_i, h_j, h_e])  → (hidden+1)
       per-node softmax attention weights
       agg = scatter_add(h_e_new × w, dst)
       node_net([h_node, agg]) → h_node'
    5. Edge decoder     : [h_i, h_j, h_e] → sigmoid score

    Parameters
    ----------
    node_dim, edge_dim : input feature dimensions
    hidden : representation width (H)
    n_iters : outer message-passing iterations
    n_gnns_per_iter : inner GNN rounds per iteration (≥1)
    recurrent : share weights across outer iterations (à la original EggNet)
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_iters: int = 4,
        n_gnns_per_iter: int = 2,
        recurrent: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_iters = n_iters
        self.n_gnns_per_iter = n_gnns_per_iter
        self.recurrent = recurrent

        self.node_encoder = MLP(node_dim, hidden, hidden, n_layers=3)
        # Initial node processing applied once before the iteration loop
        self.node_network_0 = MLP(hidden, hidden, hidden, n_layers=3)
        self.edge_encoder = MLP(edge_dim, hidden, hidden, n_layers=2)

        n_unique = 1 if recurrent else n_iters

        # j=0 edge network: input = [h_i, h_j] — no edge features on first inner pass
        self.edge_networks_first = nn.ModuleList([
            MLP(2 * hidden, hidden, hidden + 1, n_layers=3) for _ in range(n_unique)
        ])

        # j>0 edge networks: input = [h_i, h_j, h_e]
        if n_gnns_per_iter > 1:
            self.edge_networks_rest = nn.ModuleList([
                MLP(3 * hidden, hidden, hidden + 1, n_layers=3) for _ in range(n_unique)
            ])

        # node_net: [h_node, agg] → hidden  (shared across inner iters)
        self.node_networks = nn.ModuleList([
            MLP(2 * hidden, hidden, hidden, n_layers=3) for _ in range(n_unique)
        ])

        self.edge_decoder = MLP(3 * hidden, hidden, 1, n_layers=2)

    @staticmethod
    def _softmax_dst(attn: Tensor, dst: Tensor, N: int) -> Tensor:
        """Per-destination-node softmax over incoming edge attention logits."""
        attn_shifted = attn - attn.detach().max()
        attn_exp = torch.exp(attn_shifted)
        attn_sum = torch.zeros(N, device=attn.device, dtype=attn.dtype)
        attn_sum.index_add_(0, dst, attn_exp)
        return attn_exp / attn_sum[dst].clamp(min=1e-8)

    def forward(
        self,
        node_feat: Tensor,   # (N, node_dim)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, edge_dim)
    ) -> Tensor:             # (E,)
        src, dst = edge_index[0], edge_index[1]
        N = node_feat.shape[0]

        # Encode — node_network_0 provides initial processing as in original EggNet
        h_n = self.node_network_0(self.node_encoder(node_feat))  # (N, H)
        h_e = self.edge_encoder(edge_feat)                        # (E, H)

        for i in range(self.n_iters):
            k = 0 if self.recurrent else i

            for j in range(self.n_gnns_per_iter):
                # First inner iter: only node features (no accumulated edge repr)
                if j == 0:
                    e_out = self.edge_networks_first[k](
                        torch.cat([h_n[src], h_n[dst]], dim=-1)
                    )                          # (E, H+1)
                else:
                    e_out = self.edge_networks_rest[k](
                        torch.cat([h_n[src], h_n[dst], h_e], dim=-1)
                    )                          # (E, H+1)

                h_e = e_out[:, :-1]            # (E, H)
                attn = e_out[:, -1]            # (E,)

                # Attention-weighted aggregation
                w = self._softmax_dst(attn, dst, N)
                agg = torch.zeros(N, self.hidden, device=h_n.device, dtype=h_n.dtype)
                agg.index_add_(0, dst, h_e * w.unsqueeze(-1))

                # Node update
                h_n = self.node_networks[k](torch.cat([h_n, agg], dim=-1))

        score = self.edge_decoder(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))
        return score.squeeze(-1).sigmoid()


class InteractionGNNCell(nn.Module):
    """Single message-passing cell from Liu et al. (2023) HGNN.

    Edge update : [n_i, n_j, e] → e'  (with skip connection)
    Node update : aggregate(e') → agg; [n, agg] → n'  (with skip connection)
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.edge_network = MLP(3 * hidden, hidden, hidden, n_layers=3)
        self.node_network = MLP(2 * hidden, hidden, hidden, n_layers=3)

    def forward(self, nodes: Tensor, edges: Tensor, graph: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = graph[0], graph[1]
        N = nodes.shape[0]
        # Edge update with skip
        edges = self.edge_network(torch.cat([nodes[src], nodes[dst], edges], dim=-1)) + edges
        # Node update: aggregate incoming edge messages + skip
        edge_agg = torch.zeros(N, nodes.shape[-1], device=nodes.device, dtype=nodes.dtype)
        edge_agg.index_add_(0, dst, edges)
        nodes = self.node_network(torch.cat([nodes, edge_agg], dim=-1)) + nodes
        return nodes, edges


class HierarchicalGNNCell(nn.Module):
    """Hierarchical message-passing cell adapted from Liu et al. (2023).

    Updates nodes, edges, supernodes and superedges in sequence.
    Bipartite coupling: hits ↔ layer supernodes.

    Update order (following original paper):
    1. supernode ← supernode + superedge_agg + hit_agg  (bipartite, with skip)
    2. node      ← node + edge_agg + sn_msg             (bipartite, with skip)
    3. superedge ← superedge + [sn_i, sn_j, se]         (with skip)
    4. edge      ← edge + [n_i, n_j, e]                 (with skip)
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.supernode_network = MLP(3 * hidden, hidden, hidden, n_layers=3)
        self.node_network = MLP(3 * hidden, hidden, hidden, n_layers=3)
        self.superedge_network = MLP(3 * hidden, hidden, hidden, n_layers=3)
        self.edge_network = MLP(3 * hidden, hidden, hidden, n_layers=3)

    def forward(
        self,
        nodes: Tensor,              # (N, H)
        edges: Tensor,              # (E, H)
        supernodes: Tensor,         # (S, H)
        superedges: Tensor,         # (SE, H)
        graph: Tensor,              # (2, E)
        bipartite_graph: Tensor,    # (2, N)  hit_idx → layer_idx
        bipartite_weights: Tensor,  # (N, 1)
        super_graph: Tensor,        # (2, SE)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src, dst = graph[0], graph[1]
        b_src, b_dst = bipartite_graph[0], bipartite_graph[1]
        s_src, s_dst = super_graph[0], super_graph[1]
        N, S, H = nodes.shape[0], supernodes.shape[0], nodes.shape[-1]

        # 1. Supernode update: weighted hit aggregation + superedge aggregation
        hit_agg = torch.zeros(S, H, device=nodes.device, dtype=nodes.dtype)
        hit_agg.index_add_(0, b_dst, bipartite_weights * nodes[b_src])
        se_agg = torch.zeros(S, H, device=supernodes.device, dtype=supernodes.dtype)
        se_agg.index_add_(0, s_dst, superedges)
        supernodes = self.supernode_network(
            torch.cat([supernodes, se_agg, hit_agg], dim=-1)
        ) + supernodes

        # 2. Node update: edge aggregation + supernode message via bipartite
        edge_agg = torch.zeros(N, H, device=nodes.device, dtype=nodes.dtype)
        edge_agg.index_add_(0, dst, edges)
        sn_msg = bipartite_weights * supernodes[b_dst]
        nodes = self.node_network(
            torch.cat([nodes, edge_agg, sn_msg], dim=-1)
        ) + nodes

        # 3. Superedge update
        superedges = self.superedge_network(
            torch.cat([supernodes[s_src], supernodes[s_dst], superedges], dim=-1)
        ) + superedges

        # 4. Edge update
        edges = self.edge_network(
            torch.cat([nodes[src], nodes[dst], edges], dim=-1)
        ) + edges

        return nodes, edges, supernodes, superedges


class HierarchicalGNN(nn.Module):
    """Hierarchical GNN adapted from Liu et al. (2023) for E320.

    The original HGNN uses GMM clustering + cugraph (GPU-only) to construct
    supernodes dynamically.  Here we replace that with a physics-motivated
    construction: one supernode per detector layer (layer_id ∈ {0,…,n_layers-1}
    stored at node_feat[:, 0]).  The bipartite graph trivially maps each hit
    to the supernode of its layer; the super-graph is a bidirectional chain
    connecting adjacent layers.

    Architecture
    ------------
    1. Node encoder + edge encoder
    2. n_interaction_iters × InteractionGNNCell   (hit-level message passing)
    3. Supernode construction:
       - supernode[l] = mean of updated hit features on layer l
       - bipartite graph : hit i → supernode layer_ids[i]   (uniform weights)
       - super-graph     : chain 0↔1↔2↔3↔4   (bidirectional)
    4. n_hierarchical_iters × HierarchicalGNNCell (hits ↔ supernodes)
    5. Edge decoder : [n_i, n_j, e] → sigmoid score

    Parameters
    ----------
    n_layers : int
        Number of detector layers (5 for E320).
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 64,
        n_interaction_iters: int = 3,
        n_hierarchical_iters: int = 3,
        n_layers: int = 5,
    ):
        super().__init__()
        H = hidden_dim
        self.n_layers = n_layers

        self.node_encoder = MLP(node_dim, H, H, n_layers=3)
        self.edge_encoder = MLP(edge_dim, H, H, n_layers=2)

        self.interaction_cells = nn.ModuleList([
            InteractionGNNCell(H) for _ in range(n_interaction_iters)
        ])

        # Supernode encoder: mean of per-layer hit features → supernode repr
        self.supernode_encoder = MLP(H, H, H, n_layers=2)
        # Superedge encoder: pairs of adjacent-layer supernodes
        self.superedge_encoder = MLP(2 * H, H, H, n_layers=2)

        self.hierarchical_cells = nn.ModuleList([
            HierarchicalGNNCell(H) for _ in range(n_hierarchical_iters)
        ])

        self.edge_decoder = MLP(3 * H, H, 1, n_layers=2)

    def _build_super_structure(
        self, h_n: Tensor, node_feat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Construct supernodes, superedges and bipartite graph from layer IDs."""
        N, H = h_n.shape
        dev, dtype = h_n.device, h_n.dtype
        S = self.n_layers

        # Layer IDs stored in feature 0; clamp for safety
        layer_ids = node_feat[:, 0].long().clamp(0, S - 1)  # (N,)

        # Bipartite: each hit → its layer supernode
        bipartite_graph = torch.stack([
            torch.arange(N, device=dev),  # hit indices
            layer_ids,                     # supernode (layer) indices
        ])  # (2, N)

        # Supernode initial features: mean of hit representations per layer
        sn_agg = torch.zeros(S, H, device=dev, dtype=dtype)
        counts = torch.zeros(S, device=dev, dtype=dtype)
        sn_agg.index_add_(0, layer_ids, h_n)
        counts.index_add_(0, layer_ids, torch.ones(N, device=dev, dtype=dtype))
        counts.clamp_(min=1.0)
        h_sn = self.supernode_encoder(sn_agg / counts.unsqueeze(-1))  # (S, H)

        # Bipartite weights: uniform 1/count so messages sum to mean
        bipartite_weights = (1.0 / counts[layer_ids]).unsqueeze(-1)  # (N, 1)

        # Super-graph: bidirectional chain 0↔1↔…↔(S-1)
        fwd = torch.arange(S - 1, device=dev)
        bwd = torch.arange(1, S, device=dev)
        s_src = torch.cat([fwd, bwd])
        s_dst = torch.cat([bwd, fwd])
        super_graph = torch.stack([s_src, s_dst])  # (2, 2*(S-1))

        h_se = self.superedge_encoder(
            torch.cat([h_sn[s_src], h_sn[s_dst]], dim=-1)
        )  # (2*(S-1), H)

        return h_sn, h_se, bipartite_graph, bipartite_weights, super_graph

    def forward(
        self,
        node_feat: Tensor,   # (N, node_dim)
        edge_index: Tensor,  # (2, E)
        edge_feat: Tensor,   # (E, edge_dim)
    ) -> Tensor:             # (E,)
        src, dst = edge_index[0], edge_index[1]

        h_n = self.node_encoder(node_feat)   # (N, H)
        h_e = self.edge_encoder(edge_feat)   # (E, H)

        # Hit-level message passing (InteractionGNN block)
        for cell in self.interaction_cells:
            h_n, h_e = cell(h_n, h_e, edge_index)

        # Build hierarchical structure (layer-based supernodes)
        h_sn, h_se, bipartite_graph, bipartite_weights, super_graph = \
            self._build_super_structure(h_n, node_feat)

        # Hierarchical message passing
        for cell in self.hierarchical_cells:
            h_n, h_e, h_sn, h_se = cell(
                h_n, h_e, h_sn, h_se,
                edge_index, bipartite_graph, bipartite_weights, super_graph,
            )

        # Edge classification
        score = self.edge_decoder(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))
        return score.squeeze(-1).sigmoid()

