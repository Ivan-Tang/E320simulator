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
from src.layers import MLP, MultiHeadAttention, WindowedMultiHeadAttention, PositionalEncoding3D, TransformerEncoderLayer, TransformerDecoderLayer

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
    4. emb_output   : hidden → emb_dim  (L2-normalised, stored as last_embeddings)
    5. EdgeDecoder  : [h_i, h_j, h_edge] → sigmoid score
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 64,
        n_mp: int = 2,
        emb_dim: int = 8,
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

        # Intermediate embedding output (L2-normalised in forward)
        self.emb_output = MLP(hidden, hidden, emb_dim, n_layers=2)
        self.last_embeddings: Tensor | None = None

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

        self.last_embeddings = F.normalize(self.emb_output(h_n), dim=-1)  # (N, emb_dim)
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
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
    """Transformer-based edge classifier with detector-layer-stratified attention.

    Instead of global self-attention over all N hits (O(N²), N~3500 for bg=700),
    applies self-attention within each of the 5 detector layers separately
    (~700 hits/layer), reducing complexity to O(5 × (N/5)²) = O(N²/5) and
    giving a strong spatial inductive bias: same-layer hits naturally share context.
    """

    N_DETECTOR_LAYERS = 5   # E320 ALPIDE detector layers

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_len: int = 1000,   # kept for API compatibility, not used
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.d_model = d_model

        # Project raw node features to d_model
        self.input_proj = nn.Linear(node_dim, d_model)

        # Per-layer transformer encoder (shared weights across layers)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Edge feature encoder (MLP)
        self.edge_encoder = MLP(edge_dim, d_model // 2, d_model, n_layers=2)

        # Edge classification head
        self.classifier = MLP(3 * d_model, d_model, 1, n_layers=2)
        # Initialise output bias to log-prior: positive edges ~2% of candidates
        # so sigmoid(bias) ≈ 0.02 → bias ≈ log(0.02/0.98) ≈ -3.9
        # Prevents predicting ~0.5 for every edge at init (99%+ fake rate).
        nn.init.constant_(self.classifier[-1].bias, -3.9)

    def forward(
        self,
        node_feat: torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_feat: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:             # (E,)
        N = node_feat.shape[0]
        device = node_feat.device

        # Project all nodes to d_model
        x = self.input_proj(node_feat)  # (N, d_model)

        # Detector-layer-stratified self-attention
        # node_feat[:, 0] = layer_id (integer 0..N_DETECTOR_LAYERS-1)
        layer_ids = node_feat[:, 0].long()
        node_emb = torch.zeros(N, self.d_model, device=device, dtype=x.dtype)

        for layer_i in range(self.N_DETECTOR_LAYERS):
            layer_idx = (layer_ids == layer_i).nonzero(as_tuple=True)[0]
            if layer_idx.numel() == 0:
                continue
            x_layer = x[layer_idx].unsqueeze(0)   # (1, n_i, d_model)
            for enc in self.encoder_layers:
                x_layer = enc(x_layer)
            node_emb[layer_idx] = self.norm(x_layer.squeeze(0))  # (n_i, d_model)

        # Encode edge features
        edge_emb = self.edge_encoder(edge_feat)  # (E, d_model)

        # Classify edges
        src, dst = edge_index[0], edge_index[1]
        combined = torch.cat([node_emb[src], node_emb[dst], edge_emb], dim=-1)  # (E, 3*d_model)
        return self.classifier(combined).squeeze(-1).sigmoid()


class HitFilterEncoderLayer(nn.Module):
    """Transformer encoder layer with windowed self-attention for hit filtering.

    Identical structure to TransformerEncoderLayer but uses
    WindowedMultiHeadAttention for O(N×w) instead of O(N²) complexity.
    For E320 hits sorted by x_trk_mm, the window covers a contiguous
    spatial neighbourhood so same-track hits always share a window.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        window_size: int = 256,
    ):
        super().__init__()
        self.self_attn = WindowedMultiHeadAttention(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # x: (1, N, d_model)
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class E320HitFilter(nn.Module):
    """Per-hit signal/noise classifier for E320, Stage 1 of the two-stage pipeline.

    Adapted from the hit filtering stage of Van Stroud et al. (2025).
    Hits are sorted by x_trk_mm before encoding to exploit spatial locality:
    signal hits from the same track cluster near the same x position, so a
    window of size w almost certainly contains all 5 hits of one track.

    Architecture
    ------------
    1. Sort hits by x_trk_mm
    2. input_proj : node_dim → d_model
    3. n_layers × HitFilterEncoderLayer  (windowed self-attention, O(N×w))
    4. classifier : d_model → d_model → d_model//2 → 1  (3-hidden-layer dense
       network, matching the paper)
    5. Unsort output to original hit order

    Input  : (N, node_dim)  raw hit features
    Output : (N,)           per-hit signal logit
    """

    def __init__(
        self,
        node_dim:        int   = NODE_DIM,
        d_model:         int   = 64,
        n_heads:         int   = 4,
        n_layers:        int   = 3,
        dim_feedforward: int   = 128,
        window_size:     int   = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, d_model)
        self.encoder = nn.ModuleList([
            HitFilterEncoderLayer(d_model, n_heads, dim_feedforward, dropout, window_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # 3-hidden-layer dense classifier (matching the paper's design)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, node_feat: Tensor) -> Tensor:  # (N, node_dim) → (N,)
        # Sort by x_trk_mm (feature index 1) for spatial locality
        sort_idx   = torch.argsort(node_feat[:, 1])
        sorted_feat = node_feat[sort_idx]

        x = self.input_proj(sorted_feat).unsqueeze(0)   # (1, N, d_model)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x).squeeze(0)                      # (N, d_model)

        logits_sorted = self.classifier(x).squeeze(-1)  # (N,)

        # Restore original hit order
        unsort_idx = torch.argsort(sort_idx)
        return logits_sorted[unsort_idx]                 # (N,)


class MaskFormerDecoderLayer(nn.Module):
    """Single decoder layer from Van Stroud et al. (2025).

    Order: masked cross-attention → self-attention → FFN (all pre-norm).

    MaskAttention: the binary mask predicted by the previous decoder layer
    gates which hits each query is allowed to attend to.  This focuses each
    track query on the hits already assigned to its current hypothesis.
    At the first layer (no previous mask) full attention is used.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,                       # (Q, d_model)
        memory:  Tensor,                       # (N, d_model)
        mask_logits: Tensor | None = None,     # (Q, N) from previous decoder layer
    ) -> Tensor:                               # (Q, d_model)
        # Build binary attention mask from previous mask prediction
        attn_mask = None
        if mask_logits is not None:
            bin_mask = (mask_logits.sigmoid() > 0.5)   # (Q, N) bool
            # Safety: if a query masks out all hits, fall back to full attention
            all_zero = ~bin_mask.any(dim=-1, keepdim=True)  # (Q, 1)
            bin_mask = bin_mask | all_zero                   # (Q, N)
            attn_mask = bin_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, Q, N)

        # Masked cross-attention (pre-norm)
        q = self.norm1(queries).unsqueeze(0)   # (1, Q, d_model)
        m = memory.unsqueeze(0)                # (1, N, d_model)
        cross = self.cross_attn(q, m, m, mask=attn_mask)  # (1, Q, d_model)
        queries = queries + self.dropout1(cross.squeeze(0))

        # Self-attention (pre-norm)
        q = self.norm2(queries).unsqueeze(0)
        self_out = self.self_attn(q, q, q).squeeze(0)
        queries = queries + self.dropout2(self_out)

        # FFN (pre-norm)
        queries = queries + self.dropout3(self.ffn(self.norm3(queries)))
        return queries


class E320TrackFormer(nn.Module):
    """MaskFormer-style track reconstruction for the E320 5-layer detector.

    Adapted from Van Stroud et al. (2025) "Transformers for Charged Particle
    Track Reconstruction in High Energy Physics" (PRX 15, 041046).

    Differences from the original:
    - No hit-filter stage (E320 has ~100 hits/event vs. 60k at LHC)
    - Full self-attention instead of sliding-window (N is small)
    - Adapted input features (7-dim: layer_id, x, y, z, size_x, size_y, size)

    Architecture
    ------------
    1. input_proj     : node_dim → d_model
    2. Transformer encoder (n_encoder_layers × TransformerEncoderLayer)
       → hit_memory (N, d_model)
    3. Q learnable object queries
    4. Decoder: n_decoder_layers × MaskFormerDecoderLayer
       Each layer l computes intermediate mask logits M^l = query @ hit_memory.T
       and passes them as attention masks to layer l+1.
    5. class_head     : (Q, d_model) → (Q,)  track/no-track logit
    6. mask_head      : (Q, d_model) @ hit_memory.T → (Q, N)  hit-assignment logit

    Output (dict)
    -------------
    track_logits    : (Q,)           raw logit — each query being a real track
    mask_logits     : (Q, N)         final per-hit assignment logits
    aux_mask_logits : list[(Q, N)]   intermediate mask logits (for auxiliary loss)
    hit_memory      : (N, d_model)   encoder hit embeddings
    """

    def __init__(
        self,
        node_dim:         int = NODE_DIM,
        d_model:          int = 128,
        n_heads:          int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dim_feedforward:  int = 256,
        max_queries:      int = 30,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.max_queries = max_queries
        self.d_model = d_model

        # Encoder
        self.input_proj  = nn.Linear(node_dim, d_model)
        self.encoder     = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.queries     = nn.Parameter(torch.randn(max_queries, d_model))
        self.decoder     = nn.ModuleList([
            MaskFormerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output heads
        self.class_head      = nn.Linear(d_model, 1)
        self.mask_token_proj = nn.Linear(d_model, d_model)

    def _mask_logits(self, query_emb: Tensor, hit_memory: Tensor) -> Tensor:
        """Dot-product mask: (Q, d) @ (d, N) → (Q, N), scaled."""
        tokens = self.mask_token_proj(query_emb)                   # (Q, d_model)
        return torch.matmul(tokens, hit_memory.T) / math.sqrt(self.d_model)

    def forward(
        self,
        node_feat:  Tensor,              # (N, node_dim)
        edge_index: Tensor | None = None,  # unused — kept for API compatibility
        edge_feat:  Tensor | None = None,  # unused
    ) -> dict:
        # Encode all hits
        x = self.input_proj(node_feat).unsqueeze(0)  # (1, N, d_model)
        for layer in self.encoder:
            x = layer(x)
        hit_memory = self.encoder_norm(x).squeeze(0)  # (N, d_model)

        # Decode with MaskAttention
        queries = self.queries.clone()                # (Q, d_model)
        mask_logits_prev: Tensor | None = None
        aux_mask_logits: list[Tensor] = []

        for layer in self.decoder:
            queries = layer(queries, hit_memory, mask_logits=mask_logits_prev)
            mask_logits_prev = self._mask_logits(queries, hit_memory)
            aux_mask_logits.append(mask_logits_prev)

        queries = self.decoder_norm(queries)

        return {
            "track_logits":    self.class_head(queries).squeeze(-1),   # (Q,)
            "mask_logits":     self._mask_logits(queries, hit_memory),  # (Q, N)
            "aux_mask_logits": aux_mask_logits[:-1],  # all but last (recomputed above)
            "hit_memory":      hit_memory,
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

    Follows the original two-stage design more closely:
    1. InteractionGNN block → intermediate L2-normalized embeddings (emb_dim)
    2. Cosine-similarity soft bipartite weights (hit ↔ layer-mean embedding)
    3. HierarchicalGNN block with hits, edges, supernodes and superedges
    4. Edge decoder → sigmoid score

    The intermediate embeddings are stored as ``last_embeddings`` after each
    forward pass for optional embedding (HingeLoss) training.

    Parameters
    ----------
    n_layers : int
        Number of detector layers (5 for E320).
    emb_dim : int
        Dimension of intermediate L2-normalized embeddings (default 8, same as
        the original paper).
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 64,
        n_interaction_iters: int = 3,
        n_hierarchical_iters: int = 3,
        n_layers: int = 5,
        emb_dim: int = 8,
    ):
        super().__init__()
        H = hidden_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        self.node_encoder = MLP(node_dim, H, H, n_layers=3)
        self.edge_encoder = MLP(edge_dim, H, H, n_layers=2)

        self.interaction_cells = nn.ModuleList([
            InteractionGNNCell(H) for _ in range(n_interaction_iters)
        ])

        # Intermediate embedding output: hidden → emb_dim, then L2-normalized.
        # Mirrors the original paper's per-hit embedding space used for clustering.
        self.emb_output = MLP(H, H, emb_dim, n_layers=3)

        # Supernode encoder: mean of per-layer hit features → supernode repr
        self.supernode_encoder = MLP(H, H, H, n_layers=2)
        # Superedge encoder: pairs of adjacent-layer supernodes
        self.superedge_encoder = MLP(2 * H, H, H, n_layers=2)

        self.hierarchical_cells = nn.ModuleList([
            HierarchicalGNNCell(H) for _ in range(n_hierarchical_iters)
        ])

        self.edge_decoder = MLP(3 * H, H, 1, n_layers=2)

        # Set during forward; used by training loop for optional embedding loss.
        self.last_embeddings: Tensor | None = None

    def _build_super_structure(
        self, h_n: Tensor, node_feat: Tensor, embeddings: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Construct supernodes, superedges and bipartite graph from layer IDs.

        Bipartite weights are computed via cosine similarity between each hit's
        intermediate embedding and the mean embedding of its detector layer,
        matching the spirit of the original paper's dynamic graph construction.
        """
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
        counts = torch.zeros(S, device=dev, dtype=dtype)
        counts.index_add_(0, layer_ids, torch.ones(N, device=dev, dtype=dtype))
        counts.clamp_(min=1.0)

        sn_agg = torch.zeros(S, H, device=dev, dtype=dtype)
        sn_agg.index_add_(0, layer_ids, h_n)
        h_sn = self.supernode_encoder(sn_agg / counts.unsqueeze(-1))  # (S, H)

        # Soft bipartite weights via cosine similarity of intermediate embeddings.
        # Compute L2-normalised layer-mean embeddings.
        emb_agg = torch.zeros(S, self.emb_dim, device=dev, dtype=dtype)
        emb_agg.index_add_(0, layer_ids, embeddings)
        emb_layer = F.normalize(emb_agg / counts.unsqueeze(-1), dim=-1)  # (S, emb_dim)

        # Cosine similarity ∈ [0, 1] between each hit embedding and its layer mean.
        cos_sim = (embeddings * emb_layer[layer_ids]).sum(dim=-1, keepdim=True).clamp(min=0.0)  # (N, 1)

        # Normalise per layer so that aggregated messages approximate a mean.
        cos_sum = torch.zeros(S, 1, device=dev, dtype=dtype)
        cos_sum.index_add_(0, layer_ids, cos_sim)
        cos_mean = (cos_sum / counts.unsqueeze(-1))[layer_ids]  # (N, 1)
        bipartite_weights = cos_sim / cos_mean.clamp(min=1e-8)  # (N, 1)

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

        # Intermediate L2-normalised embeddings (mirrors original paper Stage 1 output)
        embeddings = F.normalize(self.emb_output(h_n), dim=-1)  # (N, emb_dim)
        self.last_embeddings = embeddings  # expose for optional embedding loss

        # Build hierarchical structure with embedding-based soft bipartite weights
        h_sn, h_se, bipartite_graph, bipartite_weights, super_graph = \
            self._build_super_structure(h_n, node_feat, embeddings)

        # Hierarchical message passing
        for cell in self.hierarchical_cells:
            h_n, h_e, h_sn, h_se = cell(
                h_n, h_e, h_sn, h_se,
                edge_index, bipartite_graph, bipartite_weights, super_graph,
            )

        # Edge classification
        score = self.edge_decoder(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))
        return score.squeeze(-1).sigmoid()

