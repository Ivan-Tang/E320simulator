import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

def MLP(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class PositionalEncoding3D(nn.Module):
    """3D sinusoidal positional encoding for hit positions.

    Based on "Attention is All You Need" positional encoding but extended to 3D.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding for x, y, z separately
        pe = torch.zeros(3, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[0, :, 0::2] = torch.sin(position * div_term)  # x: even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # x: odd indices
        pe[1, :, 0::2] = torch.sin(position * div_term)  # y: even indices
        pe[1, :, 1::2] = torch.cos(position * div_term)  # y: odd indices
        pe[2, :, 0::2] = torch.sin(position * div_term)  # z: even indices
        pe[2, :, 1::2] = torch.cos(position * div_term)  # z: odd indices

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            x, y, z: Tensors of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Positional encoding of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        # Scale coordinates to [0, max_len-1]
        x_idx = (x * (self.max_len - 1)).long().clamp(0, self.max_len - 1)
        y_idx = (y * (self.max_len - 1)).long().clamp(0, self.max_len - 1)
        z_idx = (z * (self.max_len - 1)).long().clamp(0, self.max_len - 1)

        if x.dim() == 1:
            pe_x = self.pe[0, x_idx]
            pe_y = self.pe[1, y_idx]
            pe_z = self.pe[2, z_idx]
        else:
            pe_x = self.pe[0, x_idx].view(x_idx.shape[0], x_idx.shape[1], -1)
            pe_y = self.pe[1, y_idx].view(y_idx.shape[0], y_idx.shape[1], -1)
            pe_z = self.pe[2, z_idx].view(z_idx.shape[0], z_idx.shape[1], -1)

        return pe_x + pe_y + pe_z


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        batch_size = q.size(0)

        # Linear projections and split into heads
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_linear(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Pre-LN: normalise before attention for better gradient flow
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN: normalise before FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
        # Self-attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Cross-attention
        attn_out = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(attn_out)
        x = self.norm2(x)

        # Feedforward
        ffn_out = self.ffn(x)
        x = x + self.dropout3(ffn_out)
        x = self.norm3(x)

        return x


class WindowedMultiHeadAttention(nn.Module):
    """Multi-head attention with sliding window for efficient long sequences.

    Based on the implementation in Stroud et al. (2024) for hit filtering.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, seq_len, _ = q.shape

        # Linear projections and split into heads
        q = self.q_linear(q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Initialize output tensor
        out = torch.zeros_like(q)

        # Apply sliding window attention
        for start in range(0, seq_len, self.window_size):
            end = min(start + self.window_size, seq_len)

            q_window = q[:, :, start:end, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            # Scaled dot-product attention within window
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if mask is not None:
                # Apply mask if provided (adjust window slicing)
                mask_window = mask[:, start:end, start:end] if mask.dim() == 3 else mask
                scores = scores.masked_fill(mask_window == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            window_out = torch.matmul(attn, v_window)
            out[:, :, start:end, :] = window_out

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(out)


class CylindricalPositionalEncoding(nn.Module):
    """Positional encoding for cylindrical coordinates (r, φ, z).

    Uses cyclic encoding for φ coordinate as described in Stroud et al. (2024).
    """

    def __init__(self, d_model: int, max_r: float = 1000.0, max_z: float = 3000.0):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding for r, φ, z
        # For φ, we use cyclic encoding (sin/cos of φ directly)
        # For r and z, we use standard sinusoidal encoding with scaling

        # Frequency terms for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

        # Scaling factors
        self.max_r = max_r
        self.max_z = max_z

    def forward(self, r: Tensor, phi: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            r: Radial coordinate (batch_size, seq_len) or (seq_len,)
            phi: Azimuthal angle in radians (batch_size, seq_len) or (seq_len,)
            z: Longitudinal coordinate (batch_size, seq_len) or (seq_len,)

        Returns:
            Positional encoding of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        if r.dim() == 1:
            r = r.unsqueeze(0)
            phi = phi.unsqueeze(0)
            z = z.unsqueeze(0)
            batch_size, seq_len = 1, r.shape[1]
            squeeze = True
        else:
            batch_size, seq_len = r.shape
            squeeze = False

        # Scale r and z to [0, 1] for encoding
        r_scaled = r / self.max_r
        z_scaled = z / self.max_z

        # Initialize positional encoding
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=r.device, dtype=r.dtype)

        # Sinusoidal encoding for r and z
        # Even indices: sin, odd indices: cos
        pe[:, :, 0::2] += torch.sin(r_scaled.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] += torch.cos(r_scaled.unsqueeze(-1) * self.div_term)

        pe[:, :, 0::2] += torch.sin(z_scaled.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] += torch.cos(z_scaled.unsqueeze(-1) * self.div_term)

        # Cyclic encoding for φ (direct sin/cos of φ)
        # Use different frequency bands to avoid overlap
        phi_freq = self.div_term * 10.0  # Different frequency scale for φ
        pe[:, :, 0::2] += torch.sin(phi.unsqueeze(-1) * phi_freq)
        pe[:, :, 1::2] += torch.cos(phi.unsqueeze(-1) * phi_freq)

        if squeeze:
            pe = pe.squeeze(0)

        return pe


class MaskAttention(nn.Module):
    """MaskAttention mechanism for MaskFormer-style models.

    Generates attention masks from intermediate mask proposals.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mask_proj = nn.Linear(d_model, n_heads)

    def forward(self, queries: Tensor, keys: Tensor, mask_logits: Tensor = None) -> Tensor:
        """
        Args:
            queries: (batch_size, n_queries, d_model)
            keys: (batch_size, seq_len, d_model)
            mask_logits: Optional (batch_size, n_queries, seq_len) mask logits from previous layer

        Returns:
            attention_mask: (batch_size, n_heads, n_queries, seq_len)
        """
        batch_size, n_queries, _ = queries.shape
        _, seq_len, _ = keys.shape

        if mask_logits is not None:
            # Use previous mask logits to guide attention
            # Reshape to (batch_size, n_heads, n_queries, seq_len)
            # by projecting to n_heads dimension
            mask_attention = self.mask_proj(mask_logits).view(batch_size, n_queries, self.n_heads, seq_len)
            mask_attention = mask_attention.permute(0, 2, 1, 3)  # (batch_size, n_heads, n_queries, seq_len)
            mask_attention = F.sigmoid(mask_attention)
        else:
            # Initial uniform attention
            mask_attention = torch.ones(batch_size, self.n_heads, n_queries, seq_len,
                                      device=queries.device, dtype=queries.dtype)

        return mask_attention