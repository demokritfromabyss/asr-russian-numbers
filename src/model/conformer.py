"""Conformer encoder block (Gulati et al., 2020)."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(nn.Module):
    """Macaron-style feed-forward with half-step residual."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
        return residual + 0.5 * x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return residual + self.drop(x)


class ConvModule(nn.Module):
    """Pointwise → GLU → Depthwise → BN → Swish → Pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.glu = nn.GLU(dim=-1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.glu(self.pw1(x))       # (B, T, d)
        x = x.transpose(1, 2)           # (B, d, T)
        x = self.act(self.bn(self.dw(x)))
        x = x.transpose(1, 2)           # (B, T, d)
        return residual + self.drop(self.pw2(x))


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, n_heads: int, conv_kernel: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_dim, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_dim, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)
