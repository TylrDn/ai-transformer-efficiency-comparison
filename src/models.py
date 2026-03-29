"""
models.py — Transformer variant loaders.

Variant A: Vanilla Transformer (standard scaled dot-product attention, O(n²) memory).
Variant B: Efficient Transformer (uses PyTorch's built-in memory-efficient / flash attention
           path via ``scaled_dot_product_attention`` with ``enable_flash_sdp`` where available,
           falling back to the math kernel otherwise).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward sublayer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Variant A — Vanilla multi-head attention (explicit QKV matrix multiply)
# ---------------------------------------------------------------------------


class VanillaAttention(nn.Module):
    """Standard scaled dot-product multi-head attention (O(n²) in sequence length)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, d_head)
        q = q.transpose(1, 2)  # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class VanillaTransformerLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = VanillaAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class VanillaTransformer(nn.Module):
    """Vanilla Transformer encoder (Variant A)."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [VanillaTransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(positions))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Variant B — Efficient Transformer (memory-efficient / flash attention path)
# ---------------------------------------------------------------------------


class EfficientAttention(nn.Module):
    """
    Multi-head attention using ``torch.nn.functional.scaled_dot_product_attention``.

    PyTorch ≥ 2.0 selects the most memory-efficient kernel automatically
    (Flash Attention when available, otherwise memory-efficient attention or
    the math fallback).  This avoids materialising the full (T×T) attention
    matrix, reducing peak memory from O(n²) to O(n).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's optimised kernel (Flash / memory-efficient / math)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class EfficientTransformerLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = EfficientAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class EfficientTransformer(nn.Module):
    """Efficient Transformer encoder (Variant B)."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EfficientTransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(positions))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

_VARIANTS = {
    "vanilla": VanillaTransformer,
    "efficient": EfficientTransformer,
}


def load_model(
    variant: str,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 1024,
    vocab_size: int = 32000,
    dropout: float = 0.0,
    device: Optional[str] = None,
) -> nn.Module:
    """Return an initialised Transformer variant ready for benchmarking.

    Parameters
    ----------
    variant:
        ``"vanilla"`` for Variant A, ``"efficient"`` for Variant B.
    device:
        Target device string (e.g. ``"cpu"``, ``"cuda"``).  Defaults to
        ``"cuda"`` when available, otherwise ``"cpu"``.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_VARIANTS)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _VARIANTS[variant](
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    model.eval()
    return model.to(device)
