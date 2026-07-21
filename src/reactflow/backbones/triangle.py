"""Triangle updates and attention for the ReactFlow 2.0 PairFormer.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 366-376.

These operations update the pair representation ``z_ij`` using information
from triangles ``(i, j, k)``.  They are the core of AlphaFold-style
PairFormer blocks.

Modules
-------
- :class:`TriangleMultiplicativeUpdate`: outgoing and incoming triangle
  multiplicative updates (the "triangle multiplication" of Evoformer).
- :class:`TriangleAttention`: starting/ending node attention over the pair
  representation (axial attention).
- :class:`PairTransition`: position-wise feed-forward for the pair stack.

All modules preserve pair symmetry ``z_ij = z_ji`` when the input is
symmetric: outgoing + incoming triangle updates are averaged to symmetrize,
and axial attention is applied along both axes and averaged.

Complexity
----------
- Triangle multiplicative update: ``O(L^3 * P)`` time, ``O(L^2 * P)`` memory.
- Triangle attention: ``O(L^3 * H)`` time, ``O(L^2 * H)`` memory.
- Pair transition: ``O(L^2 * P)`` time and memory.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        query: ``(..., L_q, H, D)``
        key: ``(..., L_k, H, D)``
        value: ``(..., L_k, H, D)``
        mask: optional ``(..., L_q, L_k)`` boolean (True = attend).
        dropout: optional dropout module.

    Returns:
        ``(..., L_q, H, D)``.
    """
    *batch, Lq, H, D = query.shape
    Lk = key.shape[-3]
    q = query.transpose(-2, -3)  # (..., H, Lq, D)
    k = key.transpose(-2, -3)    # (..., H, Lk, D)
    v = value.transpose(-2, -3)  # (..., H, Lk, D)
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (..., H, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-3), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)  # (..., H, Lq, D)
    out = out.transpose(-2, -3)  # (..., Lq, H, D)
    return out


# ---------------------------------------------------------------------------
# Triangle multiplicative update
# ---------------------------------------------------------------------------


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update (outgoing + incoming, symmetrized).

    Outgoing update: ``z_ij' = Linear( sum_k (a_ik ⊙ b_kj) )``
    Incoming update:  ``z_ij' = Linear( sum_k (a_kj ⊙ b_ik) )``

    The two are averaged to preserve symmetry: ``z_ij = 0.5 * (out + in)``.

    Args:
        pair_dim: pair representation dimension ``P``.
        hidden_dim: multiplicative hidden dimension (default ``P``).
        dropout: dropout rate.

    Complexity: ``O(L^3 * P)`` time, ``O(L^2 * P)`` memory.
    """

    def __init__(
        self,
        pair_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim or pair_dim
        # Gate projections (output is sigmoid gate)
        self.gate = nn.Linear(pair_dim, pair_dim)
        # Outgoing: a, b projections
        self.out_a = nn.Linear(pair_dim, self.hidden_dim)
        self.out_b = nn.Linear(pair_dim, self.hidden_dim)
        # Incoming: a, b projections (we reuse the same projections but applied
        # to the transposed pair, so this is equivalent to learning a separate
        # pair of projections for the incoming direction)
        self.in_a = nn.Linear(pair_dim, self.hidden_dim)
        self.in_b = nn.Linear(pair_dim, self.hidden_dim)
        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply triangle multiplicative update.

        Args:
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            ``(B, L, L, P)`` updated pair representation.
        """
        pair = self.norm(pair)
        B, L, _, P = pair.shape

        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
            pair = pair * pair_mask.unsqueeze(-1).float()

        # Outgoing: z_ij' = sum_k a_ik * b_kj  (i is row, k is shared)
        a_out = self.out_a(pair)  # (B, L, L, H)
        b_out = self.out_b(pair)  # (B, L, L, H)
        outgoing = torch.einsum("bikh,bkjh->bijh", a_out, b_out)  # (B, L, L, H)

        # Incoming: z_ij' = sum_k a_kj * b_ik  (j is column, k is shared)
        a_in = self.in_a(pair)   # (B, L, L, H)
        b_in = self.in_b(pair)   # (B, L, L, H)
        incoming = torch.einsum("bkjh,bikh->bijh", a_in, b_in)

        # Average outgoing and incoming, then symmetrize the update explicitly.
        # With separate projections, outgoing+incoming is NOT automatically
        # symmetric, so we enforce it here: update = 0.5 * (u + u^T).
        merged = 0.5 * (outgoing + incoming)
        merged = self.out_proj(merged)
        merged = 0.5 * (merged + merged.transpose(1, 2))  # enforce symmetry

        # Gate
        gate = torch.sigmoid(self.gate(pair))
        out = pair + self.dropout(gate * merged)
        return out


# ---------------------------------------------------------------------------
# Triangle attention
# ---------------------------------------------------------------------------


class TriangleAttention(nn.Module):
    """Axial pair attention along starting and ending nodes, symmetrized.

    Starting-node attention (attention over j for fixed i):
        z_ij' = Attention(query=z_ij, keys=z_ik, values=z_ik)

    Ending-node attention (attention over i for fixed j):
        z_ij' = Attention(query=z_ij, keys=z_kj, values=z_kj)

    The two are averaged to preserve symmetry.

    Args:
        pair_dim: pair representation dimension ``P``.
        num_heads: number of attention heads.
        head_dim: dimension per head (default ``pair_dim // num_heads``).
        dropout: dropout rate.

    Complexity: ``O(L^3 * H)`` time, ``O(L^2 * H)`` memory.
    """

    def __init__(
        self,
        pair_dim: int,
        *,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = max(1, pair_dim // num_heads)
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(pair_dim)
        # Starting node (row) attention
        self.to_q_row = nn.Linear(pair_dim, inner_dim, bias=False)
        self.to_k_row = nn.Linear(pair_dim, inner_dim, bias=False)
        self.to_v_row = nn.Linear(pair_dim, inner_dim, bias=False)
        # Ending node (column) attention
        self.to_q_col = nn.Linear(pair_dim, inner_dim, bias=False)
        self.to_k_col = nn.Linear(pair_dim, inner_dim, bias=False)
        self.to_v_col = nn.Linear(pair_dim, inner_dim, bias=False)
        # Output
        self.to_out = nn.Linear(inner_dim, pair_dim)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, L, inner) -> (B, L, L, H, D)``."""
        B, L1, L2, _ = x.shape
        return x.view(B, L1, L2, self.num_heads, self.head_dim)

    def forward(
        self,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply triangle attention.

        Args:
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            ``(B, L, L, P)`` updated pair representation.
        """
        residual = pair
        pair = self.norm(pair)
        B, L, _, P = pair.shape

        # Build key mask for attention (over the attended dimension)
        if mask is not None:
            attn_mask = mask.unsqueeze(1)  # (B, 1, L) - broadcast over query dim
            attn_mask = attn_mask.unsqueeze(-3)  # (B, 1, 1, L) for head broadcast
        else:
            attn_mask = None

        # Row attention: for each i, attend over k in z_ik
        # query: z_ij (along j), key/value: z_ik (along k)
        q_row = self._split_heads(self.to_q_row(pair))  # (B, L, L, H, D)
        k_row = self._split_heads(self.to_k_row(pair))  # (B, L, L, H, D) -- key along last L (k)
        v_row = self._split_heads(self.to_v_row(pair))
        # For row attention, query index is j, key index is k, shared index is i
        # We want: for each (i, j), attend over k in {(i, k)}
        # So we treat pair[B, i, :, :] as the sequence for each i
        # Reshape: (B, L, L, H, D) -> (B*L, L, H, D)
        q_row = q_row.reshape(B * L, L, self.num_heads, self.head_dim)
        k_row = k_row.reshape(B * L, L, self.num_heads, self.head_dim)
        v_row = v_row.reshape(B * L, L, self.num_heads, self.head_dim)
        row_mask = None
        if mask is not None:
            # Key mask (B, L) -> (B*L, 1, L) to broadcast over query dim L_q.
            # _attention expects mask (..., L_q, L_k); we pass (B*L, 1, L)
            # so mask.unsqueeze(-3) -> (B*L, 1, 1, L) broadcasts with
            # scores (B*L, H, L, L).
            row_mask = mask.unsqueeze(1).expand(B, L, L).reshape(B * L, L).unsqueeze(1)
        row_out = _attention(q_row, k_row, v_row, mask=row_mask, dropout=None)
        row_out = row_out.reshape(B, L, L, self.num_heads * self.head_dim)

        # Column attention: for each j, attend over k in z_kj
        # Transpose pair so the attended axis is the middle
        pair_t = pair.transpose(1, 2)  # (B, L, L, P) with (j, i) order
        q_col = self._split_heads(self.to_q_col(pair_t)).reshape(B * L, L, self.num_heads, self.head_dim)
        k_col = self._split_heads(self.to_k_col(pair_t)).reshape(B * L, L, self.num_heads, self.head_dim)
        v_col = self._split_heads(self.to_v_col(pair_t)).reshape(B * L, L, self.num_heads, self.head_dim)
        col_mask = None
        if mask is not None:
            col_mask = mask.unsqueeze(1).expand(B, L, L).reshape(B * L, L).unsqueeze(1)
        col_out = _attention(q_col, k_col, v_col, mask=col_mask, dropout=None)
        col_out = col_out.reshape(B, L, L, self.num_heads * self.head_dim)
        # col_out is in (j, i) order, transpose back to (i, j)
        col_out = col_out.transpose(1, 2)

        # Average row and column attention for symmetry.  With separate row/col
        # projections, 0.5*(row_out + col_out) is NOT automatically symmetric
        # (row_out[i,j] uses keys pair[i,:], col_out[j,i] uses keys pair[:,j]=
        # pair[j,:] by symmetry, but row_proj != col_proj).  We enforce
        # symmetry explicitly by averaging the merged output with its transpose.
        merged = 0.5 * (row_out + col_out)
        merged = self.to_out(merged)
        merged = 0.5 * (merged + merged.transpose(1, 2))  # enforce symmetry
        gate = torch.sigmoid(self.gate(pair))
        out = residual + self.dropout(gate * merged)
        return out


# ---------------------------------------------------------------------------
# Pair transition (FFN)
# ---------------------------------------------------------------------------


class PairTransition(nn.Module):
    """Position-wise feed-forward network for the pair representation.

    Formula: ``z_ij' = z_ij + Linear2(Dropout(GELU(Linear1(LayerNorm(z_ij)))))``

    Args:
        pair_dim: pair representation dimension ``P``.
        expansion: hidden dimension multiplier (default 4).
        dropout: dropout rate.

    Complexity: ``O(L^2 * P * expansion)`` time, ``O(L^2 * P)`` memory.
    """

    def __init__(
        self,
        pair_dim: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        hidden = pair_dim * expansion
        self.norm = nn.LayerNorm(pair_dim)
        self.linear1 = nn.Linear(pair_dim, hidden)
        self.linear2 = nn.Linear(hidden, pair_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Apply pair transition.

        Args:
            pair: ``(B, L, L, P)``.

        Returns:
            ``(B, L, L, P)`` updated pair representation.
        """
        residual = pair
        x = self.norm(pair)
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x
