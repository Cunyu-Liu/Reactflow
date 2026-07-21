"""Single-pair communication for the ReactFlow 2.0 PairFormer.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 373-374.

These modules bridge the single representation ``s_i in R^C`` and the pair
representation ``z_ij in R^P``:

- :class:`OuterProductMean`: updates the pair stack from the single stack
  via the outer-product mean (Evoformer-style).
- :class:`PairToSingleAttention`: updates the single stack by attending over
  the pair representation.

Complexity
----------
- OuterProductMean: ``O(L^2 * C * P)`` time, ``O(L^2 * P)`` memory.
- PairToSingleAttention: ``O(L^2 * C)`` time, ``O(L * C)`` memory.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class OuterProductMean(nn.Module):
    """Update the pair representation from the single representation.

    Formula (Evoformer): for each pair ``(i, j)``,

        z_ij' = Linear( mean_k ( a_i ⊗ b_j ) ) + z_ij

    where ``a_i, b_j in R^C`` are projections of the single representation
    and ``⊗`` is the outer product.  The mean is over a per-position mask.

    This module is the standard single-to-pair communication in the
    Evoformer / PairFormer architecture.

    Args:
        single_dim: dimension ``C`` of the single representation.
        pair_dim: dimension ``P`` of the pair representation.
        projection_dim: intermediate dimension ``D`` for the outer product.
            Default ``min(C, P)``.

    Complexity: ``O(L^2 * D^2)`` time, ``O(L^2 * P)`` memory.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.projection_dim = projection_dim or min(single_dim, pair_dim)
        self.norm = nn.LayerNorm(single_dim)
        self.proj_a = nn.Linear(single_dim, self.projection_dim, bias=False)
        self.proj_b = nn.Linear(single_dim, self.projection_dim, bias=False)
        # Flatten outer product D*D -> pair_dim
        self.out = nn.Linear(self.projection_dim * self.projection_dim, pair_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add the outer-product-mean update to the pair representation.

        Args:
            single: ``(B, L, C)``.
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            ``(B, L, L, P)`` updated pair representation.
        """
        s = self.norm(single)
        a = self.proj_a(s)  # (B, L, D)
        b = self.proj_b(s)  # (B, L, D)

        # Memory-efficient outer product mean.
        # We want: update[b,i,j,p] = sum_{d1,d2} a[b,i,d1] * b[b,j,d2] * W[d1,d2,p]
        # where W = self.out.weight reshaped from (P, D*D) to (P, D, D) -> (D, D, P).
        # Decompose into two einsums to avoid materializing (B, L, L, D, D):
        #   c[b,j,d1,p] = sum_{d2} b[b,j,d2] * W[d1,d2,p]   # (B, L, D, P)
        #   update[b,i,j,p] = sum_{d1} a[b,i,d1] * c[b,j,d1,p]  # (B, L, L, P)
        D = self.projection_dim
        P = self.pair_dim
        W = self.out.weight.t().reshape(D, D, P)  # (D, D, P)
        c = torch.einsum("bje,dep->bjdp", b, W)  # (B, L, D, P)
        update = torch.einsum("bid,bjdp->bijp", a, c)  # (B, L, L, P)
        update = update + self.out.bias  # add bias

        # Symmetrize the update to preserve pair symmetry
        update = 0.5 * (update + update.transpose(1, 2))

        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()
            update = update * pair_mask

        return pair + update


class PairToSingleAttention(nn.Module):
    """Update the single representation by attending over the pair stack.

    For each position ``i``, we attend over the row ``z_i, :`` (i.e., the
    pair representation of ``i`` with all other positions ``k``).  The query
    is the single representation ``s_i``, and the keys/values are derived
    from the pair representation ``z_ik``.

    Formula:

        q_i = W_q s_i,  k_ik = W_k z_ik,  v_ik = W_v z_ik
        s_i' = s_i + W_o Attention(q_i, k_i:, v_i:)

    This is the standard pair-to-single communication in the Evoformer.

    Args:
        single_dim: dimension ``C`` of the single representation.
        pair_dim: dimension ``P`` of the pair representation.
        num_heads: number of attention heads.
        head_dim: dimension per head.

    Complexity: ``O(L^2 * C)`` time, ``O(L * C)`` memory.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = max(1, single_dim // num_heads)
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim

        self.norm_single = nn.LayerNorm(single_dim)
        self.norm_pair = nn.LayerNorm(pair_dim)
        self.to_q = nn.Linear(single_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(pair_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(pair_dim, inner_dim, bias=False)
        # Bias from single representation
        self.to_bias = nn.Linear(single_dim, num_heads, bias=False)
        self.to_out = nn.Linear(inner_dim, single_dim)
        self.gate = nn.Linear(single_dim, single_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply pair-to-single attention.

        Args:
            single: ``(B, L, C)``.
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            ``(B, L, C)`` updated single representation.
        """
        residual = single
        B, L, C = single.shape
        s = self.norm_single(single)
        p = self.norm_pair(pair)

        q = self.to_q(s)  # (B, L, inner)
        # For each i, keys/values come from pair[i, :, :] -- shape (B, L, L, P) -> (B*L, L, P)
        k = self.to_k(p)  # (B, L, L, inner)
        v = self.to_v(p)  # (B, L, L, inner)
        bias = self.to_bias(s)  # (B, L, num_heads)

        # Reshape to multi-head
        q = q.view(B, L, self.num_heads, self.head_dim)             # (B, L, H, D)
        k = k.view(B, L, L, self.num_heads, self.head_dim)          # (B, L_i, L_k, H, D)
        v = v.view(B, L, L, self.num_heads, self.head_dim)

        # Reshape for attention: for each i, attend over k
        # q: (B, L_i, H, D) -> (B*L_i, H, 1, D)
        # k: (B, L_i, L_k, H, D) -> (B*L_i, H, L_k, D)
        q_h = q.transpose(1, 2).reshape(B * L, self.num_heads, 1, self.head_dim)  # (B*L, H, 1, D)
        k_h = k.permute(0, 1, 3, 2, 4).reshape(B * L, self.num_heads, L, self.head_dim)  # (B*L, H, L, D)
        v_h = v.permute(0, 1, 3, 2, 4).reshape(B * L, self.num_heads, L, self.head_dim)
        # bias: (B, L, H) -> (B*L, H, 1, 1) so it broadcasts over the key axis
        bias_h = bias.view(B * L, self.num_heads, 1, 1)  # (B*L, H, 1, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # (B*L, H, 1, L)
        scores = scores + bias_h  # add bias to all keys (broadcasts over L)

        if mask is not None:
            # For each i, the k mask is mask[B, :]
            k_mask = mask.unsqueeze(1).expand(B, L, L).reshape(B * L, L)  # (B*L, L)
            scores = scores.masked_fill(~k_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_h)  # (B*L, H, 1, D)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        out = self.to_out(out)
        gate = torch.sigmoid(self.gate(s))
        return residual + self.dropout(gate * out)


# ---------------------------------------------------------------------------
# Single (row) attention — intra-single stack communication
# ---------------------------------------------------------------------------


class SingleRowAttention(nn.Module):
    """Standard multi-head self-attention over the single stack.

    Formula: ``s_i' = s_i + W_o Softmax(QK^T / sqrt(D)) V``.

    Args:
        single_dim: dimension ``C`` of the single representation.
        num_heads: number of attention heads.
        head_dim: dimension per head.
        dropout: dropout rate.

    Complexity: ``O(L^2 * C)`` time, ``O(L * C)`` memory.
    """

    def __init__(
        self,
        single_dim: int,
        *,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = max(1, single_dim // num_heads)
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(single_dim)
        self.to_q = nn.Linear(single_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(single_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(single_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, single_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        single: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply single-row self-attention.

        Args:
            single: ``(B, L, C)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            ``(B, L, C)`` updated single representation.
        """
        residual = single
        s = self.norm(single)
        B, L, _ = s.shape
        q = self.to_q(s).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = self.to_k(s).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(s).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, L, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        out = self.to_out(out)
        return residual + self.dropout(out)


class SingleTransition(nn.Module):
    """Position-wise feed-forward network for the single representation.

    Formula: ``s_i' = s_i + Linear2(Dropout(GELU(Linear1(LayerNorm(s_i)))))``

    Args:
        single_dim: dimension ``C`` of the single representation.
        expansion: hidden dimension multiplier.
        dropout: dropout rate.

    Complexity: ``O(L * C * expansion)`` time, ``O(L * C)`` memory.
    """

    def __init__(
        self,
        single_dim: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(single_dim)
        hidden = single_dim * expansion
        self.linear1 = nn.Linear(single_dim, hidden)
        self.linear2 = nn.Linear(hidden, single_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        """Apply single transition.

        Args:
            single: ``(B, L, C)``.

        Returns:
            ``(B, L, C)`` updated single representation.
        """
        residual = single
        x = self.norm(single)
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x
