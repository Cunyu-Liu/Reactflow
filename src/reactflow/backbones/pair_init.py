"""Symmetric pair initialization for the ReactFlow 2.0 static PairFormer.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 360-364.

The pair representation ``z_ij in R^P`` is initialized from the single
representation ``s_i, s_j in R^C`` and deterministic chemistry features
(relative distance, pair compatibility).  The initialization must satisfy
``z_ij = z_ji`` by construction.

Formula
-------
The initial pair features are the concatenation of:

1. ``proj_left(h_i)``     -- left single projection
2. ``proj_right(h_j)``    -- right single projection
3. ``h_i ⊙ h_j``          -- Hadamard product (symmetric)
4. ``|h_i - h_j|``        -- absolute difference (symmetric)
5. ``dist_emb(|i - j|)``  -- relative distance embedding (symmetric)
6. ``compat(b_i, b_j)``   -- 3-channel pair compatibility (symmetric)

To guarantee ``z_ij = z_ji`` while still allowing asymmetric ``proj_left``
and ``proj_right``, the final initialization symmetrizes as:

    z_ij = 0.5 * (z_ij_raw + z_ji_raw)

where ``z_ij_raw`` is the unsymmetric concatenation.  This is cheaper than
forcing ``proj_left = proj_right`` and still gives exact symmetry.

An alternative is to use the same projection for left and right
(``proj_left = proj_right``), which makes ``z_ij_raw = z_ji_raw`` by
construction; this is controlled by the ``share_projection`` flag.

Complexity
----------
- Time: ``O(L^2 * P)`` for the pair grid.
- Memory: ``O(L^2 * P)``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .embeddings import (
    RelativeDistanceEmbedding,
    pair_compatibility_matrix,
)


class SymmetricPairInit(nn.Module):
    """Initialize a symmetric pair representation from singles.

    Args:
        single_dim: dimension ``C`` of the single representation.
        pair_dim: dimension ``P`` of the pair representation.
        share_projection: if True, use a single projection for left and
            right (guarantees ``z_ij_raw = z_ji_raw`` by construction).
            If False, use separate projections and symmetrize via
            ``z_ij = 0.5 * (z_ij_raw + z_ji_raw)``.
        use_distance_embedding: include the learned relative distance embedding.
        use_compatibility: include the 3-channel pair compatibility features.
        num_distance_bins: number of relative distance bins (only used if
            ``use_distance_embedding`` is True).

    Complexity: ``O(L^2 * P)`` time and memory.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        share_projection: bool = True,
        use_distance_embedding: bool = True,
        use_compatibility: bool = True,
        distance_boundaries: tuple = (1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256),
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.share_projection = share_projection
        self.use_distance_embedding = use_distance_embedding
        self.use_compatibility = use_compatibility

        # Determine per-component dimensions
        # left/right product: each single_dim
        # hadamard: single_dim
        # abs_diff: single_dim
        # distance: pair_dim (we project to pair_dim)
        # compatibility: 3 (we project to pair_dim)
        nuc_component_dim = 4 * single_dim  # left + right + hadamard + abs_diff
        extra_dim = 0
        if use_distance_embedding:
            extra_dim += pair_dim
        if use_compatibility:
            extra_dim += pair_dim
        total_input_dim = nuc_component_dim + extra_dim

        if share_projection:
            self.left_proj = nn.Linear(single_dim, single_dim, bias=False)
            self.right_proj = self.left_proj
        else:
            self.left_proj = nn.Linear(single_dim, single_dim, bias=False)
            self.right_proj = nn.Linear(single_dim, single_dim, bias=False)

        # Final projection from total_input_dim -> pair_dim
        self.out_proj = nn.Linear(total_input_dim, pair_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        if use_distance_embedding:
            self.distance_embedding = RelativeDistanceEmbedding(
                pair_dim, boundaries=distance_boundaries,
            )
        if use_compatibility:
            self.compat_proj = nn.Linear(3, pair_dim)
            nn.init.normal_(self.compat_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.compat_proj.bias)

        self.norm = nn.LayerNorm(pair_dim)

    def forward(
        self,
        single: torch.Tensor,
        indices: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the symmetric initial pair representation.

        Args:
            single: FloatTensor of shape ``(B, L, C)``.
            indices: LongTensor of shape ``(B, L)`` with nucleotide vocab indices.
            mask: optional BoolTensor of shape ``(B, L)`` where True indicates
                a real (non-padding) position.  Padded positions get zero pair
                features.

        Returns:
            FloatTensor of shape ``(B, L, L, P)`` with ``z_ij = z_ji``.
        """
        B, L, C = single.shape
        device = single.device

        left = self.left_proj(single)   # (B, L, C)
        right = self.right_proj(single)  # (B, L, C)

        # Hadamard and abs_diff are computed from the UNPROJECTED single
        # representation so they are symmetric: h_i * h_j = h_j * h_i.
        single_i = single.unsqueeze(2)  # (B, L, 1, C)
        single_j = single.unsqueeze(1)  # (B, 1, L, C)
        hadamard = single_i * single_j   # (B, L, L, C) symmetric
        abs_diff = torch.abs(single_i - single_j)  # (B, L, L, C) symmetric

        # Build the (i, j) and (j, i) orderings of the projected components.
        # raw_ij uses [left_i, right_j, ...] and raw_ji uses [left_j, right_i, ...].
        # After applying out_proj and averaging, the result is symmetric:
        #   pair[b,i,j] = 0.5 * (out_proj([left_i, right_j, ...]) +
        #                        out_proj([left_j, right_i, ...]))
        #   pair[b,j,i] = 0.5 * (out_proj([left_j, right_i, ...]) +
        #                        out_proj([left_i, right_j, ...]))
        # These are equal because addition commutes.
        left_i_exp = left.unsqueeze(2).expand(B, L, L, C)
        right_j_exp = right.unsqueeze(1).expand(B, L, L, C)
        left_j_exp = left.unsqueeze(1).expand(B, L, L, C)
        right_i_exp = right.unsqueeze(2).expand(B, L, L, C)

        components_ij = [left_i_exp, right_j_exp, hadamard, abs_diff]
        components_ji = [left_j_exp, right_i_exp, hadamard, abs_diff]

        if self.use_distance_embedding:
            dist_emb = self.distance_embedding(L, device=device)  # (L, L, P)
            dist_emb = dist_emb.unsqueeze(0).expand(B, -1, -1, -1)
            components_ij.append(dist_emb)
            components_ji.append(dist_emb)  # symmetric

        if self.use_compatibility:
            compat = pair_compatibility_matrix(indices)  # (B, L, L, 3)
            compat_emb = self.compat_proj(compat)  # (B, L, L, P)
            components_ij.append(compat_emb)
            components_ji.append(compat_emb)  # symmetric

        raw_ij = torch.cat(components_ij, dim=-1)  # (B, L, L, total_input)
        raw_ji = torch.cat(components_ji, dim=-1)

        # Always symmetrize.  When share_projection=True, left==right, so
        # raw_ij and raw_ji swap the first two components but the averaging
        # still produces a symmetric result (and is a no-op only if out_proj
        # treats the first two components symmetrically, which it need not).
        z_ij = self.out_proj(raw_ij)
        z_ji = self.out_proj(raw_ji)
        pair = 0.5 * (z_ij + z_ji)

        pair = self.norm(pair)

        # Apply mask: zero out pairs involving padding positions
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
            pair = pair * pair_mask.unsqueeze(-1).float()

        # Zero out diagonal (a position cannot pair with itself)
        diag = torch.eye(L, dtype=torch.bool, device=device)
        pair = pair * (~diag).unsqueeze(0).unsqueeze(-1).float()

        return pair

    def symmetry_residual(self, pair: torch.Tensor) -> torch.Tensor:
        """Compute the symmetry residual ``||z_ij - z_ji||_2`` for auditing.

        This should be at machine precision (~1e-7 for float32) if the
        initialization is correct.

        Args:
            pair: FloatTensor of shape ``(B, L, L, P)``.

        Returns:
            FloatTensor of shape ``(B,)`` with the mean L2 residual per batch.
        """
        # pair.permute(0, 2, 1, 3) swaps i and j axes
        diff = pair - pair.transpose(1, 2)
        residual = diff.pow(2).mean(dim=[1, 2, 3]).sqrt()
        return residual
