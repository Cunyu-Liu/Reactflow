"""Input embeddings for the ReactFlow 2.0 static PairFormer.

This module produces per-position single representations ``s_i in R^C`` and
per-pair initial representations ``z_ij in R^P`` from a raw RNA sequence.

Features
--------
1. **Nucleotide embedding**: learned lookup over ``{A, C, G, U, N, PAD}`` plus
   an optional one-hot fallback.
2. **Positional embedding**: sinusoidal (fixed) + learned (optional) additive.
   Rotary position embedding (RoPE) is applied inside attention layers, not
   here, because RoPE is a rotation of the query/key and not an additive
   position encoding.
3. **Relative distance bins**: a learned embedding table over binned
   ``|i - j|`` values, used to initialize the pair representation.
4. **Pair compatibility**: a 3-channel feature (canonical, wobble, illegal)
   computed deterministically from the nucleotide identities.  This is the
   only chemistry prior injected into the pair representation.

All embeddings are deterministic given the seed; dropout is supported but
disabled by default for the pilot.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 350-358.

Complexity
----------
- Single embedding: ``O(L * C)`` time and memory.
- Pair initialization: ``O(L^2 * P)`` time and memory.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCLEOTIDE_VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4, "-": 5}
VOCAB_SIZE = 6  # A, C, G, U, N, PAD
PAD_INDEX = 5

CANONICAL_PAIR_SET = frozenset({("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")})
WOBBLE_PAIR_SET = frozenset({("G", "U"), ("U", "G")})


# ---------------------------------------------------------------------------
# Nucleotide embedding
# ---------------------------------------------------------------------------


def encode_sequence(sequence: str) -> torch.Tensor:
    """Encode an RNA sequence into a LongTensor of vocab indices.

    Formula: ``idx_i = NUCLEOTIDE_VOCAB[base_i]`` with ``T -> U`` and unknown
    bases mapped to ``N``.  Complexity: ``O(L)``.

    Args:
        sequence: RNA sequence over ``{A, C, G, U, T, N, -}`` (case-insensitive).

    Returns:
        LongTensor of shape ``(L,)`` with values in ``[0, VOCAB_SIZE)``.
    """

    seq_upper = sequence.upper()
    indices = []
    for base in seq_upper:
        indices.append(NUCLEOTIDE_VOCAB.get(base, 4))  # default to N
    return torch.tensor(indices, dtype=torch.long)


def encode_batch(sequences: Tuple[str, ...]) -> torch.Tensor:
    """Encode a batch of variable-length sequences into a padded LongTensor.

    Returns a LongTensor of shape ``(B, L_max)`` right-padded with ``PAD_INDEX``.

    Complexity: ``O(sum_i L_i)``.
    """

    if not sequences:
        return torch.zeros(0, 0, dtype=torch.long)
    max_len = max(len(s) for s in sequences)
    batch = torch.full((len(sequences), max_len), PAD_INDEX, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = encode_sequence(seq)
    return batch


class NucleotideEmbedding(nn.Module):
    """Learned nucleotide embedding with optional one-hot concatenation.

    Formula: ``e_i = E[idx_i]`` where ``E in R^{V x D}`` is a learned table.
    Optionally concatenate a one-hot vector of the nucleotide.

    Complexity: ``O(L * D)`` time, ``O(V * D)`` parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        vocab_size: int = VOCAB_SIZE,
        use_onehot: bool = False,
        padding_idx: int = PAD_INDEX,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.use_onehot = use_onehot
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    @property
    def output_dim(self) -> int:
        """Return the output dimension (learned + one-hot)."""
        return self.embed_dim + (self.vocab_size if self.use_onehot else 0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Embed a batch of nucleotide indices.

        Args:
            indices: LongTensor of shape ``(B, L)``.

        Returns:
            FloatTensor of shape ``(B, L, output_dim)``.
        """
        emb = self.embedding(indices)
        if self.use_onehot:
            onehot = nn.functional.one_hot(indices, self.vocab_size).float()
            # Zero out padding one-hot
            mask = (indices == self.padding_idx).unsqueeze(-1).float()
            onehot = onehot * (1.0 - mask)
            emb = torch.cat([emb, onehot], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------


def sinusoidal_positions(max_len: int, dim: int) -> torch.Tensor:
    """Return fixed sinusoidal positional encodings.

    Formula (Vaswani et al. 2017): for position ``pos`` and dimension ``i``,

        PE(pos, 2k)   = sin(pos / 10000^{2k/d})
        PE(pos, 2k+1) = cos(pos / 10000^{2k/d})

    Complexity: ``O(L * D)``.
    """

    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalEmbedding(nn.Module):
    """Sinusoidal + optional learned positional embedding.

    Formula: ``p_i = sin_enc(i) + learn_enc(i)`` (learned optional).

    Complexity: ``O(L * D)`` time, ``O(L_max * D)`` learned parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        max_len: int = 1024,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.learnable = learnable
        register_buffer = getattr(self, "register_buffer", None)
        # Use register_buffer for the fixed sinusoidal table
        self.register_buffer("sinusoidal", sinusoidal_positions(max_len, embed_dim), persistent=False)
        if learnable:
            self.learned = nn.Parameter(torch.zeros(max_len, embed_dim))
            nn.init.normal_(self.learned, mean=0.0, std=0.02)

    def forward(self, length: int) -> torch.Tensor:
        """Return positional embeddings for a given length.

        Args:
            length: sequence length ``L`` (must be ``<= max_len``).

        Returns:
            FloatTensor of shape ``(L, embed_dim)``.
        """
        if length > self.max_len:
            raise ValueError(f"length {length} exceeds max_len {self.max_len}")
        pe = self.sinusoidal[:length]
        if self.learnable:
            pe = pe + self.learned[:length]
        return pe


# ---------------------------------------------------------------------------
# Relative distance bins
# ---------------------------------------------------------------------------


def bin_relative_distance(
    distance,
    *,
    boundaries: Tuple[int, ...] = (1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256),
) -> torch.Tensor:
    """Bin a relative distance ``|i - j|`` into a discrete bin index.

    Uses ``boundary``-based binning: bin 0 is ``d < 1`` (same position),
    bin ``k`` is ``boundaries[k-1] <= d < boundaries[k]``, and the last bin
    is ``d >= boundaries[-1]``.

    Args:
        distance: LongTensor of arbitrary shape with non-negative values,
            or a Python int (converted to a 0-d tensor).
        boundaries: strictly increasing distance boundaries.

    Returns:
        LongTensor of the same shape with bin indices in ``[0, len(boundaries)]``.
    """

    if not isinstance(distance, torch.Tensor):
        distance = torch.tensor(distance, dtype=torch.long)
    num_bins = len(boundaries) + 1
    bnd = torch.tensor(boundaries, dtype=distance.dtype, device=distance.device)
    # Each distance gets the number of boundaries it exceeds
    bin_idx = (distance.unsqueeze(-1) >= bnd).sum(dim=-1)
    return bin_idx.clamp(max=num_bins - 1)


class RelativeDistanceEmbedding(nn.Module):
    """Learned embedding of binned relative distance for pair initialization.

    Formula: ``d_ij = E_bin[bin(|i - j|)]`` where ``E_bin in R^{B x P}``.

    Complexity: ``O(L^2 * P)`` time, ``O(B * P)`` parameters.
    """

    def __init__(
        self,
        pair_dim: int,
        *,
        boundaries: Tuple[int, ...] = (1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256),
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.boundaries = boundaries
        num_bins = len(boundaries) + 1
        self.num_bins = num_bins
        self.embedding = nn.Embedding(num_bins, pair_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return the pair-distance embedding for an ``L x L`` grid.

        Args:
            length: sequence length ``L``.
            device: target device.

        Returns:
            FloatTensor of shape ``(L, L, pair_dim)``.
        """
        arange = torch.arange(length, device=device)
        dist = (arange.unsqueeze(0) - arange.unsqueeze(1)).abs()
        bins = bin_relative_distance(dist, boundaries=self.boundaries)
        return self.embedding(bins)


# ---------------------------------------------------------------------------
# Pair compatibility
# ---------------------------------------------------------------------------


def pair_compatibility_matrix(
    indices: torch.Tensor,
    *,
    padding_idx: int = PAD_INDEX,
) -> torch.Tensor:
    """Compute the 3-channel pair-compatibility matrix for a batch.

    Channels: ``[canonical, wobble, illegal]`` where ``illegal`` includes
    same-base, non-complementary, and padding positions.  The diagonal is
    all zeros (a position cannot pair with itself).

    Formula: for bases ``(b_i, b_j)`` with ``i != j`` and neither padding,
    - ``canonical = 1`` iff ``(b_i, b_j) in CANONICAL_PAIR_SET``
    - ``wobble    = 1`` iff ``(b_i, b_j) in WOBBLE_PAIR_SET``
    - ``illegal   = 1`` iff neither canonical nor wobble

    Args:
        indices: LongTensor of shape ``(B, L)`` with vocab indices.
        padding_idx: vocab index for padding positions.

    Returns:
        FloatTensor of shape ``(B, L, L, 3)``.
    """

    # Map indices back to one-letter codes for set lookup
    # 0:A 1:C 2:G 3:U 4:N 5:PAD
    idx_to_base = ["A", "C", "G", "U", "N", "-"]
    B, L = indices.shape
    device = indices.device
    comp = torch.zeros(B, L, L, 3, device=device, dtype=torch.float)

    # Vectorized: build canonical and wobble masks via index comparison
    # Canonical: (A,U) (U,A) (G,C) (C,G) => (0,3) (3,0) (2,1) (1,2)
    # Wobble: (G,U) (U,G) => (2,3) (3,2)
    i_idx = indices.unsqueeze(2)  # (B, L, 1)
    j_idx = indices.unsqueeze(1)  # (B, 1, L)

    canonical_pairs = [
        (0, 3), (3, 0), (2, 1), (1, 2),
    ]
    wobble_pairs = [
        (2, 3), (3, 2),
    ]
    canonical_mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    wobble_mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    for a, b in canonical_pairs:
        canonical_mask = canonical_mask | ((i_idx == a) & (j_idx == b))
    for a, b in wobble_pairs:
        wobble_mask = wobble_mask | ((i_idx == a) & (j_idx == b))

    pad_mask = (i_idx == padding_idx) | (j_idx == padding_idx) | (i_idx == 4) | (j_idx == 4)
    diag_mask = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    invalid_mask = pad_mask | diag_mask

    comp[..., 0] = canonical_mask.float()
    comp[..., 1] = wobble_mask.float()
    # illegal = 1 iff not canonical, not wobble, not invalid
    illegal = (~canonical_mask) & (~wobble_mask) & (~invalid_mask)
    comp[..., 2] = illegal.float()
    # Zero out invalid positions
    comp = comp * (~invalid_mask).unsqueeze(-1).float()
    return comp


# ---------------------------------------------------------------------------
# Combined input embedding
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    """Combined single representation: nucleotide + positional embedding.

    Formula: ``s_i = LayerNorm(e_i + p_i)`` where ``e_i`` is the nucleotide
    embedding and ``p_i`` is the positional embedding.

    Optionally accepts precomputed frozen single features (from an external
    foundation model) and concatenates them: ``s_i = [e_i + p_i ; frozen_i]``.

    Complexity: ``O(L * C)`` time.
    """

    def __init__(
        self,
        single_dim: int,
        *,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 1024,
        use_onehot: bool = False,
        learnable_pos: bool = False,
        frozen_feature_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.frozen_feature_dim = frozen_feature_dim
        self.nucleotide = NucleotideEmbedding(
            embed_dim=single_dim - frozen_feature_dim if frozen_feature_dim > 0 else single_dim,
            vocab_size=vocab_size,
            use_onehot=use_onehot,
        )
        # If nucleotide output_dim != single_dim, project
        nuc_out = self.nucleotide.output_dim
        if nuc_out != single_dim - frozen_feature_dim and frozen_feature_dim == 0:
            self.proj = nn.Linear(nuc_out, single_dim)
        elif frozen_feature_dim > 0 and nuc_out != single_dim - frozen_feature_dim:
            self.proj = nn.Linear(nuc_out, single_dim - frozen_feature_dim)
        else:
            self.proj = nn.Identity()
        self.positional = PositionalEmbedding(
            single_dim - frozen_feature_dim,
            max_len=max_len,
            learnable=learnable_pos,
        )
        self.norm = nn.LayerNorm(single_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        indices: torch.Tensor,
        *,
        frozen_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the single representation.

        Args:
            indices: LongTensor of shape ``(B, L)``.
            frozen_features: optional FloatTensor of shape ``(B, L, frozen_feature_dim)``.

        Returns:
            FloatTensor of shape ``(B, L, single_dim)``.
        """
        B, L = indices.shape
        nuc = self.nucleotide(indices)
        nuc = self.proj(nuc)
        pos = self.positional(L).unsqueeze(0).expand(B, -1, -1).to(nuc.device)
        single = nuc + pos
        if frozen_features is not None:
            if frozen_features.shape[-1] != self.frozen_feature_dim:
                raise ValueError(
                    f"frozen_features dim {frozen_features.shape[-1]} != {self.frozen_feature_dim}"
                )
            single = torch.cat([single, frozen_features], dim=-1)
        single = self.norm(single)
        single = self.dropout(single)
        return single
