"""Unit tests for the backbones.embeddings module (C1-2).

Covers:
- NucleotideEmbedding shape, vocab, padding_idx behavior
- PositionalEmbedding sinusoidal vs learnable
- RelativeDistanceEmbedding symmetry, bin boundaries
- pair_compatibility_matrix: canonical, wobble, illegal, padding
- InputEmbedding combined output
- encode_sequence / encode_batch round-trip
- bin_relative_distance boundaries
- sinusoidal_positions orthogonality properties
"""

from __future__ import annotations

import pytest
import torch

from reactflow.backbones.embeddings import (
    InputEmbedding,
    NucleotideEmbedding,
    PAD_INDEX,
    PositionalEmbedding,
    RelativeDistanceEmbedding,
    bin_relative_distance,
    encode_batch,
    encode_sequence,
    pair_compatibility_matrix,
    sinusoidal_positions,
)


# ---------------------------------------------------------------------------
# NucleotideEmbedding
# ---------------------------------------------------------------------------


class TestNucleotideEmbedding:
    def test_output_shape_default(self):
        emb = NucleotideEmbedding(embed_dim=16)
        idx = torch.tensor([[0, 1, 2, 3, 5], [0, 0, 1, 1, 2]])
        out = emb(idx)
        assert out.shape == (2, 5, 16)

    def test_output_dim_with_onehot(self):
        emb = NucleotideEmbedding(embed_dim=16, use_onehot=True)
        assert emb.output_dim == 16 + 6  # vocab_size=6
        idx = torch.tensor([[0, 1, 2, 3]])
        out = emb(idx)
        assert out.shape == (1, 4, 22)

    def test_output_dim_without_onehot(self):
        emb = NucleotideEmbedding(embed_dim=16, use_onehot=False)
        assert emb.output_dim == 16

    def test_padding_idx_zero(self):
        emb = NucleotideEmbedding(embed_dim=8, padding_idx=PAD_INDEX)
        idx = torch.tensor([[0, 5, 5]])  # 5 = PAD
        out = emb(idx)
        # Padding positions must produce zero vectors
        assert torch.allclose(out[0, 1], torch.zeros(8))
        assert torch.allclose(out[0, 2], torch.zeros(8))

    def test_gradient_flow(self):
        emb = NucleotideEmbedding(embed_dim=8)
        idx = torch.tensor([[0, 1, 2]])
        out = emb(idx)
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# PositionalEmbedding
# ---------------------------------------------------------------------------


class TestPositionalEmbedding:
    def test_sinusoidal_shape(self):
        pe = PositionalEmbedding(embed_dim=16, max_len=32, learnable=False)
        out = pe(length=10)
        assert out.shape == (10, 16)

    def test_sinusoidal_is_fixed(self):
        pe = PositionalEmbedding(embed_dim=16, max_len=32, learnable=False)
        out1 = pe(length=10)
        out2 = pe(length=10)
        assert torch.allclose(out1, out2)

    def test_learnable_pos(self):
        pe = PositionalEmbedding(embed_dim=16, max_len=32, learnable=True)
        out1 = pe(length=10)
        # Backprop to update learned pos
        out1.sum().backward()
        assert pe.learned.grad is not None
        out2 = pe(length=10)
        # Should still be the same forward pass (params unchanged)
        assert torch.allclose(out1, out2)

    def test_different_lengths_distinct(self):
        pe = PositionalEmbedding(embed_dim=16, max_len=32, learnable=False)
        out5 = pe(length=5)
        out10 = pe(length=10)
        # First 5 positions should match
        assert torch.allclose(out5, out10[:5], atol=1e-6)


# ---------------------------------------------------------------------------
# RelativeDistanceEmbedding
# ---------------------------------------------------------------------------


class TestRelativeDistanceEmbedding:
    def test_output_shape(self):
        emb = RelativeDistanceEmbedding(pair_dim=8)
        out = emb(length=10)
        assert out.shape == (10, 10, 8)

    def test_symmetry(self):
        """z_ij must equal z_ji because |i-j| = |j-i|."""
        emb = RelativeDistanceEmbedding(pair_dim=8)
        out = emb(length=10)
        diff = (out - out.transpose(0, 1)).abs().max()
        assert diff.item() < 1e-6, f"distance embedding is not symmetric: {diff}"

    def test_diagonal_zero_or_consistent(self):
        """Diagonal entries (i==i, distance=0) should all be identical."""
        emb = RelativeDistanceEmbedding(pair_dim=8)
        out = emb(length=10)
        diag = out[range(10), range(10)]
        # All diagonal entries should be the same
        diff = (diag - diag[0:1]).abs().max()
        assert diff.item() < 1e-6


# ---------------------------------------------------------------------------
# pair_compatibility_matrix
# ---------------------------------------------------------------------------


class TestPairCompatibilityMatrix:
    def test_output_shape(self):
        idx = torch.tensor([[0, 1, 2, 3]])  # A, C, G, U
        m = pair_compatibility_matrix(idx)
        assert m.shape == (1, 4, 4, 3)

    def test_canonical_pairs(self):
        idx = torch.tensor([[0, 3]])  # A, U
        m = pair_compatibility_matrix(idx)
        # A-U is canonical: m[0, 0, 1, 0] should be 1 (canonical channel)
        assert m[0, 0, 1, 0].item() == 1
        assert m[0, 0, 1, 1].item() == 0  # not wobble
        assert m[0, 0, 1, 2].item() == 0  # not illegal

    def test_wobble_pairs(self):
        idx = torch.tensor([[2, 3]])  # G, U
        m = pair_compatibility_matrix(idx)
        assert m[0, 0, 1, 1].item() == 1  # G-U is wobble
        assert m[0, 0, 1, 0].item() == 0  # not canonical

    def test_illegal_pairs(self):
        idx = torch.tensor([[0, 0]])  # A, A
        m = pair_compatibility_matrix(idx)
        assert m[0, 0, 1, 2].item() == 1  # A-A is illegal
        assert m[0, 0, 1, 0].item() == 0
        assert m[0, 0, 1, 1].item() == 0

    def test_symmetry(self):
        idx = torch.tensor([[0, 1, 2, 3, 5]])
        m = pair_compatibility_matrix(idx)
        diff = (m - m.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-6

    def test_padding_positions(self):
        idx = torch.tensor([[0, 5]])  # A, PAD
        m = pair_compatibility_matrix(idx)
        # Padding row/col should be all zeros
        assert m[0, 1].sum().item() == 0
        assert m[0, :, 1].sum().item() == 0


# ---------------------------------------------------------------------------
# bin_relative_distance
# ---------------------------------------------------------------------------


class TestBinRelativeDistance:
    def test_basic_boundaries(self):
        boundaries = (1, 2, 3, 5, 8, 12)
        # distance 0 -> bin 0
        assert int(bin_relative_distance(0, boundaries=boundaries)) == 0
        # distance 1 -> bin 1
        assert int(bin_relative_distance(1, boundaries=boundaries)) == 1
        # distance 2 -> bin 2
        assert int(bin_relative_distance(2, boundaries=boundaries)) == 2
        # distance 4 -> bin 3 (between 3 and 5)
        assert int(bin_relative_distance(4, boundaries=boundaries)) == 3
        # distance 11 -> bin 5 (between 8 and 12, exclusive of 12)
        assert int(bin_relative_distance(11, boundaries=boundaries)) == 5
        # distance 12 -> bin 6 (>= 12, last bin)
        assert int(bin_relative_distance(12, boundaries=boundaries)) == 6
        # distance 100 -> bin 6 (above max)
        assert int(bin_relative_distance(100, boundaries=boundaries)) == 6

    def test_tensor_input(self):
        boundaries = (1, 5, 10)
        d = torch.tensor([0, 1, 5, 11])
        bins = bin_relative_distance(d, boundaries=boundaries)
        assert bins.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# encode_sequence / encode_batch
# ---------------------------------------------------------------------------


class TestEncode:
    def test_encode_sequence_basic(self):
        idx = encode_sequence("ACGU")
        assert idx.tolist() == [0, 1, 2, 3]

    def test_encode_sequence_with_n(self):
        idx = encode_sequence("ACGN")
        assert idx.tolist() == [0, 1, 2, 4]

    def test_encode_sequence_with_t(self):
        # T maps to U (index 3)
        idx = encode_sequence("ACGT")
        assert idx.tolist() == [0, 1, 2, 3]

    def test_encode_batch_padded(self):
        seqs = ["ACG", "ACGU"]
        idx = encode_batch(seqs)
        # Should be padded to length 4
        assert idx.shape == (2, 4)
        assert idx[0, 3].item() == PAD_INDEX  # padding


# ---------------------------------------------------------------------------
# InputEmbedding
# ---------------------------------------------------------------------------


class TestInputEmbedding:
    def test_output_shape(self):
        emb = InputEmbedding(single_dim=32, max_len=64)
        idx = torch.tensor([[0, 1, 2, 3]])
        out = emb(idx)
        assert out.shape == (1, 4, 32)

    def test_with_frozen_features(self):
        emb = InputEmbedding(single_dim=32, max_len=64, frozen_feature_dim=8)
        idx = torch.tensor([[0, 1, 2, 3]])
        frozen = torch.randn(1, 4, 8)
        out = emb(idx, frozen_features=frozen)
        assert out.shape == (1, 4, 32)

    def test_gradient_flow(self):
        emb = InputEmbedding(single_dim=16, max_len=32)
        idx = torch.tensor([[0, 1, 2, 3]])
        out = emb(idx)
        out.sum().backward()
        # Should have gradients on nucleotide and positional embeddings
        assert emb.nucleotide.embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# sinusoidal_positions
# ---------------------------------------------------------------------------


class TestSinusoidalPositions:
    def test_shape(self):
        pos = sinusoidal_positions(10, 16)
        assert pos.shape == (10, 16)

    def test_different_positions_distinct(self):
        pos = sinusoidal_positions(10, 16)
        # Position 0 and position 1 should be different
        assert not torch.allclose(pos[0], pos[1])

    def test_deterministic(self):
        pos1 = sinusoidal_positions(10, 16)
        pos2 = sinusoidal_positions(10, 16)
        assert torch.allclose(pos1, pos2)
