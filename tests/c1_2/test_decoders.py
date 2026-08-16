"""Unit tests for the decoders package (C1-2).

Covers:
- threshold_decoder: shape, range, masking
- nussinov_dp_decoder: shape, legality, nested (no pseudoknots)
- mea_decoder: shape, legality, nested
- greedy_pseudoknot_decoder: shape, legality (allows crossings)
- decode dispatcher: routes correctly
- Legality: all decoders respect min_loop, canonical/wobble pairs
"""

from __future__ import annotations

import pytest
import torch

from reactflow.decoders import (
    DecoderConfig,
    decode,
    greedy_pseudoknot_decoder,
    mea_decoder,
    nussinov_dp_decoder,
    threshold_decoder,
)
from reactflow.constraints import validate_pair_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_input(L: int = 8, *, batch: int = 1):
    """Create a simple (B, L, L) logits + (B, L) indices for testing."""
    indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3][:L]] * batch)
    # Make logits favor some pairs (e.g., i pairs with L-1-i)
    logits = torch.full((batch, L, L), -2.0)
    for b in range(batch):
        for i in range(L // 2):
            j = L - 1 - i
            logits[b, i, j] = 2.0
            logits[b, j, i] = 2.0
    # Mask diagonal
    for i in range(L):
        logits[:, i, i] = float("-inf")
    return logits, indices


def _make_compatible_pairs_input(L: int = 8, *, batch: int = 1):
    """Create input where favored pairs respect canonical/wobble chemistry.

    For L=8, indices = [A, C, G, U, A, C, G, U, ...]
    Canonical pairs: A-U (0,3), C-G (1,2), G-C (2,1), U-A (3,0)
    Make pair (0, L-1) = (A, U) favored; (1, L-2) = (C, G) favored; etc.
    """
    indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3][:L]] * batch)
    logits = torch.full((batch, L, L), -5.0)
    for b in range(batch):
        for i in range(L // 2):
            j = L - 1 - i
            # Check chemistry
            a, b_ = int(indices[b, i]), int(indices[b, j])
            canonical = {(0, 3), (3, 0), (1, 2), (2, 1)}
            wobble = {(2, 3), (3, 2)}
            if (a, b_) in canonical or (a, b_) in wobble:
                logits[b, i, j] = 5.0
                logits[b, j, i] = 5.0
    for i in range(L):
        logits[:, i, i] = float("-inf")
    return logits, indices


# ---------------------------------------------------------------------------
# Threshold decoder
# ---------------------------------------------------------------------------


class TestThresholdDecoder:
    def test_output_shape(self):
        logits, indices = _make_simple_input(L=8)
        out = threshold_decoder(logits, indices=indices)
        assert out.shape == (1, 8, 8)
        assert out.dtype == torch.float32

    def test_binary_output(self):
        logits, indices = _make_simple_input(L=8)
        out = threshold_decoder(logits, indices=indices, apply_legality=False)
        unique = set(out.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_diagonal_zero(self):
        logits, indices = _make_simple_input(L=8)
        out = threshold_decoder(logits, indices=indices, apply_legality=False)
        for i in range(8):
            assert out[0, i, i].item() == 0.0

    def test_symmetry(self):
        logits, indices = _make_simple_input(L=8)
        out = threshold_decoder(logits, indices=indices, apply_legality=False)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-6

    def test_with_bpp(self):
        logits, indices = _make_simple_input(L=8)
        bpp = torch.sigmoid(logits)
        out = threshold_decoder(logits, indices=indices, use_bpp=True, bpp=bpp, apply_legality=False)
        assert out.shape == (1, 8, 8)


# ---------------------------------------------------------------------------
# Nussinov DP decoder
# ---------------------------------------------------------------------------


class TestNussinovDPDecoder:
    def test_output_shape(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        assert out.shape == (1, 8, 8)
        assert out.dtype == torch.float32

    def test_binary_output(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        unique = set(out.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_diagonal_zero(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        for i in range(8):
            assert out[0, i, i].item() == 0.0

    def test_symmetry(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-6

    def test_matching_property(self):
        """Each position must pair with at most one other position."""
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        # Sum along each row should be <= 1 (one pair per position max)
        row_sums = out.sum(dim=-1)
        assert (row_sums <= 1.0 + 1e-6).all(), f"row sums exceed 1: {row_sums}"

    def test_no_pseudoknots(self):
        """Nested DP must never produce crossing pairs."""
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        # Extract pairs (upper triangle)
        pairs = []
        for i in range(8):
            for j in range(i + 1, 8):
                if out[0, i, j].item() > 0.5:
                    pairs.append((i, j))
        # Check for crossings: (a, b) and (c, d) with a < c < b < d
        for a, b in pairs:
            for c, d in pairs:
                if (a, b) == (c, d):
                    continue
                assert not (a < c < b < d), f"crossing pair: ({a},{b}) and ({c},{d})"

    def test_legality(self):
        """Output must satisfy validate_pair_matrix with allow_pseudoknot=False."""
        logits, indices = _make_compatible_pairs_input(L=8)
        out = nussinov_dp_decoder(logits, indices=indices)
        seq = "".join("ACGU"[int(i)] for i in indices[0].tolist())
        mat = out[0].tolist()
        result = validate_pair_matrix(seq, mat, allow_pseudoknot=False)
        assert result.valid, f"invalid structure: {result.violations}"


# ---------------------------------------------------------------------------
# MEA decoder
# ---------------------------------------------------------------------------


class TestMEADecoder:
    def test_output_shape(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        assert out.shape == (1, 8, 8)
        assert out.dtype == torch.float32

    def test_binary_output(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        unique = set(out.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_diagonal_zero(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        for i in range(8):
            assert out[0, i, i].item() == 0.0

    def test_symmetry(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-6

    def test_matching_property(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        row_sums = out.sum(dim=-1)
        assert (row_sums <= 1.0 + 1e-6).all()

    def test_no_pseudoknots(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        pairs = []
        for i in range(8):
            for j in range(i + 1, 8):
                if out[0, i, j].item() > 0.5:
                    pairs.append((i, j))
        for a, b in pairs:
            for c, d in pairs:
                if (a, b) == (c, d):
                    continue
                assert not (a < c < b < d)

    def test_legality(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        bpp = torch.sigmoid(logits)
        out = mea_decoder(bpp, indices=indices)
        seq = "".join("ACGU"[int(i)] for i in indices[0].tolist())
        mat = out[0].tolist()
        result = validate_pair_matrix(seq, mat, allow_pseudoknot=False)
        assert result.valid, f"MEA produced invalid structure: {result.violations}"


# ---------------------------------------------------------------------------
# Greedy pseudoknot decoder
# ---------------------------------------------------------------------------


class TestGreedyPseudoknotDecoder:
    def test_output_shape(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = greedy_pseudoknot_decoder(logits, indices=indices)
        assert out.shape == (1, 8, 8)
        assert out.dtype == torch.float32

    def test_binary_output(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = greedy_pseudoknot_decoder(logits, indices=indices)
        unique = set(out.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_matching_property(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = greedy_pseudoknot_decoder(logits, indices=indices)
        row_sums = out.sum(dim=-1)
        assert (row_sums <= 1.0 + 1e-6).all()

    def test_legality(self):
        """Greedy with allow_pseudoknot=True should still produce legal chemistry."""
        logits, indices = _make_compatible_pairs_input(L=8)
        out = greedy_pseudoknot_decoder(logits, indices=indices)
        # Use validate_pair_matrix with allow_pseudoknot=True
        seq = "".join("ACGU"[int(i)] for i in indices[0].tolist())
        mat = out[0].tolist()
        result = validate_pair_matrix(seq, mat, allow_pseudoknot=True)
        assert result.valid, f"greedy produced invalid structure: {result.violations}"


# ---------------------------------------------------------------------------
# Decode dispatcher
# ---------------------------------------------------------------------------


class TestDecodeDispatcher:
    def test_threshold_mode(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = decode(logits, indices=indices, mode="threshold", apply_legality=False) \
            if False else decode(logits, indices=indices, mode="threshold")
        assert out.shape == (1, 8, 8)

    def test_nussinov_dp_mode(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = decode(logits, indices=indices, mode="nussinov_dp")
        assert out.shape == (1, 8, 8)

    def test_mea_mode(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = decode(logits, indices=indices, mode="mea")
        assert out.shape == (1, 8, 8)

    def test_greedy_pseudoknot_mode(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        out = decode(logits, indices=indices, mode="greedy_pseudoknot")
        assert out.shape == (1, 8, 8)

    def test_unknown_mode_raises(self):
        logits, indices = _make_compatible_pairs_input(L=8)
        with pytest.raises(ValueError):
            decode(logits, indices=indices, mode="bogus")


# ---------------------------------------------------------------------------
# DecoderConfig
# ---------------------------------------------------------------------------


class TestDecoderConfig:
    def test_defaults(self):
        cfg = DecoderConfig()
        assert cfg.min_loop == 3
        assert cfg.allow_wobble is True
        assert cfg.threshold == 0.5
        assert cfg.min_score == 0.0

    def test_custom(self):
        cfg = DecoderConfig(min_loop=4, threshold=0.7)
        assert cfg.min_loop == 4
        assert cfg.threshold == 0.7
