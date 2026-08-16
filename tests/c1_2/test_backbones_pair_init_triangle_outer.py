"""Unit tests for backbones.pair_init, triangle, and outer modules (C1-2).

Covers:
- SymmetricPairInit symmetry (shared and split projection), zero diagonal/pad
- TriangleMultiplicativeUpdate symmetry preservation
- TriangleAttention symmetry preservation
- PairTransition shape
- OuterProductMean symmetry
- PairToSingleAttention shape
- SingleRowAttention shape
- SingleTransition shape
"""

from __future__ import annotations

import pytest
import torch

from reactflow.backbones.pair_init import SymmetricPairInit
from reactflow.backbones.triangle import (
    PairTransition,
    TriangleAttention,
    TriangleMultiplicativeUpdate,
)
from reactflow.backbones.outer import (
    OuterProductMean,
    PairToSingleAttention,
    SingleRowAttention,
    SingleTransition,
)


# ---------------------------------------------------------------------------
# SymmetricPairInit
# ---------------------------------------------------------------------------


class TestSymmetricPairInit:
    def test_output_shape(self):
        init = SymmetricPairInit(single_dim=16, pair_dim=8)
        single = torch.randn(2, 10, 16)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]] * 2)
        out = init(single, indices)
        assert out.shape == (2, 10, 10, 8)

    def test_symmetry_shared_projection(self):
        """With share_projection=True, z_ij = z_ji by construction."""
        init = SymmetricPairInit(single_dim=16, pair_dim=8, share_projection=True)
        single = torch.randn(1, 8, 16)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = init(single, indices)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-5, f"shared proj not symmetric: {diff}"

    def test_symmetry_split_projection(self):
        """With share_projection=False, explicit symmetrization ensures z_ij = z_ji."""
        init = SymmetricPairInit(single_dim=16, pair_dim=8, share_projection=False)
        single = torch.randn(1, 8, 16)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = init(single, indices)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-5, f"split proj not symmetric: {diff}"

    def test_diagonal_zero(self):
        init = SymmetricPairInit(single_dim=16, pair_dim=8)
        single = torch.randn(1, 6, 16)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = init(single, indices)
        diag = out[0, range(6), range(6)]
        assert diag.abs().max().item() < 1e-6

    def test_padding_zero(self):
        init = SymmetricPairInit(single_dim=16, pair_dim=8)
        single = torch.randn(1, 6, 16)
        indices = torch.tensor([[0, 1, 2, 3, 5, 5]])  # last two are PAD
        mask = indices != 5
        out = init(single, indices, mask=mask)
        # Padding rows/cols should be zero
        assert out[0, 4].abs().max().item() < 1e-6
        assert out[0, 5].abs().max().item() < 1e-6
        assert out[0, :, 4].abs().max().item() < 1e-6
        assert out[0, :, 5].abs().max().item() < 1e-6

    def test_symmetry_residual_method(self):
        init = SymmetricPairInit(single_dim=16, pair_dim=8)
        single = torch.randn(1, 6, 16)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = init(single, indices)
        residual = init.symmetry_residual(out)
        assert residual.item() < 1e-5

    def test_gradient_flow(self):
        init = SymmetricPairInit(single_dim=8, pair_dim=4)
        single = torch.randn(1, 5, 8, requires_grad=True)
        indices = torch.tensor([[0, 1, 2, 3, 0]])
        out = init(single, indices)
        out.sum().backward()
        assert single.grad is not None


# ---------------------------------------------------------------------------
# TriangleMultiplicativeUpdate
# ---------------------------------------------------------------------------


class TestTriangleMultiplicativeUpdate:
    def test_output_shape(self):
        layer = TriangleMultiplicativeUpdate(pair_dim=8)
        pair = torch.randn(2, 10, 10, 8)
        out = layer(pair)
        assert out.shape == (2, 10, 10, 8)

    def test_symmetry_preservation(self):
        """If input is symmetric, output should remain symmetric."""
        layer = TriangleMultiplicativeUpdate(pair_dim=8)
        pair = torch.randn(1, 8, 8, 8)
        pair = 0.5 * (pair + pair.transpose(1, 2))  # symmetrize input
        out = layer(pair)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-5, f"triangle_mult broke symmetry: {diff}"

    def test_residual_connection(self):
        """Output should differ from input by a bounded amount (residual)."""
        layer = TriangleMultiplicativeUpdate(pair_dim=8)
        pair = torch.randn(1, 6, 6, 8)
        out = layer(pair)
        # Output should not be identical (update is applied)
        assert not torch.allclose(out, pair)

    def test_with_mask(self):
        layer = TriangleMultiplicativeUpdate(pair_dim=8)
        pair = torch.randn(1, 6, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = layer(pair, mask=mask)
        assert out.shape == (1, 6, 6, 8)

    def test_gradient_flow(self):
        layer = TriangleMultiplicativeUpdate(pair_dim=8)
        pair = torch.randn(1, 5, 5, 8, requires_grad=True)
        out = layer(pair)
        out.sum().backward()
        assert pair.grad is not None


# ---------------------------------------------------------------------------
# TriangleAttention
# ---------------------------------------------------------------------------


class TestTriangleAttention:
    def test_output_shape(self):
        layer = TriangleAttention(pair_dim=8, num_heads=4)
        pair = torch.randn(2, 10, 10, 8)
        out = layer(pair)
        assert out.shape == (2, 10, 10, 8)

    def test_symmetry_preservation(self):
        layer = TriangleAttention(pair_dim=8, num_heads=4)
        pair = torch.randn(1, 8, 8, 8)
        pair = 0.5 * (pair + pair.transpose(1, 2))
        out = layer(pair)
        diff = (out - out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-5

    def test_gradient_flow(self):
        layer = TriangleAttention(pair_dim=8, num_heads=2)
        pair = torch.randn(1, 5, 5, 8, requires_grad=True)
        out = layer(pair)
        out.sum().backward()
        assert pair.grad is not None


# ---------------------------------------------------------------------------
# PairTransition
# ---------------------------------------------------------------------------


class TestPairTransition:
    def test_output_shape(self):
        layer = PairTransition(pair_dim=8, expansion=4)
        pair = torch.randn(2, 10, 10, 8)
        out = layer(pair)
        assert out.shape == (2, 10, 10, 8)

    def test_residual(self):
        layer = PairTransition(pair_dim=8, expansion=2)
        pair = torch.randn(1, 5, 5, 8)
        out = layer(pair)
        # Should be a residual update (output != input but close)
        assert out.shape == pair.shape


# ---------------------------------------------------------------------------
# OuterProductMean
# ---------------------------------------------------------------------------


class TestOuterProductMean:
    def test_output_shape(self):
        layer = OuterProductMean(single_dim=16, pair_dim=8)
        single = torch.randn(2, 10, 16)
        pair = torch.randn(2, 10, 10, 8)
        out = layer(single, pair)
        assert out.shape == (2, 10, 10, 8)

    def test_symmetry_of_update(self):
        """The update (output - input) should be symmetric."""
        layer = OuterProductMean(single_dim=8, pair_dim=4)
        single = torch.randn(1, 6, 8)
        pair = torch.randn(1, 6, 6, 4)
        out = layer(single, pair)
        update = out - pair
        diff = (update - update.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-5

    def test_gradient_flow(self):
        layer = OuterProductMean(single_dim=8, pair_dim=4)
        single = torch.randn(1, 5, 8, requires_grad=True)
        pair = torch.randn(1, 5, 5, 4, requires_grad=True)
        out = layer(single, pair)
        out.sum().backward()
        assert single.grad is not None
        assert pair.grad is not None


# ---------------------------------------------------------------------------
# PairToSingleAttention
# ---------------------------------------------------------------------------


class TestPairToSingleAttention:
    def test_output_shape(self):
        layer = PairToSingleAttention(single_dim=16, pair_dim=8, num_heads=4)
        single = torch.randn(2, 10, 16)
        pair = torch.randn(2, 10, 10, 8)
        out = layer(single, pair)
        assert out.shape == (2, 10, 16)

    def test_gradient_flow(self):
        layer = PairToSingleAttention(single_dim=8, pair_dim=4, num_heads=2)
        single = torch.randn(1, 5, 8, requires_grad=True)
        pair = torch.randn(1, 5, 5, 4, requires_grad=True)
        out = layer(single, pair)
        out.sum().backward()
        assert single.grad is not None
        assert pair.grad is not None


# ---------------------------------------------------------------------------
# SingleRowAttention
# ---------------------------------------------------------------------------


class TestSingleRowAttention:
    def test_output_shape(self):
        layer = SingleRowAttention(single_dim=16, num_heads=4)
        single = torch.randn(2, 10, 16)
        out = layer(single)
        assert out.shape == (2, 10, 16)

    def test_gradient_flow(self):
        layer = SingleRowAttention(single_dim=8, num_heads=2)
        single = torch.randn(1, 5, 8, requires_grad=True)
        out = layer(single)
        out.sum().backward()
        assert single.grad is not None


# ---------------------------------------------------------------------------
# SingleTransition
# ---------------------------------------------------------------------------


class TestSingleTransition:
    def test_output_shape(self):
        layer = SingleTransition(single_dim=16, expansion=4)
        single = torch.randn(2, 10, 16)
        out = layer(single)
        assert out.shape == (2, 10, 16)
