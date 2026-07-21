"""Unit tests for the StaticPairFormer, BilinearPairHead, and CNN/UNet pair heads (C1-2).

Covers:
- StaticPairFormer forward shape, output dtype, symmetry, mask, gradient
- PairFormerConfig validation
- PairOutputHead symmetry, BPP range, calibration temperature
- BilinearPairHead symmetry, output shape, gradient
- CNNPairHead symmetry, output shape, gradient
- UNetPairHead symmetry, output shape, gradient
- Parameter count reporting
- num_parameters() method
"""

from __future__ import annotations

import pytest
import torch

from reactflow.models.static_pairformer import (
    PairFormerBlock,
    PairFormerConfig,
    PairFormerOutput,
    PairOutputHead,
    StaticPairFormer,
)
from reactflow.models.bilinear_pair_head import (
    BilinearPairHead,
    BilinearPairHeadConfig,
)
from reactflow.models.cnn_pair_head import (
    CNNPairHead,
    CNNPairHeadConfig,
    UNetPairHead,
    UNetPairHeadConfig,
)


# ---------------------------------------------------------------------------
# PairFormerConfig
# ---------------------------------------------------------------------------


class TestPairFormerConfig:
    def test_default_config(self):
        cfg = PairFormerConfig()
        assert cfg.single_dim == 256
        assert cfg.pair_dim == 64
        assert cfg.num_blocks == 8
        assert cfg.use_calibration is True

    def test_invalid_num_blocks(self):
        with pytest.raises(ValueError):
            PairFormerConfig(num_blocks=0)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            PairFormerConfig(single_dim=0)
        with pytest.raises(ValueError):
            PairFormerConfig(pair_dim=-1)


# ---------------------------------------------------------------------------
# PairFormerBlock
# ---------------------------------------------------------------------------


class TestPairFormerBlock:
    def test_output_shapes(self):
        cfg = PairFormerConfig(single_dim=16, pair_dim=8, num_blocks=1, num_heads_pair=2, num_heads_single=2)
        block = PairFormerBlock(cfg)
        single = torch.randn(2, 6, 16)
        pair = torch.randn(2, 6, 6, 8)
        single_out, pair_out = block(single, pair)
        assert single_out.shape == single.shape
        assert pair_out.shape == pair.shape

    def test_pair_symmetry_preservation(self):
        cfg = PairFormerConfig(single_dim=16, pair_dim=8, num_blocks=1, num_heads_pair=2, num_heads_single=2)
        block = PairFormerBlock(cfg)
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        pair = 0.5 * (pair + pair.transpose(1, 2))  # symmetric input
        _, pair_out = block(single, pair)
        diff = (pair_out - pair_out.transpose(1, 2)).abs().max()
        assert diff.item() < 1e-4, f"PairFormerBlock broke pair symmetry: {diff}"


# ---------------------------------------------------------------------------
# PairOutputHead
# ---------------------------------------------------------------------------


class TestPairOutputHead:
    def test_output_shape(self):
        head = PairOutputHead(pair_dim=8, single_dim=16)
        single = torch.randn(2, 10, 16)
        pair = torch.randn(2, 10, 10, 8)
        out = head(single, pair)
        assert isinstance(out, PairFormerOutput)
        assert out.logits.shape == (2, 10, 10)
        assert out.bpp.shape == (2, 10, 10)
        assert out.unpaired_logit.shape == (2, 10)
        assert out.unpaired_prob.shape == (2, 10)

    def test_logit_symmetry(self):
        head = PairOutputHead(pair_dim=8, single_dim=16)
        single = torch.randn(1, 8, 16)
        pair = torch.randn(1, 8, 8, 8)
        out = head(single, pair)
        # Symmetry residual should be ~0 (ignoring -inf diagonal)
        residual = out.symmetry_residual()
        assert residual.item() < 1e-5

    def test_bpp_range(self):
        head = PairOutputHead(pair_dim=8, single_dim=16)
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        out = head(single, pair)
        # BPP should be in [0, 1] for valid (non-masked) cells
        valid = torch.isfinite(out.logits)
        bpp_valid = out.bpp[valid]
        assert (bpp_valid >= 0).all()
        assert (bpp_valid <= 1).all()

    def test_diagonal_masked(self):
        head = PairOutputHead(pair_dim=8, single_dim=16)
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        out = head(single, pair)
        # Diagonal should be -inf
        for i in range(6):
            assert out.logits[0, i, i].item() == float("-inf")
        # BPP diagonal should be 0
        for i in range(6):
            assert out.bpp[0, i, i].item() == 0.0

    def test_padding_masked(self):
        head = PairOutputHead(pair_dim=8, single_dim=16)
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = head(single, pair, mask=mask)
        # Padding rows/cols should have -inf logits and 0 BPP
        for i in range(3, 6):
            for j in range(6):
                assert out.logits[0, i, j].item() == float("-inf")
                assert out.bpp[0, i, j].item() == 0.0

    def test_temperature_positive(self):
        head = PairOutputHead(pair_dim=8, single_dim=16, use_calibration=True, init_temperature=2.0)
        assert head.temperature.item() > 0
        # Should be close to 2.0 at init
        assert abs(head.temperature.item() - 2.0) < 1e-4

    def test_pair_type_logits_optional(self):
        head = PairOutputHead(pair_dim=8, single_dim=16, num_pair_types=4)
        single = torch.randn(1, 6, 16)
        pair = torch.randn(1, 6, 6, 8)
        out = head(single, pair)
        assert out.pair_type_logits is not None
        assert out.pair_type_logits.shape == (1, 6, 6, 4)


# ---------------------------------------------------------------------------
# StaticPairFormer
# ---------------------------------------------------------------------------


class TestStaticPairFormer:
    def test_forward_shape(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=2,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        assert isinstance(out, PairFormerOutput)
        assert out.logits.shape == (1, 8, 8)
        assert out.bpp.shape == (1, 8, 8)

    def test_logit_symmetry(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=2,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        residual = out.symmetry_residual()
        assert residual.item() < 1e-4, f"symmetry residual too large: {residual}"

    def test_padding_handling(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=2,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 5, 5]])  # last two are PAD
        out = model(indices)
        # Padding rows/cols should be -inf
        for i in range(4, 6):
            for j in range(6):
                assert out.logits[0, i, j].item() == float("-inf")
                assert out.bpp[0, i, j].item() == 0.0

    def test_bpp_range(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=2,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        valid = torch.isfinite(out.logits)
        bpp_valid = out.bpp[valid]
        assert (bpp_valid >= 0).all()
        assert (bpp_valid <= 1).all()

    def test_gradient_flow(self):
        cfg = PairFormerConfig(
            single_dim=8, pair_dim=4, num_blocks=1,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3]])
        out = model(indices)
        loss = out.bpp.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_num_parameters(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=2,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        n = model.num_parameters()
        assert n > 0

    def test_with_explicit_mask(self):
        cfg = PairFormerConfig(
            single_dim=16, pair_dim=8, num_blocks=1,
            num_heads_pair=2, num_heads_single=2,
        )
        model = StaticPairFormer(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        mask = torch.tensor([[True, True, True, True, False, False]])
        out = model(indices, mask=mask)
        assert out.logits.shape == (1, 6, 6)
        # Masked rows should be -inf
        for i in range(4, 6):
            assert out.logits[0, i, 0].item() == float("-inf")


# ---------------------------------------------------------------------------
# BilinearPairHead
# ---------------------------------------------------------------------------


class TestBilinearPairHead:
    def test_forward_shape(self):
        cfg = BilinearPairHeadConfig(single_dim=16, num_single_layers=1)
        model = BilinearPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        assert out.logits.shape == (1, 6, 6)
        assert out.bpp.shape == (1, 6, 6)

    def test_logit_symmetry(self):
        cfg = BilinearPairHeadConfig(single_dim=16, num_single_layers=1)
        model = BilinearPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        residual = out.symmetry_residual()
        assert residual.item() < 1e-4, f"bilinear head not symmetric: {residual}"

    def test_bpp_range(self):
        cfg = BilinearPairHeadConfig(single_dim=16, num_single_layers=1)
        model = BilinearPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        valid = torch.isfinite(out.logits)
        bpp_valid = out.bpp[valid]
        assert (bpp_valid >= 0).all()
        assert (bpp_valid <= 1).all()

    def test_diagonal_masked(self):
        cfg = BilinearPairHeadConfig(single_dim=16, num_single_layers=1)
        model = BilinearPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        for i in range(6):
            assert out.logits[0, i, i].item() == float("-inf")
            assert out.bpp[0, i, i].item() == 0.0

    def test_gradient_flow(self):
        cfg = BilinearPairHeadConfig(single_dim=8, num_single_layers=1)
        model = BilinearPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3]])
        out = model(indices)
        out.bpp.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_num_parameters(self):
        cfg = BilinearPairHeadConfig(single_dim=16, num_single_layers=1)
        model = BilinearPairHead(cfg)
        n = model.num_parameters()
        assert n > 0


# ---------------------------------------------------------------------------
# CNNPairHead
# ---------------------------------------------------------------------------


class TestCNNPairHead:
    def test_forward_shape(self):
        cfg = CNNPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_pair_layers=2,
        )
        model = CNNPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        assert out.logits.shape == (1, 8, 8)

    def test_logit_symmetry(self):
        cfg = CNNPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_pair_layers=2,
        )
        model = CNNPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        residual = out.symmetry_residual()
        assert residual.item() < 1e-4, f"CNN head not symmetric: {residual}"

    def test_bpp_range(self):
        cfg = CNNPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_pair_layers=2,
        )
        model = CNNPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        valid = torch.isfinite(out.logits)
        bpp_valid = out.bpp[valid]
        assert (bpp_valid >= 0).all()
        assert (bpp_valid <= 1).all()

    def test_diagonal_masked(self):
        cfg = CNNPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_pair_layers=2,
        )
        model = CNNPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        for i in range(8):
            assert out.logits[0, i, i].item() == float("-inf")
            assert out.bpp[0, i, i].item() == 0.0

    def test_gradient_flow(self):
        cfg = CNNPairHeadConfig(
            single_dim=8, pair_dim=4, pair_hidden_dim=4,
            num_single_layers=1, num_pair_layers=1,
        )
        model = CNNPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1]])
        out = model(indices)
        out.bpp.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# UNetPairHead
# ---------------------------------------------------------------------------


class TestUNetPairHead:
    def test_forward_shape(self):
        # L must be divisible by 2^num_levels for clean UNet
        cfg = UNetPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_levels=2,
        )
        model = UNetPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])  # L=16
        out = model(indices)
        assert out.logits.shape == (1, 16, 16)

    def test_logit_symmetry(self):
        cfg = UNetPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_levels=2,
        )
        model = UNetPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        residual = out.symmetry_residual()
        assert residual.item() < 1e-4, f"UNet head not symmetric: {residual}"

    def test_bpp_range(self):
        cfg = UNetPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_levels=2,
        )
        model = UNetPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        valid = torch.isfinite(out.logits)
        bpp_valid = out.bpp[valid]
        assert (bpp_valid >= 0).all()
        assert (bpp_valid <= 1).all()

    def test_diagonal_masked(self):
        cfg = UNetPairHeadConfig(
            single_dim=16, pair_dim=8, pair_hidden_dim=8,
            num_single_layers=1, num_levels=2,
        )
        model = UNetPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
        out = model(indices)
        for i in range(16):
            assert out.logits[0, i, i].item() == float("-inf")
            assert out.bpp[0, i, i].item() == 0.0

    def test_gradient_flow(self):
        cfg = UNetPairHeadConfig(
            single_dim=8, pair_dim=4, pair_hidden_dim=4,
            num_single_layers=1, num_levels=1,
        )
        model = UNetPairHead(cfg)
        indices = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])  # L=8
        out = model(indices)
        out.bpp.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Cross-model comparison (matched capacity sanity check)
# ---------------------------------------------------------------------------


class TestMatchedCapacity:
    def test_bilinear_smaller_than_pairformer(self):
        """The bilinear baseline should be smaller than the compact PairFormer.

        This is the spec's point: capacity alone is not what makes the
        PairFormer work.
        """
        pf_cfg = PairFormerConfig(
            single_dim=64, pair_dim=32, num_blocks=4,
            num_heads_pair=4, num_heads_single=4,
        )
        bl_cfg = BilinearPairHeadConfig(single_dim=64, num_single_layers=4)
        pf = StaticPairFormer(pf_cfg)
        bl = BilinearPairHead(bl_cfg)
        assert bl.num_parameters() < pf.num_parameters(), (
            f"bilinear ({bl.num_parameters()}) should be smaller than "
            f"PairFormer ({pf.num_parameters()})"
        )
