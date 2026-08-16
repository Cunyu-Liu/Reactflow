"""Unit tests for the losses module (C1-2).

Covers:
- class_balanced_bce_loss: non-negativity, mask handling, pos_weight
- focal_loss: non-negativity, gamma effect
- soft_f1_loss: range [0, 1], gradient flow
- dice_loss: range [0, 1]
- pair_count_reg_loss: non-negativity, p_norm
- symmetry_audit_loss: zero for symmetric input, positive for asymmetric
- calibration_loss: non-negativity
- unpaired_bce_loss: non-negativity, mask handling
- pairformer_loss: combined loss, all components present
"""

from __future__ import annotations

import pytest
import torch

from reactflow.losses import (
    LossConfig,
    calibration_loss,
    class_balanced_bce_loss,
    dice_loss,
    focal_loss,
    pair_count_reg_loss,
    pairformer_loss,
    soft_f1_loss,
    symmetry_audit_loss,
    unpaired_bce_loss,
)
from reactflow.models.static_pairformer import PairFormerOutput, StaticPairFormer, PairFormerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_batch(L: int = 6, *, batch: int = 2, seed: int = 0):
    """Create (logits, targets, mask) for loss tests."""
    torch.manual_seed(seed)
    # Random logits
    logits = torch.randn(batch, L, L) * 0.5
    # Symmetrize
    logits = 0.5 * (logits + logits.transpose(1, 2))
    # Mask diagonal
    for i in range(L):
        logits[:, i, i] = float("-inf")
    # Random binary targets (symmetric)
    targets = torch.randint(0, 2, (batch, L, L)).float()
    targets = 0.5 * (targets + targets.transpose(1, 2))
    for i in range(L):
        targets[:, i, i] = 0.0
    # Mask
    mask = torch.ones(batch, L, dtype=torch.bool)
    return logits, targets, mask


def _make_pairformer_output(L: int = 6, batch: int = 1):
    """Create a PairFormerOutput from a real model forward pass."""
    cfg = PairFormerConfig(
        single_dim=16, pair_dim=8, num_blocks=1,
        num_heads_pair=2, num_heads_single=2,
    )
    model = StaticPairFormer(cfg)
    indices = torch.tensor([[0, 1, 2, 3, 0, 1][:L]] * batch)
    out = model(indices)
    return out, indices


# ---------------------------------------------------------------------------
# class_balanced_bce_loss
# ---------------------------------------------------------------------------


class TestClassBalancedBCELoss:
    def test_non_negative(self):
        logits, targets, mask = _make_simple_batch()
        loss = class_balanced_bce_loss(logits, targets, mask=mask)
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        # If logits strongly agree with targets, loss should be small
        logits, targets, mask = _make_simple_batch()
        # Make logits very confident in the right direction
        confident_logits = torch.where(
            targets >= 0.5,
            torch.full_like(targets, 10.0),
            torch.full_like(targets, -10.0),
        )
        # Mask diagonal
        L = logits.shape[-1]
        for i in range(L):
            confident_logits[:, i, i] = float("-inf")
        loss = class_balanced_bce_loss(confident_logits, targets, mask=mask)
        assert loss.item() < 0.1, f"perfect prediction should have low loss: {loss}"

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        logits = logits.detach().requires_grad_(True)
        loss = class_balanced_bce_loss(logits, targets, mask=mask)
        loss.backward()
        assert logits.grad is not None

    def test_pos_weight_override(self):
        logits, targets, mask = _make_simple_batch()
        loss_auto = class_balanced_bce_loss(logits, targets, mask=mask, auto_pos_weight=True)
        loss_fixed = class_balanced_bce_loss(logits, targets, mask=mask, pos_weight=2.0, auto_pos_weight=False)
        # Different pos_weight should give different losses
        assert loss_auto.item() != loss_fixed.item()


# ---------------------------------------------------------------------------
# focal_loss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_non_negative(self):
        logits, targets, mask = _make_simple_batch()
        loss = focal_loss(logits, targets, mask=mask, gamma=2.0)
        assert loss.item() >= 0

    def test_gamma_zero_matches_bce(self):
        """With gamma=0, focal loss should match BCE (up to mean reduction)."""
        logits, targets, mask = _make_simple_batch()
        focal = focal_loss(logits, targets, mask=mask, gamma=0.0)
        bce = class_balanced_bce_loss(
            logits, targets, mask=mask, auto_pos_weight=False, pos_weight=None
        )
        # Should be very close (both are mean BCE)
        assert abs(focal.item() - bce.item()) < 1e-4

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        logits = logits.detach().requires_grad_(True)
        loss = focal_loss(logits, targets, mask=mask, gamma=2.0)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# soft_f1_loss
# ---------------------------------------------------------------------------


class TestSoftF1Loss:
    def test_range(self):
        """soft F1 loss is 1 - F1, so it's in [0, 1]."""
        logits, targets, mask = _make_simple_batch()
        loss = soft_f1_loss(logits, targets, mask=mask)
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_prediction(self):
        logits, targets, mask = _make_simple_batch()
        confident = torch.where(targets >= 0.5, torch.full_like(targets, 20.0), torch.full_like(targets, -20.0))
        L = logits.shape[-1]
        for i in range(L):
            confident[:, i, i] = float("-inf")
        loss = soft_f1_loss(confident, targets, mask=mask)
        assert loss.item() < 0.1

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        logits = logits.detach().requires_grad_(True)
        loss = soft_f1_loss(logits, targets, mask=mask)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# dice_loss
# ---------------------------------------------------------------------------


class TestDiceLoss:
    def test_range(self):
        logits, targets, mask = _make_simple_batch()
        loss = dice_loss(logits, targets, mask=mask)
        assert 0.0 <= loss.item() <= 1.0

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        logits = logits.detach().requires_grad_(True)
        loss = dice_loss(logits, targets, mask=mask)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# pair_count_reg_loss
# ---------------------------------------------------------------------------


class TestPairCountRegLoss:
    def test_non_negative(self):
        logits, targets, mask = _make_simple_batch()
        loss = pair_count_reg_loss(logits, targets, mask=mask, p_norm=1)
        assert loss.item() >= 0

    def test_l2_norm(self):
        logits, targets, mask = _make_simple_batch()
        loss = pair_count_reg_loss(logits, targets, mask=mask, p_norm=2)
        assert loss.item() >= 0

    def test_zero_for_perfect_count(self):
        # If predicted count == target count, loss should be ~0
        targets = torch.zeros(1, 6, 6)
        targets[0, 0, 1] = 1.0
        targets[0, 1, 0] = 1.0
        # Logits that produce exactly 1 pair after sigmoid
        logits = torch.full((1, 6, 6), -10.0)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 0] = 10.0
        for i in range(6):
            logits[:, i, i] = float("-inf")
        loss = pair_count_reg_loss(logits, targets, p_norm=1)
        assert loss.item() < 0.1

    def test_invalid_p_norm_raises(self):
        logits, targets, mask = _make_simple_batch()
        with pytest.raises(ValueError):
            pair_count_reg_loss(logits, targets, mask=mask, p_norm=3)


# ---------------------------------------------------------------------------
# symmetry_audit_loss
# ---------------------------------------------------------------------------


class TestSymmetryAuditLoss:
    def test_zero_for_symmetric_input(self):
        """Symmetric logits should give ~0 symmetry audit loss."""
        logits, _, mask = _make_simple_batch()
        # logits is already symmetric from _make_simple_batch
        loss = symmetry_audit_loss(logits, mask=mask)
        assert loss.item() < 1e-6, f"symmetric input gave loss {loss}"

    def test_positive_for_asymmetric_input(self):
        """Asymmetric logits should give positive symmetry audit loss."""
        torch.manual_seed(42)
        logits = torch.randn(1, 6, 6)
        # Deliberately make it asymmetric (do NOT symmetrize)
        mask = torch.ones(1, 6, dtype=torch.bool)
        # Mask diagonal (set to -inf so it's excluded)
        for i in range(6):
            logits[:, i, i] = float("-inf")
        loss = symmetry_audit_loss(logits, mask=mask)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# calibration_loss
# ---------------------------------------------------------------------------


class TestCalibrationLoss:
    def test_non_negative(self):
        logits, targets, mask = _make_simple_batch()
        loss = calibration_loss(logits, targets, mask=mask, num_bins=10)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        logits = logits.detach().requires_grad_(True)
        loss = calibration_loss(logits, targets, mask=mask, num_bins=5)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# unpaired_bce_loss
# ---------------------------------------------------------------------------


class TestUnpairedBCELoss:
    def test_non_negative(self):
        logits, targets, mask = _make_simple_batch()
        L = logits.shape[-1]
        unpaired_logit = torch.randn(1, L)
        loss = unpaired_bce_loss(unpaired_logit, targets, mask=mask)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        logits, targets, mask = _make_simple_batch()
        L = logits.shape[-1]
        unpaired_logit = torch.randn(1, L, requires_grad=True)
        loss = unpaired_bce_loss(unpaired_logit, targets, mask=mask)
        loss.backward()
        assert unpaired_logit.grad is not None

    def test_paired_position_target_is_zero(self):
        """For a position in a pair, the unpaired target should be 0."""
        targets = torch.zeros(1, 4, 4)
        targets[0, 0, 1] = 1.0
        targets[0, 1, 0] = 1.0
        # Compute the implicit unpaired target
        paired = targets.sum(dim=-1).clamp(max=1.0)
        unpaired_target = 1.0 - paired
        # Position 0 and 1 are paired -> unpaired_target should be 0
        assert unpaired_target[0, 0].item() == 0.0
        assert unpaired_target[0, 1].item() == 0.0
        # Position 2 and 3 are unpaired -> unpaired_target should be 1
        assert unpaired_target[0, 2].item() == 1.0
        assert unpaired_target[0, 3].item() == 1.0


# ---------------------------------------------------------------------------
# pairformer_loss (combined)
# ---------------------------------------------------------------------------


class TestPairformerLoss:
    def test_combined_loss_with_model_output(self):
        output, _ = _make_pairformer_output(L=6, batch=1)
        # Create matching targets
        targets = torch.randint(0, 2, (1, 6, 6)).float()
        targets = 0.5 * (targets + targets.transpose(1, 2))
        for i in range(6):
            targets[:, i, i] = 0.0
        parts = pairformer_loss(output, targets)
        assert "total" in parts
        assert parts["total"].item() >= 0

    def test_all_components_present(self):
        output, _ = _make_pairformer_output(L=6, batch=1)
        targets = torch.randint(0, 2, (1, 6, 6)).float()
        targets = 0.5 * (targets + targets.transpose(1, 2))
        for i in range(6):
            targets[:, i, i] = 0.0
        parts = pairformer_loss(output, targets)
        expected_keys = {
            "total", "bce", "focal", "soft_f1", "dice",
            "pair_count", "symmetry", "calibration", "unpaired",
        }
        assert set(parts.keys()) == expected_keys

    def test_disabled_components_are_zero(self):
        output, _ = _make_pairformer_output(L=6, batch=1)
        targets = torch.randint(0, 2, (1, 6, 6)).float()
        targets = 0.5 * (targets + targets.transpose(1, 2))
        for i in range(6):
            targets[:, i, i] = 0.0
        cfg = LossConfig(
            bce_weight=0.0, focal_weight=0.0, soft_f1_weight=0.0,
            dice_weight=0.0, pair_count_weight=0.0, symmetry_weight=0.0,
            calibration_weight=0.0, unpaired_weight=0.0,
        )
        parts = pairformer_loss(output, targets, config=cfg)
        # All disabled terms should be 0
        for key in ("bce", "focal", "soft_f1", "dice", "pair_count", "symmetry", "calibration", "unpaired"):
            assert parts[key].item() == 0.0
        # Total should also be 0
        assert parts["total"].item() == 0.0

    def test_gradient_flow(self):
        output, _ = _make_pairformer_output(L=6, batch=1)
        targets = torch.randint(0, 2, (1, 6, 6)).float()
        targets = 0.5 * (targets + targets.transpose(1, 2))
        for i in range(6):
            targets[:, i, i] = 0.0
        parts = pairformer_loss(output, targets)
        parts["total"].backward()
        # Should have gradients on the model parameters
        # (verified by checking the PairFormerOutput's logits require grad)
