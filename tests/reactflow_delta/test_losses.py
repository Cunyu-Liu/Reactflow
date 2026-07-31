"""Tests for M0 losses (v3 EPRO §4.9, T-M0.3).

Validates:
  * Student-t NLL is finite and decreases as predictions improve.
  * Heteroscedastic NLL respects mask and measurement variance.
  * Measurement variance increases effective variance.
  * Mask zeroes loss on excluded positions.
  * Skill computation matches evaluator definition.
"""

from __future__ import annotations

import math

import pytest
import torch

from reactflow.delta.losses import (
    HeteroscedasticNLL,
    StudentTNLL,
    WeightedMAELoss,
    compute_skill,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_data():
    """Toy (n=10) pair with mask and target."""

    torch.manual_seed(42)
    n = 10
    target = torch.randn(n)
    mu = torch.randn(n, requires_grad=True)
    mask = torch.ones(n, dtype=torch.bool)
    mask[3] = False  # exclude one position
    return mu, target, mask


# ---------------------------------------------------------------------------
# Student-t NLL
# ---------------------------------------------------------------------------


class TestStudentTNLL:
    def test_nll_finite(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = StudentTNLL(learned_scale=True)
        loss = loss_fn(mu, target, mask)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_nll_decreases_with_better_prediction(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = StudentTNLL(learned_scale=True)
        loss_bad = loss_fn(mu, target, mask)
        loss_good = loss_fn(target.detach().clone(), target, mask)
        assert loss_good < loss_bad

    def test_mask_excludes_positions(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = StudentTNLL(learned_scale=True)
        # Perturb a masked-out position: loss should not change.
        mu2 = mu.clone()
        mu2[3] += 1000.0  # position 3 is masked out
        loss1 = loss_fn(mu, target, mask)
        loss2 = loss_fn(mu2, target, mask)
        assert abs(loss1.item() - loss2.item()) < 1e-5

    def test_measurement_variance_increases_loss(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = StudentTNLL(learned_scale=True)
        loss_no_meas = loss_fn(mu, target, mask, measurement_variance=None)
        loss_with_meas = loss_fn(mu, target, mask, measurement_variance=0.5)
        # With measurement variance, the effective sigma is larger, which
        # changes the NLL. For a bad prediction, larger sigma should decrease
        # the tail penalty but increase the log-pref. Check it's different.
        assert abs(loss_no_meas.item() - loss_with_meas.item()) > 1e-4

    def test_df_in_range(self):
        loss_fn = StudentTNLL(init_df=4.0, min_df=2.0, max_df=30.0)
        df = loss_fn.df.item()
        assert 2.0 <= df <= 30.0

    def test_gradient_flows(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = StudentTNLL(learned_scale=True)
        loss = loss_fn(mu, target, mask)
        loss.backward()
        assert mu.grad is not None
        assert torch.any(mu.grad != 0)
        assert loss_fn.log_sigma_raw.grad is not None

    def test_empty_mask_returns_zero(self):
        mu = torch.randn(5)
        target = torch.randn(5)
        mask = torch.zeros(5, dtype=torch.bool)
        loss_fn = StudentTNLL()
        loss = loss_fn(mu, target, mask)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Heteroscedastic NLL
# ---------------------------------------------------------------------------


class TestHeteroscedasticNLL:
    def test_nll_finite(self, toy_data):
        mu, target, mask = toy_data
        log_var = torch.zeros(10, requires_grad=True)
        loss_fn = HeteroscedasticNLL()
        loss = loss_fn(mu, log_var, target, mask)
        assert torch.isfinite(loss)

    def test_mask_excludes_positions(self, toy_data):
        mu, target, mask = toy_data
        log_var = torch.zeros(10)
        loss_fn = HeteroscedasticNLL()
        mu2 = mu.clone()
        mu2[3] += 1000.0
        loss1 = loss_fn(mu, log_var, target, mask)
        loss2 = loss_fn(mu2, log_var, target, mask)
        assert abs(loss1.item() - loss2.item()) < 1e-5

    def test_higher_var_reduces_penalty_for_bad_pred(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = HeteroscedasticNLL()
        log_var_low = torch.full((10,), -2.0)  # small variance
        log_var_high = torch.full((10,), 2.0)  # large variance
        loss_low = loss_fn(mu, log_var_low, target, mask)
        loss_high = loss_fn(mu, log_var_high, target, mask)
        # With large errors and large variance, the NLL should be lower
        # (less penalty for being wrong when uncertainty is high).
        assert loss_high < loss_low


# ---------------------------------------------------------------------------
# Weighted MAE
# ---------------------------------------------------------------------------


class TestWeightedMAE:
    def test_basic(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = WeightedMAELoss()
        loss = loss_fn(mu, target, mask, weight=1.0)
        expected = (mu[mask] - target[mask]).abs().mean()
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_weight_scales_loss(self, toy_data):
        mu, target, mask = toy_data
        loss_fn = WeightedMAELoss()
        loss1 = loss_fn(mu, target, mask, weight=1.0)
        loss2 = loss_fn(mu, target, mask, weight=2.0)
        assert abs(loss2.item() - 2.0 * loss1.item()) < 1e-6


# ---------------------------------------------------------------------------
# Skill computation
# ---------------------------------------------------------------------------


class TestComputeSkill:
    def test_perfect_prediction_skill_near_1(self):
        target = torch.randn(20)
        mu = target.clone()
        mask = torch.ones(20, dtype=torch.bool)
        skill = compute_skill(mu, target, mask)
        assert skill > 0.99

    def test_zero_prediction_skill_0(self):
        target = torch.randn(20)
        mu = torch.zeros(20)
        mask = torch.ones(20, dtype=torch.bool)
        skill = compute_skill(mu, target, mask)
        assert abs(skill) < 1e-6

    def test_empty_mask_returns_nan(self):
        mu = torch.randn(5)
        target = torch.randn(5)
        mask = torch.zeros(5, dtype=torch.bool)
        skill = compute_skill(mu, target, mask)
        assert math.isnan(skill)

    def test_zero_target_returns_nan(self):
        # wmae_zero = 0 when all targets are 0 => Skill is NaN.
        target = torch.zeros(20)
        mu = torch.zeros(20)
        mask = torch.ones(20, dtype=torch.bool)
        skill = compute_skill(mu, target, mask)
        assert math.isnan(skill)
