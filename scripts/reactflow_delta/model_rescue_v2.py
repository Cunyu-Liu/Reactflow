#!/usr/bin/env python3
"""Mean-first, calibration-second models for Model Rescue v2."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v1 import AlignedDeltaModel
from scripts.reactflow_delta.run_p3_lrso_v3 import ALPHA, SCALE_FLOOR


MEAN_CANDIDATE = "b1_mean_aligned"
CALIBRATED_CANDIDATE = "b1_mean_aligned_calibrated_residual"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v2_prediction.v1"


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1e-8)))


class MeanAlignedModel(AlignedDeltaModel):
    """The B1 K_rank=0 backbone with an explicit delta-mean interface."""

    def __init__(self, d: int = 96, heads: int = 4, hidden: int = 64) -> None:
        super().__init__(k_rank=0, sparse=False, d=d, heads=heads, hidden=hidden)

    @staticmethod
    def _mutation_one_hot(
        refs: list[str], alts: list[str], device: torch.device
    ) -> torch.Tensor:
        ref_idx = torch.tensor([ALPHA.get(x, 3) for x in refs], device=device)
        alt_idx = torch.tensor([ALPHA.get(x, 3) for x in alts], device=device)
        out = torch.zeros(len(refs), 8, device=device)
        out.scatter_(1, ref_idx[:, None], 1.0)
        out.scatter_(1, alt_idx[:, None] + 4, 1.0)
        return out

    def forward_mean_and_features(
        self,
        H: torch.Tensor,
        edit_idx: torch.Tensor,
        dists: torch.Tensor,
        refs: list[str],
        alts: list[str],
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return delta mean and the frozen direct features used by calibration."""
        device = H.device
        batch = edit_idx.shape[0]
        length = H.shape[0]
        hn = self.ctx_norm(H)
        hp = hn[edit_idx]
        mutation = self._mutation_one_hot(refs, alts, device)
        hp_e = hp.unsqueeze(1).expand(batch, length, -1)
        h_e = hn.unsqueeze(0).expand(batch, -1, -1)
        mutation_e = mutation.unsqueeze(1).expand(batch, length, -1)
        features = torch.cat([hp_e, h_e, dists.unsqueeze(-1), mutation_e], dim=-1)
        delta_mean = self.bdirect(features).squeeze(-1).masked_fill(~masks, 0.0)
        same = torch.tensor(
            [ref == alt for ref, alt in zip(refs, alts)],
            device=device,
            dtype=torch.bool,
        )
        delta_mean = delta_mean.masked_fill(same[:, None], 0.0)
        return delta_mean, features

    def forward_mean(
        self,
        H: torch.Tensor,
        edit_idx: torch.Tensor,
        dists: torch.Tensor,
        refs: list[str],
        alts: list[str],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_mean_and_features(H, edit_idx, dists, refs, alts, masks)[0]


class GlobalResidualCalibrator(nn.Module):
    """One train-only zero-mean Gaussian residual scale."""

    def __init__(self, initial_scale: float = 0.3) -> None:
        super().__init__()
        raw = _inverse_softplus(max(initial_scale - SCALE_FLOOR, 1e-5))
        self.raw_scale = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self, delta_mean: torch.Tensor) -> tuple[torch.Tensor, ...]:
        scale = SCALE_FLOOR + torch.nn.functional.softplus(self.raw_scale)
        weights = torch.ones(*delta_mean.shape, 1, device=delta_mean.device)
        locations = delta_mean.unsqueeze(-1)
        scales = scale.expand_as(delta_mean).unsqueeze(-1)
        return weights, locations, scales


class ConditionalScaleMixtureCalibrator(nn.Module):
    """A conditional narrow/wide zero-mean residual scale mixture."""

    def __init__(self, feature_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        with torch.no_grad():
            self.head[-1].bias[0] = 0.0
            self.head[-1].bias[1] = _inverse_softplus(0.1)
            self.head[-1].bias[2] = _inverse_softplus(0.2)

    def forward(
        self, delta_mean: torch.Tensor, frozen_features: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        raw = self.head(frozen_features.detach())
        probability_narrow = torch.sigmoid(raw[..., 0])
        narrow = SCALE_FLOOR + torch.nn.functional.softplus(raw[..., 1])
        wide = narrow + torch.nn.functional.softplus(raw[..., 2])
        weights = torch.stack([probability_narrow, 1.0 - probability_narrow], dim=-1)
        locations = torch.stack([delta_mean.detach(), delta_mean.detach()], dim=-1)
        scales = torch.stack([narrow, wide], dim=-1)
        return weights, locations, scales


def freeze_mean_model(model: MeanAlignedModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def target_delta(
    target: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        qualified_mask,
        target - wt_filled[None, :],
        torch.zeros_like(target),
    )


def cell_balanced_l1(
    delta_mean: torch.Tensor,
    target: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
) -> torch.Tensor:
    """Position -> mutant -> one puzzle-method cell L1."""
    delta = target_delta(target, qualified_mask, wt_filled)
    loss = torch.abs(delta_mean - delta).masked_fill(~qualified_mask, 0.0)
    counts = qualified_mask.float().sum(-1)
    valid_mutants = counts > 0
    if not bool(valid_mutants.any()):
        return loss.sum() * 0.0
    per_mutant = loss.sum(-1) / counts.clamp(min=1.0)
    return per_mutant[valid_mutants].mean()


def expected_abs_normal(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.clamp(min=1e-12)
    z = mean / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return 2.0 * scale * phi + mean * (2.0 * cdf - 1.0)


def gaussian_mixture_crps_torch(
    locations: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Closed-form row-wise CRPS for Gaussian mixtures."""
    scales = scales.clamp(min=SCALE_FLOOR)
    weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-12)
    first = torch.zeros_like(target)
    components = locations.shape[-1]
    for i in range(components):
        first = first + weights[..., i] * expected_abs_normal(
            target - locations[..., i], scales[..., i]
        )
    second = torch.zeros_like(target)
    for i in range(components):
        for j in range(components):
            pair_scale = torch.sqrt(scales[..., i].square() + scales[..., j].square())
            second = second + weights[..., i] * weights[..., j] * expected_abs_normal(
                locations[..., i] - locations[..., j], pair_scale
            )
    return first - 0.5 * second


def cell_balanced_crps(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
) -> torch.Tensor:
    delta = target_delta(target, qualified_mask, wt_filled)
    crps = gaussian_mixture_crps_torch(locations, scales, weights, delta)
    crps = crps.masked_fill(~qualified_mask, 0.0)
    counts = qualified_mask.float().sum(-1)
    valid_mutants = counts > 0
    if not bool(valid_mutants.any()):
        return crps.sum() * 0.0
    per_mutant = crps.sum(-1) / counts.clamp(min=1.0)
    return per_mutant[valid_mutants].mean()


def mean_state_snapshot(model: MeanAlignedModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_mean_state_unchanged(
    before: dict[str, torch.Tensor], model: MeanAlignedModel
) -> None:
    after = model.state_dict()
    for name, expected in before.items():
        if not torch.equal(expected, after[name].detach().cpu()):
            raise RuntimeError(f"calibration changed frozen mean parameter {name}")
