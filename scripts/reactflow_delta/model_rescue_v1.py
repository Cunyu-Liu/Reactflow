#!/usr/bin/env python3
"""M2 model-rescue candidates with train/evaluation distribution alignment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.run_p3_lrso_v3 import (
    ALPHA,
    LRSOv3,
    SCALE_FLOOR,
    _wt_ctx_tensors,
)


def aligned_wt_ctx_tensors(univ: Any, construct_id: str, device: str):
    """Per-construct WT standardization using outcome-blind available inputs."""
    seq, _react, prec, obs_token, pos, region = _wt_ctx_tensors(univ, construct_id, device)
    c = univ.get_construct(construct_id)
    obs = c.wt_observed.astype(bool)
    raw = c.wt_reactivity.astype(np.float32)
    if obs.any():
        mean = float(np.mean(raw[obs]))
        std = max(float(np.std(raw[obs])), 1e-3)
        normalized = np.where(obs, (raw - mean) / std, 0.0).astype(np.float32)
    else:
        normalized = np.zeros_like(raw, dtype=np.float32)
    return seq, torch.tensor(normalized, device=device), prec, obs_token, pos, region


class AlignedDeltaModel(LRSOv3):
    """Aligned direct/low-rank model, optionally with a zero/change mixture.

    The sparse candidate is trained without changer labels.  It predicts a
    differentiable two-component distribution over mutation deltas:
    zero component N(0, sigma_zero) and change component N(mu, sigma_change).
    """

    def __init__(self, k_rank: int, sparse: bool = False, d: int = 96, heads: int = 4, hidden: int = 64) -> None:
        super().__init__(k_rank=k_rank, d=d, heads=heads, hidden=hidden, likelihood="gaussian")
        self.sparse = bool(sparse)
        if self.sparse:
            self.change_gate = nn.Sequential(
                nn.Linear(d + d + 1 + 8, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            nn.init.constant_(self.change_gate[-1].bias, -1.5)
            # softplus(-2.97) is close to 0.05; learned end-to-end, no labels.
            self.zero_scale_raw = nn.Parameter(torch.tensor(-2.97))

    @staticmethod
    def _mutation_one_hot(refs: list[str], alts: list[str], device: torch.device) -> torch.Tensor:
        ref_idx = torch.tensor([ALPHA.get(x, 3) for x in refs], device=device)
        alt_idx = torch.tensor([ALPHA.get(x, 3) for x in alts], device=device)
        out = torch.zeros(len(refs), 8, device=device)
        out.scatter_(1, ref_idx[:, None], 1.0)
        out.scatter_(1, alt_idx[:, None] + 4, 1.0)
        return out

    def forward_distribution(self, H, edit_idx, dists, refs, alts, masks):
        """Return weights, delta locations, and scales with shape (B,L,C)."""
        device = H.device
        batch = edit_idx.shape[0]
        length = H.shape[0]
        hn = self.ctx_norm(H)
        hp = hn[edit_idx]
        ra = self._mutation_one_hot(refs, alts, device)
        source = self.src(torch.cat([hp, ra], dim=-1))
        receiver = self.recv(hn)
        modulation = self.gmod(dists.unsqueeze(-1))
        hp_e = hp.unsqueeze(1).expand(batch, length, -1)
        h_e = hn.unsqueeze(0).expand(batch, -1, -1)
        ra_e = ra.unsqueeze(1).expand(batch, length, -1)
        direct_input = torch.cat([hp_e, h_e, dists.unsqueeze(-1), ra_e], dim=-1)
        direct = self.bdirect(direct_input).squeeze(-1)
        if self.k_rank == 0:
            low_rank = torch.zeros_like(direct)
        else:
            low_rank = (source.unsqueeze(1) * receiver.unsqueeze(0) * modulation).sum(-1)
        change_location = (direct + low_rank).masked_fill(~masks, 0.0)
        same = torch.tensor([r == a for r, a in zip(refs, alts)], device=device, dtype=torch.bool)
        change_location[same] = 0.0
        change_scale = self.scale_head(hn).clamp(min=SCALE_FLOOR)

        if not self.sparse:
            weights = torch.ones(batch, length, 1, device=device)
            locations = change_location.unsqueeze(-1)
            scales = change_scale[None, :, None].expand(batch, -1, -1)
            return weights, locations, scales

        probability_change = torch.sigmoid(self.change_gate(direct_input).squeeze(-1))
        probability_change = torch.where(masks, probability_change, torch.zeros_like(probability_change))
        probability_change[same] = 0.0
        weights = torch.stack([1.0 - probability_change, probability_change], dim=-1)
        locations = torch.stack([torch.zeros_like(change_location), change_location], dim=-1)
        zero_scale = SCALE_FLOOR + torch.nn.functional.softplus(self.zero_scale_raw)
        scales = torch.stack(
            [
                zero_scale.expand(batch, length),
                change_scale[None, :].expand(batch, -1),
            ],
            dim=-1,
        )
        return weights, locations, scales


def aligned_mixture_loss(
    model: AlignedDeltaModel,
    H: torch.Tensor,
    edit_idx: torch.Tensor,
    dists: torch.Tensor,
    refs: list[str],
    alts: list[str],
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
    huber_lambda: float = 0.0,
) -> torch.Tensor:
    """Masked Gaussian-mixture NLL plus optional mean-delta Huber penalty."""
    weights, locations, scales = model.forward_distribution(
        H, edit_idx, dists, refs, alts, prediction_mask
    )
    target_delta = torch.where(
        qualified_mask,
        target - wt_filled[None, :],
        torch.zeros_like(target),
    )
    y = target_delta.unsqueeze(-1)
    scales = scales.clamp(min=SCALE_FLOOR)
    log_normal = -0.5 * ((y - locations) / scales) ** 2 - torch.log(scales) - 0.5 * math.log(2.0 * math.pi)
    log_weights = torch.log(weights.clamp(min=1e-8))
    nll = -torch.logsumexp(log_weights + log_normal, dim=-1)
    nll = nll.masked_fill(~qualified_mask, 0.0)
    denom = qualified_mask.float().sum(-1).clamp(min=1.0)
    per_mutant = nll.sum(-1) / denom
    loss = per_mutant.mean()
    if huber_lambda > 0:
        mean_delta = (weights * locations).sum(-1)
        huber = torch.nn.functional.smooth_l1_loss(
            mean_delta[qualified_mask], target_delta[qualified_mask], reduction="mean", beta=0.1
        )
        loss = loss + float(huber_lambda) * huber
    return loss


def weighted_gaussian_mixture_crps(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Exact CRPS for row-wise weighted Gaussian mixtures.

    locations/scales/weights are (N,C); target is (N,).
    """
    from scipy.special import ndtr

    def expected_abs_normal(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
        sd = np.maximum(sd, 1e-12)
        z = mean / sd
        return 2.0 * sd * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi) + mean * (2.0 * ndtr(z) - 1.0)

    locations = np.asarray(locations, dtype=float)
    scales = np.asarray(scales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = weights / weights.sum(axis=1, keepdims=True)
    first = np.zeros(len(target), dtype=float)
    for i in range(locations.shape[1]):
        first += weights[:, i] * expected_abs_normal(target - locations[:, i], scales[:, i])
    second = np.zeros(len(target), dtype=float)
    for i in range(locations.shape[1]):
        for j in range(locations.shape[1]):
            sd = np.sqrt(scales[:, i] ** 2 + scales[:, j] ** 2)
            second += weights[:, i] * weights[:, j] * expected_abs_normal(locations[:, i] - locations[:, j], sd)
    return first - 0.5 * second
