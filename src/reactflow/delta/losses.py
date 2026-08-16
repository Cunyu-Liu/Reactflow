"""Robust NLL losses for EPRO M0 (v3 EPRO §4.9, Phase M0 T-M0.3).

Implements:
  1. Student-t NLL (robust to heavy-tailed reactivity noise).
  2. Heteroscedastic Gaussian NLL (model predicts mean + variance).
  3. Measurement variance integration (per-pair noise from RDAT registry).

The loss is computed only on endpoint-mask positions (§12.1): non-edit,
non-missing, probe-eligible. Positions outside the mask contribute zero.

All losses are differentiable and support variable-length sequences via
the ``mask`` argument.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES_SCHEMA_VERSION = "reactflow-delta-m0-losses-v1"


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over masked positions. Returns 0 if mask is all False."""

    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return (x * mask).sum() / mask.sum().clamp(min=1)


class StudentTNLL(nn.Module):
    """Student-t negative log-likelihood (robust NLL, §4.9 EPRO-Lite).

    NLL = -log p(y | mu, sigma, df)
        = -log Gamma((df+1)/2) + log Gamma(df/2)
          + 0.5 * log(df * pi * sigma^2)
          + ((df+1)/2) * log(1 + (y - mu)^2 / (df * sigma^2))

    The model predicts ``mu`` (delta_r_hat). ``sigma`` is either:
      - a learned per-pair scale (if ``learned_scale=True``), or
      - a fixed scale + measurement_variance.

    ``df`` (degrees of freedom) is a learned parameter controlling tail weight.
    Small df = heavy tails (robust to outliers).
    """

    def __init__(self, *, learned_scale: bool = True, init_df: float = 4.0,
                 min_df: float = 2.0, max_df: float = 30.0,
                 min_sigma: float = 1e-4, fixed_sigma: float = 1.0):
        super().__init__()
        self.learned_scale = learned_scale
        self.min_df = min_df
        self.max_df = max_df
        self.min_sigma = min_sigma
        self._fixed_sigma = fixed_sigma

        # Learned degrees of freedom (constrained to [min_df, max_df]).
        self.logit_df = nn.Parameter(torch.tensor(math.log((init_df - min_df) / (max_df - init_df))))

        if learned_scale:
            # Learned global log-scale.
            self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def df(self) -> torch.Tensor:
        """Degrees of freedom in [min_df, max_df]."""

        return self.min_df + (self.max_df - self.min_df) * torch.sigmoid(self.logit_df)

    @property
    def sigma(self) -> torch.Tensor:
        """Global scale sigma."""

        if self.learned_scale:
            return F.softplus(self.log_sigma_raw) + self.min_sigma
        return torch.tensor(self._fixed_sigma)

    def forward(self, mu: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor,
                measurement_variance: torch.Tensor | float | None = None,
                ) -> torch.Tensor:
        """Compute Student-t NLL.

        Parameters
        ----------
        mu : (n,) — predicted delta_r.
        target : (n,) — ground-truth delta_r.
        mask : (n,) bool — True for positions to include in loss.
        measurement_variance : (n,) or scalar or None — per-position measurement
            noise variance. Added to sigma^2.

        Returns
        -------
        loss : scalar — mean NLL over masked positions.
        """

        if mask.sum() == 0:
            return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

        mu_m = mu[mask]
        y_m = target[mask]
        n = mu_m.shape[0]

        df = self.df
        sigma = self.sigma

        # Total variance = model variance + measurement variance.
        if measurement_variance is not None:
            if isinstance(measurement_variance, (int, float)):
                meas_var = torch.tensor(float(measurement_variance), dtype=mu.dtype, device=mu.device)
            else:
                meas_var = measurement_variance[mask]
            sigma2 = sigma ** 2 + meas_var
        else:
            sigma2 = sigma ** 2

        sigma2 = sigma2.clamp(min=self.min_sigma ** 2)

        # Student-t NLL.
        log_gamma_num = torch.lgamma(0.5 * (df + 1.0))
        log_gamma_den = torch.lgamma(0.5 * df)
        log_pref = 0.5 * torch.log(df * math.pi * sigma2)
        residual_sq = (y_m - mu_m) ** 2
        log_tail = (0.5 * (df + 1.0)) * torch.log1p(residual_sq / (df * sigma2))

        nll = -log_gamma_num + log_gamma_den + log_pref + log_tail
        return nll.mean()


class HeteroscedasticNLL(nn.Module):
    """Heteroscedastic Gaussian NLL (§4.9 EPRO-Core output).

    The model predicts both mean ``mu`` and log-variance ``log_var``.
    NLL = 0.5 * log(sigma^2) + 0.5 * (y - mu)^2 / sigma^2

    Measurement variance is added to the predicted variance.
    """

    def __init__(self, *, min_var: float = 1e-6):
        super().__init__()
        self.min_var = min_var

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor, mask: torch.Tensor,
                measurement_variance: torch.Tensor | float | None = None,
                ) -> torch.Tensor:
        """Compute heteroscedastic NLL.

        Parameters
        ----------
        mu : (n,) — predicted mean.
        log_var : (n,) — predicted log-variance.
        target : (n,) — ground-truth.
        mask : (n,) bool.
        measurement_variance : per-position measurement noise variance.

        Returns
        -------
        loss : scalar.
        """

        if mask.sum() == 0:
            return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

        mu_m = mu[mask]
        lv_m = log_var[mask]
        y_m = target[mask]

        var = F.softplus(lv_m) + self.min_var

        if measurement_variance is not None:
            if isinstance(measurement_variance, (int, float)):
                meas_var = torch.tensor(float(measurement_variance), dtype=mu.dtype, device=mu.device)
            else:
                meas_var = measurement_variance[mask]
            var = var + meas_var

        nll = 0.5 * torch.log(var) + 0.5 * (y_m - mu_m) ** 2 / var
        return nll.mean()


class WeightedMAELoss(nn.Module):
    """Pair-quality-weighted MAE (for monitoring, matches evaluator Skill metric)."""

    def forward(self, mu: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
        mu_m = mu[mask]
        y_m = target[mask]
        return weight * (mu_m - y_m).abs().mean()


class WeightedMSELoss(nn.Module):
    """Pair-quality-weighted MSE (for overfit training, provides gradient
    proportional to error magnitude unlike MAE)."""

    def forward(self, mu: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
        mu_m = mu[mask]
        y_m = target[mask]
        return weight * ((mu_m - y_m) ** 2).mean()


def compute_skill(mu: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> float:
    """Compute pair-level Skill = 1 - WMAE(pred, true) / WMAE(0, true).

    Matches the evaluator's per-pair Skill (§12.1). Returns NaN if
    WMAE(0, true) == 0.
    """

    if mask.sum() == 0:
        return float("nan")
    mu_m = mu[mask].detach().cpu().numpy()
    y_m = target[mask].detach().cpu().numpy()
    wmae_pred = abs(mu_m - y_m).mean()
    wmae_zero = abs(y_m).mean()
    if wmae_zero == 0:
        return float("nan")
    return float(1.0 - wmae_pred / wmae_zero)
