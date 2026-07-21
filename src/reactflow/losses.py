"""Training losses for the C1-2 static PairFormer.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 379-384.

Loss components
---------------
- :func:`class_balanced_bce_loss` -- BCE on pair logits with positive-class
  weight to counter the heavy class imbalance (most L*L cells are non-pairs).
- :func:`focal_loss` -- focal loss (Lin et al.) with gamma focusing parameter.
- :func:`soft_f1_loss` / :func:`dice_loss` -- differentiable F1 / Dice loss
  computed from soft TP/FP/FN over the upper-triangle candidate cells.
- :func:`pair_count_reg_loss` -- L1/L2 regularizer pulling the predicted pair
  count toward the target count.
- :func:`symmetry_audit_loss` -- ``||logits - logits^T||_2`` to penalize any
  asymmetry that drifts in through float arithmetic (should be ~0 by
  construction, but is enforced as a safety term).
- :func:`calibration_loss` -- Expected Calibration Error (ECE) computed over
  reliability bins, plus a Brier score term.
- :func:`unpaired_bce_loss` -- BCE on per-position unpaired probabilities
  against the implicit unpaired target (``1 - sum_j P_ij^*``).
- :func:`pairformer_loss` -- combined weighted sum of all the above.

All losses accept batched inputs ``(B, L, L)`` for pair tensors and ``(B, L)``
for single tensors.  Padding positions are excluded via the mask.

Complexity
----------
All loss terms are ``O(B * L^2)`` except ECE which is ``O(B * L^2 + num_bins)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .models.static_pairformer import PairFormerOutput


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Configuration for :func:`pairformer_loss`.

    Weights default to a reasonable starting point; tune via the model config
    YAML.  Setting a weight to 0 disables that term.

    Attributes:
        bce_weight: weight for class-balanced BCE.
        focal_weight: weight for focal loss.
        soft_f1_weight: weight for soft F1 loss.
        dice_weight: weight for Dice loss.
        pair_count_weight: weight for pair-count regularizer.
        symmetry_weight: weight for symmetry audit loss.
        calibration_weight: weight for ECE + Brier calibration loss.
        unpaired_weight: weight for unpaired BCE loss.
        focal_gamma: focusing parameter for focal loss.
        bce_pos_weight: positive-class weight for BCE (override auto-compute).
        bce_pos_weight_auto: if True, compute pos_weight from the batch.
        ece_num_bins: number of reliability bins for ECE.
    """

    bce_weight: float = 1.0
    focal_weight: float = 0.5
    soft_f1_weight: float = 0.5
    dice_weight: float = 0.0
    pair_count_weight: float = 0.1
    symmetry_weight: float = 0.01
    calibration_weight: float = 0.1
    unpaired_weight: float = 0.5

    focal_gamma: float = 2.0
    bce_pos_weight: Optional[float] = None
    bce_pos_weight_auto: bool = True
    ece_num_bins: int = 15

    def __post_init__(self) -> None:
        if self.ece_num_bins < 1:
            raise ValueError("ece_num_bins must be >= 1")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upper_triangle_mask(L: int, *, device: torch.device) -> torch.Tensor:
    """Return a ``(L, L)`` bool mask that is True for upper-triangle cells (i<j)."""
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


def _pair_mask_and_diag(
    mask: Optional[torch.Tensor],
    L: int,
    *,
    device: torch.device,
) -> tuple:
    """Return (pair_cell_mask, diag_mask) for the batch.

    ``pair_cell_mask`` is True for valid (i, j) upper-triangle cells with
    both endpoints real.  ``diag_mask`` is True on the diagonal.
    """
    diag = torch.eye(L, dtype=torch.bool, device=device)
    upper = _upper_triangle_mask(L, device=device)
    if mask is not None:
        both_real = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
        pair_cell_mask = both_real & upper.unsqueeze(0)
    else:
        pair_cell_mask = upper.unsqueeze(0).expand(1, L, L)
    return pair_cell_mask, diag


def _safe_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Numerically stable BCE with logits, supporting pos_weight per element."""
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )


# ---------------------------------------------------------------------------
# Class-balanced BCE loss
# ---------------------------------------------------------------------------


def class_balanced_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[float] = None,
    auto_pos_weight: bool = True,
) -> torch.Tensor:
    """Class-balanced BCE on pair logits.

    Args:
        logits: ``(B, L, L)`` pair logits (symmetric, diagonal = -inf).
        targets: ``(B, L, L)`` binary target pair matrix (symmetric).
        mask: optional ``(B, L)`` real-position mask.
        pos_weight: positive-class weight (overrides auto).
        auto_pos_weight: if True and ``pos_weight`` is None, compute pos_weight
            as ``num_negatives / max(num_positives, 1)`` over the valid cells.

    Returns:
        Scalar loss (mean over valid upper-triangle cells).

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    # Use upper triangle only (symmetric targets/logits)
    flat_mask = pair_cell_mask.reshape(-1)
    flat_logits = logits.reshape(-1)[flat_mask]
    flat_targets = targets.reshape(-1)[flat_mask].to(flat_logits.dtype)

    if pos_weight is None and auto_pos_weight:
        num_pos = flat_targets.sum().clamp(min=1.0)
        num_neg = (1.0 - flat_targets).sum().clamp(min=1.0)
        pw = (num_neg / num_pos).clamp(max=1e4)
    elif pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=device, dtype=flat_logits.dtype)
    else:
        pw = None

    loss = _safe_bce_with_logits(flat_logits, flat_targets, pos_weight=pw)
    return loss.mean()


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Focal loss (Lin et al.) on pair logits.

    Formula: ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``, where
    ``p_t = sigmoid(logit)`` if target=1 else ``1 - sigmoid(logit)``.

    Args:
        logits: ``(B, L, L)`` pair logits.
        targets: ``(B, L, L)`` binary target.
        mask: optional ``(B, L)``.
        gamma: focusing parameter (default 2.0).
        alpha: optional balancing weight (applied to positive class).

    Returns:
        Scalar loss.

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    flat_mask = pair_cell_mask.reshape(-1)
    flat_logits = logits.reshape(-1)[flat_mask]
    flat_targets = targets.reshape(-1)[flat_mask].to(flat_logits.dtype)

    # Numerically stable sigmoid
    p = torch.sigmoid(flat_logits)
    p_t = p * flat_targets + (1.0 - p) * (1.0 - flat_targets)
    # Focal modulator
    modulator = (1.0 - p_t).clamp(min=1e-8) ** gamma
    # Cross-entropy per element
    bce = F.binary_cross_entropy_with_logits(
        flat_logits, flat_targets, reduction="none"
    )
    loss = modulator * bce
    if alpha is not None:
        alpha_t = alpha * flat_targets + (1.0 - alpha) * (1.0 - flat_targets)
        loss = alpha_t * loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Soft F1 / Dice loss
# ---------------------------------------------------------------------------


def soft_f1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable F1 loss = ``1 - soft_F1``.

    Soft TP/FP/FN are computed by integrating ``sigmoid(logit) * target``,
    ``sigmoid(logit) * (1 - target)`` and ``(1 - sigmoid(logit)) * target``
    over valid upper-triangle cells.  The result is a smooth surrogate that can
    be optimized directly with gradient descent.

    Args:
        logits: ``(B, L, L)`` pair logits.
        targets: ``(B, L, L)`` binary target.
        mask: optional ``(B, L)``.
        eps: numerical stability.

    Returns:
        Scalar loss.

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    probs = torch.sigmoid(logits)
    # Use upper triangle only
    probs_m = probs * pair_cell_mask.float()
    targets_m = targets.to(probs.dtype) * pair_cell_mask.float()

    tp = (probs_m * targets_m).sum()
    fp = (probs_m * (1.0 - targets_m)).sum()
    fn = ((1.0 - probs_m) * targets_m).sum()

    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return 1.0 - f1


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable Dice loss = ``1 - soft_Dice``.

    Dice = ``2 * TP / (2 * TP + FP + FN)`` (same as F1 up to smoothing).  We
    expose it separately for the config-weighted combination, but it is
    mathematically equivalent to :func:`soft_f1_loss` modulo the smoothing
    constant.  The difference is that Dice typically uses ``probs * targets``
    and ``probs + targets`` style smoothing, which we follow here.

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    probs = torch.sigmoid(logits)
    probs_m = probs * pair_cell_mask.float()
    targets_m = targets.to(probs.dtype) * pair_cell_mask.float()

    intersection = (probs_m * targets_m).sum()
    denom = probs_m.sum() + targets_m.sum()
    dice = (2.0 * intersection) / (denom + eps)
    return 1.0 - dice


# ---------------------------------------------------------------------------
# Pair-count regularizer
# ---------------------------------------------------------------------------


def pair_count_reg_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    p_norm: int = 1,
) -> torch.Tensor:
    """Regularizer pulling the predicted pair count toward the target count.

    Args:
        logits: ``(B, L, L)`` pair logits.
        targets: ``(B, L, L)`` binary target.
        mask: optional ``(B, L)``.
        p_norm: 1 for L1, 2 for L2.

    Returns:
        Scalar loss.

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    probs = torch.sigmoid(logits)
    probs_m = probs * pair_cell_mask.float()
    targets_m = targets.to(probs.dtype) * pair_cell_mask.float()

    # Count predicted pairs (sum over upper triangle, divide by 2 since matrix
    # is symmetric — but we restricted to upper triangle so no division).
    pred_count = probs_m.sum(dim=[1, 2])  # (B,)
    target_count = targets_m.sum(dim=[1, 2])  # (B,)

    diff = pred_count - target_count
    if p_norm == 1:
        return diff.abs().mean()
    if p_norm == 2:
        return (diff ** 2).mean()
    raise ValueError(f"unsupported p_norm: {p_norm}")


# ---------------------------------------------------------------------------
# Symmetry audit loss
# ---------------------------------------------------------------------------


def symmetry_audit_loss(logits: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalty for asymmetry in the pair logits.

    Formula: ``mean((logit_ij - logit_ji)^2)`` over valid off-diagonal cells.

    This should be ~0 by construction (the model symmetrizes its output), but
    including it as a safety term guards against any drift from custom
    modifications.

    Complexity: ``O(B * L^2)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    _, diag = _pair_mask_and_diag(mask, L, device=device)
    # Include both (i,j) and (j,i) — full off-diagonal — for the symmetry check.
    if mask is not None:
        both_real = mask.unsqueeze(2) & mask.unsqueeze(1)
    else:
        both_real = torch.ones(L, L, dtype=torch.bool, device=device).unsqueeze(0)
    # Valid off-diagonal cells: both endpoints real, not diagonal, and both
    # logits finite (excludes -inf masked cells where diff would be NaN).
    off_diag = both_real & ~diag.unsqueeze(0) & torch.isfinite(logits) & torch.isfinite(logits.transpose(1, 2))
    # Use torch.where to avoid NaN from (-inf) - (-inf) on the diagonal.
    diff = logits - logits.transpose(1, 2)
    diff_safe = torch.where(off_diag, diff, torch.zeros_like(diff))
    return (diff_safe ** 2).sum() / off_diag.float().sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Calibration loss (ECE + Brier)
# ---------------------------------------------------------------------------


def calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    num_bins: int = 15,
    brier_weight: float = 0.5,
) -> torch.Tensor:
    """ECE + Brier score loss for probability calibration.

    ECE = ``sum_b (|B_b|/N) |acc(B_b) - conf(B_b)|`` where ``acc`` is the
    empirical positive rate and ``conf`` the mean predicted probability in bin
    ``b``.

    Brier = ``mean((p - y)^2)``.

    Args:
        logits: ``(B, L, L)`` pair logits.
        targets: ``(B, L, L)`` binary target.
        mask: optional ``(B, L)``.
        num_bins: number of reliability bins for ECE.
        brier_weight: weight for the Brier term.

    Returns:
        Scalar loss.

    Complexity: ``O(B * L^2 + num_bins)``.
    """
    B, L, _ = logits.shape
    device = logits.device
    pair_cell_mask, _ = _pair_mask_and_diag(mask, L, device=device)
    pair_cell_mask = pair_cell_mask & torch.isfinite(logits)

    flat_mask = pair_cell_mask.reshape(-1)
    flat_logits = logits.reshape(-1)[flat_mask]
    flat_targets = targets.reshape(-1)[flat_mask].to(flat_logits.dtype)

    probs = torch.sigmoid(flat_logits)

    # ECE (differentiable relaxation: bin by predicted prob, compute mean)
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    ece = torch.zeros(1, device=device)
    for b in range(num_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        in_bin = (probs >= lo) & (probs < hi if b < num_bins - 1 else probs <= hi)
        if in_bin.sum() > 0:
            acc = flat_targets[in_bin].mean()
            conf = probs[in_bin].mean()
            weight = in_bin.float().sum() / flat_targets.numel()
            ece = ece + weight * (acc - conf).abs()

    # Brier score
    brier = ((probs - flat_targets) ** 2).mean()

    return ece + brier_weight * brier


# ---------------------------------------------------------------------------
# Unpaired BCE loss
# ---------------------------------------------------------------------------


def unpaired_bce_loss(
    unpaired_logit: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE on per-position unpaired logits.

    The implicit unpaired target is ``1 - sum_j P_ij^*`` (1 if position i is
    not in any pair, 0 if it is paired).

    Args:
        unpaired_logit: ``(B, L)`` per-position unpaired logits.
        targets: ``(B, L, L)`` binary target pair matrix (symmetric).
        mask: optional ``(B, L)``.

    Returns:
        Scalar loss.

    Complexity: ``O(B * L^2)`` to compute the target, ``O(B * L)`` for BCE.
    """
    # Compute per-position "is unpaired" target
    # A position i is unpaired iff sum_j targets[i, j] == 0
    paired_indicator = targets.sum(dim=-1).clamp(max=1.0)  # (B, L), 1 if paired
    unpaired_target = 1.0 - paired_indicator  # (B, L)

    if mask is not None:
        # Filter to valid positions BEFORE computing BCE to avoid NaN from
        # -inf logits on padding positions (inf * 0 = NaN in IEEE 754).
        valid = mask
        flat_logits = unpaired_logit[valid]
        flat_targets = unpaired_target[valid]
        if flat_logits.numel() == 0:
            return unpaired_logit.new_zeros(())
        # Replace any remaining -inf/inf with 0 for safety (shouldn't happen
        # after filtering, but guards against edge cases).
        flat_logits = torch.where(
            torch.isfinite(flat_logits), flat_logits, flat_logits.new_zeros(())
        )
        return F.binary_cross_entropy_with_logits(
            flat_logits, flat_targets, reduction="mean"
        )
    else:
        return F.binary_cross_entropy_with_logits(
            unpaired_logit, unpaired_target, reduction="mean"
        )


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def pairformer_loss(
    output: PairFormerOutput,
    targets: torch.Tensor,
    *,
    config: Optional[LossConfig] = None,
) -> dict:
    """Combined weighted loss for the static PairFormer.

    Args:
        output: PairFormerOutput (logits, bpp, unpaired_logit, mask, temperature).
        targets: ``(B, L, L)`` binary target pair matrix (symmetric, 0 on diagonal).
        config: loss configuration (uses default if None).

    Returns:
        Dict with keys: ``total``, ``bce``, ``focal``, ``soft_f1``, ``dice``,
        ``pair_count``, ``symmetry``, ``calibration``, ``unpaired``.  Each value
        is a scalar tensor (0 for disabled terms).

    Complexity: ``O(B * L^2)`` overall (dominated by the BCE term).
    """
    cfg = config or LossConfig()
    logits = output.logits
    mask = output.mask
    B, L, _ = logits.shape

    parts = {
        "bce": torch.tensor(0.0, device=logits.device),
        "focal": torch.tensor(0.0, device=logits.device),
        "soft_f1": torch.tensor(0.0, device=logits.device),
        "dice": torch.tensor(0.0, device=logits.device),
        "pair_count": torch.tensor(0.0, device=logits.device),
        "symmetry": torch.tensor(0.0, device=logits.device),
        "calibration": torch.tensor(0.0, device=logits.device),
        "unpaired": torch.tensor(0.0, device=logits.device),
    }

    if cfg.bce_weight > 0:
        parts["bce"] = class_balanced_bce_loss(
            logits, targets, mask=mask,
            pos_weight=cfg.bce_pos_weight, auto_pos_weight=cfg.bce_pos_weight_auto,
        )

    if cfg.focal_weight > 0:
        parts["focal"] = focal_loss(
            logits, targets, mask=mask, gamma=cfg.focal_gamma,
        )

    if cfg.soft_f1_weight > 0:
        parts["soft_f1"] = soft_f1_loss(logits, targets, mask=mask)

    if cfg.dice_weight > 0:
        parts["dice"] = dice_loss(logits, targets, mask=mask)

    if cfg.pair_count_weight > 0:
        parts["pair_count"] = pair_count_reg_loss(
            logits, targets, mask=mask, p_norm=1,
        )

    if cfg.symmetry_weight > 0:
        parts["symmetry"] = symmetry_audit_loss(logits, mask=mask)

    if cfg.calibration_weight > 0:
        parts["calibration"] = calibration_loss(
            logits, targets, mask=mask, num_bins=cfg.ece_num_bins,
        )

    if cfg.unpaired_weight > 0 and output.unpaired_logit is not None:
        parts["unpaired"] = unpaired_bce_loss(
            output.unpaired_logit, targets, mask=mask,
        )

    total = (
        cfg.bce_weight * parts["bce"]
        + cfg.focal_weight * parts["focal"]
        + cfg.soft_f1_weight * parts["soft_f1"]
        + cfg.dice_weight * parts["dice"]
        + cfg.pair_count_weight * parts["pair_count"]
        + cfg.symmetry_weight * parts["symmetry"]
        + cfg.calibration_weight * parts["calibration"]
        + cfg.unpaired_weight * parts["unpaired"]
    )
    parts["total"] = total
    return parts
