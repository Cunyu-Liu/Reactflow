"""Teacher BPP distillation loss component.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies, distillation).

This module is **not a fusion strategy in the usual sense**; it does not
combine backbone feature stacks.  Instead it wraps a frozen *teacher* model
(typically a strong baseline PairFormer or a pre-trained structure predictor)
and computes a knowledge-distillation loss between the teacher's base-pair
probability (BPP) predictions and the student's BPP predictions.

The distillation loss is the symmetric KL divergence scaled by
``temperature^2`` (Hinton et al., 2015), which preserves the gradient
magnitude when the student operates at a different temperature than the
teacher::

    distillation_loss = T^2 * KL(softmax(student_logits / T) || softmax(teacher_logits / T))

The teacher's parameters are frozen (``requires_grad=False``) so only the
student receives gradients.  The teacher forward runs under ``torch.no_grad``
to avoid retaining its activation graph.

Formula
-------
For each valid cell ``(i, j)`` (upper-triangle, both endpoints real)::

    p_s = softmax([logit_s / T, 1 - logit_s / T])  # student soft label
    p_t = softmax([logit_t / T, 1 - logit_t / T])  # teacher soft label
    loss_ij = T^2 * (p_t * (log p_t - log p_s)).sum()

where the sum is over the two classes (pair / non-pair).  The result is
averaged over all valid cells.

Complexity
----------
- Time: ``O(B * L^2)`` for the KL divergence (dominated by the teacher forward
  which is backbone-specific).
- Memory: ``O(B * L^2)`` for the teacher BPP (no gradients retained).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import FusionConfig, FusionStrategy


class TeacherBPPDistillation(FusionStrategy):
    """Knowledge-distillation loss from a frozen teacher's BPP predictions.

    This class inherits :class:`FusionStrategy` for package consistency, but
    its :meth:`forward` is an identity pass-through (it returns the first
    backbone's features unchanged).  The real functionality is in
    :meth:`distillation_loss`, which runs the frozen teacher and computes the
    symmetric KL divergence between teacher and student BPPs.

    Args:
        config: :class:`FusionConfig`.  ``single_dim`` and ``pair_dim`` should
            match the teacher's output dimensions.
        teacher: the frozen teacher :class:`~torch.nn.Module` that maps
            ``(indices, mask)`` to a BPP tensor of shape ``(B, L, L)``.  The
            teacher must expose either a ``forward(indices, mask)`` returning a
            tensor, or a ``predict_bpp(indices, mask)`` method.  Its
            parameters are frozen on construction.
        temperature: distillation temperature ``T``.  Higher ``T`` produces
            softer distributions.  Default ``1.0``.

    Complexity: construction ``O(P_teacher)`` to freeze params; the teacher
    forward complexity is model-specific.
    """

    def __init__(
        self,
        config: FusionConfig,
        teacher: nn.Module,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(config)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.teacher: nn.Module = teacher
        self.temperature: float = temperature

        # Freeze the teacher so only the student trains.
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    # -- FusionStrategy interface (identity pass-through) ------------------

    def forward(
        self,
        single_features: List[torch.Tensor],
        pair_features: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Identity pass-through (this is not a real fusion strategy).

        Returns the first backbone's single and pair features unchanged.
        The actual functionality is in :meth:`distillation_loss`.

        Args:
            single_features: list of length ``N``; the first element is
                returned as the fused single.
            pair_features: list of length ``N``; the first element is
                returned as the fused pair (may be ``None``).
            mask: BoolTensor ``(B, L)`` passed through unchanged.

        Returns:
            Tuple ``(single_features[0], pair_features[0])``.

        Complexity: ``O(1)`` (no computation).
        """
        if not single_features:
            raise ValueError("single_features must not be empty")
        fused_single = single_features[0]
        fused_pair = pair_features[0] if pair_features else None
        return fused_single, fused_pair

    # -- Distillation ------------------------------------------------------

    def _compute_teacher_bpp(
        self,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the frozen teacher to produce BPP predictions.

        The teacher is expected to accept ``(indices, mask)`` and return a
        BPP tensor of shape ``(B, L, L)``.  If the teacher's forward returns
        a structured output with a ``bpp`` field (e.g. a
        :class:`~reactflow.models.static_pairformer.PairFormerOutput`), that
        field is extracted.

        Args:
            indices: LongTensor ``(B, L)`` nucleotide vocab indices.
            mask: BoolTensor ``(B, L)`` real-position mask.

        Returns:
            FloatTensor ``(B, L, L)`` teacher BPP probabilities.

        Complexity: teacher-model-specific.
        """
        with torch.no_grad():
            out = self.teacher(indices, mask)
            if isinstance(out, torch.Tensor):
                bpp = out
            elif hasattr(out, "bpp"):
                bpp = out.bpp
            elif hasattr(out, "logits"):
                bpp = torch.sigmoid(out.logits)
            else:
                raise TypeError(
                    f"Unsupported teacher output type {type(out)}: expected a "
                    f"Tensor or an object with a 'bpp' or 'logits' attribute"
                )
        return bpp

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the symmetric KL distillation loss.

        Formula::

            L = T^2 * mean_{i<j, valid} KL(p_teacher || p_student)

        where ``p = softmax([logit/T, -logit/T])`` over the binary
        (pair / non-pair) axis.

        Args:
            student_logits: FloatTensor ``(B, L, L)`` raw pair logits from
                the student model.
            indices: LongTensor ``(B, L)`` nucleotide vocab indices (passed
                to the teacher).
            mask: BoolTensor ``(B, L)`` real-position mask.

        Returns:
            Scalar distillation loss.

        Complexity: ``O(B * L^2)`` plus the teacher forward.
        """
        B, L, _ = student_logits.shape
        device = student_logits.device
        T = self.temperature

        teacher_bpp = self._compute_teacher_bpp(indices, mask)  # (B, L, L)

        # Convert teacher BPP to logits (inverse sigmoid, numerically stable).
        teacher_bpp = teacher_bpp.clamp(1e-7, 1.0 - 1e-7)
        teacher_logits = torch.log(teacher_bpp / (1.0 - teacher_bpp))

        # Build the valid-cell mask: upper triangle, both endpoints real,
        # and student logits finite (excludes -inf diagonal).
        upper = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )
        both_real = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
        valid = both_real & upper.unsqueeze(0)  # (B, L, L)
        valid = valid & torch.isfinite(student_logits)

        # Soft labels via temperature scaling.
        # Treat each cell as a binary classification: [pair, non-pair].
        s_logit = student_logits / T  # (B, L, L)
        t_logit = teacher_logits / T

        # Stack to (B, L, L, 2) so softmax is over the class axis.
        s_pair = torch.stack([s_logit, -s_logit], dim=-1)  # (B, L, L, 2)
        t_pair = torch.stack([t_logit, -t_logit], dim=-1)

        log_p_s = F.log_softmax(s_pair, dim=-1)
        log_p_t = F.log_softmax(t_pair, dim=-1)

        # KL(p_t || p_s) = sum_c p_t * (log p_t - log p_s)
        p_t = log_p_t.exp()
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # (B, L, L)

        # Average over valid cells.
        valid_f = valid.to(kl.dtype)
        count = valid_f.sum().clamp(min=1.0)
        loss = (kl * valid_f).sum() / count
        loss = loss * (T * T)

        return loss
