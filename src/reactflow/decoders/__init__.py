"""Decoders that convert pair logits / BPP into binary pair matrices.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 385-389 (C1-2 decoders).

Decoders provided
-----------------
- :func:`threshold_decoder` -- cell-wise threshold on BPP, then legality
  projection (greedy).
- :func:`nussinov_dp_decoder` -- exact maximum-weight nested matching (Nussinov
  DP) on log-odds scores.  This is the **default** decoder per the frozen
  ``static_v1.yaml`` evaluator contract.
- :func:`mea_decoder` -- Maximum Expected Accuracy DP on BPP, maximizing the
  expected number of correct (paired + unpaired) decisions.
- :func:`greedy_pseudoknot_decoder` -- greedy matching that allows pseudoknots
  (used only for explicit pseudoknot-allowed experiments).

All decoders accept batched inputs ``(B, L, L)`` and return binary pair matrices
``(B, L, L)`` as ``torch.Tensor`` of dtype ``torch.float32``.  Legality
constraints (canonical/wobble pairs, ``min_loop``) are enforced by routing
through the legacy ``reactflow.constraints`` projection routines where needed.

Complexity
----------
- threshold: ``O(L^2)`` for thresholding + ``O(L^2 log L)`` for greedy
  projection.
- Nussinov DP: ``O(L^3)`` time, ``O(L^2)`` memory per sequence.
- MEA: ``O(L^3)`` time, ``O(L^2)`` memory per sequence.
- greedy pseudoknot: ``O(L^2 log L)`` time, ``O(L^2)`` memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..constraints import (  # type: ignore[import-not-found]
    project_greedy_matching,
    project_max_weight_nested,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DecoderConfig:
    """Common configuration for all decoders.

    Attributes:
        min_loop: minimum loop length (default 3, per ``static_v1.yaml``).
        allow_wobble: whether to allow GU/UG pairs (default True).
        threshold: cell-wise threshold for the threshold decoder (default 0.5).
        min_score: log-odds threshold for Nussinov DP (default 0.0).
        seq_pad_index: vocab index used for padding (default 5).
    """

    min_loop: int = 3
    allow_wobble: bool = True
    threshold: float = 0.5
    min_score: float = 0.0
    seq_pad_index: int = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VOCAB_INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N", 5: "-"}


def _indices_to_sequence(indices: torch.Tensor, *, pad_index: int = 5) -> str:
    """Convert a 1-D LongTensor of vocab indices to an uppercase RNA sequence.

    Padding positions are replaced with ``N`` so they never form legal pairs.
    """
    out_chars = []
    for idx in indices.tolist():
        if idx == pad_index:
            out_chars.append("N")
            continue
        out_chars.append(_VOCAB_INDEX_TO_BASE.get(int(idx), "N"))
    return "".join(out_chars)


def _scores_to_list_matrix(scores: torch.Tensor) -> Tuple[Tuple[float, ...], ...]:
    """Convert a 2-D tensor to a tuple-of-tuples (for the legacy projectors)."""
    rows = scores.tolist()
    return tuple(tuple(row) for row in rows)


def _binary_matrix_from_indices(
    pairs: Tuple[Tuple[int, ...], ...],
    length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert a tuple-of-tuples binary matrix to a torch tensor."""
    mat = torch.zeros(length, length, dtype=torch.float32, device=device)
    for i, row in enumerate(pairs):
        for j, val in enumerate(row):
            if val:
                mat[i, j] = 1.0
    return mat


# ---------------------------------------------------------------------------
# Threshold decoder
# ---------------------------------------------------------------------------


def threshold_decoder(
    logits: torch.Tensor,
    *,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: Optional[DecoderConfig] = None,
    use_bpp: bool = False,
    bpp: Optional[torch.Tensor] = None,
    apply_legality: bool = True,
) -> torch.Tensor:
    """Cell-wise threshold decoder.

    Args:
        logits: ``(B, L, L)`` pair logits.
        indices: ``(B, L)`` nucleotide vocab indices.
        mask: optional ``(B, L)`` real-position mask.
        config: decoder config.
        use_bpp: if True, threshold the BPP instead of logits.
        bpp: pre-computed BPP (B, L, L).  Required if ``use_bpp=True``.
        apply_legality: if True, project the thresholded matrix to a legal
            matching via the greedy projector.  If False, return the raw
            thresholded matrix (which may have illegal pairs and
            non-matching).

    Returns:
        Binary pair matrix ``(B, L, L)`` as float32 tensor.

    Complexity: ``O(B * L^2)`` thresholding + ``O(B * L^2 log L)`` if legality
    projection is applied.
    """
    cfg = config or DecoderConfig()
    B, L, _ = logits.shape
    device = logits.device

    if mask is None:
        mask = indices != cfg.seq_pad_index

    if use_bpp:
        if bpp is None:
            raise ValueError("use_bpp=True requires bpp argument")
        scores = bpp
    else:
        scores = torch.sigmoid(logits)

    thresholded = (scores > cfg.threshold).float()

    # Zero out diagonal and padding
    diag = torch.eye(L, dtype=torch.bool, device=device)
    thresholded = thresholded * (~diag).unsqueeze(0).float()
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    thresholded = thresholded * pair_mask.float()

    if not apply_legality:
        return thresholded

    # Project to legal matching per sequence
    out = torch.zeros_like(thresholded)
    for b in range(B):
        seq = _indices_to_sequence(indices[b], pad_index=cfg.seq_pad_index)
        scores_b = thresholded[b].cpu().tolist()
        # Use greedy matching with pseudoknot=False to enforce nested legality
        # We treat the thresholded matrix as the score (1.0 for selected, 0.0 otherwise)
        # and set min_score=0.5 so only thresholded cells are candidates.
        legal = project_greedy_matching(
            seq,
            scores_b,
            min_loop=cfg.min_loop,
            allow_wobble=cfg.allow_wobble,
            allow_pseudoknot=False,
            min_score=0.5,
        )
        out[b] = _binary_matrix_from_indices(legal, L, device=device)
    return out


# ---------------------------------------------------------------------------
# Nussinov DP decoder (default per static_v1.yaml)
# ---------------------------------------------------------------------------


def nussinov_dp_decoder(
    logits: torch.Tensor,
    *,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: Optional[DecoderConfig] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact maximum-weight nested matching decoder (Nussinov DP).

    This is the **default** decoder per the frozen ``static_v1.yaml`` evaluator
    contract.  It takes the pair logits, optionally divides by a temperature,
    and solves the exact maximum-weight nested matching via
    :func:`reactflow.constraints.project_max_weight_nested`.

    Args:
        logits: ``(B, L, L)`` pair logits.
        indices: ``(B, L)`` nucleotide vocab indices.
        mask: optional ``(B, L)`` real-position mask.
        config: decoder config.
        temperature: optional scalar tensor to divide logits before DP.

    Returns:
        Binary pair matrix ``(B, L, L)`` as float32 tensor.

    Complexity: ``O(B * L^3)`` time, ``O(B * L^2)`` memory.
    """
    cfg = config or DecoderConfig()
    B, L, _ = logits.shape
    device = logits.device

    if mask is None:
        mask = indices != cfg.seq_pad_index

    if temperature is not None:
        scores = logits / temperature.clamp(min=1e-6)
    else:
        scores = logits

    # Mask out diagonal and padding with -inf so they are never selected
    diag = torch.eye(L, dtype=torch.bool, device=device)
    scores = scores.masked_fill(diag.unsqueeze(0), float("-inf"))
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    scores = scores.masked_fill(~pair_mask, float("-inf"))

    out = torch.zeros(B, L, L, dtype=torch.float32, device=device)
    for b in range(B):
        seq = _indices_to_sequence(indices[b], pad_index=cfg.seq_pad_index)
        scores_b = scores[b].cpu().tolist()
        legal = project_max_weight_nested(
            seq,
            scores_b,
            min_loop=cfg.min_loop,
            allow_wobble=cfg.allow_wobble,
            min_score=cfg.min_score,
        )
        out[b] = _binary_matrix_from_indices(legal, L, device=device)
    return out


# ---------------------------------------------------------------------------
# Maximum Expected Accuracy (MEA) decoder
# ---------------------------------------------------------------------------


def mea_decoder(
    bpp: torch.Tensor,
    *,
    indices: torch.Tensor,
    unpaired_prob: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    config: Optional[DecoderConfig] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Maximum Expected Accuracy decoder.

    Solves the MEA DP on BPP to maximize the expected number of correct
    decisions (paired + unpaired).  The recurrence (for nested structures) is

        W[i][j] = max(
            W[i][j-1] + s_unpaired(j),                    # j unpaired
            max_{k in [i, j-min_loop-1], (k,j) legal}
                W[i][k-1] + gamma * 2 * P_kj + W[k+1][j-1]  # j pairs with k
        ),

    where ``s_unpaired(j) = 2 * P_unpaired_j - 1`` if ``unpaired_prob`` is
    provided, else ``0``.  ``gamma`` is the standard MEA scaling factor that
    trades off sensitivity vs specificity.

    Args:
        bpp: ``(B, L, L)`` base-pair probabilities in ``[0, 1]``.
        indices: ``(B, L)`` nucleotide vocab indices.
        unpaired_prob: optional ``(B, L)`` per-position unpaired probabilities.
        mask: optional ``(B, L)`` real-position mask.
        config: decoder config (uses ``min_loop``, ``allow_wobble``).
        gamma: MEA scaling factor (default 1.0).

    Returns:
        Binary pair matrix ``(B, L, L)`` as float32 tensor.

    Complexity: ``O(B * L^3)`` time, ``O(B * L^2)`` memory.
    """
    from ..constraints import is_allowed_pair  # type: ignore[import-not-found]

    cfg = config or DecoderConfig()
    B, L, _ = bpp.shape
    device = bpp.device

    if mask is None:
        mask = indices != cfg.seq_pad_index

    # Clamp BPP to avoid log(0) issues
    bpp_safe = bpp.clamp(min=1e-8, max=1.0 - 1e-8)

    out = torch.zeros(B, L, L, dtype=torch.float32, device=device)
    min_loop = cfg.min_loop
    allow_wobble = cfg.allow_wobble

    for b in range(B):
        seq = _indices_to_sequence(indices[b], pad_index=cfg.seq_pad_index)
        P = bpp_safe[b].cpu().tolist()
        if unpaired_prob is not None:
            P_unp = unpaired_prob[b].cpu().tolist()
        else:
            P_unp = [0.5] * L

        # Precompute legality mask
        legal = [[False] * L for _ in range(L)]
        for i in range(L):
            for j in range(i + min_loop + 1, L):
                if is_allowed_pair(seq[i], seq[j], allow_wobble=allow_wobble):
                    legal[i][j] = True

        # DP table: W[i][j] = max expected accuracy for interval [i, j]
        # Negative infinity for invalid intervals
        # We use 0-indexed inclusive intervals.  W[i][j] for i > j is 0 (empty).
        W = [[0.0] * L for _ in range(L)]
        # Backpointer: -1 means j unpaired, else k means j pairs with k
        bp = [[-1] * L for _ in range(L)]

        # Length 1 intervals: single unpaired position
        for i in range(L):
            s_unp = 2.0 * float(P_unp[i]) - 1.0
            W[i][i] = s_unp
            bp[i][i] = -1

        # Fill by increasing interval length
        for length in range(2, L + 1):
            for i in range(L - length + 1):
                j = i + length - 1
                # Branch 1: j unpaired
                s_unp_j = 2.0 * float(P_unp[j]) - 1.0
                best = W[i][j - 1] + s_unp_j if j - 1 >= i else s_unp_j
                best_k = -1
                # Branch 2: j pairs with k
                for k in range(i, j - min_loop):
                    if not legal[k][j]:
                        continue
                    s_pair = gamma * 2.0 * float(P[k][j])
                    left = W[i][k - 1] if k - 1 >= i else 0.0
                    inner = W[k + 1][j - 1] if k + 1 <= j - 1 else 0.0
                    cand = left + s_pair + inner
                    if cand > best + 1e-12:
                        best = cand
                        best_k = k
                W[i][j] = best
                bp[i][j] = best_k

        # Traceback (iterative stack)
        pairs_list = []
        stack: list = [(0, L - 1)]
        while stack:
            i, j = stack.pop()
            if i > j:
                continue
            if i == j:
                continue
            k = bp[i][j]
            if k == -1:
                # j unpaired, recurse on [i, j-1]
                if i <= j - 1:
                    stack.append((i, j - 1))
            else:
                # j pairs with k
                pairs_list.append((k, j))
                if i <= k - 1:
                    stack.append((i, k - 1))
                if k + 1 <= j - 1:
                    stack.append((k + 1, j - 1))

        for i, j in pairs_list:
            out[b, i, j] = 1.0
            out[b, j, i] = 1.0

    # Apply mask
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    out = out * pair_mask.float()
    diag = torch.eye(L, dtype=torch.bool, device=device)
    out = out * (~diag).unsqueeze(0).float()
    return out


# ---------------------------------------------------------------------------
# Greedy pseudoknot-allowed decoder
# ---------------------------------------------------------------------------


def greedy_pseudoknot_decoder(
    logits: torch.Tensor,
    *,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: Optional[DecoderConfig] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy matching decoder that allows pseudoknots.

    Uses :func:`reactflow.constraints.project_greedy_matching` with
    ``allow_pseudoknot=True``.  This decoder is only for explicit
    pseudoknot-allowed experiments; the default evaluator pipeline uses
    :func:`nussinov_dp_decoder`.

    Args:
        logits: ``(B, L, L)`` pair logits.
        indices: ``(B, L)`` nucleotide vocab indices.
        mask: optional ``(B, L)`` real-position mask.
        config: decoder config.
        temperature: optional scalar tensor to divide logits before greedy.

    Returns:
        Binary pair matrix ``(B, L, L)`` as float32 tensor.

    Complexity: ``O(B * L^2 log L)`` time, ``O(B * L^2)`` memory.
    """
    cfg = config or DecoderConfig()
    B, L, _ = logits.shape
    device = logits.device

    if mask is None:
        mask = indices != cfg.seq_pad_index

    if temperature is not None:
        scores = logits / temperature.clamp(min=1e-6)
    else:
        scores = logits

    diag = torch.eye(L, dtype=torch.bool, device=device)
    scores = scores.masked_fill(diag.unsqueeze(0), float("-inf"))
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    scores = scores.masked_fill(~pair_mask, float("-inf"))

    out = torch.zeros(B, L, L, dtype=torch.float32, device=device)
    for b in range(B):
        seq = _indices_to_sequence(indices[b], pad_index=cfg.seq_pad_index)
        scores_b = scores[b].cpu().tolist()
        legal = project_greedy_matching(
            seq,
            scores_b,
            min_loop=cfg.min_loop,
            allow_wobble=cfg.allow_wobble,
            allow_pseudoknot=True,
            min_score=cfg.min_score,
        )
        out[b] = _binary_matrix_from_indices(legal, L, device=device)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def decode(
    logits: torch.Tensor,
    *,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bpp: Optional[torch.Tensor] = None,
    unpaired_prob: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
    config: Optional[DecoderConfig] = None,
    mode: str = "nussinov_dp",
) -> torch.Tensor:
    """Dispatch to a decoder by name.

    Args:
        logits: ``(B, L, L)`` pair logits.
        indices: ``(B, L)`` nucleotide vocab indices.
        mask: optional ``(B, L)`` real-position mask.
        bpp: optional ``(B, L, L)`` BPP (required for MEA / threshold-with-bpp).
        unpaired_prob: optional ``(B, L)`` (required for MEA with unpaired).
        temperature: optional scalar tensor.
        config: decoder config.
        mode: one of ``"threshold"``, ``"nussinov_dp"``, ``"mea"``,
            ``"greedy_pseudoknot"``.

    Returns:
        Binary pair matrix ``(B, L, L)`` as float32 tensor.
    """
    if mode == "threshold":
        return threshold_decoder(
            logits,
            indices=indices,
            mask=mask,
            config=config,
            use_bpp=bpp is not None,
            bpp=bpp,
        )
    if mode == "nussinov_dp":
        return nussinov_dp_decoder(
            logits,
            indices=indices,
            mask=mask,
            config=config,
            temperature=temperature,
        )
    if mode == "mea":
        if bpp is None:
            # Compute BPP from logits + temperature
            t = temperature if temperature is not None else torch.tensor(1.0, device=logits.device)
            safe = logits.clamp(min=-30.0)
            bpp = torch.sigmoid(safe / t.clamp(min=1e-6))
        return mea_decoder(
            bpp,
            indices=indices,
            unpaired_prob=unpaired_prob,
            mask=mask,
            config=config,
        )
    if mode == "greedy_pseudoknot":
        return greedy_pseudoknot_decoder(
            logits,
            indices=indices,
            mask=mask,
            config=config,
            temperature=temperature,
        )
    raise ValueError(f"unknown decoder mode: {mode!r}")
