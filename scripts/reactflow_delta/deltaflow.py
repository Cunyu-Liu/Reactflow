#!/usr/bin/env python3
"""DeltaFlow conditional flow-matching residual-profile generator (Stage 2).

The candidate denoiser models the joint distribution of the full residual
profile ``r = target_delta - stage1_point`` conditioned on the frozen
condition vector c (WT sequence, WT 2A3-MaP profile with error/missing
mask, SNV ref/alt/source-position context, RNet2 teacher embedding, and
the Stage-1 per-position mean ``m``).  The matched null keeps the
identical parameter universe, operator family, initialization, and
training protocol while restricting every position-mixing operator to the
diagonal, which makes it a per-position generator at unchanged capacity.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

FLOW_CHECKPOINT_SCHEMA = "reactflow_delta.deltaflow_flow_checkpoint.v1"

POINT_CHANNELS = 18
PAIR_CHANNELS = 28
TEACHER_WIDTH = 384
SINGLE_WIDTH = 64
PAIR_WIDTH = 48
ATTENTION_BLOCKS = 4
ATTENTION_HEADS = 8
FFN_WIDTH = 256
CONV_KERNEL = 3
TIME_DIMS = 16
DROPOUT = 0.1
OUT_CHANNELS = 1

POINT_SIGNED_DISTANCE = 0
POINT_ABSOLUTE_DISTANCE = 1
POINT_LOG_DISTANCE = 2
POINT_EDIT_POSITION = 3
POINT_RECEIVER_POSITION = 4
POINT_SAME_SITE = 5
POINT_DESIGN_REGION = 6
POINT_REF_ONEHOT = 7
POINT_ALT_ONEHOT = 11
POINT_WT_REACTIVITY = 15
POINT_WT_ERROR = 16
POINT_WT_OBSERVED = 17

PAIR_CHANNEL_NAMES: tuple[str, ...] = (
    "base_AA", "base_AC", "base_AG", "base_AU",
    "base_CA", "base_CC", "base_CG", "base_CU",
    "base_GA", "base_GC", "base_GG", "base_GU",
    "base_UA", "base_UC", "base_UG", "base_UU",
    "pair_signed_distance",
    "pair_absolute_distance",
    "pair_log_distance",
    "pair_same_site",
    "same_side_of_edit",
    "wt_reactivity_product",
    "wt_error_product",
    "wt_observed_and",
    "wt_observed_row",
    "wt_observed_column",
    "mutation_distance_product",
    "mutation_min_distance",
)
if len(PAIR_CHANNEL_NAMES) != PAIR_CHANNELS:
    raise RuntimeError("pair channel name list does not match PAIR_CHANNELS")


class _DiagCache:
    """Cache diagonal and center-tap masks keyed by (length, device)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str, bool], torch.Tensor] = {}

    def eye(self, length: int, device: torch.device) -> torch.Tensor:
        key = (int(length), str(device), True)
        cached = self._cache.get(key)
        if cached is None:
            cached = torch.eye(int(length), dtype=torch.bool, device=device)
            self._cache[key] = cached
        return cached

    def center_kernel(self, device: torch.device) -> torch.Tensor:
        key = (CONV_KERNEL, str(device), False)
        cached = self._cache.get(key)
        if cached is None:
            cached = torch.zeros(
                CONV_KERNEL, CONV_KERNEL, dtype=torch.bool, device=device
            )
            cached[CONV_KERNEL // 2, CONV_KERNEL // 2] = True
            self._cache[key] = cached
        return cached


_DIAG = _DiagCache()


def time_features(t: torch.Tensor, dims: int = TIME_DIMS) -> torch.Tensor:
    """Sinusoidal time embedding for scalar t in [0, 1]."""
    if t.ndim != 1:
        raise ValueError("flow time must be a per-row vector")
    exponent = torch.arange(dims // 2, dtype=torch.float32, device=t.device)
    angles = t[:, None] * math.pi * (2.0 ** (exponent / max(dims // 2 - 1, 1)))[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class _PairNorm(nn.Module):
    """LayerNorm over the channel axis of a [B, C, L, L] pair state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.norm(pair.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DeltaFlowBlock(nn.Module):
    """One denoiser block: single attention/FFN plus pair triangle/conv/FFN."""

    def __init__(self, *, diagonal: bool) -> None:
        super().__init__()
        self.diagonal = bool(diagonal)
        self.attention_norm = nn.LayerNorm(SINGLE_WIDTH)
        self.qkv = nn.Linear(SINGLE_WIDTH, 3 * SINGLE_WIDTH)
        self.attention_output = nn.Linear(SINGLE_WIDTH, SINGLE_WIDTH)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.residual_dropout = nn.Dropout(DROPOUT)
        self.ffn_norm = nn.LayerNorm(SINGLE_WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(SINGLE_WIDTH, FFN_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_WIDTH, SINGLE_WIDTH),
        )
        self.triangle_norm = nn.LayerNorm(SINGLE_WIDTH)
        self.triangle_left = nn.Linear(SINGLE_WIDTH, PAIR_WIDTH)
        self.triangle_right = nn.Linear(SINGLE_WIDTH, PAIR_WIDTH)
        self.pair_norm = _PairNorm(PAIR_WIDTH)
        self.pair_conv = nn.Conv2d(
            PAIR_WIDTH, PAIR_WIDTH, CONV_KERNEL, padding=CONV_KERNEL // 2
        )
        self.pair_ffn_norm = _PairNorm(PAIR_WIDTH)
        self.pair_ffn = nn.Sequential(
            nn.Conv2d(PAIR_WIDTH, FFN_WIDTH, 1),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(FFN_WIDTH, PAIR_WIDTH, 1),
        )
        self.readout_norm = _PairNorm(PAIR_WIDTH)
        self.readout = nn.Linear(PAIR_WIDTH, SINGLE_WIDTH)

    def _pair_conv(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self.diagonal:
            return self.pair_conv(normalized)
        weight = self.pair_conv.weight * _DIAG.center_kernel(
            normalized.device
        ).float()[None, None]
        return F.conv2d(
            normalized,
            weight,
            bias=self.pair_conv.bias,
            stride=self.pair_conv.stride,
            padding=self.pair_conv.padding,
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _width = single.shape
        normalized = self.attention_norm(single)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, ATTENTION_HEADS, SINGLE_WIDTH // ATTENTION_HEADS
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            SINGLE_WIDTH // ATTENTION_HEADS
        )
        invalid_key = (~mask).unsqueeze(1).unsqueeze(2)
        eye = _DIAG.eye(length, single.device)[None, None]
        logits = logits.masked_fill(invalid_key & ~eye, float("-inf"))
        if self.diagonal:
            logits = logits.masked_fill(~eye, float("-inf"))
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, length, SINGLE_WIDTH
        )
        single = single + self.residual_dropout(self.attention_output(context))
        single = single + self.residual_dropout(self.ffn(self.ffn_norm(single)))

        triangle = self.triangle_norm(single)
        outer = torch.einsum(
            "bic,bjc->bcij", self.triangle_left(triangle), self.triangle_right(triangle)
        )
        if self.diagonal:
            outer = outer * _DIAG.eye(length, single.device)[None, None]
        pair = pair + outer
        pair = pair + self._pair_conv(self.pair_norm(pair))
        pair = pair + self.pair_ffn(self.pair_ffn_norm(pair))

        readout = self.readout_norm(pair)
        counts = mask.float().sum(-1).clamp(min=1.0)
        if self.diagonal:
            eye = _DIAG.eye(length, single.device)
            diagonal_state = readout * eye[None, None]
            pooled = diagonal_state.sum(dim=-1) / counts[:, None, None]
        else:
            valid = mask.float()[:, None, :, None]
            pooled = (readout * valid).sum(dim=-1) / counts[:, None, None]
        single = single + self.residual_dropout(self.readout(pooled.transpose(1, 2)))
        return single, pair


class DeltaFlowDenoiser(nn.Module):
    """Velocity network v(r_t, t, c) over the residual profile.

    The flow state r_t enters through a dedicated per-position projection so
    that the learned velocity depends on the current state.  Without state
    dependence the ODE trajectories would be independent of the initial
    noise, and the sampler could not represent the conditional distribution
    p(r | c).

    The condition vector c follows the frozen spec (§4.1): WT sequence and
    2A3-MaP profile (error/missing mask), SNV ref/alt/source-position
    context (point_in), WT pair context (pair_in, derived from the same
    frozen 18 channels), RNet2 teacher embedding, and the Stage-1
    per-position mean ``stage1`` (m_i), each with a dedicated projection.
    """

    def __init__(self, *, diagonal: bool = False) -> None:
        super().__init__()
        self.diagonal = bool(diagonal)
        self.state_in = nn.Linear(OUT_CHANNELS, SINGLE_WIDTH)
        self.point_in = nn.Linear(POINT_CHANNELS, SINGLE_WIDTH)
        self.stage1_in = nn.Linear(OUT_CHANNELS, SINGLE_WIDTH)
        self.teacher_in = nn.Linear(TEACHER_WIDTH, SINGLE_WIDTH)
        self.time_in = nn.Sequential(
            nn.Linear(TIME_DIMS, FFN_WIDTH),
            nn.GELU(),
            nn.Linear(FFN_WIDTH, SINGLE_WIDTH),
        )
        self.pair_in = nn.Conv2d(PAIR_CHANNELS, PAIR_WIDTH, 1)
        self.blocks = nn.ModuleList(
            DeltaFlowBlock(diagonal=self.diagonal)
            for _ in range(ATTENTION_BLOCKS)
        )
        self.output_norm = nn.LayerNorm(SINGLE_WIDTH)
        self.output = nn.Sequential(
            nn.Linear(SINGLE_WIDTH, FFN_WIDTH),
            nn.GELU(),
            nn.Linear(FFN_WIDTH, OUT_CHANNELS),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        point_in: torch.Tensor,
        pair_in: torch.Tensor,
        teacher: torch.Tensor,
        stage1: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if point_in.ndim != 3 or point_in.shape[-1] != POINT_CHANNELS:
            raise ValueError("flow point inputs must have shape [B,L,18]")
        batch, length, _ = point_in.shape
        if state.ndim != 2 or state.shape != (batch, length):
            raise ValueError("flow state must have shape [B,L] matching point rows")
        if pair_in.ndim != 4 or pair_in.shape != (batch, PAIR_CHANNELS, length, length):
            raise ValueError("flow pair inputs must have shape [B,28,L,L]")
        if teacher.shape != (batch, length, TEACHER_WIDTH):
            raise ValueError("flow teacher inputs must have shape [B,L,384]")
        if stage1.ndim != 2 or stage1.shape != (batch, length):
            raise ValueError("stage1 conditioning must have shape [B,L]")
        if mask.shape != (batch, length) or mask.dtype != torch.bool:
            raise ValueError("flow mask must have shape [B,L] and bool dtype")
        if t.shape != (batch,):
            raise ValueError("flow time must have shape [B]")
        single = self.point_in(point_in) + self.teacher_in(teacher)
        single = single + self.state_in(state.unsqueeze(-1))
        single = single + self.stage1_in(stage1.unsqueeze(-1))
        single = single + self.time_in(time_features(t))[:, None, :]
        pair = self.pair_in(pair_in)
        for block in self.blocks:
            single, pair = block(single, pair, mask)
        velocity = self.output(self.output_norm(single)).squeeze(-1)
        return velocity * mask.float()


def build_pair_channels(
    point_in: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Construct the frozen 28-channel pair input from the 18 point channels.

    Channel layout (frozen, see PAIR_CHANNEL_NAMES):
      0-15   one-hot base pair (row base x column base) from the ACGU ref
             one-hot rows of the point input, order (row base, column base)
             with A<C<G<U, i.e. AA, AC, AG, AU, CA, ..., UU.
      16     pair_signed_distance: (pos_i - pos_j) / (L - 1)
      17     pair_absolute_distance: |pos_i - pos_j| / (L - 1)
      18     pair_log_distance: log(1 + |pos_i - pos_j|) / log(L)
      19     pair_same_site: (pos_i == pos_j) as float
      20     same_side_of_edit: 1.0 if sign(pos_i - edit) == sign(pos_j -
             edit) else -1.0, where positions exactly at the edit form
             their own side (sign 0 -> 0.0).
      21     wt_reactivity_product: point wt_reactivity_i * wt_reactivity_j
      22     wt_error_product: point wt_error_i * wt_error_j
      23     wt_observed_and: wt_observed_i AND wt_observed_j
      24     wt_observed_row: wt_observed_i broadcast over columns
      25     wt_observed_column: wt_observed_j broadcast over rows
      26     mutation_distance_product: signed_distance_i * signed_distance_j
             (the per-position receiver-edit distances)
      27     mutation_min_distance: min(|d_i|, |d_j|) over the per-position
             receiver-edit distances

    ``point_in`` follows the frozen 18-channel BASELINE_FEATURE_NAMES layout
    (model_rescue_v5_probe.baseline_features): [0] signed distance to edit,
    [1] absolute distance to edit, [2] log distance, [3] edit position,
    [4] receiver position, [5] same site, [6] design region, [7..10] ref
    one-hot, [11..14] alt one-hot, [15] wt reactivity, [16] wt error
    precision, [17] wt observed.  Channels are zeroed where either the row
    or the column is outside the validity mask.
    """
    if point_in.ndim != 3 or point_in.shape[-1] != POINT_CHANNELS:
        raise ValueError("pair construction needs point input [B,L,18]")
    batch, length, _ = point_in.shape
    if mask.shape != (batch, length) or mask.dtype != torch.bool:
        raise ValueError("pair construction needs mask [B,L] bool")

    onehot = point_in[..., POINT_REF_ONEHOT : POINT_REF_ONEHOT + 4]
    base_pair = torch.einsum("bia,bjc->bijac", onehot, onehot).reshape(
        batch, length, length, 16
    )

    def rowwise(channel: torch.Tensor) -> torch.Tensor:
        return channel[:, :, None].expand(-1, -1, length)

    def columnwise(channel: torch.Tensor) -> torch.Tensor:
        return channel[:, None, :].expand(-1, length, -1)

    receiver_position = point_in[..., POINT_RECEIVER_POSITION]
    edit_position = point_in[..., POINT_EDIT_POSITION]
    signed_distance = point_in[..., POINT_SIGNED_DISTANCE]
    absolute_distance = point_in[..., POINT_ABSOLUTE_DISTANCE]
    wt_reactivity = point_in[..., POINT_WT_REACTIVITY]
    wt_error = point_in[..., POINT_WT_ERROR]
    wt_observed = point_in[..., POINT_WT_OBSERVED]

    position_row = receiver_position[:, :, None] * (length - 1)
    position_column = receiver_position[:, None, :] * (length - 1)
    pair_distance = position_row - position_column
    pair_abs_distance = pair_distance.abs()
    log_norm = math.log(max(length, 2))
    edit_index = edit_position[:, :1, None] * (length - 1)
    sign_row = torch.sign(position_row - edit_index)
    sign_column = torch.sign(position_column - edit_index)
    same_side = torch.where(
        (sign_row == 0) | (sign_column == 0),
        torch.zeros_like(sign_row),
        torch.where(
            sign_row == sign_column,
            torch.ones_like(sign_row),
            -torch.ones_like(sign_row),
        ),
    )
    channels = torch.stack(
        [
            base_pair[..., 0], base_pair[..., 1], base_pair[..., 2],
            base_pair[..., 3], base_pair[..., 4], base_pair[..., 5],
            base_pair[..., 6], base_pair[..., 7], base_pair[..., 8],
            base_pair[..., 9], base_pair[..., 10], base_pair[..., 11],
            base_pair[..., 12], base_pair[..., 13], base_pair[..., 14],
            base_pair[..., 15],
            pair_distance / max(length - 1, 1),
            pair_abs_distance / max(length - 1, 1),
            torch.log1p(pair_abs_distance) / log_norm,
            (pair_distance == 0).to(torch.float32),
            same_side,
            rowwise(wt_reactivity) * columnwise(wt_reactivity),
            rowwise(wt_error) * columnwise(wt_error),
            rowwise(wt_observed) * columnwise(wt_observed),
            rowwise(wt_observed),
            columnwise(wt_observed),
            rowwise(signed_distance) * columnwise(signed_distance),
            torch.minimum(rowwise(absolute_distance), columnwise(absolute_distance)),
        ],
        dim=1,
    )
    channels = channels * mask[:, None, :, None] * mask[:, None, None, :]
    return channels.to(torch.float32)


def make_flow_pair(
    *, seed: int, device: str | torch.device
) -> tuple[DeltaFlowDenoiser, DeltaFlowDenoiser]:
    """Create the candidate/diagonal-null pair from one initialization."""

    torch.manual_seed(int(seed))
    candidate = DeltaFlowDenoiser(diagonal=False).to(device)
    null = copy.deepcopy(candidate)
    assert_flow_pair_initial_match(candidate, null)
    return candidate, null


def assert_flow_pair_initial_match(
    candidate: DeltaFlowDenoiser, null: DeltaFlowDenoiser
) -> None:
    candidate_state = candidate.state_dict()
    null_state = null.state_dict()
    if candidate_state.keys() != null_state.keys():
        raise RuntimeError("candidate/null flow state names differ")
    for name in candidate_state:
        if not torch.equal(candidate_state[name], null_state[name]):
            raise RuntimeError(f"candidate/null flow initialization differs at {name}")
    if flow_parameter_count(candidate) != flow_parameter_count(null):
        raise RuntimeError("candidate/null flow parameter counts differ")


def flow_parameter_count(model: DeltaFlowDenoiser) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def interpolate_flow_state(
    noise: torch.Tensor, residual: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Rectified-flow interpolation r_t = (1 - t) * noise + t * residual.

    Callers must query the velocity network at this state during training;
    ``masked_flow_loss`` only defines the regression target, it does not
    build the query point.
    """
    if t.ndim != 1 or t.shape[0] != noise.shape[0]:
        raise ValueError("interpolation time must be a per-row vector")
    fraction = t[:, None]
    state = (1.0 - fraction) * noise + fraction * residual
    return state * mask.float()


def masked_flow_loss(
    velocity: torch.Tensor,
    residual: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mutant-balanced rectified-flow regression loss.

    Positions are averaged within one mutant, then mutants are averaged
    equally, mirroring the method-cell-balanced discipline of Stage 1.
    """
    target_velocity = residual - noise
    squared = ((velocity - target_velocity) ** 2).masked_fill(~mask, 0.0)
    counts = mask.float().sum(-1)
    valid = counts > 0
    if not bool(valid.any()):
        return squared.sum() * 0.0
    per_mutant = squared.sum(-1) / counts.clamp(min=1.0)
    return per_mutant[valid].mean()


def euler_flow_sample(
    model: DeltaFlowDenoiser,
    *,
    point_in: torch.Tensor,
    pair_in: torch.Tensor,
    teacher: torch.Tensor,
    stage1: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Deterministic Euler integration of the rectified-flow ODE.

    The state is initialized from noise and each step queries the velocity
    network at the current state, so trajectories genuinely depend on the
    initial draw.
    """

    if steps < 1:
        raise ValueError("flow sampling requires at least one Euler step")
    model.eval()
    length = point_in.shape[1]
    state = torch.randn(
        point_in.shape[0], length, device=point_in.device, generator=generator
    )
    state = state * mask.float()
    dt = 1.0 / int(steps)
    with torch.no_grad():
        for step in range(int(steps)):
            t = torch.full((point_in.shape[0],), step * dt, device=point_in.device)
            velocity = model(state, point_in, pair_in, teacher, stage1, mask, t)
            state = (state + dt * velocity) * mask.float()
    return state


def gaussian_expected_absolute(
    mean: "torch.Tensor | float", scale: "torch.Tensor | float"
) -> "torch.Tensor | float":
    """Closed-form E|X| for X ~ N(mean, scale)."""

    z = mean / scale
    return scale * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * z * z) + mean * (
        1.0 - 2.0 * torch.erf(z / math.sqrt(2.0))
    )


def samples_to_mixture(
    samples: torch.Tensor, mask: torch.Tensor, *, scale_floor: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse joint samples into per-position single-Gaussian marginals.

    ``samples`` has shape [S, B, L]; each of the B*L masked entries gets its
    own location/scale statistics computed across the S sampling draws only.
    Returns (weights [N,1], locations [N,1], scales [N,1], expected absolute
    delta [N]) where N is the number of masked (batch, position) rows, in
    row-major (batch-first, position-last) order.
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape [S,B,L]")
    if mask.shape != samples.shape[1:] or mask.dtype != torch.bool:
        raise ValueError("mask must have shape [B,L] and bool dtype")
    if samples.shape[0] < 2:
        raise ValueError("mixture statistics need at least two draws")
    batch, length = mask.shape
    flat_mask = mask.reshape(-1)
    flat_samples = samples.permute(1, 2, 0).reshape(-1, samples.shape[0])
    selected = flat_samples[flat_mask]
    n = int(selected.shape[0])
    if n == 0:
        empty = torch.zeros(0, 1, device=samples.device, dtype=samples.dtype)
        return (
            empty,
            empty,
            empty,
            torch.zeros(0, device=samples.device, dtype=samples.dtype),
        )
    location = selected.mean(dim=1, keepdim=True)
    variance = selected.var(dim=1, unbiased=False, keepdim=True)
    scale = variance.clamp(min=scale_floor * scale_floor).sqrt()
    weights = torch.ones_like(location)
    expected = gaussian_expected_absolute(location, scale).squeeze(-1)
    return weights, location, scale, expected
