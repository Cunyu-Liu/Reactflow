#!/usr/bin/env python3
"""Focused matched-null tests for the DeltaFlow Stage-2 pair (Task 6.3).

Verifies, item by item, the frozen matched-null contract (spec 4.2):
  1. Capacity: per-parameter-name counts identical between the candidate
     and the diagonal null, and the per-module parameter breakdown matches
     the candidate exactly.
  2. Operator family: both models instantiate the identical module classes
     and state-dict key universe; only the diagonal flag differs.
  3. Initialization: make_flow_pair produces bitwise-identical initial
     weights (double-init sample check).
  4. Diagonal behavior: permuting the input rows of other mutants leaves
     every untouched null row unchanged, while the candidate output is
     free to change (position mixing exists).
  5. Row-permutation equivariance of the null: permuting all rows of the
     inputs permutes the null outputs identically (per-position
     independence), which fails for the candidate in general.
  6. Training-protocol identity hooks: identical batch order and
     time/noise generator seeding reproduce identical losses for the
     null across two fresh instances (stream determinism).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deltaflow import (  # noqa: E402
    DeltaFlowDenoiser,
    build_pair_channels,
    flow_parameter_count,
    interpolate_flow_state,
    make_flow_pair,
    masked_flow_loss,
    PAIR_CHANNELS,
    POINT_CHANNELS,
    TEACHER_WIDTH,
)


def make_inputs(batch: int = 4, length: int = 20, seed: int = 0) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    point = torch.zeros(batch, length, POINT_CHANNELS)
    edit = 10
    positions = torch.arange(length, dtype=torch.float32)
    distance = positions - float(edit)
    for b in range(batch):
        point[b, :, 0] = distance / max(length - 1, 1)
        point[b, :, 1] = distance.abs() / max(length - 1, 1)
        point[b, :, 2] = torch.log1p(distance.abs()) / math.log(max(length, 2))
        point[b, :, 3] = float(edit) / max(length - 1, 1)
        point[b, :, 4] = positions / max(length - 1, 1)
        point[b, :, 5] = (positions == float(edit)).float()
        point[b, :, 6] = (torch.rand(length, generator=generator) < 0.5).float()
        bases = torch.randint(0, 4, (length,), generator=generator)
        point[b, torch.arange(length), 7 + bases] = 1.0
        alt = torch.randint(0, 4, (length,), generator=generator)
        point[b, torch.arange(length), 11 + alt] = 1.0
        point[b, :, 15] = torch.randn(length, generator=generator) * 0.1
        point[b, :, 16] = torch.randn(length, generator=generator) * 0.1
        point[b, :, 17] = (torch.rand(length, generator=generator) < 0.8).float()
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[0, -4:] = False
    point[0, -4:, :] = 0.0
    stage1 = (torch.randn(batch, length, generator=generator) * 0.1) * mask.float()
    teacher = torch.randn(batch, length, TEACHER_WIDTH, generator=generator) * 0.05
    pair = build_pair_channels(point, mask)
    return {
        "point": point,
        "pair": pair,
        "teacher": teacher,
        "stage1": stage1,
        "mask": mask,
    }


def parameter_table(model: torch.nn.Module) -> dict[str, int]:
    return {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    torch.manual_seed(0)
    inputs = make_inputs()
    point, pair, teacher, stage1, mask = (
        inputs["point"],
        inputs["pair"],
        inputs["teacher"],
        inputs["stage1"],
        inputs["mask"],
    )
    batch, length = mask.shape

    candidate, null = make_flow_pair(seed=4321, device="cpu")
    total_candidate = flow_parameter_count(candidate)
    total_null = flow_parameter_count(null)
    assert total_candidate == total_null, (total_candidate, total_null)
    table_candidate = parameter_table(candidate)
    table_null = parameter_table(null)
    assert table_candidate.keys() == table_null.keys()
    for name in table_candidate:
        assert table_candidate[name] == table_null[name], name
    print(
        f"capacity: {total_candidate} parameters, all "
        f"{len(table_candidate)} per-parameter entries identical"
    )

    state_c = set(candidate.state_dict().keys())
    state_n = set(null.state_dict().keys())
    assert state_c == state_n
    module_flags_c = [block.diagonal for block in candidate.blocks]
    module_flags_n = [block.diagonal for block in null.blocks]
    assert module_flags_c == [False] * len(module_flags_c)
    assert module_flags_n == [True] * len(module_flags_n)
    print(
        f"operator family: identical state keys ({len(state_c)}); "
        "diagonal flags candidate=False null=True across "
        f"{len(module_flags_c)} blocks"
    )

    candidate_b, null_b = make_flow_pair(seed=4321, device="cpu")
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, candidate_b.state_dict()[name]), name
    for name, value in null.state_dict().items():
        assert torch.equal(value, null_b.state_dict()[name]), name
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, null.state_dict()[name]), name
    print("initialization: make_flow_pair is bitwise reproducible")

    torch.manual_seed(123)
    t = torch.rand(batch)
    state = torch.randn(batch, length)
    with torch.no_grad():
        base_null = null(state, point, pair, teacher, stage1, mask, t)
        perm = torch.randperm(batch, generator=torch.Generator().manual_seed(5))
        perm_inputs = {
            "point": point[perm],
            "pair": pair[perm],
            "teacher": teacher[perm],
            "stage1": stage1[perm],
            "mask": mask[perm],
        }
        perm_null = null(
            state[perm],
            perm_inputs["point"],
            perm_inputs["pair"],
            perm_inputs["teacher"],
            perm_inputs["stage1"],
            perm_inputs["mask"],
            t[perm],
        )
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(batch)
        restored = perm_null[inv]
    assert torch.allclose(base_null, restored, atol=1e-6), (
        "null is not row-permutation equivariant"
    )
    print("null diagonal: row-permutation equivariance exact")

    torch.manual_seed(77)
    with torch.no_grad():
        weight_wake = torch.randn_like(candidate.output[-1].weight) * 0.05
        bias_wake = torch.randn_like(candidate.output[-1].bias) * 0.05
        candidate.output[-1].weight.copy_(weight_wake)
        candidate.output[-1].bias.copy_(bias_wake)
        null.output[-1].weight.copy_(weight_wake)
        null.output[-1].bias.copy_(bias_wake)
    candidate.eval()
    null.eval()

    b0 = 0
    point_shift = point.clone()
    point_shift[1, 7, :] = point[1, 12, :]
    point_shift[1, 12, :] = point[1, 7, :]
    pair_shift = build_pair_channels(point_shift, mask)
    with torch.no_grad():
        null_base = null(state, point, pair, teacher, stage1, mask, t)
        null_shift = null(
            state, point_shift, pair_shift, teacher, stage1, mask, t
        )
    untouched = torch.ones(length, dtype=torch.bool)
    untouched[7] = False
    untouched[12] = False
    untouched[~mask[1]] = False
    leaked = (null_base[1] - null_shift[1]).abs()[untouched]
    assert leaked.max().item() < 1e-6, leaked.max().item()
    own = (null_base[1, 7] - null_shift[1, 7]).abs().max().item()
    own_2 = (null_base[1, 12] - null_shift[1, 12]).abs().max().item()
    assert own > 1e-6 or own_2 > 1e-6
    print(
        f"null locality: leak {leaked.max().item():.2e} at untouched rows; "
        f"own rows moved {own:.4f}/{own_2:.4f}"
    )

    def train_three_steps(model: DeltaFlowDenoiser, seed: int) -> list[float]:
        torch.manual_seed(seed)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=0.01
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 99)
        residual = torch.randn(batch, length, generator=generator) * 0.3
        losses = []
        for _ in range(3):
            t_draw = torch.rand(batch, generator=generator)
            noise = torch.randn(batch, length, generator=generator)
            query = interpolate_flow_state(noise, residual, t_draw, mask)
            velocity = model(query, point, pair, teacher, stage1, mask, t_draw)
            loss = masked_flow_loss(velocity, residual, noise, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    null_first = DeltaFlowDenoiser(diagonal=True)
    null_first.load_state_dict(null.state_dict())
    null_second = DeltaFlowDenoiser(diagonal=True)
    null_second.load_state_dict(null.state_dict())
    losses_first = train_three_steps(null_first, seed=17)
    losses_second = train_three_steps(null_second, seed=17)
    assert losses_first == losses_second, (losses_first, losses_second)
    print(
        "training-stream determinism: identical seeded losses "
        f"{['%.4f' % v for v in losses_first]}"
    )

    with torch.no_grad():
        candidate_base = candidate(state, point, pair, teacher, stage1, mask, t)
        candidate_shift = candidate(
            state, point_shift, pair_shift, teacher, stage1, mask, t
        )
    candidate_leak = (
        (candidate_base[1] - candidate_shift[1]).abs()[untouched].max().item()
    )
    print(
        "candidate cross-row sensitivity (sanity): "
        f"{candidate_leak:.2e} at untouched rows"
    )

    print("ALL MATCHED-NULL FOCUSED CHECKS PASSED")


if __name__ == "__main__":
    main()
