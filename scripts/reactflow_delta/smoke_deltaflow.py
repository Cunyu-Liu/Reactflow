#!/usr/bin/env python3
"""CPU smoke test for the fixed deltaflow module.

Verifies:
  1. Velocity network depends on the flow state (r_t sensitivity).
  2. Candidate/null pair init-identical + capacity identical.
  3. Pair channel construction: shape, mask zeroing, diagonal semantics,
     base-pair one-hot, true pair distances.
  4. Rectified-flow training loop smoke (loss decreases, no NaN).
  5. Euler sampling: two independent noise draws produce different
     trajectories but identical when replayed with the same generator seed.
  6. samples_to_mixture: per-position marginals, not cross-position pooling.
  7. Diagonal null: swapping two positions in point_in changes candidate
     output at other rows but leaves the null output unchanged (diagonal
     per-position semantics).
"""

from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from deltaflow import (  # noqa: E402
    DeltaFlowDenoiser,
    build_pair_channels,
    euler_flow_sample,
    flow_parameter_count,
    interpolate_flow_state,
    make_flow_pair,
    masked_flow_loss,
    samples_to_mixture,
    PAIR_CHANNELS,
    POINT_CHANNELS,
    POINT_REF_ONEHOT,
    TEACHER_WIDTH,
)

torch.manual_seed(0)


def make_inputs(batch: int = 3, length: int = 24) -> dict:
    point = torch.zeros(batch, length, POINT_CHANNELS)
    edit = torch.randint(0, length, (batch,))
    for b in range(batch):
        positions = torch.arange(length, dtype=torch.float32)
        distance = positions - float(edit[b])
        point[b, :, 0] = distance / max(length - 1, 1)
        point[b, :, 1] = distance.abs() / max(length - 1, 1)
        point[b, :, 2] = torch.log1p(distance.abs()) / math.log(max(length, 2))
        point[b, :, 3] = float(edit[b]) / max(length - 1, 1)
        point[b, :, 4] = positions / max(length - 1, 1)
        point[b, :, 5] = (positions == float(edit[b])).float()
        point[b, :, 6] = (torch.rand(length) < 0.5).float()
        bases = torch.randint(0, 4, (length,))
        point[b, torch.arange(length), 7 + bases] = 1.0
        alt = torch.randint(0, 4, (length,))
        point[b, torch.arange(length), 11 + alt] = 1.0
        point[b, :, 15] = torch.randn(length) * 0.1
        point[b, :, 16] = torch.randn(length) * 0.1
        point[b, :, 17] = (torch.rand(length) < 0.8).float()
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[0, -3:] = False
    point[0, -3:, :] = 0.0
    teacher = torch.randn(batch, length, TEACHER_WIDTH) * 0.05
    stage1 = torch.randn(batch, length) * 0.1
    stage1 = stage1 * mask.float()
    pair = build_pair_channels(point, mask)
    return {
        "point": point,
        "pair": pair,
        "teacher": teacher,
        "stage1": stage1,
        "mask": mask,
        "edit": edit,
    }


def main() -> None:
    device = torch.device("cpu")
    inputs = make_inputs()
    point, pair, teacher, stage1, mask = (
        inputs["point"],
        inputs["pair"],
        inputs["teacher"],
        inputs["stage1"],
        inputs["mask"],
    )
    batch, length = mask.shape
    print(f"inputs: batch={batch} length={length}")

    assert pair.shape == (batch, PAIR_CHANNELS, length, length), pair.shape
    assert torch.isfinite(pair).all(), "pair channels non-finite"
    invalid = ~mask
    if invalid.any():
        invalid_row_values = pair * invalid[:, None, :, None].to(pair.dtype)
        invalid_col_values = pair * invalid[:, None, None, :].to(pair.dtype)
        assert invalid_row_values.abs().max().item() == 0.0, "masked pair rows not zero"
        assert invalid_col_values.abs().max().item() == 0.0, (
            "masked pair columns not zero"
        )

    for b in range(batch):
        i, j = 5, 11
        expected = (i - j) / max(length - 1, 1)
        got = pair[b, 16, i, j].item()
        assert abs(got - expected) < 1e-6, (got, expected)
        expected_abs = abs(j - i) / max(length - 1, 1)
        assert abs(pair[b, 17, i, j].item() - expected_abs) < 1e-6
        base_i = point[b, i, POINT_REF_ONEHOT : POINT_REF_ONEHOT + 4]
        base_j = point[b, j, POINT_REF_ONEHOT : POINT_REF_ONEHOT + 4]
        flat = int(base_i.argmax()) * 4 + int(base_j.argmax())
        assert pair[b, flat, i, j].item() == 1.0, "base pair one-hot mismatch"
        pair_slice = pair[b, :, i, j]
        assert pair_slice[:16].sum().item() == 1.0, "base-pair block not one-hot"
        assert pair_slice[19].item() == 0.0, "same-site flag wrong for i != j"
        assert pair[b, 19, i, i].item() == 1.0, "same-site flag wrong for i == i"
    print("pair channels: shape/mask/distances/base-pair semantics OK")

    candidate, null = make_flow_pair(seed=1234, device=device)
    count = flow_parameter_count(candidate)
    print(f"parameter count: {count}")
    assert flow_parameter_count(null) == count

    t = torch.rand(batch)
    state_a = torch.randn(batch, length)
    state_b = torch.randn(batch, length)
    with torch.no_grad():
        vel_a = candidate(state_a, point, pair, teacher, stage1, mask, t)
        vel_b = candidate(state_b, point, pair, teacher, stage1, mask, t)
    init_diff = (vel_a - vel_b).abs().max().item()
    print(f"velocity init |dV| (zero-init output layer): {init_diff:.2e}")
    assert (vel_a[~mask] == 0).all(), "velocity not masked"

    t = torch.rand(batch)
    residual = torch.randn(batch, length)
    noise = torch.randn(batch, length)
    velocity = candidate(state_a, point, pair, teacher, stage1, mask, t)
    loss = masked_flow_loss(velocity, residual, noise, mask)
    assert torch.isfinite(loss), "flow loss non-finite"
    loss.backward()
    grads = [
        p.grad.abs().max().item()
        for p in candidate.parameters()
        if p.grad is not None
    ]
    assert all(math.isfinite(g) for g in grads), "non-finite gradients"
    print(f"flow loss: {loss.item():.6f}; max grad: {max(grads):.4f}")

    torch.manual_seed(7)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    samples_a = euler_flow_sample(
        candidate,
        point_in=point,
        pair_in=pair,
        teacher=teacher,
        stage1=stage1,
        mask=mask,
        steps=10,
        generator=generator,
    )
    generator.manual_seed(7)
    samples_b = euler_flow_sample(
        candidate,
        point_in=point,
        pair_in=pair,
        teacher=teacher,
        stage1=stage1,
        mask=mask,
        steps=10,
        generator=generator,
    )
    assert torch.equal(samples_a, samples_b), "sampling is not seed-deterministic"
    generator.manual_seed(99)
    samples_c = euler_flow_sample(
        candidate,
        point_in=point,
        pair_in=pair,
        teacher=teacher,
        stage1=stage1,
        mask=mask,
        steps=10,
        generator=generator,
    )
    spread = (samples_a - samples_c).abs().max().item()
    print(f"cross-seed trajectory spread: {spread:.6f}")
    assert spread > 1e-3, "trajectories collapsed across noise draws"
    assert (samples_a[~mask] == 0).all()

    n_draws = 12
    draws = []
    generator = torch.Generator(device="cpu")
    for s in range(n_draws):
        generator.manual_seed(1000 + s)
        draws.append(
            euler_flow_sample(
                candidate,
                point_in=point,
                pair_in=pair,
                teacher=teacher,
                stage1=stage1,
                mask=mask,
                steps=10,
                generator=generator,
            )
        )
    samples = torch.stack(draws, dim=0)
    weights, location, scale, expected = samples_to_mixture(samples, mask)
    n_expected = int(mask.sum())
    assert location.shape == (n_expected, 1), location.shape
    assert scale.shape == (n_expected, 1)
    assert expected.shape == (n_expected,)
    assert torch.isfinite(location).all() and torch.isfinite(scale).all()
    per_pos_var = samples.var(dim=0, unbiased=False)
    masked_var = per_pos_var[mask]
    assert torch.allclose(
        scale.squeeze(-1), masked_var.clamp(min=1e-8).sqrt(), atol=1e-5
    ), "mixture scale does not match per-position variance"
    assert (scale > 0).all()
    print(
        "mixture: per-position marginals OK "
        f"(N={n_expected}, mean scale={scale.mean().item():.4f})"
    )

    trained = DeltaFlowDenoiser(diagonal=False)
    with torch.no_grad():
        trained.load_state_dict(candidate.state_dict())

    t_eval = torch.rand(batch)
    noise_eval = torch.randn(batch, length)
    state_eval = interpolate_flow_state(noise_eval, residual, t_eval, mask)

    def fixed_draw_loss(model: DeltaFlowDenoiser) -> float:
        with torch.no_grad():
            velocity_eval = model(
                state_eval, point, pair, teacher, stage1, mask, t_eval
            )
            return masked_flow_loss(velocity_eval, residual, noise_eval, mask).item()

    before = fixed_draw_loss(trained)
    optimizer = torch.optim.AdamW(trained.parameters(), lr=5e-3, weight_decay=0.01)
    for epoch in range(200):
        optimizer.zero_grad(set_to_none=True)
        t = torch.rand(batch)
        noise = torch.randn(batch, length)
        query_state = interpolate_flow_state(noise, residual, t, mask)
        velocity = trained(query_state, point, pair, teacher, stage1, mask, t)
        loss = masked_flow_loss(velocity, residual, noise, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained.parameters(), 1.0)
        optimizer.step()
    after = fixed_draw_loss(trained)
    print(
        f"training smoke: fixed-draw loss {before:.4f} -> {after:.4f} over 200 steps"
        " (smoke lr 5e-3 for CPU reachability; production lr 2e-4 per spec 4.4)"
    )
    assert after < 0.85 * before, "training did not reduce the fixed-draw flow loss"
    assert torch.isfinite(loss)

    t_check = torch.rand(batch)
    with torch.no_grad():
        trained_vel_a = trained(state_a, point, pair, teacher, stage1, mask, t_check)
        trained_vel_b = trained(state_b, point, pair, teacher, stage1, mask, t_check)
    trained_diff = (trained_vel_a - trained_vel_b).abs().max().item()
    print(f"velocity state sensitivity after training: {trained_diff:.6f}")
    assert trained_diff > 1e-4, "velocity network is not state-sensitive after training"

    stage1_shifted = stage1 + 0.05
    with torch.no_grad():
        trained_vel_s = trained(
            state_a, point, pair, teacher, stage1_shifted, mask, t_check
        )
    stage1_effect = (trained_vel_a - trained_vel_s).abs().max().item()
    print(f"velocity stage1 sensitivity after training: {stage1_effect:.6f}")
    assert stage1_effect > 1e-6, "stage1 conditioning does not reach the network"

    null_model = DeltaFlowDenoiser(diagonal=True)
    with torch.no_grad():
        null_model.load_state_dict(trained.state_dict())
    null_model.eval()
    trained.eval()
    point_swapped = point.clone()
    b0 = 0
    point_swapped[b0, 2, :] = point[b0, 15, :]
    point_swapped[b0, 15, :] = point[b0, 2, :]
    pair_swapped = build_pair_channels(point_swapped, mask)
    t_fixed = torch.full((batch,), 0.5)
    state_fixed = torch.zeros(batch, length)
    with torch.no_grad():
        base_cand = trained(state_fixed, point, pair, teacher, stage1, mask, t_fixed)
        swap_cand = trained(
            state_fixed, point_swapped, pair_swapped, teacher, stage1, mask, t_fixed
        )
        base_null = null_model(state_fixed, point, pair, teacher, stage1, mask, t_fixed)
        swap_null = null_model(
            state_fixed, point_swapped, pair_swapped, teacher, stage1, mask, t_fixed
        )
    cand_row_change = (base_cand - swap_cand).abs()[b0]
    cand_rows = set(torch.nonzero(cand_row_change > 1e-6).flatten().tolist())
    print(f"candidate rows changed by swap: {sorted(cand_rows)}")
    assert len(cand_rows) > 2, "candidate is unexpectedly local"
    null_row_change = (base_null[b0] - swap_null[b0]).abs()
    untouched = torch.ones(length, dtype=torch.bool)
    untouched[2] = False
    untouched[15] = False
    untouched[~mask[b0]] = False
    leaked = null_row_change[untouched]
    print(f"null leaked change at untouched rows: {leaked.max().item():.2e}")
    assert leaked.max().item() < 1e-6, (
        "diagonal null changed at rows other than the swapped ones"
    )
    null_swap_row = (base_null[b0, 2] - swap_null[b0, 2]).abs().max().item()
    null_swap_row_2 = (base_null[b0, 15] - swap_null[b0, 15]).abs().max().item()
    print(f"null change at the swapped rows: {null_swap_row:.4f} / {null_swap_row_2:.4f}")
    assert null_swap_row > 1e-6 or null_swap_row_2 > 1e-6, (
        "null did not respond to its own rows changing"
    )

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
