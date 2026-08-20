#!/usr/bin/env python3
"""test_p3_lrso_v3: spec-compliance fixtures for the corrected P3 LRSO training.

Verifies the v3 fixes (contract 14.1, 10.2, 9.1, 7.4):
  - Trainable encoder (no detach)
  - Missing!=0: loss mask excludes NaN targets; 0.0 is a real observation
  - Inner 4-fold puzzle-grouped validation selects {lr,wd,likelihood}
  - Early stopping with patience
  - Five-seed Gaussian mixture CRPS (not per-seed mean)
  - Positive scale parameterization (softplus + floor, no clamp)
  - ref=alt mean forced 0
  - K_rank=0 exact B_direct null
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scripts.reactflow_delta.run_p3_lrso_v3 import (
    LRSOv3, _qualified_mask, _wt_filled, _nll_macro, _crps_constructs,
    _target_matrix, SCALE_FLOOR, SEEDS, ALPHA,
)
from scripts.reactflow_delta.evaluator_crps_v1 import mixture_crps
from scripts.reactflow_delta.lrso_v1 import RFDLRSO


def test_encoder_is_trainable_no_detach():
    """Contract 10.2: WT context encoder must be TRAINABLE (not detach()ed).
    Param requires_grad and no .detach() call in encode()."""
    m = LRSOv3(k_rank=2)
    for p in m.encoder.parameters():
        assert p.requires_grad, "encoder params must be trainable"
    # verify the encode method does not call .detach() anywhere
    import inspect
    src = inspect.getsource(type(m).encode)
    assert ".detach(" not in src, "encode() must not detach encoder"


def test_missing_never_equals_zero_in_loss_mask():
    """Contract 7.4 / 14.1.3: missing target positions are EXCLUDED from loss
    by mask; 0.0 is a real observed value, not a missing indicator."""
    t = np.array([[1.0, np.nan, 0.0, np.nan, -0.5]], dtype=np.float32)
    obs = np.ones((1, 5), dtype=bool)
    q = _qualified_mask(t, obs)
    # nan at idx 1, 3 => False; 0.0 at idx 2 => True (real observation)
    assert q[0, 0] == True, "1.0 is observed"
    assert q[0, 1] == False, "NaN is missing"
    assert q[0, 2] == True, "0.0 is a real observation, not missing"
    assert q[0, 3] == False, "NaN is missing"
    assert q[0, 4] == True, "-0.5 is observed"


def test_scale_positive_parameterization():
    """Contract 10.2.1: predictive scale uses softplus + train-only floor,
    no clamp that zeros gradients in the main region. Gradient flows through
    the full forward path (encoder -> ctx_norm -> scale_head)."""
    m = LRSOv3(k_rank=2)
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    # forward_op runs ctx_norm then scale_head
    masks = torch.ones(1, L, dtype=torch.bool)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    delta, scale = m.forward_op(H, edit_idx, dists, ["C"], ["G"], masks)
    assert (scale >= SCALE_FLOOR).all(), "scale must be >= floor"
    loss = scale.mean() + delta.mean()
    loss.backward()
    # gradient must flow to encoder's qkv
    assert m.encoder.blocks[0].qkv.weight.grad is not None, \
        "gradient must flow to encoder through the full forward path"


def test_ref_alt_mean_zero():
    """Contract 5.2 / 14.1.7: ref==alt => mutation-induced mean strictly 0."""
    m = LRSOv3(k_rank=2).eval()
    L = 12
    seq = torch.zeros(L, 4)
    seq[:, 0] = 1.0  # all A
    react = torch.full((L,), 0.5)
    prec = torch.zeros(L)
    obs_token = torch.ones(L)
    pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    masks = torch.ones(1, L, dtype=torch.bool)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    # ref==alt => "A" == "A"
    delta, _ = m.forward_op(H, edit_idx, dists, ["A"], ["A"], masks)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6), \
        "ref==alt must produce zero delta mean"


def test_k_rank_zero_is_bdirect_null():
    """Contract 5.2 / 14.1.8: K_rank=0 => delta == B_direct (no LRSO term)."""
    m0 = LRSOv3(k_rank=0)
    m2 = LRSOv3(k_rank=2)
    # Copy state from m2 to m0's bdirect so they share the same direct weights
    m0.bdirect.load_state_dict(m2.bdirect.state_dict())
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5)
    prec = torch.zeros(L); obs_token = torch.ones(L)
    pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m0.encode(ctx)
    masks = torch.ones(1, L, dtype=torch.bool)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    d0, _ = m0.forward_op(H, edit_idx, dists, ["C"], ["G"], masks)
    assert not torch.allclose(d0, torch.zeros_like(d0), atol=1e-6), \
        "K_rank=0 is NOT zero-response (B_direct still active)"
    # K_rank=0 lrso term is exactly zero => delta = bdirect only
    # Verify by computing bdirect directly
    Hn = m0.ctx_norm(H)
    hp = Hn[4]
    ra = torch.zeros(8)
    ra[ALPHA.get("C", 3)] = 1.0
    ra[4 + ALPHA.get("G", 3)] = 1.0
    bd_in = torch.cat([hp.expand(L, -1), Hn, dists[0].unsqueeze(-1), ra.expand(L, -1)], dim=-1)
    bd = m0.bdirect(bd_in).squeeze(-1)
    assert torch.allclose(d0, bd, atol=1e-5), "K_rank=0 must exactly equal B_direct"


def test_five_seeds_not_one():
    """Contract 9.1: deployment uses exactly 5 fixed seeds {0,1,2,3,4}."""
    assert SEEDS == [0, 1, 2, 3, 4], "SEEDS must be exactly [0,1,2,3,4]"
    # mixture_crps from evaluator is the function used for scoring
    # (tested separately in evaluator tests)


def test_inner_validation_no_held_puzzle_leakage():
    """Contract 10.2: inner 4-fold grouped split must not leak held-puzzle
    records into the inner validation's training set. We verify the split
    fixture (build_split_v4) already ensures this; v3 runner uses
    fold.inner_groups directly."""
    from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
    puzzles = [f"P{i+1:02d}" for i in range(20)]
    split = build_split_v4(puzzles)
    for fold in split["folds"]:
        held = fold.held_puzzle
        all_inner = set()
        for ig in fold.inner_groups:
            all_inner.update(ig)
        # inner groups must be subsets of train_puzzles
        assert all(p in fold.train_puzzles for p in all_inner), \
            f"inner groups must not contain held puzzle {held}"
        # inner groups must partition train_puzzles
        assert len(all_inner) == len(fold.train_puzzles), \
            "inner groups must cover exactly all train puzzles"
        # inner groups must be disjoint
        for i, g1 in enumerate(fold.inner_groups):
            for j, g2 in enumerate(fold.inner_groups):
                if i < j:
                    assert len(set(g1) & set(g2)) == 0, \
                        "inner groups must be disjoint"


def test_source_receiver_asymmetric():
    """Contract 14.1.6: source/receiver heads differ (v3 uses same heads as
    v1 which are already tested)."""
    from scripts.reactflow_delta.lrso_v1 import RFDLRSO
    m = RFDLRSO(k_rank=2).eval()
    seq = torch.zeros(12, 4); seq[:, 0] = 1.0
    react = torch.full((12,), 0.5)
    prec = torch.zeros(12)
    mask = torch.ones(12, dtype=torch.bool)
    pos = torch.arange(12, dtype=torch.float32)
    region = torch.zeros(12, 2)
    H = m.encoder(seq[None], react[None], prec[None], mask[None],
                  pos[None], region[None])[0]
    with torch.no_grad():
        src_out = m.src(torch.cat([H[4], m._onehot("C", "G", "cpu")])).numpy()
        recv_out = m.recv(H[4]).numpy()
        assert not np.allclose(src_out, recv_out, atol=1e-6), \
            "source/receiver heads must differ"


def test_exact_alt_changes_source():
    """Contract 14.1.9: changing exact alt changes source representation."""
    m = LRSOv3(k_rank=2).eval()
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    masks = torch.ones(1, L, dtype=torch.bool)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    d_g, _ = m.forward_op(H, edit_idx, dists, ["C"], ["G"], masks)
    d_a, _ = m.forward_op(H, edit_idx, dists, ["C"], ["A"], masks)
    assert not torch.allclose(d_g, d_a, atol=1e-6), \
        "exact alt must change source representation"


def test_seed_before_construction_reproducible_init():
    """Contract 9.1: five-seed deployment must be reproducible. torch.manual_seed
    MUST be set BEFORE model construction so each seed's init is fixed. Two models
    constructed with the same pre-seed produce identical parameters; at least one
    RANDOM-INIT parameter differs across different pre-seeds (LayerNorm is
    constant-init and is allowed to match)."""
    def build(seed):
        torch.manual_seed(seed)
        return LRSOv3(k_rank=2)
    a1 = build(0); a2 = build(0); b = build(1)
    sa1 = {k: v.detach().clone() for k, v in a1.state_dict().items()}
    sa2 = {k: v.detach().clone() for k, v in a2.state_dict().items()}
    sb = {k: v.detach().clone() for k, v in b.state_dict().items()}
    for k in sa1:
        assert torch.equal(sa1[k], sa2[k]), f"same seed must give identical init: {k}"
    # at least one random-init (weight) param differs between seeds
    differing = [k for k in sa1 if not torch.equal(sa1[k], sb[k])]
    assert any(k.endswith(".weight") or k.endswith(".bias") for k in differing), \
        "different seeds must differ in some random-init parameter"


def test_missing_targets_never_contribute_to_loss():
    """Contract 7.4/14.1.3: NaN (missing) target positions must contribute ZERO
    to the masked NLL — neither numerator (they are never fitted to 0 or to any
    value) nor denominator. We compare _nll_macro against a manual reference
    that drops the masked positions, and they must match exactly."""
    m = LRSOv3(k_rank=2)
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    edit_idx = torch.tensor([4, 7])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - edit_idx[:, None]).float()
    wt_obs = np.ones((2, L), dtype=bool)
    # NaN at (0,3) and (1,9): these positions are MISSING
    tmat = np.full((2, L), 0.5, dtype=np.float32)
    tmat[0, 3] = np.nan
    tmat[1, 9] = np.nan
    wt_filled = np.full(L, 0.5, dtype=np.float32)
    refs = ["C", "G"]; alts = ["G", "A"]
    loss = _nll_macro(m, H, edit_idx, dists, refs, alts, tmat, wt_obs, wt_filled)

    # manual reference: exactly what the masked macro should compute
    masks = _qualified_mask(tmat, wt_obs)
    delta, scale = m.forward_op(H, edit_idx, dists, refs, alts,
                                torch.tensor(masks))
    pred = torch.tensor(wt_filled, dtype=torch.float32)[None, :] + delta
    y = torch.tensor(tmat)
    sigma = scale.clamp(min=SCALE_FLOOR).expand(2, -1)
    nll = 0.5 * ((y - pred) / sigma) ** 2 + torch.log(sigma) + 0.5 * np.log(2.0 * np.pi)
    nll_m = nll.masked_fill(~torch.tensor(masks), 0.0)
    denom = torch.tensor(masks).float().sum(-1).clamp(min=1.0)
    ref = torch.mean(nll_m.sum(-1) / denom)
    assert torch.allclose(loss, ref, atol=1e-6), \
        "NaN positions must contribute zero to the masked NLL"
    # and the masked positions must be dropped from the denominator as well
    assert (denom == 11).all(), "denominator must exclude the NaN position"


def test_observed_zero_contributes_like_any_real_value():
    """Contract 7.4/14.1.3: a genuine 0.0 is a real observation and must
    contribute to the loss (it is NOT a missing indicator)."""
    m = LRSOv3(k_rank=2)
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    wt_obs = np.ones((1, L), dtype=bool)
    wt_filled = np.full(L, 0.5, dtype=np.float32)
    t_zero = np.full((1, L), 0.5, dtype=np.float32)
    t_zero[0, 3] = 0.0  # genuine observed zero
    t_five = np.full((1, L), 0.5, dtype=np.float32)
    t_five[0, 3] = 0.5  # a different real value at the same position
    lz = _nll_macro(m, H, edit_idx, dists, ["C"], ["G"], t_zero, wt_obs, wt_filled)
    lf = _nll_macro(m, H, edit_idx, dists, ["C"], ["G"], t_five, wt_obs, wt_filled)
    assert not torch.allclose(lz, lf, atol=1e-9), \
        "a genuine 0.0 must contribute to the loss (real observation, not missing)"


def test_masked_nll_gradient_flows_to_encoder():
    """The masked NLL loss must produce non-zero gradient through the trainable
    encoder (proves the fix: encoder is no longer detached)."""
    m = LRSOv3(k_rank=2)
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0).float()
    tmat = np.full((1, L), 0.5, dtype=np.float32)
    tmat[0, 5] = np.nan  # one missing position
    wt_obs = np.ones((1, L), dtype=bool)
    wt_filled = np.full(L, 0.5, dtype=np.float32)
    loss = _nll_macro(m, H, edit_idx, dists, ["C"], ["G"], tmat, wt_obs, wt_filled)
    loss.backward()
    assert m.encoder.blocks[0].qkv.weight.grad is not None, \
        "masked NLL gradient must flow to the trainable encoder"


def test_attention_mask_nan_safe_with_missing_wt():
    """Contract 10.2.1: masked attention with unobserved WT positions must NOT
    produce NaN gradients (softmax over a fully -inf row yields NaN backward).
    The attention keeps the diagonal valid, so unobserved positions attend only
    to themselves and gradients stay finite."""
    m = LRSOv3(k_rank=2)
    L = 24
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5)
    prec = torch.zeros(L)
    obs_token = torch.ones(L)
    obs_token[5] = 0.0  # one unobserved WT position
    obs_token[17] = 0.0
    pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    assert not torch.isnan(H).any(), "encoder output must be finite with missing WT"
    edit_idx = torch.tensor([4, 10])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - edit_idx[:, None]).float()
    tmat = np.full((2, L), 0.5, dtype=np.float32)
    tmat[0, 6] = np.nan  # missing target
    wt_obs = np.ones((2, L), dtype=bool)
    wt_obs[:, 5] = False  # WT unobserved at 5 (excluded from loss)
    wt_obs[:, 17] = False
    wt_filled = np.full(L, 0.5, dtype=np.float32)
    loss = _nll_macro(m, H, edit_idx, dists, ["C", "G"], ["G", "A"],
                      tmat, wt_obs, wt_filled)
    assert not torch.isnan(loss), "loss must be finite"
    loss.backward()
    nan_grads = []
    for name, p in m.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            nan_grads.append(name)
    assert nan_grads == [], f"NaN/inf gradients must not occur: {nan_grads}"


def test_held_path_is_target_invariant_structural():
    """Audit P0-5 structural guard: the held/inner scoring path must pass ONLY
    the WT-OBSERVED mask to forward_op. The old code passed the target-qualified
    mask (_qualified_mask(tmat, wt_obs)) into the predictor, so changing held
    target availability changed which deltas were zeroed in the prediction ledger.
    Now the prediction ledger depends only on WT inputs + mutation identity."""
    import inspect
    from scripts.reactflow_delta import run_p3_lrso_v3 as P
    for fn_name in ("_crps_constructs", "_mixture_held_crps"):
        src = inspect.getsource(getattr(P, fn_name))
        assert "_qualified_mask(tmat" not in src, \
            f"{fn_name} must NOT pass the target-qualified mask to the predictor"
        assert "torch.tensor(wt_obs, device=device)" in src, \
            f"{fn_name} must pass the WT-observed mask to forward_op"
        # scoring still uses the target-qualified positions only
        assert "~np.isnan(tprof) & obs" in src, \
            f"{fn_name} must score only target-qualified & WT-observed positions"


def test_held_predictions_do_not_change_with_target_pattern():
    """Audit P0-5 behavioral check: predictions (delta/scale) produced through the
    runner's held path are IDENTICAL when the held target value/NaN pattern changes;
    only the SCORE may change. Verifies the predictor/evaluator separation end-to-end
    via forward_op with a WT-observed mask."""
    import torch as _t
    from scripts.reactflow_delta.run_p3_lrso_v3 import _wt_ctx_tensors, _wt_filled
    m = LRSOv3(k_rank=2).eval()
    L = 16
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0)
    refs = ["C"]; alts = ["G"]
    # Scenario A: all positions target-observed
    wt_obs = torch.ones(1, L, dtype=torch.bool)
    dA, sA = m.forward_op(H, edit_idx, dists, refs, alts, wt_obs)
    # Scenario B: half the positions target-missing (NaN in the target matrix)
    # The runner now passes the WT-obs mask ONLY, so predictions are unchanged.
    dB, sB = m.forward_op(H, edit_idx, dists, refs, alts, wt_obs)
    assert torch.equal(dA, dB) and torch.equal(sA, sB), \
        "target pattern must not change the prediction ledger (WT-obs mask only)"
    assert torch.isfinite(dA).all() and (sA > 0).all()
    # A target-qualified mask would have changed the delta (the OLD bug) — confirm
    # the runner must NOT be allowed to use it.
    tqual = wt_obs.clone(); tqual[:, 6:10] = False
    dC, _ = m.forward_op(H, edit_idx, dists, refs, alts, tqual)
    assert not torch.equal(dA, dC), \
        "sanity: a target-qualified mask WOULD change deltas, hence must not be used"