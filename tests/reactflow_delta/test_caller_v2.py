"""Tests for the R3 fold-local caller v2 (ReactFlowDelta §13.2 R3, §13.3 items 2 & 5).

Covers:
  1. determinism           — same input + seed -> byte-identical output (manifest & results)
  2. fold-locality         — reading an outer (validation/test) row raises a hard error
  3. sliding cluster       — synthetic hotspot is called changer vs background nonchanger
  4. spatial-block null    — null is reproducible given seed, differs across seeds
  5. low reliability       — ICC below threshold / insufficient replicates -> NO_CALL
  6. eligibility mask      — only eligible positions contribute to the cluster statistic
  7. consumption/exposure  — outer-row access writes a consumption event; seal can never
                             be restored (aggregate must not restore sealed)
"""
import importlib.util
import json
import math
import os
import random

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALLER_PY = os.path.join(ROOT, "scripts/reactflow_delta/caller_v2.py")


def _load_caller():
    import sys
    spec = importlib.util.spec_from_file_location("caller_v2", CALLER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["caller_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


caller_mod = _load_caller()


def _mk_profile(seed, length=100, base_amp=0.5, noise=0.01, error=0.05):
    """Deterministic WT profile (base varies across positions + tiny noise)."""
    rng = random.Random(seed)
    base = [0.5 + base_amp * math.sin(i / 7.0) for i in range(length)]
    return [b + rng.gauss(0, noise) for b in base], [error] * length


def _reliable_group(study="STUDYA", probe=("1M7",), n_rep=4, length=100):
    """A high-ICC replicate group (small noise -> replicates agree well)."""
    profs = [_mk_profile(1000 + r, length=length, noise=0.01)[0] for r in range(n_rep)]
    errs = [[0.05] * length for _ in range(n_rep)]
    return caller_mod.ReplicateGroup(
        group_key=(study, tuple(probe)), wt_profiles=profs, wt_errors=errs,
        eligibility_mask=[1] * length, study=study)


def _noisy_group(study="STUDYB", probe=("2A3",), n_rep=4, length=100):
    """A low-ICC replicate group (noise >> signal -> replicates disagree)."""
    profs = [_mk_profile(2000 + r, length=length, base_amp=0.01, noise=2.0)[0]
             for r in range(n_rep)]
    errs = [[0.05] * length for _ in range(n_rep)]
    return caller_mod.ReplicateGroup(
        group_key=(study, tuple(probe)), wt_profiles=profs, wt_errors=errs,
        eligibility_mask=[1] * length, study=study)


def _pair(pair_id, wt, mut, werr, merr, mask, group_key, role="train"):
    return caller_mod.PairFeatures(
        pair_id=pair_id, wt_reactivity=wt, mutant_reactivity=mut,
        wt_error=werr, mutant_error=merr, eligibility_mask=mask,
        group_key=group_key, role=role)


def _base_pairs_and_groups(length=100):
    """Reusable train-fold: several reliable replicate groups + a hotspot/background pair."""
    groups = [_reliable_group(f"STUDY{i}", n_rep=4, length=length) for i in range(8)]
    # reliable group for the test pair
    g0 = groups[0]
    gk = g0.group_key
    wt, werr = g0.wt_profiles[0], g0.wt_errors[0]
    # hotspot: strong delta at positions 40..44 (eligible)
    mut_hot = list(wt)
    for i in range(40, 45):
        mut_hot[i] = wt[i] + 0.5
    hotspot = _pair("hot", wt, mut_hot, werr, werr, [1] * length, gk)
    # background: no change
    bg = _pair("bg", wt, list(wt), werr, werr, [1] * length, gk)
    return groups, hotspot, bg


def test_determinism_byte_identical():
    """Same input + same seed -> identical manifest/results (byte-level)."""
    groups, hotspot, bg = _base_pairs_and_groups()
    pairs = [hotspot, bg]

    c1 = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, pairs)
    c2 = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, pairs)
    r1 = [c1.call(p).to_dict() for p in pairs]
    r2 = [c2.call(p).to_dict() for p in pairs]
    m1 = c1.manifest([c1.call(p) for p in pairs]).to_dict()
    m2 = c2.manifest([c2.call(p) for p in pairs]).to_dict()

    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert m1["input_sha256"] == m2["input_sha256"]


def test_determinism_different_seed_differs():
    """Different seed -> different (deterministic) null sequence."""
    groups, hotspot, bg = _base_pairs_and_groups()
    pairs = [hotspot, bg]
    c1 = caller_mod.CallerV2(seed=1).fit(groups, pairs)
    c2 = caller_mod.CallerV2(seed=2).fit(groups, pairs)
    assert c1.null_distribution != c2.null_distribution
    assert len(c1.null_distribution) == len(c2.null_distribution)


def test_fold_locality_outer_row_hard_fail():
    """Reading a validation/test row raises OuterFoldAccessError (hard fail)."""
    split_roles = {"TRAIN_S": "train", "TEST_S": "test", "VAL_S": "validation"}
    loader = caller_mod.FoldLocalLoader(split_roles, train_roles=("train",))
    # train row is fine
    loader.assert_train("pair_train", "TRAIN_S")
    assert loader.seal.is_sealed
    # outer row -> hard error + exposure recorded + seal broken
    with pytest.raises(caller_mod.OuterFoldAccessError):
        loader.assert_train("pair_test", "TEST_S")
    assert not loader.seal.is_sealed
    assert loader.ledger.has_outer_access()


def test_fold_locality_structurally_prevented_via_guard():
    """The loader rejects any non-train study before pair construction."""
    split_roles = {"ADD140": "train", "SL5CV2": "test"}
    loader = caller_mod.FoldLocalLoader(split_roles, train_roles=("train",))
    with pytest.raises(caller_mod.OuterFoldAccessError):
        loader.assert_train("SL5CV2_pair", "SL5CV2")


def test_sliding_cluster_hotspot_vs_background():
    """Synthetic hotspot (eligible) -> changer; background -> nonchanger."""
    groups, hotspot, bg = _base_pairs_and_groups()
    caller = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, [hotspot, bg])
    res_hot = caller.call(hotspot)
    res_bg = caller.call(bg)
    assert res_hot.label == "1", f"expected changer, got {res_hot.to_dict()}"
    assert res_bg.label == "0", f"expected nonchanger, got {res_bg.to_dict()}"
    assert res_hot.statistic > res_bg.statistic
    # reliability gate must be satisfied for both (they map to reliable group)
    assert res_hot.reliability is not None and res_hot.reliability >= caller.icc_threshold


def test_spatial_block_null_reproducible():
    """Null distribution is deterministic given seed; differs across seeds."""
    groups = [_reliable_group(f"STUDY{i}", n_rep=4) for i in range(8)]
    profs, mask = caller_mod._null_z_profiles(groups)
    n1 = caller_mod.spatial_block_null(profs, mask, n_null=500, seed=7)
    n2 = caller_mod.spatial_block_null(profs, mask, n_null=500, seed=7)
    assert n1 == n2
    assert len(n1) == 500
    n3 = caller_mod.spatial_block_null(profs, mask, n_null=500, seed=8)
    assert n1 != n3


def test_spatial_block_pvalue_plus_one():
    """p-value uses (b+1)/(B+1) per audit §13.3.5."""
    null = [1.0, 2.0, 3.0]
    p = caller_mod._p_value(null, 3.0, plus_one=True)
    assert p == pytest.approx((1 + 1) / (3 + 1))
    p0 = caller_mod._p_value(null, 10.0, plus_one=True)
    assert p0 == pytest.approx((0 + 1) / (3 + 1))


def test_low_reliability_no_call():
    """ICC below threshold / insufficient replicates -> NO_CALL (no forced label)."""
    groups, hotspot, _ = _base_pairs_and_groups()
    # replace the pair's group with a noisy (low-ICC) one
    noisy = _noisy_group("STUDYB", n_rep=4)
    groups = groups[1:] + [noisy]  # keep >= min_replicate_groups reliable groups
    caller = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, [hotspot])
    # pair maps to noisy low-ICC group
    low_pair = caller_mod.PairFeatures(
        pair_id="low", wt_reactivity=hotspot.wt_reactivity,
        mutant_reactivity=hotspot.mutant_reactivity,
        wt_error=hotspot.wt_error, mutant_error=hotspot.mutant_error,
        eligibility_mask=hotspot.eligibility_mask, group_key=noisy.group_key)
    res = caller.call(low_pair)
    assert res.label == "NO_CALL"
    assert res.reliability is not None and res.reliability < caller.icc_threshold


def test_no_call_when_group_has_insufficient_replicates():
    """A group with < min_replicates has reliability None -> NO_CALL."""
    groups, hotspot, _ = _base_pairs_and_groups()
    single = caller_mod.ReplicateGroup(
        group_key=("SOLO", ("1M7",)),
        wt_profiles=[hotspot.wt_reactivity],
        wt_errors=[hotspot.wt_error],
        eligibility_mask=hotspot.eligibility_mask, study="SOLO")
    groups = groups + [single]
    caller = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, [hotspot])
    solo_pair = caller_mod.PairFeatures(
        pair_id="solo", wt_reactivity=hotspot.wt_reactivity,
        mutant_reactivity=hotspot.mutant_reactivity,
        wt_error=hotspot.wt_error, mutant_error=hotspot.mutant_error,
        eligibility_mask=hotspot.eligibility_mask, group_key=single.group_key)
    res = caller.call(solo_pair)
    assert res.label == "NO_CALL"


def test_eligibility_mask_honored():
    """Strong delta at INELIGIBLE positions must not contribute -> nonchanger."""
    groups, hotspot, _ = _base_pairs_and_groups()
    gk = groups[0].group_key
    wt, werr = groups[0].wt_profiles[0], groups[0].wt_errors[0]
    # strong delta at positions 40..44 but those are INELIGIBLE (mask=0)
    mut = list(wt)
    for i in range(40, 45):
        mut[i] = wt[i] + 0.5
    mask = [1] * len(wt)
    for i in range(40, 45):
        mask[i] = 0
    masked_pair = _pair("masked", wt, mut, werr, werr, mask, gk)
    caller = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, [masked_pair])
    res = caller.call(masked_pair)
    # with the hotspot masked out the statistic is not significant
    assert res.label == "0"
    assert res.statistic == pytest.approx(0.0)
    # now unmask -> changer
    mask2 = [1] * len(wt)
    unmasked = _pair("unmasked", wt, mut, werr, werr, mask2, gk)
    res2 = caller.call(unmasked)
    assert res2.label == "1"


def test_consumption_event_and_seal_never_restored():
    """Outer access writes a consumption event; seal can never be restored."""
    split_roles = {"TRAIN_S": "train", "TEST_S": "test"}
    loader = caller_mod.FoldLocalLoader(split_roles, train_roles=("train",))
    loader.assert_train("t1", "TRAIN_S")
    with pytest.raises(caller_mod.OuterFoldAccessError):
        loader.assert_train("t2", "TEST_S")
    events = loader.ledger.events
    # both a train access and an outer access are recorded
    assert any(e["event_type"] == "TRAIN_ROW_ACCESS" for e in events)
    assert any(e["event_type"] == "OUTER_ROW_ACCESS" and e["row_id"] == "t2" for e in events)
    # seal is permanently broken; aggregate must NOT restore it
    assert not loader.seal.is_sealed
    with pytest.raises(caller_mod.SealViolationError):
        loader.seal.restore()
    assert not loader.seal.is_sealed


def test_manifest_reports_seal_outer_access():
    """Manifest surfaces whether outer access occurred (aggregate transparency)."""
    split_roles = {"TRAIN_S": "train", "TEST_S": "test"}
    loader = caller_mod.FoldLocalLoader(split_roles, train_roles=("train",))
    groups, hotspot, bg = _base_pairs_and_groups()
    caller = caller_mod.CallerV2(seed=caller_mod.RNG_SEED).fit(groups, [hotspot, bg])
    results = [caller.call(p) for p in [hotspot, bg]]
    man = caller.manifest(results)
    # no outer access in the fit-only path
    assert man.seal_outer_access is False
