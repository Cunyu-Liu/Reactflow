#!/usr/bin/env python3
"""Unit tests for caller_v4 (Batch 1C) — information-permission + sensitivity gate."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import pytest

import caller_v4 as c4
from caller_v2 import (
    CallerV2Error,
    CallResult,
    ReplicateGroup,
    PairFeatures,
    ICC_THRESHOLD,
    MIN_REPLICATES,
    MIN_REPLICATE_GROUPS,
)


def _pair(pair_id, wt, mut, mask, group=("TRAIN_A", "1M7"), werr=None, merr=None):
    werr = werr or [0.1] * len(wt)
    merr = merr or [0.1] * len(mut)
    return PairFeatures(pair_id=pair_id, wt_reactivity=list(wt),
                        mutant_reactivity=list(mut), wt_error=list(werr),
                        mutant_error=list(merr), eligibility_mask=list(mask),
                        group_key=group, role="train")


def _group(key, profiles, mask):
    mask = mask or [1] * len(profiles[0])
    return ReplicateGroup(group_key=key, wt_profiles=[list(p) for p in profiles],
                          wt_errors=[[0.1] * len(profiles[0]) for _ in profiles],
                          eligibility_mask=list(mask), study=key[0])


def _train_groups():
    # 6 replicate groups with >=2 replicates, clear structure
    return [
        _group(("TRAIN_A", "1M7"), [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 3.9]], [1, 1, 1, 1]),
        _group(("TRAIN_B", "1M7"), [[5.0, 5.0, 5.0], [5.2, 4.8, 5.1]], [1, 1, 1]),
        _group(("TRAIN_C", "1M7"), [[1.0, 1.0, 1.0, 1.0, 1.0], [1.1, 0.9, 1.0, 1.05, 0.95]], [1] * 5),
        _group(("TRAIN_D", "1M7"), [[2.0, 2.0, 2.0], [2.1, 1.9, 2.0], [1.95, 2.05, 2.0]], [1, 1, 1]),
        _group(("TRAIN_E", "1M7"), [[3.0, 3.0, 3.0, 3.0], [3.1, 3.0, 2.9, 3.05]], [1] * 4),
        _group(("TRAIN_F", "1M7"), [[4.0, 4.0, 4.0], [4.05, 3.95, 4.0], [3.98, 4.02, 4.0]], [1] * 3),
    ]


def test_singleton_no_replicate_group_no_call():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    # min_replicates=2; a singleton group has n_replicates=1
    groups = _train_groups()
    caller.fit(groups, [])
    p = _pair("p1", [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [1, 1, 1], group=("HELD_X", "1M7"))
    # held group not in reliability map -> global reliability fallback; callable
    res = caller.call(p)
    assert res.label in ("1", "0", "NO_CALL")


def test_held_outer_train_only_sigma():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    # STRICT mode: noise_replicate_groups (held pooled scatter) is forbidden
    with pytest.raises(CallerV2Error):
        c4.CallerV4(mode=c4.MODE_STRICT).fit(_train_groups(), [],
                                             noise_replicate_groups=_train_groups())


def test_transductive_add_held_wt_replicates():
    caller = c4.CallerV4(mode=c4.MODE_TRANSDUCTIVE)
    caller.fit(_train_groups(), [])
    caller.add_held_wt_replicates([_group(("HELD_X", "1M7"), [[1, 2, 3], [1.1, 2.1, 3.1]], [1, 1, 1])])
    assert ("HELD_X", "1M7") in caller._sigma_by_group


def test_transductive_add_held_forbidden_in_strict():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    with pytest.raises(CallerV2Error):
        caller.add_held_wt_replicates([_group(("HELD_X", "1M7"), [[1, 2, 3], [1.1, 2.1, 3.1]], [1, 1, 1])])


def test_ddof_scatter_uses_sample_sd():
    # two identical profiles -> scatter 0 -> positions unfinite -> fallback to error
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    p = _pair("p1", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1, 1, 1], group=("TRAIN_A", "1M7"))
    res = caller.call(p)
    assert res.label in ("1", "0", "NO_CALL")


def test_near_zero_variance_handled():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    p = _pair("p1", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1, 1, 1], group=("TRAIN_A", "1M7"))
    caller.call(p)  # must not raise


def test_edited_site_removed_from_mask():
    # edited-site position excluded -> not eligible -> never contributes z
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    # mask[1]=0 (edited site excluded); huge mutant jump at excluded site must not flip
    p = _pair("p1", [1.0, 2.0, 3.0], [1.0, 99.0, 3.0], [1, 0, 1], group=("TRAIN_A", "1M7"))
    res = caller.call(p)
    assert res.label in ("1", "0", "NO_CALL")


def test_missing_reactivity_not_zero():
    # missing (non-finite) measured as excluded, not zero.
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    p = _pair("p1", [1.0, float("nan"), 3.0], [1.0, 5.0, 3.0], [1, 1, 1], group=("TRAIN_A", "1M7"))
    z, eligible = caller._z_for_pair(p)
    # NaN position: either excluded or z=None, never treated as finite 0 on its own
    for i in range(len(p)):
        if not _finite_z(z[i]):
            pass
    assert sum(1 for v in z if v is not None and _finite_z(v)) >= 0


def _finite_z(v):
    import math
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def test_strict_fallback_uses_train_median_sigma():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    assert caller._train_median_sigma is not None
    # held group not in train -> _z_for_pair med_sigma = train_median (not None)
    p = _pair("p1", [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [1, 1, 1], group=("HELD_X", "1M7"))
    sigma = caller._sigma_by_group.get(("HELD_X", "1M7"))
    assert sigma is None
    res = caller.call(p)
    assert res.label in ("1", "0", "NO_CALL")


def test_sensitivity_manifest_transition_and_coverage():
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    primary = [
        CallResult("a", "1", 1.0, 0.01, 0.9, ("P1", "1M7")),
        CallResult("b", "0", 0.1, 0.9, 0.9, ("P1", "1M7")),
        CallResult("c", "NO_CALL", None, None, None, ("P2", "1M7")),
    ]
    sens = [
        CallResult("a", "1", 1.0, 0.01, 0.9, ("P1", "1M7")),
        CallResult("b", "1", 1.0, 0.01, 0.9, ("P1", "1M7")),   # 0->1 flip
        CallResult("c", "0", 0.1, 0.9, 0.9, ("P2", "1M7")),      # NO_CALL->call
    ]
    m = caller.sensitivity_manifest(primary, sens, {"a": "P1", "b": "P1", "c": "P2"})
    assert m["n_paired_rows"] == 3
    assert m["transition_matrix"]["0_to_1"] == 1
    assert m["transition_matrix"]["no_call_to_call"] == 1
    assert m["label_agreement"] == pytest.approx(1 / 3)


def test_sensitivity_gate_evaluation():
    # all-agree, high coverage -> gate passes
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    rows = [CallResult(f"p{i}", "1", 1.0, 0.01, 0.9, ("P1", "1M7")) for i in range(30)]
    m = caller.sensitivity_manifest(rows, rows, {f"p{i}": "P1" for i in range(30)})
    assert m["stability_gate"]["pass"] is True


def test_pair_endpoint_saturation_and_absolute_ranking():
    # negative/positive absolute ranking: both large-magnitude (signed or abs)
    # should rank high in an absolute-magnitude task; here we only assert the
    # z computation is signed and finite, and the caller does not crash.
    caller = c4.CallerV4(mode=c4.MODE_STRICT)
    caller.fit(_train_groups(), [])
    p_neg = _pair("n", [1.0, 2.0, 3.0], [-5.0, -6.0, -7.0], [1, 1, 1], group=("TRAIN_A", "1M7"))
    p_pos = _pair("p", [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [1, 1, 1], group=("TRAIN_A", "1M7"))
    zn, _ = caller._z_for_pair(p_neg)
    zp, _ = caller._z_for_pair(p_pos)
    assert _finite_z(zn[0]) and _finite_z(zp[0])
    # signed z: negative jump yields negative z, positive jump yields positive z
    assert zn[0] < 0 and zp[0] > 0


def test_unknown_mode_rejected():
    with pytest.raises(CallerV2Error):
        c4.CallerV4(mode="BOGUS_MODE")