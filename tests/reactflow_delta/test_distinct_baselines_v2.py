#!/usr/bin/env python3
"""test_distinct_baselines_v2: audit P0-3 acceptance.

  * distinct baseline objects have distinct identity and non-identical predictions
  * train_median is REALLY computed from the train fold (not a copy of WT anchor)
  * the old v2 alias pattern (nonlinear/flat_mlp/rfd_direct sharing one object) is
    structurally impossible now: every class owns its own parameters
  * RFDDirectRank0 disables the low-rank term (k_rank=0) — the nested-null bridge
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta import baseline_v1 as B
from scripts.reactflow_delta.run_p3_lrso_v3 import LRSOv3


def _make_synthetic_csv(path: Path) -> Path:
    seq = "A" * 8 + "C" * 4 + "G" * 4 + "U" * 4  # 20 nt
    rows = []
    for puzzle in ["P01", "P02"]:
        for method in ["Eterna", "Rosetta"]:
            rec = {"id": f"{puzzle}_{method}_wt", "sequence": seq,
                   "experiment_type": "2A3_MaP", "dataset_name": "X",
                   "puzzle": puzzle, "method": method, "sub_start": 9,
                   "sub_end": 16, "design_length": 8,
                   "design_sequence": seq[8:16],
                   "target_structure": "", "mutA": 0,
                   "M2_structure": "AAAA"}
            for i in range(1, 21):
                rec[f"reactivity_{i:04d}"] = float(i) / 20
                rec[f"reactivity_error_{i:04d}"] = 0.1
            rows.append(rec)
            for design_pos in range(8):
                full_pos = 8 + design_pos
                ref = seq[full_pos]
                for alt in [base for base in "ACGU" if base != ref][:2]:
                    m = dict(rec)
                    m["id"] = (
                        f"{puzzle}_{method}_mm_{design_pos}_{ref}_{alt}"
                    )
                    m["sequence"] = (
                        seq[:full_pos] + alt + seq[full_pos + 1:]
                    )
                    m["mutA"] = design_pos + 1
                    for i in range(1, 21):
                        m[f"reactivity_{i:04d}"] = (
                            float((i + full_pos) % 20) / 20
                        )
                        m[f"reactivity_error_{i:04d}"] = 0.1
                    rows.append(m)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def universe():
    with tempfile.TemporaryDirectory() as td:
        csv = _make_synthetic_csv(Path(td) / "m2.csv")
        u = M2Universe(csv)
        u.build()
        yield u


@pytest.fixture()
def train_held(universe):
    recs = universe.get_records()
    train = [r for r in recs if r.puzzle == "P01"]
    held = [r for r in recs if r.puzzle == "P02"]
    return train, held


def test_objects_are_distinct_instances(universe):
    z = B.ZeroResponse(); tm = B.TrainMedian()
    rd = B.RidgeDirect(); nl = B.NonlinearDirect(); r0 = B.RFDDirectRank0()
    objs = [z, tm, rd, nl, r0]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            assert objs[i] is not objs[j]
            assert objs[i].name != objs[j].name


def test_predictions_differ_between_distinct_classes(universe, train_held):
    train, held = train_held
    device = "cpu"
    zero = B.ZeroResponse(); zero.fit(universe, train, device)
    tm = B.TrainMedian(); tm.fit(universe, train, device)
    rd = B.RidgeDirect(); rd.fit(universe, train, device)
    nl = B.NonlinearDirect(); nl.fit(universe, train, device)

    p_zero = zero.predict_full_profile(universe, held, device)
    p_tm = tm.predict_full_profile(universe, held, device)
    p_rd = rd.predict_full_profile(universe, held, device)
    p_nl = nl.predict_full_profile(universe, held, device)

    keys = sorted(p_zero)
    locs = {nm: [p[k][0] for k in keys] for nm, p in
            [("zero", p_zero), ("train_median", p_tm), ("ridge", p_rd), ("nonlinear", p_nl)]}
    names = ["zero", "train_median", "ridge", "nonlinear"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(locs[names[i]], locs[names[j]], atol=1e-6), \
                f"{names[i]} and {names[j]} predictions are identical (alias!): " \
                f"z={locs[names[i]][:5]} {names[j]}={locs[names[j]][:5]}"


def test_train_median_is_real_trainfold_statistic(universe, train_held):
    train, held = train_held
    tm = B.TrainMedian(); tm.fit(universe, train, "cpu")
    # expected per-position median over train mutant target values
    expected: dict[int, list] = {}
    for r in train:
        tp, _ = universe.mutant_full_profile(
            r.wt_id, r.design_pos, r.ref, r.alt
        )
        if tp is None:
            continue
        for i in range(len(tp)):
            if not np.isnan(tp[i]):
                expected.setdefault(i, []).append(float(tp[i]))
    # sample a few positions
    for pos in (0, 9, 15):
        med = float(np.median(expected[pos]))
        assert tm.median[pos] == pytest.approx(med, abs=1e-9)
    # and it is NOT the WT anchor at the same position
    assert not np.allclose(list(tm.median.values()), 1.0)


def test_rfddirect_rank0_disables_low_rank_term():
    """k_rank=0 must produce delta identical to the direct branch alone (low-rank term
    exactly zero), while k_rank=2 can differ — this is the nested-null bridge."""
    torch_ok = True
    try:
        import torch
        m0 = LRSOv3(k_rank=0)
        m2 = LRSOv3(k_rank=2)
        m0.eval(); m2.eval()
        L = 16
        seq = torch.zeros(L, 4); seq[:, 0] = 1.0
        react = torch.full((L,), 0.5)
        prec = torch.zeros(L); obs = torch.ones(L)
        pos = torch.arange(L, dtype=torch.float32)
        region = torch.zeros(L, 2)
        ctx = (seq, react, prec, obs, pos, region)
        H0 = m0.encode(ctx); H2 = m2.encode(ctx)
        edit_idx = torch.tensor([4])
        dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0)
        masks = torch.ones(1, L, dtype=torch.bool)
        d0, s0 = m0.forward_op(H0, edit_idx, dists, ["C"], ["G"], masks)
        d2, s2 = m2.forward_op(H2, edit_idx, dists, ["C"], ["G"], masks)
        # rank0's low-rank contribution is structurally zero: delta = bdirect only.
        # Replicate forward_op's bdirect input exactly (ctx_norm, ref/alt one-hot).
        from scripts.reactflow_delta.run_p3_lrso_v3 import ALPHA as _ALPHA
        Hn = m0.ctx_norm(H0)
        hp = Hn[4].unsqueeze(0)
        ra = torch.zeros(1, 8)
        ra[0, _ALPHA["C"]] = 1.0                    # ref "C" -> ALPHA["C"] = 1
        ra[0, 4 + _ALPHA["G"]] = 1.0                # alt "G" -> ALPHA["G"] = 2 -> slot 6
        hp_e = hp.unsqueeze(1).expand(1, L, -1)
        H_e = Hn.unsqueeze(0).expand(1, -1, -1)
        ra_e = ra.unsqueeze(1).expand(1, L, -1)
        bd = m0.bdirect(torch.cat([hp_e, H_e, dists.unsqueeze(-1), ra_e], dim=-1)).squeeze(-1)
        assert torch.allclose(d0, bd, atol=1e-6), "rank0 must equal direct branch"
        # scale heads shared architecture => finite positive
        assert torch.isfinite(d0).all() and (s0 > 0).all()
        # k_rank=2 has additional capacity => delta differs in general (non-degenerate)
        assert not torch.allclose(d0, d2, atol=1e-3)
    except ImportError:
        torch_ok = False
    assert torch_ok, "torch must be available for the rank0 null test"


def test_no_shared_instance_structural_guard(universe, train_held):
    """Regression guard: the v2 pattern of pointing three names at ONE object must be
    impossible — each BASELINES entry constructs a NEW instance."""
    import inspect
    for name, cls in B.BASELINES.items():
        inst1 = cls(); inst2 = cls()
        assert inst1 is not inst2, f"BASELINES['{name}'] must construct fresh instances"
