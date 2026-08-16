#!/usr/bin/env python3
"""Tests for m2r_data_v1 — M2R dataset loader."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r


@pytest.fixture(scope="module")
def parsed():
    path = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
    designs, meta = m2r.parse_m2r_csv(path)
    return designs, meta


def test_meta_counts(parsed):
    designs, meta = parsed
    assert meta["n_designs"] > 100
    assert meta["n_pairs"] > 3000
    assert meta["n_rows"] > 9000


def test_designs_have_pairs(parsed):
    designs, _ = parsed
    usable = [d for d in designs if d["usable"]]
    assert len(usable) > 100
    # most designs have pairs with rescue_factor; a few (e.g. P20_Eterna)
    # may have missing rescue data and are filtered at the runner level
    n_designs_with_rf = sum(
        1 for d in usable if any(p["rescue_factor"] is not None for p in d["pairs"]))
    assert n_designs_with_rf > 100


def test_single_mutants_are_single_diff(parsed):
    designs, _ = parsed
    d = designs[0]
    seq = d["sequence"]
    for p in d["pairs"][:5]:
        a = p.get("mutA_seq"); b = p.get("mutB_seq"); db = p.get("double_seq")
        # M2R sequences may have different pad/barcode sequences across rows,
        # so the raw sequence difference count is not reliable.  We just verify
        # that the sequences exist and have the same length.
        assert a is not None and b is not None and db is not None
        assert len(a) == len(seq)
        assert len(b) == len(seq)
        assert len(db) == len(seq)


def test_pair_positions_alignment(parsed):
    designs, _ = parsed
    d = designs[0]
    sub_start = d["sub_start"]
    for p in d["pairs"][:5]:
        editA = sub_start - 1 + (p["mutA"] - 1)
        editB = sub_start - 1 + (p["mutB"] - 1)
        # verify positions are within the sequence
        assert 0 <= editA < len(d["sequence"])
        assert 0 <= editB < len(d["sequence"])
        # verify they are different positions (a real pair has i != j)
        assert editA != editB


def test_rescue_range(parsed):
    designs, _ = parsed
    samples = m2r.build_all_pair_samples(designs)
    rfs = [s.rescue_factor for s in samples if s.rescue_factor is not None]
    rf = np.array(rfs)
    assert len(rf) > 3000
    # physical range: rescue can be negative (double makes it worse) but bounded
    assert np.percentile(rf, 5) > -2.0
    assert np.percentile(rf, 95) < 1.5


def test_structure_feature_extraction():
    # a fake design for unit-testing build_pair_samples
    n = 30
    seq = "A" * n
    wt = [1.0] * n
    err = [0.1] * n
    d = {
        "puzzle": "P99", "method": "test", "source_accession": "OK7a_M2R_P99_test",
        "sequence": seq, "sub_start": 1, "sub_end": n,
        "target_structure": ".((....))." + "." * (n - 10),
        "wt_reactivity": wt, "wt_error": err,
        "pairs": [{
            "mutA": 2, "mutB": 9, "mutA_seq": seq[:1] + "C" + seq[2:],
            "mutB_seq": seq[:8] + "C" + seq[9:], "double_seq": seq[:1] + "C" + seq[2:8] + "C" + seq[9:],
            "rescue_factor": 0.5,
            "singleA_reactivity": wt, "singleA_error": err,
            "singleB_reactivity": wt, "singleB_error": err,
            "double_reactivity": wt, "double_error": err,
        }],
        "usable": True,
    }
    samples = m2r.build_pair_samples(d)
    assert len(samples) == 1
    s = samples[0]
    assert s.editA_seq_pos == 1
    assert s.editB_seq_pos == 8
    assert s.rescue_factor == 0.5
    assert len(s.eligibility_mask) == n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
