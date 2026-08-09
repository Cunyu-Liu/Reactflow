#!/usr/bin/env python3
"""Unit tests for the keyed prediction schema v2 (fixes Scheme-3 positional misalignment).

Covers the prediction_v2.schema.json contract as enforced by
validate_prediction_artifact_v2.validate_rows:
  * every row carries biological keys + provenance hashes;
  * (pair_id, fold_id, seed, model_variant) uniqueness;
  * raw vs transformed prediction kept separate;
  * tool failure / unsupported / missing / abstention are explicit coverage
    statuses, never coerced to zero prediction;
  * row-count semantics: n_rows == n_unique_pairs x n_seeds per model_variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))
sys.path.insert(0, str(_ROOT / "src"))

import pytest
import validate_prediction_artifact_v2 as vp


def _row(pair_id="p1", fold="f0", seed=0, mv="cand", i=0, cov="CALLED"):
    return {
        "pair_id": pair_id, "asset_id": "a", "study_id": "s",
        "publication_id": "pub:P1", "parent_id": "pa", "lineage_id": "li",
        "fold_id": fold, "split_role": "development",
        "endpoint_version": "endpoint_v6", "caller_version": "caller_v4",
        "seed": seed, "model_id": "m", "model_variant": mv,
        "y": float(i % 2), "weight": 1.0, "raw_prediction": 0.5,
        "transformed_prediction": 0.5, "coverage_status": cov,
        "data_hash": "d", "split_hash": "s", "caller_hash": "c",
        "model_config_hash": "mc", "source_commit": "sc",
    }


def _candidate_baseline(n_pair=6, n_seed=2):
    rows = []
    for mv in ("cand", "base"):
        for s in range(n_seed):
            for i in range(n_pair):
                rows.append(_row(pair_id=f"p{i}", seed=s, mv=mv, i=i))
    return rows


def test_missing_required_field_fails():
    rows = [_row()]
    del rows[0]["publication_id"]
    with pytest.raises(ValueError, match="MISSING_FIELD"):
        vp.validate_rows(rows)


def test_empty_artifact_fails():
    with pytest.raises(ValueError, match="EMPTY_PREDICTION_ARTIFACT"):
        vp.validate_rows([])


def test_duplicate_key_fails():
    rows = [_row(pair_id="p1", fold="f0", seed=0, mv="cand"),
            _row(pair_id="p1", fold="f0", seed=0, mv="cand")]
    with pytest.raises(ValueError, match="DUPLICATE_KEY_FIELDS"):
        vp.validate_rows(rows)


def test_same_pair_same_seed_different_variant_ok():
    rows = [_row(pair_id="p1", seed=0, mv="cand"),
            _row(pair_id="p1", seed=0, mv="base")]
    m = vp.validate_rows(rows)
    assert m["models"] == ["base", "cand"]


def test_invalid_coverage_status_fails():
    rows = [_row(cov="MADE_UP")]
    with pytest.raises(ValueError, match="INVALID_COVERAGE"):
        vp.validate_rows(rows)


def test_invalid_split_role_fails():
    rows = [_row()]
    rows[0]["split_role"] = "test"
    with pytest.raises(ValueError, match="INVALID_SPLIT_ROLE"):
        vp.validate_rows(rows)


def test_wrong_endpoint_version_fails():
    rows = [_row()]
    rows[0]["endpoint_version"] = "endpoint_v5"
    with pytest.raises(ValueError, match="ENDPOINT_MISMATCH"):
        vp.validate_rows(rows)


def test_wrong_caller_version_fails():
    rows = [_row()]
    rows[0]["caller_version"] = "caller_v3"
    with pytest.raises(ValueError, match="CALLER_MISMATCH"):
        vp.validate_rows(rows)


def test_noncall_with_value_fails():
    # TOOL_FAILURE / ABSTAIN / etc. must carry raw_prediction=None, never 0
    rows = [_row(cov="TOOL_FAILURE")]
    rows[0]["raw_prediction"] = 0.0
    with pytest.raises(ValueError, match="COVERAGE_NONCALL_WITH_VALUE"):
        vp.validate_rows(rows)


@pytest.mark.parametrize("cov", ["NO_CALL", "ABSTAIN", "UNSUPPORTED", "MISSING", "TOOL_FAILURE"])
def test_noncall_none_raw_prediction_ok(cov):
    rows = [_row(cov=cov)]
    rows[0]["raw_prediction"] = None
    rows[0]["transformed_prediction"] = None
    m = vp.validate_rows(rows)
    assert m["coverage_counts"][cov] == 1


def test_row_count_semantics_consistent():
    rows = _candidate_baseline(n_pair=6, n_seed=2)
    m = vp.validate_rows(rows, expect_seeds=2)
    assert m["row_count"] == 24
    assert m["row_count_semantics"]["cand"]["consistent"] is True


def test_row_count_semantics_inconsistent_fails():
    rows = _candidate_baseline(n_pair=6, n_seed=2)
    # drop one cand row -> cand rows != pairs*seeds
    cand_idx = next(i for i, r in enumerate(rows) if r["model_variant"] == "cand")
    del rows[cand_idx]
    with pytest.raises(ValueError, match="ROW_COUNT_SEMANTICS_FAIL"):
        vp.validate_rows(rows)


def test_seed_count_mismatch_fails():
    rows = _candidate_baseline(n_pair=6, n_seed=2)
    with pytest.raises(ValueError, match="SEED_COUNT_MISMATCH"):
        vp.validate_rows(rows, expect_seeds=3)


def test_seed_duplication_does_not_increase_uniqueness():
    # same pair repeated under same seed for same variant is a duplicate
    rows = [(lambda r: (r.update(seed=0), r)[1])(_row(pair_id="p1", mv="cand"))
            for _ in range(2)]
    with pytest.raises(ValueError, match="DUPLICATE_KEY_FIELDS"):
        vp.validate_rows(rows)


def test_manifest_counts():
    rows = _candidate_baseline(n_pair=6, n_seed=2)
    m = vp.validate_rows(rows)
    assert m["unique_pairs"] == 6
    assert m["unique_seeds"] == 2
    assert m["models"] == ["base", "cand"]