#!/usr/bin/env python3
"""Tests for m2r_submission_horizontal_table_v1 (definitive M2R table).

Verifies the table builder runs on the real committed audit artifacts and that
the numbers are internally consistent:
  * headline (multi-seed strong 3-way + v2 + MFE) skill/R2 match the MFE
    multi-seed report exactly
  * headline is the best row (higher skill than strong 3-way + v2, multi-seed,
    and the L1 full-stack blend)
  * puzzle-level rows read from the puzzle reports (incl. the MFE puzzle row)
  * output JSON + MD both written
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_submission_horizontal_table_v1 as sub

BASE = "/mnt/cunyuliu"
TRANSFER = f"{BASE}/m2r_transfer_20260816/m2r_transfer_report.json"
ROBUST = f"{BASE}/m2r_robust_objective_20260817/m2r_robust_objective_report.json"
ROBUST_P = f"{BASE}/m2r_robust_objective_20260817/m2r_robust_permtest.json"
PUZZLE = f"{BASE}/m2r_transfer_puzzle_20260817/m2r_transfer_puzzle_report.json"
NOISE = f"{BASE}/m2r_noise_floor_20260817/m2r_noise_floor.json"
THREEWAY = f"{BASE}/m2r_3way_ensemble_20260817/m2r_3way_ensemble_report.json"
THREEWAY_P = f"{BASE}/m2r_3way_ensemble_20260817/m2r_3way_permtest.json"
THREEWAY_PZ = f"{BASE}/m2r_3way_puzzle_20260817/m2r_3way_puzzle_report.json"
THREEWAY_S = f"{BASE}/m2r_3way_strong_20260817/m2r_3way_strong_report.json"
THREEWAY_SP = f"{BASE}/m2r_3way_strong_20260817/m2r_3way_strong_permtest.json"
THREEWAY_SPZ = f"{BASE}/m2r_3way_strong_puzzle_20260817/m2r_3way_strong_puzzle_report.json"
CEILING = f"{BASE}/m2r_ceiling_audit_lean_20260817/m2r_ceiling_audit_report.json"
V2 = f"{BASE}/m2r_features_v2_ablation_20260817/m2r_features_v2_ablation_report.json"
V2_P = f"{BASE}/m2r_features_v2_ablation_20260817/m2r_features_v2_permtest.json"
V2_PZ = f"{BASE}/m2r_features_v2_ablation_20260817/m2r_features_v2_puzzle_report.json"
MULTISEED = f"{BASE}/m2r_multiseed_20260817/m2r_multiseed_report.json"
MULTISEED_PZ = f"{BASE}/m2r_multiseed_puzzle_20260817/m2r_multiseed_puzzle_report.json"
MFE = f"{BASE}/m2r_mfe_multiseed_20260818/m2r_mfe_multiseed_report.json"
MFE_PZ = f"{BASE}/m2r_mfe_puzzle_20260818/m2r_mfe_puzzle_report.json"


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("sub")
    sub.build_table(TRANSFER, ROBUST, ROBUST_P, PUZZLE, NOISE, str(out),
                    THREEWAY, THREEWAY_P, THREEWAY_PZ,
                    THREEWAY_S, THREEWAY_SP, THREEWAY_SPZ, CEILING,
                    V2, V2_P, V2_PZ,
                    multiseed_report=MULTISEED,
                    multiseed_puzzle_report=MULTISEED_PZ,
                    mfe_report=MFE,
                    mfe_puzzle_report=MFE_PZ)
    d = json.loads((out / "submission_horizontal_table_m2r.json").read_text(encoding="utf-8"))
    return d, out


def test_headline_matches_mfe_report(report):
    d, _ = report
    mfeR = json.loads(Path(MFE).read_text(encoding="utf-8"))["results"]["mfe_multiseed_3way"]
    hl = [r for r in d["rows"] if r.get("headline")][0]
    assert hl["model"] == "multi-seed strong 3-way + v2 + MFE (NEW headline)"
    assert abs(hl["skill"] - mfeR["skill"]) < 1e-12
    assert abs(hl["r2"] - mfeR["r2"]) < 1e-12
    assert abs(hl["r2"] - 0.4178) < 0.005
    assert abs(hl["skill"] - 0.3033) < 0.005


def test_headline_is_best(report):
    d, _ = report
    skills = [r["skill"] for r in d["rows"] if r.get("skill") is not None]
    hl = [r for r in d["rows"] if r.get("headline")][0]
    assert hl["skill"] == max(skills)
    # MFE headline must beat the strong 3-way + v2 (258 dims), multi-seed (258)
    # and the L1 full-stack blend
    v2 = [r for r in d["rows"] if r["model"] == "strong 3-way + v2 features (NEW headline)"][0]
    ms = [r for r in d["rows"] if r["model"].startswith("multi-seed strong 3-way + v2 (K=5")][0]
    l1 = [r for r in d["rows"] if r["model"] == "L1 full-stack blend (a=0.80)"][0]
    assert hl["skill"] > v2["skill"] > l1["skill"]
    assert hl["skill"] > ms["skill"]


def test_puzzle_rows_present(report):
    d, _ = report
    pz_rows = [r for r in d["rows"] if r["split"] == "puzzle-level LOO"]
    # L2 blend, strong 3-way + v2, multi-seed, MFE
    assert len(pz_rows) == 4
    assert all(r["skill"] > 0.24 for r in pz_rows)
    mfe_pz = [r for r in pz_rows if "MFE" in r["model"]][0]
    mfeR_pz = json.loads(Path(MFE_PZ).read_text(encoding="utf-8"))["results"]["mfe_multiseed_3way"]
    assert abs(mfe_pz["skill"] - mfeR_pz["skill"]) < 1e-12


def test_artifacts_written(report):
    d, out = report
    assert (out / "submission_horizontal_table_m2r.json").exists()
    assert (out / "submission_horizontal_table_m2r.md").exists()
    md = (out / "submission_horizontal_table_m2r.md").read_text(encoding="utf-8")
    assert "+30.33%" in md
    assert "0.418" in md
    assert "ceiling audit" in md


def test_ceiling_audit_in_noise_floor(report):
    d, _ = report
    nf = d["noise_floor"]
    assert nf["ceiling_audit"] is not None
    assert nf["ceiling_audit"]["oracle_strong_r2"] > 0.5
    assert nf["ceiling_audit"]["legal_strong_r2"] < 0.45


def test_fail_closed_includes_mfe_extensions(report):
    d, _ = report
    fca = d["claim_matrix"]["fail_closed_audited"]
    joined = " | ".join(fca)
    assert "BPP partition-function" in joined
    assert "SHAPE-guided" in joined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
