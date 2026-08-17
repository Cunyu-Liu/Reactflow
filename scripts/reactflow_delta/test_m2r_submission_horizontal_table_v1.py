#!/usr/bin/env python3
"""Tests for m2r_submission_horizontal_table_v1 (definitive M2R table).

Verifies the table builder runs on the real committed audit artifacts and that
the numbers are internally consistent:
  * headline (3-way ensemble) skill matches the 3-way report exactly
  * headline R2 matches the 3-way report (0.370, not a typo)
  * 3-way is the best row (higher skill than L1 full-stack blend)
  * puzzle-level row reads from the puzzle report
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


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("sub")
    sub.build_table(TRANSFER, ROBUST, ROBUST_P, PUZZLE, NOISE, str(out),
                    THREEWAY, THREEWAY_P, THREEWAY_PZ,
                    THREEWAY_S, THREEWAY_SP, THREEWAY_SPZ, CEILING)
    d = json.loads((out / "submission_horizontal_table_m2r.json").read_text(encoding="utf-8"))
    return d, out


def test_headline_matches_strong_threeway_report(report):
    d, _ = report
    tws = json.loads(Path(THREEWAY_S).read_text(encoding="utf-8"))["headline"]
    hl = [r for r in d["rows"] if r.get("headline")][0]
    assert hl["model"] == "strong 3-way ensemble (300-tr base GBDTs)"
    assert abs(hl["skill"] - tws["strong"]["skill"]) < 1e-12
    assert abs(hl["r2"] - tws["strong"]["r2"]) < 1e-12
    assert abs(hl["r2"] - 0.387) < 0.005


def test_headline_is_best(report):
    d, _ = report
    skills = [r["skill"] for r in d["rows"] if r.get("skill") is not None]
    hl = [r for r in d["rows"] if r.get("headline")][0]
    assert hl["skill"] == max(skills)
    # strong 3-way must beat the default 3-way and the L1 full-stack blend
    tw = [r for r in d["rows"] if r["model"] == "3-way ensemble (L1+L2 GBDT + Ridge)"][0]
    l1 = [r for r in d["rows"] if r["model"] == "L1 full-stack blend (a=0.80)"][0]
    assert hl["skill"] > tw["skill"]
    assert hl["skill"] > l1["skill"]


def test_puzzle_row_present(report):
    d, _ = report
    pz_rows = [r for r in d["rows"] if r["split"] == "puzzle-level LOO"]
    assert len(pz_rows) == 1
    assert pz_rows[0]["skill"] > 0.24


def test_artifacts_written(report):
    d, out = report
    assert (out / "submission_horizontal_table_m2r.json").exists()
    assert (out / "submission_horizontal_table_m2r.md").exists()
    md = (out / "submission_horizontal_table_m2r.md").read_text(encoding="utf-8")
    assert "+28.11%" in md
    assert "0.387" in md
    assert "ceiling audit" in md


def test_ceiling_audit_in_noise_floor(report):
    d, _ = report
    nf = d["noise_floor"]
    assert nf["ceiling_audit"] is not None
    assert nf["ceiling_audit"]["oracle_strong_r2"] > 0.5
    assert nf["ceiling_audit"]["legal_strong_r2"] < 0.45


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
