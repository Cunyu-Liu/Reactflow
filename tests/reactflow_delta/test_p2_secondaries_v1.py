#!/usr/bin/env python3
"""Fixtures for the P2 mandatory secondaries analyzer (contract 9.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.analyze_p2_secondaries_v1 import mae, main, _crps_gaussian, _mutant_puzzle_macro


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _synthetic_rows(n: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.RandomState(seed)
    rows = []
    regions = ["design_region", "other_assay_region", "flank"]
    for k in range(n):
        dist = int(k % 21)
        target = float(rng.normal(0.0, 0.3))
        # direct pred slightly better than zero (lower |resid|)
        pred_direct = float(target + rng.normal(0.0, 0.15))
        rows.append({
            "puzzle": f"P{k % 20 + 1:02d}", "method": "M2", "construct": f"C{k}",
            "edit_pos": 10, "ref": "A", "alt": "G", "pos": k,
            "region": regions[k % len(regions)], "dist": dist,
            "wt": float(rng.normal(0.0, 0.3)), "target": target,
            "pred_direct": pred_direct,
            "pred_zero": float(target + rng.normal(0.0, 0.3)),
        })
    return rows


def test_mae_masks_nan():
    pred = np.array([1.0, np.nan, 3.0])
    tgt = np.array([0.0, 2.0, 5.0])
    assert mae(pred, tgt) == pytest.approx((1.0 + 2.0) / 2.0)


def test_analyzer_report_schema_and_sections(tmp_path):
    rows = _synthetic_rows()
    rows_path = tmp_path / "rows.jsonl"
    out_path = tmp_path / "report.json"
    _write_rows(rows_path, rows)
    main(["--rows-jsonl", str(rows_path), "--out", str(out_path)])

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "reactflow_delta.p2_secondaries.v1"
    assert report["n_position_rows"] == len(rows)

    # region breakdown covers all present regions
    assert set(report["region_mae"]) == {"design_region", "other_assay_region", "flank"}
    for v in report["region_mae"].values():
        assert v["n"] > 0
        assert "mae_direct" in v and "mae_zero" in v

    # signed-delta region breakdown present with wmae
    assert set(report["region_signed_delta"]) == {"design_region", "other_assay_region", "flank"}
    for v in report["region_signed_delta"].values():
        assert v["n"] > 0
        for k in ("mae_direct", "mae_zero", "wmae_direct", "wmae_zero"):
            assert k in v

    # distance bands: edit site (dist 0) and >=20 present
    assert "0_edit_site" in report["distance_mae"]
    assert "20plus_far" in report["distance_mae"]
    # dist 21 rows roll into 20plus_far; dist <= 19 split correctly
    assert sum(report["distance_mae"][b]["n"] for b in report["distance_mae"]) == len(rows)

    # overall signed delta mandatory secondary present and finite
    od = report["signed_delta_overall"]
    assert od["n"] > 0
    for k in ("mae_direct", "mae_zero", "wmae_direct", "wmae_zero"):
        assert not math.isnan(od[k])
    assert "skill_mae_pct" in od and "skill_wmae_pct" in od

    # calibration diagnostic present and finite
    cd = report["calibration_diagnostic"]
    assert cd["nominal_gaussian_scale"] == 0.3
    assert not math.isnan(cd["empirical_residual_sd"])
    assert 0.0 <= cd["nominal_95pct_coverage_at_fixed_scale"] <= 1.0

    # CRPS scale-sensitivity: monotone decrease of scale entries present
    ss = report["crps_scale_sensitivity"]
    assert len(ss["scales"]) == 8
    for srow in ss["scales"]:
        for k in ("scale", "crps_direct_macro", "crps_zero_macro", "D_p_macro_mean"):
            assert not math.isnan(srow[k])
    assert "reconciliation_note" in report


def test_analyzer_handles_missing_band(tmp_path):
    rows = _synthetic_rows(n=40)
    # force only small distances so some bands are absent
    for r in rows:
        r["dist"] = int(r["pos"] % 5)
    rows_path = tmp_path / "rows.jsonl"
    out_path = tmp_path / "report.json"
    _write_rows(rows_path, rows)
    main(["--rows-jsonl", str(rows_path), "--out", str(out_path)])
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert "5_9_mid" not in report["distance_mae"]
    assert "20plus_far" not in report["distance_mae"]


def test_analyzer_direct_better_than_zero_synthetic(tmp_path):
    rng = np.random.RandomState(1)
    rows = []
    for k in range(200):
        target = float(rng.normal(0.0, 0.3))
        rows.append({
            "puzzle": "P01", "method": "M2", "construct": f"C{k}",
            "edit_pos": 10, "ref": "A", "alt": "G", "pos": k,
            "region": "design_region", "dist": 3, "wt": 0.0, "target": target,
            "pred_direct": float(target + rng.normal(0.0, 0.1)),
            "pred_zero": float(target + rng.normal(0.0, 0.4)),
        })
    rows_path = tmp_path / "rows.jsonl"
    out_path = tmp_path / "report.json"
    _write_rows(rows_path, rows)
    main(["--rows-jsonl", str(rows_path), "--out", str(out_path)])
    report = json.loads(out_path.read_text(encoding="utf-8"))
    v = report["region_mae"]["design_region"]
    assert v["mae_direct"] < v["mae_zero"]


def test_signed_delta_excludes_missing_wt_anchor(tmp_path):
    rows = _synthetic_rows(n=30)
    rows[0]["wt"] = None  # WT anchor missing => signed delta must exclude this row
    rows_path = tmp_path / "rows.jsonl"
    out_path = tmp_path / "report.json"
    _write_rows(rows_path, rows)
    main(["--rows-jsonl", str(rows_path), "--out", str(out_path)])
    report = json.loads(out_path.read_text(encoding="utf-8"))
    # mutant-reactivity rows count keeps all 30; signed-delta only 29
    assert report["n_position_rows"] == 30
    assert report["signed_delta_overall"]["n"] == 29


def test_crps_gaussian_matches_scalar():
    # vectorized form must match the exact scalar energy form
    from scipy import stats as _st
    loc = np.array([0.0, 1.0])
    y = np.array([0.3, -0.5])
    v = _crps_gaussian(loc, 0.3, y)
    for i in range(2):
        m = y[i] - loc[i]
        e_abs = (0.3 * np.sqrt(2.0 / np.pi) * np.exp(-m * m / (2 * 0.3 * 0.3))
                 + m * (2 * _st.norm.cdf(m / 0.3) - 1))
        assert v[i] == pytest.approx(e_abs - 0.3 / np.sqrt(np.pi))


def test_mutant_puzzle_macro_positions_then_mutants_then_puzzles():
    # per-position CRPS grouped by mutant then puzzle; verify macro weights
    # each mutant equally regardless of its number of positions
    values = np.array([1.0, 1.0, 5.0])  # mutant A: 2 pos (mean 1.0), mutant B: 1 pos (5.0)
    mid = np.array(["P1|A|0|A|G", "P1|A|0|A|G", "P1|B|1|C|T"])
    puz = np.array(["P1", "P1", "P1"])
    macro, mut_mean = _mutant_puzzle_macro(values, mid, puz)
    assert mut_mean[0] == pytest.approx(1.0)  # mutant A mean over 2 positions
    assert mut_mean[1] == pytest.approx(5.0)  # mutant B
    assert macro["P1"] == pytest.approx(3.0)  # (1.0 + 5.0) / 2, NOT (1+1+5)/3
