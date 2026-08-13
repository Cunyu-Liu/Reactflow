#!/usr/bin/env python3
"""Fixtures for the P2 mandatory secondaries analyzer (contract 9.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.analyze_p2_secondaries_v1 import mae, main


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

    # distance bands: edit site (dist 0) and >=20 present
    assert "0_edit_site" in report["distance_mae"]
    assert "20plus_far" in report["distance_mae"]
    # dist 21 rows roll into 20plus_far; dist <= 19 split correctly
    assert sum(report["distance_mae"][b]["n"] for b in report["distance_mae"]) == len(rows)

    # calibration diagnostic present and finite
    cd = report["calibration_diagnostic"]
    assert cd["nominal_gaussian_scale"] == 0.3
    assert not math.isnan(cd["empirical_residual_sd"])
    assert 0.0 <= cd["nominal_95pct_coverage_at_fixed_scale"] <= 1.0


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
