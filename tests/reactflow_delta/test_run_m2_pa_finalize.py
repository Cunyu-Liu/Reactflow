#!/usr/bin/env python3
"""test_run_m2_pa_finalize — unit tests for the position-aware finalize driver's
report summarizers (_hrow/_drow), ensuring the integrated method summary rows are
correctly extracted from horizontal and deviation reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import run_m2_pa_finalize as fin


def _mk_horizontal(skill=0.10, n_designs=159, pct=1.0):
    return {
        "mu_ensemble": {
            "skill": skill, "ci_low": skill - 0.01, "ci_high": skill + 0.01,
            "permutation_p": 0.0033, "n_designs": n_designs, "n_positions": 270000,
            "per_design": {"pct_positive": pct},
            "wmae_model": 0.60, "wmae_baseline": 0.67,
        }
    }


def _mk_deviation(rho=0.40, auroc=0.70):
    return {
        "mu_ensemble": {
            "spearman_abs": rho, "auroc_abs": auroc, "permutation_p": 0.0033,
            "n_designs": 159, "n_positions": 270000,
            "per_design": {"mean": rho + 0.005, "pct_positive": 1.0},
            "robustness": {"pooled_rho_min_over_loo": rho - 0.002},
        },
        "per_position": [{"position": 10, "spearman_abs": rho + 0.03}],
    }


def test_hrow_extracts_summary_row():
    row = fin._hrow(_mk_horizontal(skill=0.101, n_designs=159))
    assert row["skill"] == 0.101
    assert row["n_designs"] == 159
    assert row["pct_positive"] == 1.0
    assert "ci_low" in row and "wmae_model" in row


def test_drow_extracts_summary_row():
    row = fin._drow(_mk_deviation(rho=0.409, auroc=0.701))
    assert row["spearman_rho"] == 0.409
    assert row["auroc"] == 0.701
    assert row["loo_rho_min"] == 0.407
    assert row["pos10_rho"] == 0.439
    assert row["pct_positive_designs"] == 1.0


def test_summary_roundtrip(tmp_path):
    """The two summarizers produce a coherent method-summary JSON."""
    summary = {
        "wmae_skill": {"plain_residual_mlp": fin._hrow(_mk_horizontal(0.0888)),
                       "position_aware": fin._hrow(_mk_horizontal(0.1010))},
        "deviation_detection": {"plain_residual_mlp": fin._drow(_mk_deviation(0.3767, 0.686)),
                                "position_aware": fin._drow(_mk_deviation(0.4091, 0.701))},
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["wmae_skill"]["position_aware"]["skill"] == 0.101
    assert back["wmae_skill"]["position_aware"]["skill"] > \
           back["wmae_skill"]["plain_residual_mlp"]["skill"]
    assert back["deviation_detection"]["position_aware"]["spearman_rho"] > \
           back["deviation_detection"]["plain_residual_mlp"]["spearman_rho"]
