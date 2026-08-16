#!/usr/bin/env python3
"""Unit fixtures for build_p6_cards_v1 (P6 code/data/model cards)."""

from __future__ import annotations

import scripts.reactflow_delta.build_p6_cards_v1 as G


def _p2():
    return {"verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT",
            "p2_ci20": {"mean": 0.0127, "ci_low": 0.0079, "ci_high": 0.0175}}


def _p3():
    return {"verdict": {"2": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT",
                        "4": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT",
                        "8": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT"}}


def _horizontal():
    return {"method_table": [
        {"method": "reg_direct", "mean_held_crps": 0.2023, "skill_vs_zero_pct": 5.92},
        {"method": "zero", "mean_held_crps": 0.2150, "skill_vs_zero_pct": 0.0}]}


def _p4():
    return {"verdict": "P4_EXTERNAL_STATISTICAL_PASS", "K_preaccess": 24,
            "K_preaccess_single_snv": 3237,
            "ci_zero": {"mean": 0.0410, "ci_low": 0.0153, "ci_high": 0.0667}}


def _p5():
    return {"verdict": "MECHANISM_NOT_ESTABLISHED"}


def _calib():
    return {"verdict": "CALIBRATION_ACCEPTABLE",
            "pooled": {"empirical_residual_sd": 0.61, "cov_95": 0.874}}


def _replay():
    return {"verdict": "REPLAY_CONSISTENT"}


def test_build_cards_fields():
    cards = G.build_cards(_p2(), _p3(), _horizontal(), _p4(), _p5(), _calib(),
                          _replay(), {"numpy": "1.26.4"}, {"remote": "r", "branch": "b", "head": "h"})
    assert "model_card" in cards and "data_card" in cards and "code_card" in cards
    mc = cards["model_card"]
    assert mc["dev_performance"]["mean_held_crps"] == 0.2023
    assert "P4_EXTERNAL_STATISTICAL_PASS" in mc["external_performance"]["p4_verdict"]
    assert mc["mechanism_verdict"] == "MECHANISM_NOT_ESTABLISHED"
    dc = cards["data_card"]
    assert dc["external"]["components"] == 24
    assert dc["external"]["single_snv"] == 3237
    cc = cards["code_card"]
    assert "run_p4_external_v1.py" in cc["entrypoints"]["P4"]


def test_render_env():
    out = G.render_env({"numpy": "1.26.4", "python": "3.10"})
    assert "name: editflow" in out
    assert "numpy=1.26.4" in out
