#!/usr/bin/env python3
"""Unit fixtures for generate_p6_tables_figures_v1.

Covers:
  - build_tables: maps result artifacts into the four main tables with correct
    counts and pooled statistics.
  - render_markdown / render_latex: emits table headers and no placeholder text.
"""

from __future__ import annotations

import scripts.reactflow_delta.generate_p6_tables_figures_v1 as G


def _p2():
    return {"verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT",
            "per_puzzle_D_p2": {f"P{i:02d}": 0.01 + 0.001 * i for i in range(1, 21)},
            "p2_ci20": {"mean": 0.0127, "ci_low": 0.0079, "ci_high": 0.0175},
            "sign_flip": {"p_value": 1.9e-6}}


def _p3():
    return {"verdict": {"2": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT",
                        "4": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT",
                        "8": "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT"}}


def _horizontal():
    return {"method_table": [
        {"method": "reg_direct", "mean_held_crps": 0.2023, "skill_vs_zero_pct": 5.92},
        {"method": "zero", "mean_held_crps": 0.2150, "skill_vs_zero_pct": 0.0}]}


def _p4():
    return {"verdict": "P4_EXTERNAL_STATISTICAL_PASS",
            "component_rows": [{"wt_name": "WT_A", "n_scored": 100, "n_positions": 5000,
                                "D_vs_zero": 0.03, "D_vs_median": 0.02}],
            "ci_zero": {"mean": 0.0410, "ci_low": 0.0153, "ci_high": 0.0667},
            "ci_median": {"mean": 0.0446, "ci_low": 0.0112, "ci_high": 0.0781},
            "holm_bonferroni_p_values": [0.0015, 0.0055], "fwer_pass": True,
            "leave_dominant_out_ci": {"mean": 0.0327, "ci_low": 0.0127, "ci_high": 0.0527}}


def _p5():
    return {"verdict": "MECHANISM_NOT_ESTABLISHED",
            "band_stats": {"edit_site": {"mean": 0.0311, "ci_low": 0.0056, "ci_high": 0.0567}},
            "distance_heterogeneity": {"D_edit_minus_vfar": {"mean": -0.009, "ci_low": -0.0199, "ci_high": 0.0019}},
            "negative_control": {"permuted_edit_D": {"mean": -0.1107, "ci_high": -0.0624}, "pass": True},
            "region_strata": {"M2SL5_2A3_0000": {"n_components": 3, "mean_D_edit": -0.02}}}


def _calib():
    return {"verdict": "CALIBRATION_ACCEPTABLE",
            "pooled": {"cov_68": 0.699, "cov_95": 0.874}}


def _replay():
    return {"verdict": "REPLAY_CONSISTENT"}


def test_build_tables_counts_and_stats():
    tables = G.build_tables(_p2(), _p3(), _horizontal(), _p4(), _p5(), _calib(), _replay(),
                            {"P0": "PASS"})
    assert "table1_development_horizontal" in tables
    assert len(tables["table1_development_horizontal"]["rows"]) == 2
    assert len(tables["table2_p4_external"]["rows"]) == 1
    assert "P4_EXTERNAL_STATISTICAL_PASS" in tables["table2_p4_external"]["note"]
    assert "MECHANISM_NOT_ESTABLISHED" in tables["table3_p5_mechanism"]["note"]
    assert tables["table4_gates"]["rows"][0] == {"phase": "P0", "verdict": "PASS"}


def test_render_markdown_no_placeholder():
    tables = G.build_tables(_p2(), _p3(), _horizontal(), _p4(), _p5(), _calib(), _replay(), {})
    md = G.render_markdown(tables)
    assert "| method |" in md
    assert "0.2023" in md
    assert "TODO" not in md and "placeholder" not in md.lower()


def test_render_latex():
    tables = G.build_tables(_p2(), _p3(), _horizontal(), _p4(), _p5(), _calib(), _replay(), {})
    tex = G.render_latex(tables)
    assert "\\begin{document}" in tex
    assert "\\toprule" in tex
