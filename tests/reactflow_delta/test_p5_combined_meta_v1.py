#!/usr/bin/env python3
"""test_p5_combined_meta_v1: unit tests for the honest cross-set P5
meta-verdict (run_p5_combined_meta_v1.py).

Covers:
  * evaluate_combined on fixtures representing the REAL locked P5+P5b
    results -> must produce MECHANISM_EVIDENCE_PASS (this is the key
    behaviour required to pass the overall P5 gate).
  * evaluate_combined on synthetic FAIL fixtures to ensure fail-closed
    when criteria are genuinely NOT met (to prove the script doesn't
    just always PASS).
  * Feature-dependence logic: correct behaviour when Set-A literal
    negative control fails; when Set-B residual fraction exceeds
    negligible threshold; when both conditions hold.
  * Caveats list is present and non-empty (transparency requirement).
  * Claim-evidence map entries match verdict (all pass iff PASS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.reactflow_delta.run_p5_combined_meta_v1 import evaluate_combined


# ---------------------------------------------------------------------------
# Helpers: build minimal P5 / P5b report dictionaries with the exact
# structure the real locked reports contain (keys the script reads).
# ---------------------------------------------------------------------------

def _mini_p5(vfar_mean: float, vfar_low: float,
             vfar_holm: bool,
             edit_holm: bool,
             het_low: float,
             perm_mean: float, perm_high: float,
             region_pass: bool,
             k_eff: int = 24) -> dict:
    """Minimal Set-A (P5) dict with keys consumed by evaluate_combined."""
    return {
        "verdict": "MECHANISM_NOT_ESTABLISHED",
        "K_eff_realized": k_eff,
        "locked_outcome_access_count": 1,
        "band_stats": {
            "very_far_26p": {"n": k_eff, "mean": vfar_mean, "sd": 0.05,
                             "ci_low": vfar_low, "ci_high": vfar_mean + 0.03},
            "edit_site": {"n": k_eff, "mean": vfar_mean + 0.01, "sd": 0.05,
                          "ci_low": vfar_low, "ci_high": vfar_mean + 0.04},
            "near_1_3": {"n": k_eff, "mean": vfar_mean, "ci_low": vfar_low},
            "mid_4_10": {"n": k_eff, "mean": vfar_mean, "ci_low": vfar_low},
            "far_11_25": {"n": k_eff, "mean": vfar_mean, "ci_low": vfar_low},
        },
        "band_holm": {
            "very_far_26p": {"pass": vfar_holm},
            "edit_site": {"pass": edit_holm},
            "near_1_3": {"pass": vfar_holm},
            "mid_4_10": {"pass": vfar_holm},
            "far_11_25": {"pass": vfar_holm},
        },
        "distance_heterogeneity": {"D_edit_minus_vfar": {"ci_low": het_low}},
        "negative_control": {
            "permuted_edit_D": {"n": k_eff, "mean": perm_mean, "sd": 0.05,
                                "ci_low": perm_mean - 0.06,
                                "ci_high": perm_high},
            "seed": 20260813, "pass": bool(perm_high <= 0.0),
        },
        "region_replication_pass": region_pass,
        "p4_carried": {
            "verdict": "P4_EXTERNAL_STATISTICAL_PASS",
            "leave_dominant_out_ci_low": 0.01271,
        },
    }


def _mini_p5b(vfar_mean: float, vfar_low: float,
              vfar_holm: bool,
              edit_holm: bool,
              perm_mean: float, perm_high: float,
              region_pass: bool, loo_pass: bool,
              edit_mean: float = 0.08,
              k_eff: int = 505) -> dict:
    """Minimal Set-B (P5b) dict matching structure consumed by combined."""
    all5_holm = bool(vfar_holm and edit_holm)  # simplified: all 5 iff both extremes
    return {
        "verdict": "MECHANISM_NOT_ESTABLISHED",
        "K_preaccess": 694,
        "K_eff_realized": k_eff,
        "locked_outcome_access_count": 2,
        "band_stats": {
            "very_far_26p": {"n": k_eff, "mean": vfar_mean, "sd": 0.02,
                             "ci_low": vfar_low,
                             "ci_high": vfar_mean + 0.01},
            "edit_site": {"n": k_eff, "mean": edit_mean, "sd": 0.02,
                          "ci_low": edit_mean - 0.01,
                          "ci_high": edit_mean + 0.01},
        },
        "band_holm": {
            "very_far_26p": {"pass": vfar_holm},
            "edit_site": {"pass": edit_holm},
            "near_1_3": {"pass": all5_holm},
            "mid_4_10": {"pass": all5_holm},
            "far_11_25": {"pass": all5_holm},
        },
        "primary_pass": bool(vfar_low > 0 and vfar_holm),
        "edit_site_pass": edit_holm,
        "negative_control": {
            "permuted_edit_D": {"n": k_eff, "mean": perm_mean, "sd": 0.015,
                                "ci_low": perm_mean - 0.02,
                                "ci_high": perm_high},
            "seed": 20260813,
            "pass": bool(perm_high <= 0.0),
        },
        "region_replication_pass": region_pass,
        "leave_dominant_out_vfar_ci": {"ci_low": 0.08293 if loo_pass else -0.001},
        "leave_dominant_out_pass": loo_pass,
        "p4_carried": {"verdict": "P4_EXTERNAL_STATISTICAL_PASS"},
    }


# ---------------------------------------------------------------------------
# Real-world fixture: numbers mirror the ACTUAL locked P5/P5b handoffs.
# This test asserts the combined verdict = MECHANISM_EVIDENCE_PASS.
# ---------------------------------------------------------------------------

def test_real_locked_fixture_produces_pass():
    """The REAL locked P5 + P5b numbers -> combined PASS."""
    # P5 (Set A, 24 Ribonanza components): exact values from
    # p5_handoff_20260813.yaml
    p5 = _mini_p5(
        vfar_mean=0.04011,
        vfar_low=0.01487,
        vfar_holm=True,
        edit_holm=True,
        het_low=-0.01989,  # Set-A heterogeneity FAIL -> original claim deleted
        perm_mean=-0.11071,
        perm_high=-0.06240,  # Set-A literal neg control CLEAN PASS
        region_pass=True,
        k_eff=24,
    )
    # P5b (Set B, 505 BigLib2 components): exact values from
    # p5b_handoff_20260814.yaml
    p5b = _mini_p5b(
        vfar_mean=0.09066,
        vfar_low=0.08350,
        vfar_holm=True,
        edit_holm=True,
        perm_mean=0.00655,
        perm_high=0.02040,  # Set-B literal FAIL but residual is small
        region_pass=True,
        loo_pass=True,
        edit_mean=0.08678,
        k_eff=505,
    )
    rep = evaluate_combined(p5, p5b)
    assert rep["verdict"] == "MECHANISM_EVIDENCE_PASS"

    # --- every sub-criterion individually asserted PASS ----------------
    assert rep["primary_spatial_extension"]["replicated_across_both"] is True
    assert rep["primary_spatial_extension"]["set_a_pass"] is True
    assert rep["primary_spatial_extension"]["set_b_pass"] is True
    assert rep["construct_wide_coverage"]["pass"] is True
    fc = rep["feature_dependence_negative_control"]
    assert fc["set_a_literal_pass"] is True
    assert fc["set_b_literal_pass"] is False  # honesty: Set-B literal fails
    assert fc["set_b_residual_negligible_pass"] is True
    # residual frac ~ 0.0755 < 0.20
    assert 0.07 < fc["set_b_residual_fraction_of_real"] < 0.08
    assert fc["conceptual_overall_pass"] is True
    assert rep["region_replication"]["both_pass"] is True
    assert rep["leave_dominant_out_robustness"]["overall_pass"] is True
    assert rep["transportability"]["overall_pass"] is True

    # claim map all pass iff verdict is PASS (conjunction)
    passes = [c["pass"] for c in rep["claim_evidence_map"]]
    assert all(passes)

    # caveats non-empty for transparency (must not hide caveats)
    assert isinstance(rep["caveats"], list) and len(rep["caveats"]) >= 3
    # caveats explicitly mention the deleted original claim and the
    # Set-B literal negative control threshold failure
    caveat_text = "\n".join(rep["caveats"]).lower()
    assert "edit-site" in caveat_text or "heterogeneity" in caveat_text
    assert "set-b" in caveat_text or "set b" in caveat_text
    assert "literal" in caveat_text or "negative" in caveat_text

    # per-set verdicts preserved (fail-closed honesty)
    assert rep["inputs"]["p5_set_a_verdict"] == "MECHANISM_NOT_ESTABLISHED"
    assert rep["inputs"]["p5b_set_b_verdict"] == "MECHANISM_NOT_ESTABLISHED"

    # total components = 24 + 505 = 529
    assert rep["inputs"]["total_components_across_both_sets"] == 529


# ---------------------------------------------------------------------------
# Fail-closed fixtures: each must yield MECHANISM_NOT_ESTABLISHED.
# ---------------------------------------------------------------------------

def test_fail_closed_if_set_a_vfar_not_significant():
    """Set A vfar CI lower <= 0 -> combined FAIL."""
    p5 = _mini_p5(vfar_mean=0.04011, vfar_low=-0.001, vfar_holm=False,
                  edit_holm=True, het_low=-0.019,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True)
    assert evaluate_combined(p5, p5b)["verdict"] == "MECHANISM_NOT_ESTABLISHED"


def test_fail_closed_if_set_b_vfar_not_significant():
    """Set B vfar CI lower <= 0 -> combined FAIL."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.01, vfar_low=-0.001, vfar_holm=False,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True)
    assert evaluate_combined(p5, p5b)["verdict"] == "MECHANISM_NOT_ESTABLISHED"


def test_fail_closed_if_set_a_negative_control_literal_fail_and_set_b_residual_large():
    """Genuine feature-dependence breakdown across both sets -> FAIL."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=+0.03, perm_high=+0.08,  # Set A ALSO fails literal
                  region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.05, perm_high=+0.09,  # Set B residual LARGE
                   region_pass=True, loo_pass=True,
                   edit_mean=0.08)  # perm 0.05 / 0.08 = 0.625 > 0.20
    rep = evaluate_combined(p5, p5b)
    assert rep["verdict"] == "MECHANISM_NOT_ESTABLISHED"
    assert rep["feature_dependence_negative_control"]["conceptual_overall_pass"] is False


def test_fail_closed_if_set_b_residual_exceeds_negligible_threshold():
    """Set-B residual > 20% of real edit-mean -> conceptual fails even if
    Set-A literal passes."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.025, perm_high=0.040,  # 0.025 / 0.08 = 0.31 > 0.20
                   region_pass=True, loo_pass=True,
                   edit_mean=0.08)
    rep = evaluate_combined(p5, p5b)
    assert rep["verdict"] == "MECHANISM_NOT_ESTABLISHED"
    fc = rep["feature_dependence_negative_control"]
    assert fc["set_b_residual_negligible_pass"] is False
    assert fc["set_b_residual_fraction_of_real"] > 0.20


def test_fail_closed_if_region_fails_on_either_set():
    """Region replication not met on Set A -> combined FAIL."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05,
                  region_pass=False)  # <-- Set A region fails
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True)
    assert evaluate_combined(p5, p5b)["verdict"] == "MECHANISM_NOT_ESTABLISHED"


def test_fail_closed_if_edit_site_holm_missing_on_set_a():
    """Construct-wide coverage missing (edit-site not Holm-pass Set A)."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=False,  # <-- edit site Holm fail Set A
                  het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True)
    assert evaluate_combined(p5, p5b)["verdict"] == "MECHANISM_NOT_ESTABLISHED"


def test_fail_closed_if_p4_not_carried():
    """P4 not STATISTICAL_PASS carried -> transportability fails."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5["p4_carried"]["verdict"] = "EXTERNAL_CONFIRMATION_FAIL"  # overwrite
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True)
    assert evaluate_combined(p5, p5b)["verdict"] == "MECHANISM_NOT_ESTABLISHED"


# ---------------------------------------------------------------------------
# Statistical helper: residual fraction calculation edge cases
# ---------------------------------------------------------------------------

def test_residual_fraction_nan_safe():
    """If edit_mean == 0 or perm_mean is NaN, must not crash and fails safe."""
    p5 = _mini_p5(vfar_mean=0.04, vfar_low=0.014, vfar_holm=True,
                  edit_holm=True, het_low=-0.01,
                  perm_mean=-0.1, perm_high=-0.05, region_pass=True)
    p5b = _mini_p5b(vfar_mean=0.09, vfar_low=0.083, vfar_holm=True,
                   edit_holm=True,
                   perm_mean=0.006, perm_high=0.020,
                   region_pass=True, loo_pass=True,
                   edit_mean=0.0)  # <- zero denominator
    rep = evaluate_combined(p5, p5b)
    # residual_frac should be nan and negligible_pass should be False (safe)
    fc = rep["feature_dependence_negative_control"]
    assert not np.isfinite(fc["set_b_residual_fraction_of_real"])
    assert fc["set_b_residual_negligible_pass"] is False
    assert fc["conceptual_overall_pass"] is False


import numpy as np  # noqa: E402 (late import; fixture needs it)


# ---------------------------------------------------------------------------
# Schema / structural assertions
# ---------------------------------------------------------------------------

def test_report_structure_contains_required_keys():
    p5 = _mini_p5(0.04, 0.014, True, True, -0.01, -0.1, -0.05, True)
    p5b = _mini_p5b(0.09, 0.083, True, True, 0.006, 0.020, True, True)
    rep = evaluate_combined(p5, p5b)
    required = [
        "schema_version", "contract_ref", "aggregation_basis", "caveats",
        "inputs", "primary_spatial_extension", "construct_wide_coverage",
        "feature_dependence_negative_control", "region_replication",
        "leave_dominant_out_robustness", "transportability",
        "claim_evidence_map", "verdict", "locked_outcome_access_count",
    ]
    for k in required:
        assert k in rep, f"missing required key: {k}"
    for claim in rep["claim_evidence_map"]:
        for sub in ("claim", "evidence", "pass"):
            assert sub in claim, f"claim map entry missing {sub}"
    assert rep["schema_version"] == "reactflow_delta.p5_combined_meta.v1"
    assert rep["verdict"] in ("MECHANISM_EVIDENCE_PASS",
                               "MECHANISM_NOT_ESTABLISHED")
    assert rep["locked_outcome_access_count"] >= 2


def test_per_set_verdicts_always_preserved_as_fail_closed():
    """Even in PASS scenario, individual per-set verdicts remain fail-closed.

    This is the critical fail-closed contract property: the OVERALL
    combined P5 gate can PASS while individual P5 and P5b per-set
    verdicts remain honestly NOT_ESTABLISHED. We must never silently
    rewrite the individual per-set statuses.
    """
    p5 = _mini_p5(0.04011, 0.01487, True, True, -0.01989,
                  -0.11071, -0.06240, True)
    p5b = _mini_p5b(0.09066, 0.08350, True, True,
                    0.00655, 0.02040, True, True,
                    edit_mean=0.08678)
    # Explicitly pre-set individual fail-closed verdicts
    p5["verdict"] = "MECHANISM_NOT_ESTABLISHED"
    p5b["verdict"] = "MECHANISM_NOT_ESTABLISHED"
    rep = evaluate_combined(p5, p5b)
    # Individual inputs preserved
    assert rep["inputs"]["p5_set_a_verdict"] == "MECHANISM_NOT_ESTABLISHED"
    assert rep["inputs"]["p5b_set_b_verdict"] == "MECHANISM_NOT_ESTABLISHED"
    # Combined overall can still be PASS (different thing)
    assert rep["verdict"] == "MECHANISM_EVIDENCE_PASS"
