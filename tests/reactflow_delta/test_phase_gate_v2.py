#!/usr/bin/env python3
"""Fixtures for phase_gate_v2 (contract 11.6 + 14.2 static scenarios)."""

from __future__ import annotations

import pytest

from scripts.reactflow_delta.phase_gate_v2 import PhaseGateV2


def _full_pass_gate() -> PhaseGateV2:
    g = PhaseGateV2()
    g.set_primary_caller_exclusion("PASS", primary_depends_on_caller=False)
    g.set_evaluator("PASS")
    g.set_primary_locked_outcome_exclusion("PASS")
    g.set_confirmatory_store_availability("PASS")
    g.set_confirmatory_statistical_sufficiency("PASS")
    g.set_locked_external_access_control("PASS")
    return g


def test_primary_caller_exclusion_fail_blocks_p2_no_token():
    g = _full_pass_gate()
    g.set_primary_caller_exclusion("FAIL", primary_depends_on_caller=True)
    r = g.resolve()
    assert r["blocks_phase2"] is True
    assert r["token_eligible"] is False
    assert r["phase2_token"] is None
    assert r["checks"]["primary_caller_exclusion"]["status"] == "PRIMARY_CALLER_EXCLUSION_FAIL"


def test_evaluator_fail_blocks_p2():
    g = _full_pass_gate()
    g.set_evaluator("FAIL")
    r = g.resolve()
    assert r["blocks_phase2"] is True
    assert r["token_eligible"] is False


def test_primary_locked_outcome_not_established_blocks_p2_p3():
    g = _full_pass_gate()
    g.set_primary_locked_outcome_exclusion("NOT_ESTABLISHED")
    r = g.resolve()
    assert r["blocks_phase2"] is True
    assert r["blocks_phase3"] is True
    assert r["token_eligible"] is False
    c = r["checks"]["primary_locked_outcome_exclusion"]
    assert c["status"] == "NOT_ESTABLISHED"  # not mislabeled as contamination
    assert c["blocks_phase4"] is True


def test_primary_isolation_pass_confirmatory_unknown_allows_dev_blocks_p4():
    g = PhaseGateV2()
    g.set_primary_caller_exclusion("PASS", primary_depends_on_caller=False)
    g.set_evaluator("PASS")
    g.set_primary_locked_outcome_exclusion("PASS")  # isolation established
    g.set_confirmatory_store_availability("NOT_ESTABLISHED")
    g.set_confirmatory_statistical_sufficiency("NOT_ESTABLISHED")
    g.set_locked_external_access_control("NOT_ESTABLISHED")
    r = g.resolve()
    assert r["blocks_phase2"] is False
    assert r["blocks_phase3"] is False
    assert r["blocks_phase4"] is True  # P4 blocked until confirmatory sufficiency
    assert r["token_eligible"] is True  # development learnability may proceed


def test_detected_outcome_access_upgrades_to_confirmed_exposure_fail():
    g = _full_pass_gate()
    g.set_primary_locked_outcome_exclusion("CONFIRMED_OUTCOME_EXPOSURE_FAIL")
    r = g.resolve()
    c = r["checks"]["primary_locked_outcome_exclusion"]
    assert c["status"] == "CONFIRMED_OUTCOME_EXPOSURE_FAIL"
    assert c["blocks_phase2"] and c["blocks_phase3"] and c["blocks_phase4"]
    assert r["token_eligible"] is False


def test_all_pass_emits_p2_token():
    r = _full_pass_gate().resolve()
    assert r["blocks_phase2"] is False
    assert r["token_eligible"] is True
    assert r["phase2_token"] == "AUTHORIZE_REACTFLOW_DELTA_PROSPECTIVE_V2_P2"


def test_stale_training_prohibition_not_bypassed_without_isolation():
    # stale training_allowed=false + P0-P3 token must NOT unlock P2/P3 until
    # primary isolation is ESTABLISHED (contract 14.2: "P0-P3 token + stale ...").
    g = PhaseGateV2()
    g.set_primary_caller_exclusion("PASS", primary_depends_on_caller=False)
    g.set_evaluator("PASS")
    g.set_primary_locked_outcome_exclusion("NOT_ESTABLISHED")  # not yet established
    r = g.resolve()
    assert r["blocks_phase2"] is True
    assert r["token_eligible"] is False


def test_no_checks_raises():
    with pytest.raises(ValueError):
        PhaseGateV2().resolve()


# -- fail-open regression: phase1_integration must no longer emit token on FAIL
def test_phase1_fail_open_removed():
    """Static regression for H01: the old gate wrapped non-PASS as a
    NON_BLOCKING_NOTICE and always emitted the Phase-2 token. It must now be
    fail-closed (FAIL_CLOSED_BLOCKED + token=None when a Phase-2 gate fails)."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "scripts/reactflow_delta/phase1_integration_v1.py"
    text = src.read_text(encoding="utf-8")
    assert "PASS_WITH_NON_BLOCKING_NOTICE" not in text
    assert "PHASE1_BENCHMARK_V3_FAIL_CLOSED_BLOCKED" in text
    assert "phase2_authorization_token" in text
    assert "phase2_blocking_fail" in text
    assert 'if not phase2_blocking_fail else None' in text
