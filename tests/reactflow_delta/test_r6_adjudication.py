"""Tests for the R6 GO/STOP scientific adjudication logic (ReactFlowDelta §13.2 R6 / §13.4).

Verifies the contract's terminal-decision invariants:
  * any gate that is FAIL / UNKNOWN / NOT_RUN / MISSING => overall decision STOP
    (manual override to GO is never allowed);
  * P2 not GO => overall decision is NOT GO (STOP / STOP_METHOD_ROUTE);
  * the emitted decision manifest conforms to the schema.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reactflow_delta.r6_go_stop_adjudicate import (  # noqa: E402
    ALL_GATES,
    build_manifest,
    decide_overall,
    render_markdown,
    run_all,
)


def _all_pass() -> dict:
    return {g: "PASS" for g in ALL_GATES}


# ---------------------------------------------------------------------------
# 1. Any gate missing => STOP / UNKNOWN (never manual override to GO)
# ---------------------------------------------------------------------------

def test_missing_gate_never_overrides_to_go():
    statuses = _all_pass()
    # drop one gate entirely
    statuses.pop("CALLER_V2_FOLD_LOCAL_AND_RELIABLE")
    dec = decide_overall(statuses)
    assert dec["decision"] == "STOP"
    assert "CALLER_V2_FOLD_LOCAL_AND_RELIABLE" in dec["blocking"]


def test_unknown_gate_blocks():
    statuses = _all_pass()
    statuses["ASSET_DISPOSITION_1024_OF_1024"] = "UNKNOWN"
    dec = decide_overall(statuses)
    assert dec["decision"] == "STOP"
    assert "ASSET_DISPOSITION_1024_OF_1024" in dec["blocking"]


def test_not_run_gate_blocks():
    statuses = _all_pass()
    statuses["PRIMARY_MASK_V2_PASS"] = "NOT_RUN"
    dec = decide_overall(statuses)
    assert dec["decision"] == "STOP"


def test_fail_gate_blocks():
    statuses = _all_pass()
    statuses["AUTHORITY_CLOSED_PASS"] = "FAIL"
    dec = decide_overall(statuses)
    assert dec["decision"] == "STOP"
    assert "AUTHORITY_CLOSED_PASS" in dec["blocking"]


# ---------------------------------------------------------------------------
# 2. P2 not GO => overall not GO (STOP / STOP_METHOD_ROUTE)
# ---------------------------------------------------------------------------

def test_p2_not_go_overall_stop():
    statuses = _all_pass()
    statuses["P2_LEARNABILITY_GO"] = "FAIL"
    dec = decide_overall(statuses)
    assert dec["decision"] == "STOP"
    assert dec["route"] == "STOP_METHOD_ROUTE"
    assert "P2_LEARNABILITY_GO" in dec["blocking"]


def test_p2_not_go_never_reports_go():
    # even if every OTHER gate passes, a non-GO P2 must keep overall = STOP
    for bad in ("FAIL", "UNKNOWN", "NOT_RUN"):
        statuses = _all_pass()
        statuses["P2_LEARNABILITY_GO"] = bad
        dec = decide_overall(statuses)
        assert dec["decision"] == "STOP"


def test_all_pass_is_go_sanity():
    dec = decide_overall(_all_pass())
    assert dec["decision"] == "GO"
    assert dec["blocking"] == []


# ---------------------------------------------------------------------------
# 3. Manifest schema
# ---------------------------------------------------------------------------

def test_manifest_schema():
    results = {g: ("PASS", {"evidence": "disk-checked"}) for g in ALL_GATES}
    results["P2_LEARNABILITY_GO"] = ("FAIL", {"verdict": "STOP_METHOD_ROUTE"})
    m = build_manifest(results, "blocked")
    assert m["schema"] == "reactflow_delta.r6_go_stop_adjudication.v1"
    assert m["run_id"] == "r6_go_stop_20260807"
    assert m["overall_decision"] == "STOP"
    assert m["route"] == "STOP_METHOD_ROUTE"
    assert "P2_LEARNABILITY_GO" in m["blocking_gates"]
    assert set(m["gates"].keys()) == set(ALL_GATES)
    for g in ALL_GATES:
        assert "status" in m["gates"][g]
        assert "evidence" in m["gates"][g]
    # serializable
    json.dumps(m)


def test_manifest_marks_p2_not_go():
    results = {g: ("PASS", {"evidence": "disk-checked"}) for g in ALL_GATES}
    results["P2_LEARNABILITY_GO"] = ("FAIL", {"verdict": "STOP_METHOD_ROUTE"})
    m = build_manifest(results, "blocked")
    assert m["gates"]["P2_LEARNABILITY_GO"]["status"] == "FAIL"


def test_render_markdown_runs():
    results = {g: ("PASS", {"evidence": "disk-checked"}) for g in ALL_GATES}
    results["P2_LEARNABILITY_GO"] = ("FAIL", {"verdict": "STOP_METHOD_ROUTE"})
    m = build_manifest(results, "blocked")
    md = render_markdown(m)
    assert "STOP" in md
    assert "STOP_METHOD_ROUTE" in md


# ---------------------------------------------------------------------------
# 4. Integration: the on-disk adjudicator must honor the R5 verdict (NOT_GO)
# ---------------------------------------------------------------------------

def test_p2_gate_assessed_not_go():
    manifest = run_all()
    gate = manifest["gates"]["P2_LEARNABILITY_GO"]
    assert gate["status"] == "FAIL"
    assert gate["evidence"].get("assessment") == "NOT_GO"


def test_overall_decision_is_stop_when_p2_not_go():
    manifest = run_all()
    # per R5 verdict P2 is NOT GO -> overall cannot be GO
    assert manifest["gates"]["P2_LEARNABILITY_GO"]["status"] == "FAIL"
    assert manifest["overall_decision"] == "STOP"
