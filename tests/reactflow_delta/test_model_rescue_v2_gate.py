from __future__ import annotations

from pathlib import Path

from scripts.reactflow_delta.model_rescue_v2_gate import validate_contract


ROOT = Path(__file__).resolve().parents[2]


def test_v2_contract_and_active_authority_are_consistent():
    result = validate_contract(ROOT)
    assert result["status"] == "PASS", result
    assert all(result["checks"].values())
    assert result["checks"]["terminal_failure_handoff"] is True
