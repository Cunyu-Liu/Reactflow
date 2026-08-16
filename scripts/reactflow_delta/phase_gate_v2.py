#!/usr/bin/env python3
"""phase_gate_v2: fail-closed phase gate with phase-specific blockers.

Implements ReactFlow-Delta prospective-v2 contract section 11.6:
  - every check exposes blocks_phase2/blocks_phase3/blocks_phase4/token_eligible
  - physical isolation is split into real call-path checks, not one boolean
  - primary is caller-free; a caller dependency hard-blocks P2
  - no non-PASS is silently wrapped as a non-blocking notice
  - NOT_ESTABLISHED isolation blocks P2/P3/P4; isolation PASS with confirmatory
    sufficiency unknown allows development but blocks P4; detected outcome access
    is upgraded to CONFIRMED_OUTCOME_EXPOSURE_FAIL
Static/outcome-blind: no data, no training, no locked-outcome read.
"""

from __future__ import annotations

from typing import Any


def _check(check_id: str, status: str, evidence: str, reason: str,
           blocks_phase2: bool, blocks_phase3: bool, blocks_phase4: bool) -> dict[str, Any]:
    """Build one gate check record with explicit per-phase blockers."""
    return {
        "check_id": check_id,
        "status": status,
        "evidence": evidence,
        "reason": reason,
        "blocks_phase2": blocks_phase2,
        "blocks_phase3": blocks_phase3,
        "blocks_phase4": blocks_phase4,
    }


class PhaseGateV2:
    """Compute fail-closed phase gates from observed check statuses."""

    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def register(self, check: dict[str, Any]) -> None:
        self.checks[check["check_id"]] = check

    # -- physical isolation (split per real call path) ----------------------
    def set_primary_locked_outcome_exclusion(self, status: str) -> None:
        """P2/P3 loader/cache/features/normalizer/selector/evaluator cannot read locked outcomes."""
        if status == "PASS":
            self.register(_check(
                "primary_locked_outcome_exclusion", "PASS",
                "primary call path proven unable to read locked mutant outcomes",
                "isolation established on real call path", False, False, False))
        elif status == "CONFIRMED_OUTCOME_EXPOSURE_FAIL":
            self.register(_check(
                "primary_locked_outcome_exclusion", "CONFIRMED_OUTCOME_EXPOSURE_FAIL",
                "actual locked outcome access detected",
                "real exposure failure; isolate affected assets", True, True, True))
        else:  # NOT_ESTABLISHED
            self.register(_check(
                "primary_locked_outcome_exclusion", "NOT_ESTABLISHED",
                "no proof loader cannot reach locked store; not equivalent to contamination",
                "isolation unproven", True, True, True))

    def set_confirmatory_store_availability(self, status: str) -> None:
        self.register(_check(
            "confirmatory_store_availability", status,
            "clean external store existence is not established",
            "affects Phase 4 only once primary isolation PASS", False, False,
            status != "PASS"))

    def set_confirmatory_statistical_sufficiency(self, status: str) -> None:
        self.register(_check(
            "confirmatory_statistical_sufficiency", status,
            "preaccess sufficiency is not established",
            "affects Phase 4 only", False, False, status != "PASS"))

    def set_locked_external_access_control(self, status: str) -> None:
        """Phase4 token / preaccess gate / access counter."""
        self.register(_check(
            "locked_external_access_control", status,
            "Phase4 token/preaccess/access-counter gate",
            "controls Phase 4 locked read", False, False, status != "PASS"))

    # -- caller exclusion (primary must be caller-free) ---------------------
    def set_primary_caller_exclusion(self, status: str, primary_depends_on_caller: bool) -> None:
        if primary_depends_on_caller or status != "PASS":
            self.register(_check(
                "primary_caller_exclusion", "PRIMARY_CALLER_EXCLUSION_FAIL",
                "primary path still depends on Caller or exclusion not established",
                "hard-blocks P2", True, True, True))
            self.checks["primary_caller_exclusion"]["status"] = "PRIMARY_CALLER_EXCLUSION_FAIL"
        else:
            self.register(_check(
                "primary_caller_exclusion", "PASS",
                "primary is fully caller-free",
                "legacy oracle-only Caller FAIL blocks oracle secondary only",
                False, False, False))

    def set_evaluator(self, status: str) -> None:
        self.register(_check(
            "evaluator", status,
            "unique frozen evaluator available and validator-before-metric",
            "blocks P2 if FAIL", status != "PASS", status != "PASS", status != "PASS"))

    # -- resolution ----------------------------------------------------------
    def resolve(self) -> dict[str, Any]:
        if not self.checks:
            raise ValueError("no checks registered")
        any_p2_block = any(c["blocks_phase2"] and c["status"] != "PASS" for c in self.checks.values())
        any_p3_block = any(c["blocks_phase3"] and c["status"] != "PASS" for c in self.checks.values())
        any_p4_block = any(c["blocks_phase4"] and c["status"] != "PASS" for c in self.checks.values())
        token_eligible = not any_p2_block  # fail-closed: no P2 token unless no P2-blocking gate fails
        return {
            "schema_version": "reactflow_delta.phase_gate_v2.v1",
            "verdict": ("FAIL_CLOSED_OPEN" if token_eligible else "FAIL_CLOSED_BLOCKED"),
            "token_eligible": token_eligible,
            "phase2_token": "AUTHORIZE_REACTFLOW_DELTA_PROSPECTIVE_V2_P2" if token_eligible else None,
            "blocks_phase2": any_p2_block,
            "blocks_phase3": any_p3_block,
            "blocks_phase4": any_p4_block,
            "checks": self.checks,
        }
