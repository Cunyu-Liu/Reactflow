#!/usr/bin/env python3
"""Epoch-17 Route B follow-up — mark the unselected Route C (POSITION-GRANULARITY)
phase as SUPERSEDED in active_contract.yaml phase_graph, then rebuild the authority
bundle + sentinel.

Fixes the governance inconsistency left after ep17_routeB_finalize.py: the Route B
finalization marked Route C as SUPERSEDED_UNSELECTED in the governance_resolution and
sentinel, but the phase_graph entry for POSITION-GRANULARITY was still RUNNING with
execution_authorized=True. For audit integrity, that phase must not claim RUNNING /
authorized when it has been superseded by the Route B GO.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from datetime import datetime, timezone

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
ENDPOINT_V5 = CFG / "endpoint_v5.yaml"
AMEND_DIR = ROOT / "docs/contracts/amendments"
APPROVAL_DIR = ROOT / "docs/approvals"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"

APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808.yaml"
AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808.yaml"
BUNDLE = CFG / "authority_epoch_17.bundle.sha256"
SENTINEL = CFG / "authority_epoch_17.sentinel.yaml"

NOW_UTC = datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    found = False
    for ph in ac["phase_graph"]:
        if ph.get("phase_id") == "POSITION-GRANULARITY":
            found = True
            ph["lifecycle_status"] = "SUPERSEDED_UNSELECTED"
            ph["gate_result"] = "NOT_RUN"
            ph["execution_authorized"] = False
            ph["authority_epoch"] = 17
            ph["note"] = (
                "Route C (position granularity) superseded by Route B "
                "(conditional magnitude) GO; artifacts kept in "
                "docs/contracts/amendments/_unselected_routeC_epoch17/")
    if not found:
        raise SystemExit("POSITION-GRANULARITY phase not found in phase_graph")
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("[updated] POSITION-GRANULARITY -> SUPERSEDED_UNSELECTED (execution_authorized=False)")

    # rebuild bundle + sentinel (same members as ep17_routeB_finalize)
    members = [ACTIVE, APPROVAL, AMENDMENT, ENDPOINT_V5, CONTRACT_DOC]
    lines = []
    for m in members:
        lines.append(f"{sha256(m)}  {m.relative_to(ROOT)}")
    bundle_text = "\n".join(lines) + "\n"
    BUNDLE.write_text(bundle_text, encoding="utf-8")
    bundle_sha = sha256(BUNDLE)
    print(f"[written] {BUNDLE.name} ({len(members)} members)")
    for ln in lines:
        print("   " + ln)

    sentinel = {
        "schema_version": "reactflow_delta.authority_epoch_sentinel.v1",
        "authority_epoch": 17,
        "current_authority_state": "CONDITIONAL_MAGNITUDE_EPOCH17_ENDPOINT_V5_GO_TERMINAL",
        "current_phase": "CONDITIONAL-MAGNITUDE",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_17.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW_UTC,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v5",
        "metric_semantics": "CONDITIONAL_WMAE_SKILL",
        "primary_v4_verdict": "STOP_FROZEN_EPOCH16",
        "route_b_result": "GO",
        "route_c_status": "SUPERSEDED_UNSELECTED_PARALLEL_BRANCH_KEPT",
        "phase_graph_fix": "POSITION_GRANULARITY_MARKED_SUPERSEDED_EPOCH17",
    }
    SENTINEL.write_text(yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
