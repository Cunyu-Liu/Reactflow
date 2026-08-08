#!/usr/bin/env python3
"""Finalize authority epoch 16: bind the terminal STOP verdict into active_contract,
regenerate the epoch-16 bundle/sentinel, and write endpoint_v4 detached ledger.

This is the terminal record-keeping step AFTER r6_p2v4_go_stop.py wrote
P2_learnability_terminal_v4.json (verdict STOP).  It does NOT change any science.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
ENDPOINT_V4 = CFG / "endpoint_v4.yaml"
APPROVAL = ROOT / "docs/approvals/reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808.yaml"
AMENDMENT = ROOT / "docs/contracts/amendments/reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808.yaml"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"
BUNDLE = CFG / "authority_epoch_16.bundle.sha256"
SENTINEL = CFG / "authority_epoch_16.sentinel.yaml"
RUN_DIR = ROOT / "results/p2_v3_learnability_20260808b"
TERMINAL = RUN_DIR / "P2_learnability_terminal_v4.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not TERMINAL.exists():
        raise SystemExit(f"MISSING_TERMINAL: {TERMINAL}")
    term = json.loads(TERMINAL.read_text(encoding="utf-8"))
    gate_result = term["gate_result"]  # GO / STOP

    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    au = ac["authority"]
    au["authority_epoch"] = 16
    au["current_phase"] = "REBUILD-P2"
    au["current_authority_state"] = (
        f"REBUILD_P2_TERMINAL_{gate_result}_EPOCH16")
    au["current_runnable_phase"] = "NONE"
    az = ac["authorization"]
    az["runnable_phases"] = []
    az["training_allowed"] = False
    for ph in ac["phase_graph"]:
        if ph["phase_id"] == "REBUILD-P2":
            ph["lifecycle_status"] = "TERMINAL"
            ph["gate_result"] = gate_result
            ph["execution_authorized"] = False
            ph["terminal_manifest_path"] = (
                "results/p2_v3_learnability_20260808b/P2_learnability_terminal_v4.json")
            ph["endpoint"] = "endpoint_v4"
            ph["next_on_pass"] = ["REBUILD-P3"]
            ph["next_on_fail"] = ["REPORT-X"]
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_16.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_16.sentinel.yaml"
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> REBUILD-P2 TERMINAL {gate_result}")

    # ---- endpoint_v4 detached ledger ----
    ep_ledger = CFG / "endpoint_v4.sha256"
    ep_ledger.write_text(f"{sha256(ENDPOINT_V4)}  configs/reactflow_delta/endpoint_v4.yaml\n", encoding="utf-8")
    print(f"[written] {ep_ledger.name}")

    # ---- regenerate epoch-16 bundle (bind terminal active_contract) ----
    members = [ACTIVE, APPROVAL, AMENDMENT, ENDPOINT_V4, CONTRACT_DOC]
    lines = []
    for m in members:
        lines.append(f"{sha256(m)}  {m.relative_to(ROOT)}")
    BUNDLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle_sha = sha256(BUNDLE)
    print(f"[rewritten] {BUNDLE.name} ({len(members)} members, terminal binding)")
    for ln in lines:
        print("   " + ln)

    # ---- regenerate epoch-16 sentinel (terminal) ----
    sentinel = {
        "schema_version": "reactflow_delta.authority_epoch_sentinel.v1",
        "authority_epoch": 16,
        "current_authority_state": f"REBUILD_P2_TERMINAL_{gate_result}_EPOCH16",
        "current_phase": "REBUILD-P2",
        "gate_result": gate_result,
        "endpoint_version": "endpoint_v4",
        "metric_semantics": "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_16.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "terminal_manifest_path": "results/p2_v3_learnability_20260808b/P2_learnability_terminal_v4.json",
        "created_at": term["adjudicated_at_utc"],
        "emitted_after": "TERMINAL_BINDING",
        "embedded_self_sha256": "FORBIDDEN",
    }
    SENTINEL.write_text(yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[rewritten] {SENTINEL.name} (terminal)")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
