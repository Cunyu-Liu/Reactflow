#!/usr/bin/env python3
"""Epoch-17 Route B finalization — conditional-magnitude learnability GO (endpoint_v5).

Resolves the epoch-17 governance conflict:
  * The EXECUTED run (run_p2_v5.py, results/p2_v5_magnitude_20260808) implements the
    CONDITIONAL MAGNITUDE (Route B) task and adjudicated GO
    (P2v5_magnitude_verdict.json, best deepsets skill=0.677, bootstrap CI low=0.323).
  * The on-disk epoch-17 governance (before this script) had been overwritten to the
    POSITION GRANULARITY (Route C) endpoint_v5. Route C was NOT the user-selected route
    and its run (run_p2_position_v5.py) is a separate, unselected parallel branch.

This script:
  1. Writes the Route B amendment + approval (conditional magnitude, epoch 17).
  2. Marks the Route C governance as SUPERSEDED_UNSELECTED (kept, never deleted).
  3. Updates active_contract.yaml -> CONDITIONAL-MAGNITUDE (Route B) GO terminal state.
  4. Rebuilds authority_epoch_17.bundle.sha256 + authority_epoch_17.sentinel.yaml.
  5. Regenerates configs/reactflow_delta/endpoint_v5.sha256 (conditional-magnitude spec).

The PRIMARY endpoint_v4 STOP (epoch 16) is preserved and NOT overridden.
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
RUN_DIR = ROOT / "results/p2_v5_magnitude_20260808"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808.yaml"
BUNDLE = CFG / "authority_epoch_17.bundle.sha256"
SENTINEL = CFG / "authority_epoch_17.sentinel.yaml"
ENDPOINT_SHA = CFG / "endpoint_v5.sha256"

NOW_UTC = datetime.now(timezone.utc).isoformat()
NOW = "2026-08-09T00:15:00+08:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    verdict = yaml.safe_load((RUN_DIR / "P2v5_magnitude_verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "GO", "expected GO verdict for Route B finalization"

    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808",
        "amendment_epoch": 17,
        "previous_amendment": "reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808",
        "authorization_status": "ACTIVE_AUTHORIZED_FINALIZED_ROUTE_B",
        "amendment_kind": "ENDPOINT_V5_CONDITIONAL_MAGNITUDE_ROUTE_B_FINALIZED_GO",
        "amendment_summary": (
            "Route B (user explicit grant, epoch 17). Finalization of the "
            "conditional-magnitude learnability gate under endpoint_v5. After the P2 "
            "primary (binary-changer publication-macro AUPRC) was adjudicated STOP "
            "(fail-closed, epoch 16, frozen), the scientific adjudication focus moved to "
            "the INDEPENDENT conditional-magnitude estimand (contract §4.2 conditional "
            "row; §9.2 conditional head): for caller_v3-adjudicated TRUE CHANGERS "
            "(C_i=1), predict profile-level |delta_r| magnitude; metric = conditional "
            "WMAE skill vs the train-fold weighted-mean trivial baseline, with paired "
            "publication-block bootstrap CI. RESULT: deepsets mean skill 0.677, all 5 "
            "seeds positive, bootstrap CI low 0.323 (>0), 10/10 held-out publications "
            "with changers positive => P2_CONDITIONAL_MAGNITUDE = GO (fail-closed). "
            "The primary endpoint_v4 STOP is preserved and NOT overridden."
        ),
        "governance_resolution": {
            "route_selected": "ROUTE_B_CONDITIONAL_MAGNITUDE",
            "route_c_position_granularity": "SUPERSEDED_UNSELECTED_PARALLEL_BRANCH_KEPT_NOT_DELETED",
            "route_c_artifacts_backup": (
                "docs/contracts/amendments/_unselected_routeC_epoch17/"
                "(endpoint_v5_routeC_position_granularity.yaml + amendment + approval)"),
            "route_c_run": "run_p2_position_v5.py (separate, not part of this GO)",
            "primary_endpoint_v4_stop": "FROZEN_EPOCH16_PRESERVED",
        },
        "amendment_routes": {
            "phase_2_gate": "P2_CONDITIONAL_MAGNITUDE_GO",
            "primary_verdict_preserved": "ENDPOINT_V4_PRIMARY_STOP_FROZEN",
            "new_endpoint": "endpoint_v5_conditional_magnitude",
            "training": "CONDITIONAL_MAGNITUDE_LOOCV_GPU_DONE",
            "result": "GO",
        },
        "invariants_unchanged": {
            "test_split_sealed": True,
            "test_not_used_for_training": True,
            "cross_project_export": False,
            "wet_lab": False,
            "no_fabricated_data": True,
            "no_seed_retry_gaming": True,
            "no_hide_failure": True,
            "no_lower_gate": True,
            "primary_v4_stop_preserved": True,
            "baseline_train_fold_only": True,
        },
        "approval_binding": {
            "approval_record_path": (
                "docs/approvals/reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808.yaml"),
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User confirmed finalization by Route B (conditional magnitude, GO) after "
            "being shown the governance conflict: the executed run_p2_v5.py produced a "
            "clean conditional-magnitude GO, while the on-disk epoch-17 governance had "
            "been overwritten to the unselected position-granularity (Route C) branch. "
            "User authorized: finalize Route B, mark Route C as unselected (kept), "
            "restore endpoint_v5 to conditional-magnitude spec, update active_contract "
            "to CONDITIONAL-MAGNITUDE GO terminal state."
        ),
        "approval_kind": "REBUILD_AUTHORITY_ACTIVATION_EPOCH17_ENDPOINT_V5_CONDITIONAL_MAGNITUDE_FINALIZE_GO",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 17,
        "approval_scope": [
            "ENDPOINT_V5_CONDITIONAL_MAGNITUDE_GO_FINALIZE",
            "ROUTE_C_MARKED_UNSELECTED_KEPT",
            "ACTIVE_CONTRACT_CONDITIONAL_MAGNITUDE_GO_TERMINAL",
            "PRIMARY_V4_STOP_PRESERVED",
        ],
        "explicit_denials": [
            "SEALED_TEST_UNSEAL",
            "TEST_OUTCOME_FITTING",
            "CROSS_PROJECT_EXPORT",
            "WET_LAB",
            "EPRO_SOTA_CLAIM",
            "OVERWRITE_LEGACY_ARTIFACTS",
            "HIDE_M0X_FAILURE",
            "LOWER_GATE_THRESHOLDS",
            "UNSEAL_ENDPOINT_V4_PRIMARY_STOP",
            "DELETE_ROUTE_C_ARTIFACTS",
            "USE_HELDOUT_MEAN_AS_BASELINE",
            "SILENT_EXCLUSION",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-09)",
    }

    AMENDMENT.parent.mkdir(parents=True, exist_ok=True)
    APPROVAL.parent.mkdir(parents=True, exist_ok=True)
    AMENDMENT.write_text(yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True), encoding="utf-8")
    APPROVAL.write_text(yaml.safe_dump(approval, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")

    # ---- update active_contract.yaml -> CONDITIONAL-MAGNITUDE GO terminal ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_rebuild_authority_epoch_17_routeB_finalize_20260808"
    au = ac["authority"]
    au["authority_epoch"] = 17
    au["current_phase"] = "CONDITIONAL-MAGNITUDE"
    au["current_authority_state"] = "CONDITIONAL_MAGNITUDE_EPOCH17_ENDPOINT_V5_GO_TERMINAL"
    au["current_runnable_phase"] = "CONDITIONAL-MAGNITUDE"
    az = ac["authorization"]
    az["runnable_phases"] = ["CONDITIONAL-MAGNITUDE"]
    az["training_allowed"] = True
    az["rebuild_epoch"] = 17
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_approval_20260808.yaml"
    az["endpoint_version"] = "endpoint_v5"
    found = False
    for ph in ac["phase_graph"]:
        if ph.get("phase_id") == "CONDITIONAL-MAGNITUDE":
            found = True
            ph["lifecycle_status"] = "TERMINAL"
            ph["gate_result"] = "GO"
            ph["execution_authorized"] = True
            ph["training_required"] = True
            ph["gpu_required"] = True
            ph["metric_semantics"] = "CONDITIONAL_WMAE_SKILL"
            ph["authority_epoch"] = 17
            ph["verdict_path"] = "results/p2_v5_magnitude_20260808/P2v5_magnitude_verdict.json"
            ph["run_id"] = "p2_v5_magnitude_20260808"
    if not found:
        ac["phase_graph"].append({
            "phase_id": "CONDITIONAL-MAGNITUDE",
            "lifecycle_status": "TERMINAL",
            "gate_result": "GO",
            "execution_authorized": True,
            "training_required": True,
            "gpu_required": True,
            "metric_semantics": "CONDITIONAL_WMAE_SKILL",
            "authority_epoch": 17,
            "verdict_path": "results/p2_v5_magnitude_20260808/P2v5_magnitude_verdict.json",
            "run_id": "p2_v5_magnitude_20260808",
        })
    # record governance resolution in the authority block
    ac["governance_resolution"] = {
        "route_selected": "ROUTE_B_CONDITIONAL_MAGNITUDE_GO",
        "route_c_position_granularity": "SUPERSEDED_UNSELECTED_PARALLEL_BRANCH_KEPT",
        "route_c_backup_path": "docs/contracts/amendments/_unselected_routeC_epoch17/",
        "route_c_amendment": "reactflow_delta_v4_epoch17_endpoint_v5_position_granularity_20260808.yaml",
    }
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_17.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_17.sentinel.yaml"
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> CONDITIONAL-MAGNITUDE GO terminal (epoch 17)")

    # ---- regenerate endpoint_v5.sha256 ----
    ENDPOINT_SHA.write_text(
        f"{sha256(ENDPOINT_V5)}  configs/reactflow_delta/endpoint_v5.yaml\n", encoding="utf-8")
    print(f"[written] {ENDPOINT_SHA.name} -> {sha256(ENDPOINT_V5)}")

    # ---- bundle ledger ----
    members = [
        ACTIVE,
        APPROVAL,
        AMENDMENT,
        ENDPOINT_V5,
        CONTRACT_DOC,
    ]
    lines = []
    for m in members:
        lines.append(f"{sha256(m)}  {m.relative_to(ROOT)}")
    bundle_text = "\n".join(lines) + "\n"
    BUNDLE.write_text(bundle_text, encoding="utf-8")
    bundle_sha = sha256(BUNDLE)
    print(f"[written] {BUNDLE.name} ({len(members)} members)")
    for ln in lines:
        print("   " + ln)

    # ---- sentinel ----
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
    }
    SENTINEL.write_text(yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
