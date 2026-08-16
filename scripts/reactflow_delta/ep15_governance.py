#!/usr/bin/env python3
"""Generate epoch-15 authority for the endpoint_v3/caller_v3 calibration fix.

Creates (idempotent-ish; refuses to overwrite existing epoch-15 artifacts):
  * docs/contracts/amendments/reactflow_delta_v4_epoch15_caller_v3_20260808.yaml
  * docs/approvals/reactflow_delta_v4_epoch15_caller_v3_approval_20260808.yaml
Updates active_contract.yaml -> epoch 15, REBUILD-P2 (runnable), P2 gate.
Writes authority_epoch_15.bundle.sha256 + authority_epoch_15.sentinel.yaml.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
AMEND_DIR = ROOT / "docs/contracts/amendments"
APPROVAL_DIR = ROOT / "docs/approvals"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch15_caller_v3_20260808.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch15_caller_v3_approval_20260808.yaml"
BUNDLE = CFG / "authority_epoch_15.bundle.sha256"
SENTINEL = CFG / "authority_epoch_15.sentinel.yaml"

NOW = "2026-08-08T18:00:00+08:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE: {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch15_caller_v3_20260808",
        "amendment_epoch": 15,
        "previous_amendment": "reactflow_delta_v4_rebuild_20260807",
        "authorization_status": "ACTIVE_AUTHORIZED",
        "amendment_kind": "ENDPOINT_V3_CALLER_V3_CALIBRATION_FIX",
        "amendment_summary": (
            "Per the endpoint_v3 proposal and explicit user grant (epoch 15), "
            "this amendment authorizes the calibration fix for the R5 P2 "
            "learnability gate: (1) creates endpoint_v3 + caller_v3 "
            "(empirical-scatter noise recalibration) fixing the R5 "
            "near-constant label (3 changers/6359) caused by reported-error "
            "miscalibration and cross-study reactivity scale heterogeneity; "
            "(2) reruns the P2 learnability gate (nested "
            "leave-one-publication-out) with caller_v3 labels and the "
            "unchanged evaluate_v2/split_v2; (3) re-adjudicates "
            "P2_LEARNABILITY_GO/STOP."
        ),
        "amendment_routes": {
            "phase_2_gate": "P2_LEARNABILITY_GO_OR_STOP_METHOD_ROUTE",
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
            "unit_label_score_metric_unchanged": True,
            "null_train_fold_only": True,
        },
        "approval_binding": {
            "approval_record_path": "docs/approvals/reactflow_delta_v4_epoch15_caller_v3_approval_20260808.yaml",
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch15_caller_v3_approval_20260808",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User approved activating endpoint_v3/caller_v3 (epoch 15): "
            "per-study normalization + empirical-scatter error recalibration "
            "-> caller_v3 re-derives binary labels -> rerun the P2 learnability "
            "gate with unchanged evaluate_v2/split_v2."
        ),
        "approval_kind": "REBUILD_AUTHORITY_ACTIVATION_EPOCH15_ENDPOINT_V3_CALLER_V3",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 15,
        "approval_scope": [
            "ENDPOINT_V3_FREEZE",
            "CALLER_V3_CALIBRATION_FIX",
            "P2_LEARNABILITY_GATE_RERUN",
            "GPU_TRAINING_ON_FREE_VRAM",
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
            "UNIT_LABEL_SCORE_METRIC_CHANGE",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-08)",
    }

    write_once(AMENDMENT, yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True))
    write_once(APPROVAL, yaml.safe_dump(approval, sort_keys=False, allow_unicode=True))
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")

    # ---- update active_contract.yaml (epoch 15, REBUILD-P2 runnable) ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_rebuild_authority_epoch_15_20260808"
    au = ac["authority"]
    au["authority_epoch"] = 15
    au["current_phase"] = "REBUILD-P2"
    au["current_authority_state"] = "REBUILD_P2_AUTHORIZED"
    au["current_runnable_phase"] = "REBUILD-P2"
    az = ac["authorization"]
    az["runnable_phases"] = ["REBUILD-P2"]
    az["training_allowed"] = True
    az["rebuild_epoch"] = 15
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch15_caller_v3_20260808.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch15_caller_v3_approval_20260808.yaml"
    for ph in ac["phase_graph"]:
        if ph["phase_id"] == "REBUILD-P1":
            ph["lifecycle_status"] = "TERMINAL"
            ph["gate_result"] = "PASS"
            ph["next_on_pass"] = ["REBUILD-P2"]
            ph["next_on_fail"] = ["REPORT-X"]
        if ph["phase_id"] == "REBUILD-P2":
            ph["lifecycle_status"] = "RUNNING"
            ph["gate_result"] = "NOT_RUN"
            ph["execution_authorized"] = True
            ph["training_required"] = True
            ph["gpu_required"] = True
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_15.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_15.sentinel.yaml"
    ACTIVE.write_text(
        yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> epoch 15, REBUILD-P2")

    # ---- bundle ledger ----
    members = [
        ACTIVE,
        APPROVAL,
        AMENDMENT,
        CONTRACT_DOC,
    ]
    lines = []
    for m in members:
        lines.append(f"{sha256(m)}  {m.relative_to(ROOT)}")
    bundle_text = "\n".join(lines) + "\n"
    write_once(BUNDLE, bundle_text)
    bundle_sha = sha256(BUNDLE)
    print(f"[written] {BUNDLE.name} ({len(members)} members)")
    for ln in lines:
        print("   " + ln)

    # ---- sentinel ----
    sentinel = {
        "schema_version": "reactflow_delta.authority_epoch_sentinel.v1",
        "authority_epoch": 15,
        "current_authority_state": "REBUILD_P2_AUTHORIZED",
        "current_phase": "REBUILD-P2",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_15.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "legacy_m0x": "TERMINALIZED_FAIL_NO_PASS_SENTINEL",
    }
    write_once(SENTINEL, yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True))
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
