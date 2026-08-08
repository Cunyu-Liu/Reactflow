#!/usr/bin/env python3
"""Generate epoch-16 authority for the endpoint_v4 non-degenerate macro (Route A).

Creates (idempotent-ish; refuses to overwrite existing epoch-16 artifacts):
  * docs/contracts/amendments/reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808.yaml
  * docs/approvals/reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808.yaml
  * configs/reactflow_delta/endpoint_v4.yaml (frozen relaxed-macro metric spec)
Updates active_contract.yaml -> epoch 16 (REBUILD-P2 adjudication, metric semantics
only; no new training required).  Writes authority_epoch_16.bundle.sha256 +
authority_epoch_16.sentinel.yaml.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
ENDPOINT_V4 = CFG / "endpoint_v4.yaml"
AMEND_DIR = ROOT / "docs/contracts/amendments"
APPROVAL_DIR = ROOT / "docs/approvals"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808.yaml"
BUNDLE = CFG / "authority_epoch_16.bundle.sha256"
SENTINEL = CFG / "authority_epoch_16.sentinel.yaml"

NOW = "2026-08-08T22:30:00+08:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE: {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if not ENDPOINT_V4.exists():
        raise SystemExit(f"MISSING_ENDPOINT_V4: {ENDPOINT_V4}")

    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808",
        "amendment_epoch": 16,
        "previous_amendment": "reactflow_delta_v4_epoch15_caller_v3_20260808",
        "authorization_status": "ACTIVE_AUTHORIZED",
        "amendment_kind": "ENDPOINT_V4_NON_DEGENERATE_MACRO_ROUTE_A",
        "amendment_summary": (
            "Route A (user explicit grant, epoch 16). Relaxes ONLY the PRIMARY "
            "metric degeneracy policy of the P2 learnability gate: endpoint_v4 "
            "computes publication-macro AUPRC over NON-DEGENERATE (mixed-label) "
            "publications, with constant-label publications explicitly listed "
            "and excluded (no silent dropping). This addresses the R5/P2_v3 "
            "UNIDENTIFIABLE verdict: 8/10 eligible publications show strong "
            "per-pub AP but 2 constant-label publications (pmid_25883046, "
            "pmid_35982307) triggered the frozen evaluate_v2 fail-closed macro. "
            "unit/label/score/mask/information-permission/caller UNCHANGED from "
            "endpoint_v3. Recomputes the P2 gate metric from existing held-out "
            "predictions (no retraining), re-adjudicates P2_LEARNABILITY_GO/STOP."
        ),
        "amendment_routes": {
            "phase_2_gate": "P2_LEARNABILITY_GO_OR_STOP_METHOD_ROUTE",
            "metric_change": "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS_ONLY",
            "recompute": "FROM_EXISTING_HELDOUT_PREDICTIONS_NO_RETRAIN",
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
            "unit_label_score_unchanged": True,
            "caller_unchanged": True,
            "excluded_publications_documented": True,
        },
        "approval_binding": {
            "approval_record_path": "docs/approvals/reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808.yaml",
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User chose Route A (relax macro): new authority epoch + amendment to "
            "compute publication-macro AUPRC over non-degenerate (mixed-label) "
            "publications with the 2 constant-label publications explicitly "
            "excluded and documented, then re-adjudicate P2_LEARNABILITY_GO/STOP "
            "on the recalibrated caller_v3 labels."
        ),
        "approval_kind": "REBUILD_AUTHORITY_ACTIVATION_EPOCH16_ENDPOINT_V4_NON_DEGENERATE_MACRO",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 16,
        "approval_scope": [
            "ENDPOINT_V4_FREEZE",
            "NON_DEGENERATE_MACRO_RELAXED_POLICY",
            "P2_LEARNABILITY_GATE_RECOMPUTE_NO_RETRAIN",
            "EXCLUDED_PUBLICATIONS_DOCUMENTED",
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
            "SILENT_DROP_OF_DEGENERATE_PUBLICATIONS",
            "RETRAIN_OR_CHANGE_MODEL_PREDICTIONS",
            "MODIFY_FROZEN_EVALUATE_V2",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-08)",
    }

    write_once(AMENDMENT, yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True))
    write_once(APPROVAL, yaml.safe_dump(approval, sort_keys=False, allow_unicode=True))
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")

    # ---- update active_contract.yaml (epoch 16, REBUILD-P2 adjudication) ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_rebuild_authority_epoch_16_20260808"
    au = ac["authority"]
    au["authority_epoch"] = 16
    au["current_phase"] = "REBUILD-P2"
    au["current_authority_state"] = "REBUILD_P2_EPOCH16_NON_DEGENERATE_MACRO_AUTHORIZED"
    au["current_runnable_phase"] = "REBUILD-P2"
    az = ac["authorization"]
    az["runnable_phases"] = ["REBUILD-P2"]
    az["training_allowed"] = True
    az["rebuild_epoch"] = 16
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch16_endpoint_v4_non_degenerate_macro_20260808.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch16_endpoint_v4_approval_20260808.yaml"
    az["endpoint_version"] = "endpoint_v4"
    for ph in ac["phase_graph"]:
        if ph["phase_id"] == "REBUILD-P2":
            ph["lifecycle_status"] = "RUNNING"
            ph["gate_result"] = "NOT_RUN"
            ph["execution_authorized"] = True
            ph["training_required"] = False
            ph["gpu_required"] = False
            ph["metric_semantics"] = "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS"
            ph["authority_epoch"] = 16
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_16.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_16.sentinel.yaml"
    ACTIVE.write_text(
        yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> epoch 16, endpoint_v4, REBUILD-P2")

    # ---- bundle ledger ----
    members = [
        ACTIVE,
        APPROVAL,
        AMENDMENT,
        ENDPOINT_V4,
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
        "authority_epoch": 16,
        "current_authority_state": "REBUILD_P2_EPOCH16_NON_DEGENERATE_MACRO_AUTHORIZED",
        "current_phase": "REBUILD-P2",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_16.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v4",
        "metric_semantics": "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS",
        "recompute": "FROM_EXISTING_HELDOUT_PREDICTIONS_NO_RETRAIN",
    }
    write_once(SENTINEL, yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True))
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
