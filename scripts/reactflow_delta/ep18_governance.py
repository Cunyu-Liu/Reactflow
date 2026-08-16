#!/usr/bin/env python3
"""Generate epoch-18 authority for Phase 3 model architecture iteration.

Creates (refuses to overwrite existing epoch-18 artifacts):
  * docs/contracts/amendments/reactflow_delta_v4_epoch18_phase3_20260808.yaml
  * docs/approvals/reactflow_delta_v4_epoch18_phase3_approval_20260808.yaml
Updates active_contract.yaml -> epoch 18 (PHASE3-ARCH, endpoint_v5) with the
per-scheme budgets / max-2-rounds / stop rules preregistered in the amendment.
Writes authority_epoch_18.bundle.sha256 + authority_epoch_18.sentinel.yaml.

Phase 3 = 模型架构迭代 (contract §Phase 3). Primary scheme = conditional-magnitude
pair head (方案一). Prereq Phase 2 GO satisfied (P2 conditional-magnitude GO, epoch 17).
Endpoint_v5 input/output frozen. Primary endpoint_v4 STOP preserved (epoch 16).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
ENDPOINT_V5 = CFG / "endpoint_v5.yaml"
AMEND_DIR = ROOT / "docs/contracts/amendments"
APPROVAL_DIR = ROOT / "docs/approvals"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch18_phase3_20260808.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch18_phase3_approval_20260808.yaml"
BUNDLE = CFG / "authority_epoch_18.bundle.sha256"
SENTINEL = CFG / "authority_epoch_18.sentinel.yaml"

NOW = "2026-08-08T21:00:00+08:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE: {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if not ENDPOINT_V5.exists():
        raise SystemExit(f"MISSING_ENDPOINT_V5: {ENDPOINT_V5}")

    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch18_phase3_20260808",
        "amendment_epoch": 18,
        "previous_amendment": "reactflow_delta_v4_epoch17_endpoint_v5_conditional_magnitude_20260808",
        "authorization_status": "ACTIVE_AUTHORIZED",
        "amendment_kind": "PHASE3_MODEL_ARCHITECTURE_ITERATION",
        "amendment_summary": (
            "Phase 3 model architecture iteration (contract §Phase 3) after the "
            "P2 conditional-magnitude learnability GO (epoch 17, endpoint_v5). "
            "Primary scheme (方案一) = conditional-magnitude pair head, evaluated "
            "against a capacity-matched generic baseline under nested "
            "leave-one-publication-out, 5 deterministic seeds. Only contract-listed "
            "capabilities are tested: pair-level alignment, exact-alt x WT-state "
            "generic interaction, and controlled nonlocal propagation. Endpoint_v5 "
            "input/output frozen; primary endpoint_v4 STOP preserved (epoch 16)."
        ),
        "phase3_budget_and_stop_rules": {
            "schemes": [
                "pair_v1_conditional_magnitude_head (primary)",
                "exact_alt_v1_generic_interaction",
                "epro_v2_repaired_nonlocal_propagation",
            ],
            "seeds": [0, 1, 2, 3, 4],
            "selection": "preregistered best by mean conditional WMAE skill on development outer folds",
            "capacity_matching": "per-scheme params/FLOPs/train-time matched to capacity-matched generic",
            "one_capability_at_a_time": True,
            "max_iteration_rounds_per_scheme": 2,
            "paired_ablation": "5 seeds paired; 5-fold paired CI",
            "stop_rules": [
                "scheme with development outer-fold CI low <= 0 vs capacity-matched generic retires immediately",
                "if all schemes fail, use simplest generic and pivot to benchmark/resource route; NO dev13+ free search",
                "no post-hoc aggregation; no test/held-out tuning",
            ],
            "gpu_required": True,
            "cuda_unavailable_policy": "STOP_AND_PRESERVE_EVIDENCE (no CPU silent fallback)",
        },
        "acceptance_criteria": {
            "candidate_beats_capacity_matched_generic_ci_low_gt_0": True,
            "ablations_match_preregistered_direction": True,
            "no_gradient_or_convergence_failure": True,
            "gain_exceeds_seed_variance": True,
            "no_post_hoc_aggregation": True,
        },
        "amendment_routes": {
            "phase": "PHASE3-ARCH",
            "gate": "PHASE3_DEVELOPMENT_WINNER",
            "primary_verdict_preserved": "ENDPOINT_V4_PRIMARY_STOP_FROZEN",
            "endpoint": "endpoint_v5_frozen",
            "training": "PHASE3_LOOCV_GPU",
            "module_new": [
                "reactflow_delta/models/pair_v1.py",
                "reactflow_delta/models/exact_alt_v1.py",
                "reactflow_delta/models/epro_v2.py",
                "reactflow_delta/train_v2.py",
                "reactflow_delta/samplers.py",
                "tests/reactflow_delta/test_model_invariants_v2.py",
            ],
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
            "no_certified_untouched_confirmatory_publications": True,
            "confirmatory_test_access": False,
        },
        "approval_binding": {
            "approval_record_path": "docs/approvals/reactflow_delta_v4_epoch18_phase3_approval_20260808.yaml",
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch18_phase3_approval_20260808",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User explicitly authorized authority epoch 18, Phase 3 model architecture "
            "iteration (conditional-magnitude pair head as primary scheme), with the "
            "preregistered budgets / max-2-rounds / stop rules from the Phase 3 proposal."
        ),
        "approval_kind": "REBUILD_AUTHORITY_ACTIVATION_EPOCH18_PHASE3_ARCHITECTURE",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 18,
        "approval_scope": [
            "PHASE3_MODEL_ARCHITECTURE_ITERATION",
            "CONDITIONAL_MAGNITUDE_PAIR_HEAD_PRIMARY",
            "CAPACITY_MATCHED_GENERIC_PAIRED_ABLATION",
            "5_SEEDS_LOOCV_GPU",
            "MAX_2_ROUNDS_PER_SCHEME",
            "ENDPOINT_V5_FROZEN",
            "DEVELOPMENT_WINNER_PREREGISTERED_FREEZE",
        ],
        "explicit_denials": [
            "SEALED_TEST_UNSEAL",
            "TEST_OUTCOME_FITTING",
            "CROSS_PROJECT_EXPORT",
            "WET_LAB",
            "EPRO_SOTA_CLAIM",
            "CONFIRMATORY_TEST_ACCESS",
            "OVERWRITE_LEGACY_ARTIFACTS",
            "HIDE_M0X_FAILURE",
            "LOWER_GATE_THRESHOLDS",
            "UNSEAL_ENDPOINT_V4_PRIMARY_STOP",
            "DEV13_PLUS_FREE_SEARCH_IF_ALL_SCHEMES_FAIL",
            "POST_HOC_AGGREGATION",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-09)",
    }

    write_once(AMENDMENT, yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True))
    write_once(APPROVAL, yaml.safe_dump(approval, sort_keys=False, allow_unicode=True))
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")

    # ---- update active_contract.yaml (epoch 18, PHASE3-ARCH) ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_rebuild_authority_epoch_18_phase3_20260808"
    au = ac["authority"]
    au["authority_epoch"] = 18
    au["current_phase"] = "PHASE3-ARCH"
    au["current_authority_state"] = "PHASE3_ARCH_EPOCH18_ENDPOINT_V5_AUTHORIZED"
    au["current_runnable_phase"] = "PHASE3-ARCH"
    az = ac["authorization"]
    az["runnable_phases"] = ["PHASE3-ARCH"]
    az["training_allowed"] = True
    az["rebuild_epoch"] = 18
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch18_phase3_20260808.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch18_phase3_approval_20260808.yaml"
    az["endpoint_version"] = "endpoint_v5"
    found = False
    for ph in ac["phase_graph"]:
        if ph.get("phase_id") == "PHASE3-ARCH":
            found = True
            ph["lifecycle_status"] = "RUNNING"
            ph["gate_result"] = "NOT_RUN"
            ph["execution_authorized"] = True
            ph["training_required"] = True
            ph["gpu_required"] = True
            ph["metric_semantics"] = "CONDITIONAL_WMAE_SKILL"
            ph["authority_epoch"] = 18
            ph["endpoint"] = "endpoint_v5"
            ph["budget_and_stop_rules"] = amendment["phase3_budget_and_stop_rules"]
    if not found:
        ac["phase_graph"].append({
            "phase_id": "PHASE3-ARCH",
            "dependencies": ["CONDITIONAL-MAGNITUDE"],
            "lifecycle_status": "RUNNING",
            "gate_result": "NOT_RUN",
            "execution_authorized": True,
            "training_required": True,
            "gpu_required": True,
            "metric_semantics": "CONDITIONAL_WMAE_SKILL",
            "authority_epoch": 18,
            "endpoint": "endpoint_v5",
            "budget_and_stop_rules": amendment["phase3_budget_and_stop_rules"],
        })
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_18.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_18.sentinel.yaml"
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> epoch 18, endpoint_v5, PHASE3-ARCH")

    # ---- bundle ledger ----
    members = [ACTIVE, APPROVAL, AMENDMENT, ENDPOINT_V5, CONTRACT_DOC]
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
        "authority_epoch": 18,
        "current_authority_state": "PHASE3_ARCH_EPOCH18_ENDPOINT_V5_AUTHORIZED",
        "current_phase": "PHASE3-ARCH",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_18.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v5",
        "metric_semantics": "CONDITIONAL_WMAE_SKILL",
        "primary_v4_verdict": "STOP_FROZEN_EPOCH16",
        "phase3_max_rounds_per_scheme": 2,
        "phase3_primary_scheme": "pair_v1_conditional_magnitude_head",
    }
    write_once(SENTINEL, yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True))
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
