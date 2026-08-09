#!/usr/bin/env python3
"""Generate epoch-19 authority: terminalize PHASE3-ARCH fail-closed + benchmark/resource pivot.

Closes the last open phase in active_contract.yaml (PHASE3-ARCH), which was left RUNNING
after all three authorized architecture schemes (DeepSets / exact-alt interaction /
repaired EPRO propagation) were preregistered and retired fail-closed (CI low <= 0 vs the
capacity-matched generic). Per the Phase 3 stop rule the project pivots to the
benchmark/resource route (NO dev13+ free search) and retains the simplest generic.

Note: M0-X was already terminalized as FAIL (`M0X_FAILED_NO_PASS_SENTINEL`) in a prior
epoch; this finalizer leaves that frozen record untouched and only closes PHASE3-ARCH.

Creates (refuses to overwrite existing epoch-19 artifacts):
  * docs/contracts/amendments/reactflow_delta_v4_epoch19_phase3_closure_20260809.yaml
  * docs/approvals/reactflow_delta_v4_epoch19_phase3_closure_approval_20260809.yaml
Updates active_contract.yaml -> epoch 19 (BENCHMARK-RESOURCE, fail-closed closure).
Writes authority_epoch_19.bundle.sha256 + authority_epoch_19.sentinel.yaml.
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
CLOSURE_DOC = ROOT / "docs/audits/reactflow_delta_phase3_closure_fail_closed_20260809.md"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch19_phase3_closure_20260809.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch19_phase3_closure_approval_20260809.yaml"
BUNDLE = CFG / "authority_epoch_19.bundle.sha256"
SENTINEL = CFG / "authority_epoch_19.sentinel.yaml"

NOW = "2026-08-09T08:30:00+08:00"


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
    if not CLOSURE_DOC.exists():
        raise SystemExit(f"MISSING_CLOSURE_DOC: {CLOSURE_DOC}")

    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch19_phase3_closure_20260809",
        "amendment_epoch": 19,
        "previous_amendment": "reactflow_delta_v4_epoch18_phase3_20260808",
        "authorization_status": "ACTIVE_AUTHORIZED",
        "amendment_kind": "PHASE3_FAIL_CLOSED_CLOSURE_AND_BENCHMARK_RESOURCE_PIVOT",
        "amendment_summary": (
            "Terminalize the last open phase PHASE3-ARCH as FAIL_CLOSED_RETIRED. All three "
            "authorized architecture schemes (pair_v1 DeepSets primary, exact_alt_v1 generic "
            "interaction, epro_v2 repaired nonlocal propagation) were executed under epoch 18 "
            "with nested leave-one-publication-out, 5 paired seeds, capacity-matched generic; "
            "each retired fail-closed because the paired publication-block CI low <= 0 vs the "
            "same-capacity generic. Per the preregistered Phase 3 stop rule, the project pivots "
            "to the benchmark/resource route (NO dev13+ free search) and retains the simplest "
            "generic as the development default. M0-X was already terminalized FAIL in a prior "
            "epoch; this finalizer closes only PHASE3-ARCH and rebinds authority to epoch 19."
        ),
        "phase3_closure_verdict": {
            "gate": "PHASE3_DEVELOPMENT_WINNER",
            "result": "FAIL_CLOSED_RETIRED_ALL_SCHEMES",
            "route_pivot": "BENCHMARK_RESOURCE_ROUTE",
            "retained_default": "simplest_capacity_matched_generic",
            "endpoint": "endpoint_v5_frozen",
            "primary_v4_verdict_preserved": "ENDPOINT_V4_PRIMARY_STOP_FROZEN",
            "per_scheme_mean_skill": {
                "pair_v1_deepsets": 0.684,
                "exact_alt_v1": 0.722,
                "epro_v2_repaired": 0.657,
                "generic": 0.679,
            },
            "closure_audit_doc": "docs/audits/reactflow_delta_phase3_closure_fail_closed_20260809.md",
        },
        "benchmark_resource_route": {
            "purpose": (
                "Resource/negative-result evidence chain: caller reliability (50% NO_CALL, "
                "global ICC 0.686), publication label shift (changers rate 0.048-1.0, 20.7x), "
                "magnitude-vs-replicate-noise floor (44.7% below 1x, 61.95% below 1.96x), and "
                "per-publication uniformity of the negative result."),
            "gpu_required": False,
            "training_required": False,
            "no_dev13_plus_free_search": True,
        },
        "invariants_unchanged": {
            "test_split_sealed": True,
            "test_not_used_for_training": True,
            "cross_project_export": False,
            "wet_lab": False,
            "no_fabricated_data": True,
            "no_hide_failure": True,
            "no_lower_gate": True,
            "primary_v4_stop_preserved": True,
            "no_seed_retry_gaming": True,
            "m0x_fail_record_preserved": True,
        },
        "approval_binding": {
            "approval_record_path": "docs/approvals/reactflow_delta_v4_epoch19_phase3_closure_approval_20260809.yaml",
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch19_phase3_closure_approval_20260809",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User explicitly authorized repair of authority integrity: terminalize the last "
            "open Phase 3 (PHASE3-ARCH) as fail-closed retired per the preregistered Phase 3 "
            "stop rule, pivot to the benchmark/resource route, and rebind authority to a new "
            "epoch 19 bundle+sentinel. M0-X was already terminalized FAIL; leave that frozen."
        ),
        "approval_kind": "REBUILD_AUTHORITY_CLOSURE_EPOCH19_PHASE3_FAIL_CLOSED_BENCHMARK_RESOURCE",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 19,
        "approval_scope": [
            "PHASE3_ARCH_FAIL_CLOSED_RETIRED",
            "BENCHMARK_RESOURCE_ROUTE_PIVOT",
            "EPOCH19_BUNDLE_SENTINEL_REBIND",
            "M0X_FAIL_RECORD_PRESERVED",
            "NO_DEV13_PLUS_FREE_SEARCH",
        ],
        "explicit_denials": [
            "SEALED_TEST_UNSEAL",
            "TEST_OUTCOME_FITTING",
            "CROSS_PROJECT_EXPORT",
            "WET_LAB",
            "SOTA_CLAIM",
            "CONFIRMATORY_TEST_ACCESS",
            "OVERWRITE_LEGACY_ARTIFACTS",
            "HIDE_FAILURE",
            "LOWER_GATE_THRESHOLDS",
            "UNSEAL_ENDPOINT_V4_PRIMARY_STOP",
            "DEV13_PLUS_FREE_SEARCH",
            "POST_HOC_AGGREGATION",
            "RERUN_PHASE3_AFTER_CLOSURE",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-09)",
    }

    write_once(AMENDMENT, yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True))
    write_once(APPROVAL, yaml.safe_dump(approval, sort_keys=False, allow_unicode=True))
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")

    # ---- update active_contract.yaml (epoch 19, BENCHMARK-RESOURCE) ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_rebuild_authority_epoch_19_phase3_closure_20260809"
    au = ac["authority"]
    au["authority_epoch"] = 19
    au["current_phase"] = "BENCHMARK-RESOURCE"
    au["current_authority_state"] = "PHASE3_FAIL_CLOSED_BENCHMARK_RESOURCE_EPOCH19"
    au["current_runnable_phase"] = "BENCHMARK-RESOURCE"
    az = ac["authorization"]
    az["runnable_phases"] = ["BENCHMARK-RESOURCE"]
    az["training_allowed"] = False
    az["rebuild_epoch"] = 19
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch19_phase3_closure_20260809.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch19_phase3_closure_approval_20260809.yaml"
    az["endpoint_version"] = "endpoint_v5"

    # close PHASE3-ARCH
    found_ph = False
    for ph in ac["phase_graph"]:
        if ph.get("phase_id") == "PHASE3-ARCH":
            found_ph = True
            ph["lifecycle_status"] = "TERMINAL"
            ph["gate_result"] = "FAIL"
            ph["execution_authorized"] = False
            ph["training_required"] = False
            ph["gpu_required"] = False
            ph["terminal_sentinel_status"] = "PHASE3_ARCH_FAIL_CLOSED_RETIRED"
            ph["closure_audit_doc"] = "docs/audits/reactflow_delta_phase3_closure_fail_closed_20260809.md"
            ph["next_on_pass"] = ["BENCHMARK-RESOURCE"]
            ph["next_on_fail"] = ["BENCHMARK-RESOURCE"]
    if not found_ph:
        raise SystemExit("PHASE3-ARCH not found in phase_graph; aborting")
    # add BENCHMARK-RESOURCE phase (idempotent)
    if not any(ph.get("phase_id") == "BENCHMARK-RESOURCE" for ph in ac["phase_graph"]):
        ac["phase_graph"].append({
            "phase_id": "BENCHMARK-RESOURCE",
            "dependencies": ["PHASE3-ARCH"],
            "lifecycle_status": "RUNNING",
            "gate_result": "NOT_RUN",
            "execution_authorized": True,
            "training_required": False,
            "gpu_required": False,
            "authority_epoch": 19,
            "endpoint": "endpoint_v5",
            "route": "BENCHMARK_RESOURCE_ROUTE",
            "no_dev13_plus_free_search": True,
            "evidence_manifest_path": "results/phase3_benchmark_resource_20260809/manifest.json",
            "verdict_path": "results/phase3_diagnostic_20260809/phase3_diagnostic_table.json",
        })
    ac["governance_resolution"] = {
        "phase3_route": "PHASE3_FAIL_CLOSED_RETIRED_ALL_SCHEMES",
        "phase3_pivot": "BENCHMARK_RESOURCE_ROUTE",
        "phase3_retained_default": "simplest_capacity_matched_generic",
        "m0x_status": "ALREADY_TERMINAL_FAIL_PRESERVED",
        "current_phase": "BENCHMARK-RESOURCE",
    }
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_19.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_19.sentinel.yaml"
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> epoch 19, BENCHMARK-RESOURCE, PHASE3-ARCH closed")

    # ---- bundle ledger ----
    members = [ACTIVE, APPROVAL, AMENDMENT, ENDPOINT_V5, CONTRACT_DOC, CLOSURE_DOC]
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
        "authority_epoch": 19,
        "current_authority_state": "PHASE3_FAIL_CLOSED_BENCHMARK_RESOURCE_EPOCH19",
        "current_phase": "BENCHMARK-RESOURCE",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_19.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v5",
        "metric_semantics": "CONDITIONAL_WMAE_SKILL",
        "phase3_closure": "FAIL_CLOSED_RETIRED_ALL_SCHEMES",
        "route_pivot": "BENCHMARK_RESOURCE_ROUTE",
        "m0x_status": "ALREADY_TERMINAL_FAIL_PRESERVED",
        "primary_v4_verdict": "STOP_FROZEN_EPOCH16",
    }
    write_once(SENTINEL, yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True))
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
