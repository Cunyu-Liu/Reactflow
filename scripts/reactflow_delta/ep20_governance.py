#!/usr/bin/env python3
"""Generate epoch-20 authority: PHASE1_BENCHMARK_V3_ONLY in the isolated worktree.

Supersedes the stale semantic fields of epoch 19 (which left
BENCHMARK-RESOURCE with RUNNING/NOT_RUN scope ambiguity). This epoch binds the
benchmark_v3 rebuild to the isolated worktree
`/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809` on branch
`codex/reactflow-delta-benchmark-v3-20260809` at the plan source commit.

Scope (authorized): provenance, parser/disposition, physical stores, Caller
diagnostics, schema/evaluator fixtures, endpoint_v6 + CallerV4 information-
permission freeze, split v3, physical test isolation, statistical design.
Training/candidate/test-outcome actions are DENIED.

Creates (refuses to overwrite existing epoch-20 artifacts):
  * docs/contracts/amendments/reactflow_delta_v4_epoch20_benchmark_v3_20260809.yaml
  * docs/approvals/reactflow_delta_v4_epoch20_benchmark_v3_approval_20260809.yaml
  * docs/audits/reactflow_delta_phase1_benchmark_v3_plan_20260809.md
Updates active_contract.yaml -> epoch 20 (PHASE1_BENCHMARK_V3_ONLY).
Writes authority_epoch_20.bundle.sha256 + authority_epoch_20.sentinel.yaml.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path("/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809")
CFG = ROOT / "configs/reactflow_delta"
ACTIVE = CFG / "active_contract.yaml"
ENDPOINT_V6 = CFG / "endpoint_v6.yaml"
ENDPOINT_V5 = CFG / "endpoint_v5.yaml"
SPLIT_V3 = CFG / "split_v3.yaml"
AMEND_DIR = ROOT / "docs/contracts/amendments"
APPROVAL_DIR = ROOT / "docs/approvals"
AUDIT_DIR = ROOT / "docs/audits"
CONTRACT_DOC = ROOT / "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"
STAT_DESIGN = ROOT / "docs/reactflow_delta/statistical_design_v1.md"
SPLIT_POLICY = ROOT / "docs/reactflow_delta/split_policy_v3.md"

AMENDMENT = AMEND_DIR / "reactflow_delta_v4_epoch20_benchmark_v3_20260809.yaml"
APPROVAL = APPROVAL_DIR / "reactflow_delta_v4_epoch20_benchmark_v3_approval_20260809.yaml"
PLAN_AUDIT = AUDIT_DIR / "reactflow_delta_phase1_benchmark_v3_plan_20260809.md"
BUNDLE = CFG / "authority_epoch_20.bundle.sha256"
SENTINEL = CFG / "authority_epoch_20.sentinel.yaml"

NOW = "2026-08-09T00:00:00+00:00"

ARTIFACT_ROOT = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/benchmark_v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE: {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    for p in (ENDPOINT_V6, SPLIT_V3, STAT_DESIGN, SPLIT_POLICY, CONTRACT_DOC):
        if not p.exists():
            raise SystemExit(f"MISSING_REQUIRED: {p}")
    commit = source_commit()
    print(f"[source commit] {commit}")

    plan_audit = (
        "# ReactFlow-Delta Phase 1 (benchmark_v3) — plan & authority plan\n\n"
        "- date: 2026-08-09\n"
        "- authority epoch: 20\n"
        "- scope: PHASE1_BENCHMARK_V3_ONLY (no training, no candidate model, no confirmatory outcome)\n"
        "- worktree: `/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809`\n"
        "- branch: `codex/reactflow-delta-benchmark-v3-20260809`\n"
        f"- source commit: `{commit}`\n"
        f"- artifact root: `{ARTIFACT_ROOT}`\n\n"
        "## Mandated Phase-1 batches\n\n"
        "- Batch 0A fresh preflight (done, epoch 19 untouched)\n"
        "- Batch 0B isolated worktree + epoch 20 authority (this record)\n"
        "- Batch 1A asset disposition v3 + pair publication registry v1 + sequence/lineage leakage audit\n"
        "- Batch 1B split v3 + physical test isolation + exposure ledger + statistical design\n"
        "- Batch 1C endpoint_v6 + CallerV4 information-permission freeze + sensitivity gate\n"
        "- Batch 1D keyed prediction schema v2 + evaluate_v6 fixtures\n\n"
        "## Hard constraints\n\n"
        "- No learned baseline trained (Phase 2 needs `AUTHORIZE_PHASE2_LEARNABILITY`)\n"
        "- No confirmatory outcome opened (outcome-blind metadata only)\n"
        "- No push/PR; focused local commits only\n"
        "- Same PMID/DOI never crosses split; UNKNOWN never adds confirmatory N\n"
        "- target eligibility mask only in label/evaluator, never model input\n"
        "- fail closed at every scientific gate\n"
    )
    write_once(PLAN_AUDIT, plan_audit)

    amendment = {
        "contract_schema": "reactflow_delta.contract.v4",
        "amendment_id": "reactflow_delta_v4_epoch20_benchmark_v3_20260809",
        "amendment_epoch": 20,
        "previous_amendment": "reactflow_delta_v4_epoch19_phase3_closure_20260809",
        "authorization_status": "ACTIVE_AUTHORIZED",
        "amendment_kind": "BENCHMARK_V3_REBUILD_PHASE1_ONLY",
        "amendment_summary": (
            "Rebuild a publication-verified, replicate-aware, physically isolated test, "
            "biological-key-aligned ReactFlow-Delta benchmark/evaluator under the isolated "
            "worktree benchmark_v3_20260809. Supersedes the stale semantic fields of epoch 19 "
            "(BENCHMARK-RESOURCE RUNNING/NOT_RUN ambiguity). Scope is strictly Phase 1 "
            "(provenance, disposition, split, physical stores, Caller diagnostics, endpoint_v6 "
            "CallerV4 freeze, keyed prediction schema v2, evaluate_v6). NO training, NO "
            "candidate model, NO confirmatory outcome access. Only after a strong simple/generic "
            "model proves cross-real-publication learnability (Phase 2) may method modeling be "
            "re-authorized."
        ),
        "source_commit": commit,
        "worktree": str(ROOT),
        "branch": "codex/reactflow-delta-benchmark-v3-20260809",
        "artifact_root": ARTIFACT_ROOT,
        "scope": "PHASE1_BENCHMARK_V3_ONLY",
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "confirmatory_test_outcome_access_allowed": False,
        "benchmark_resource_epoch19": {
            "state": "SUPERSEDED_BY_EPOCH20_FOR_SCOPE_SEMANTICS",
            "note": "epoch 19 bundle/sentinel hashes remain valid; only the ambiguous "
                    "BENCHMARK-RESOURCE scope/exit semantics are superseded by epoch 20.",
        },
        "historical_development_evidence": {
            "route_c": "HISTORICAL_DEVELOPMENT_EVIDENCE_ONLY",
            "scheme_1_to_3": "HISTORICAL_DEVELOPMENT_EVIDENCE_ONLY",
            "note": "Old Route-C and Scheme 1-3 results are retained as historical "
                    "development evidence, never as confirmatory or prospective claims.",
        },
        "phase1_authorized_items": [
            "provenance_crosswalk",
            "parser_disposition",
            "physical_stores",
            "caller_diagnostics",
            "endpoint_v6_caller_v4_freeze",
            "split_v3_physical_test_isolation",
            "keyed_prediction_schema_v2",
            "evaluate_v6_fixtures",
            "statistical_design",
        ],
        "invariants_unchanged": {
            "test_not_used_for_training": True,
            "primary_v4_stop_preserved": True,
            "m0x_fail_record_preserved": True,
            "no_fabricated_data": True,
            "no_hide_failure": True,
            "no_lower_gate": True,
            "no_seed_retry_gaming": True,
            "no_push_pr": True,
            "no_cleanup_untracked": True,
        },
        "approval_binding": {
            "approval_record_path": "docs/approvals/reactflow_delta_v4_epoch20_benchmark_v3_approval_20260809.yaml",
            "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
            "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        },
    }
    approval = {
        "schema_version": "reactflow_delta.approval_record.v1",
        "approval_id": "reactflow_delta_v4_epoch20_benchmark_v3_approval_20260809",
        "project_task_id": "reactflow_delta",
        "approved_at": NOW,
        "approval_source": "EXPLICIT_USER_MESSAGE_CONTEXT_VERIFIED",
        "user_instruction": (
            "User submitted the post-audit recovery master execution prompt and required its "
            "implementation from Phase 0 through Phase 1 complete gate. This authorizes Phase 0 "
            "fresh preflight and Phase 1 benchmark/evaluator/provenance/Caller rebuild in the "
            "isolated worktree, plus focused local commits. It does NOT authorize Phase 2+ "
            "(which need per-phase tokens), training, test-outcome access, push/PR, or cleanup."
        ),
        "approval_kind": "BENCHMARK_V3_REBUILD_PHASE1_ONLY_EPOCH20",
        "approved_contract_path": "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md",
        "approved_epoch": 20,
        "approval_scope": [
            "PHASE0_FRESH_PREFLIGHT",
            "PHASE1_BENCHMARK_V3_REBUILD",
            "ENDPOINT_V6_CALLER_V4_FREEZE",
            "SPLIT_V3_PHYSICAL_TEST_ISOLATION",
            "KEYED_PREDICTION_SCHEMA_V2",
            "EVALUATE_V6_FIXTURES",
            "FOCUSED_LOCAL_COMMITS_ONLY",
        ],
        "explicit_denials": [
            "TRAINING",
            "CANDIDATE_MODEL_TRAINING",
            "CONFIRMATORY_TEST_OUTCOME_ACCESS",
            "PHASE2_LEARNABILITY",
            "PHASE3_METHOD",
            "PHASE4_LOCKED_TEST",
            "PHASE5_6_PUBLICATION_RELEASE",
            "PUSH",
            "PR",
            "SEALED_TEST_UNSEAL",
            "SOTA_CLAIM",
            "OVERWRITE_LEGACY_ARTIFACTS",
            "CLEANUP_UNTACKED",
            "HIDE_FAILURE",
            "LOWER_GATE_THRESHOLDS",
        ],
        "signature_status": "SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "external_identity_status": "VERIFIED_VIA_EXPLICIT_USER_MESSAGE",
        "signed_by": "user (explicit message 2026-08-09)",
    }

    write_once(AMENDMENT, yaml.safe_dump(amendment, sort_keys=False, allow_unicode=True))
    write_once(APPROVAL, yaml.safe_dump(approval, sort_keys=False, allow_unicode=True))
    print(f"[written] {AMENDMENT.name}")
    print(f"[written] {APPROVAL.name}")
    print(f"[written] {PLAN_AUDIT.name}")

    # ---- update active_contract.yaml (epoch 20, PHASE1_BENCHMARK_V3_ONLY) ----
    ac = yaml.safe_load(ACTIVE.read_text(encoding="utf-8"))
    ac["manifest_id"] = "reactflow_delta_v4_authority_epoch_20_benchmark_v3_20260809"
    au = ac["authority"]
    au["authority_epoch"] = 20
    au["current_phase"] = "PHASE1_BENCHMARK_V3"
    au["current_authority_state"] = "PHASE1_BENCHMARK_V3_REBUILD_EPOCH20"
    au["current_runnable_phase"] = "PHASE1_BENCHMARK_V3"
    az = ac["authorization"]
    az["runnable_phases"] = ["PHASE1_BENCHMARK_V3"]
    az["training_allowed"] = False
    az["candidate_model_training_allowed"] = False
    az["confirmatory_test_outcome_access_allowed"] = False
    az["rebuild_epoch"] = 20
    az["rebuild_amendment_path"] = "docs/contracts/amendments/reactflow_delta_v4_epoch20_benchmark_v3_20260809.yaml"
    az["rebuild_approval_record_path"] = "docs/approvals/reactflow_delta_v4_epoch20_benchmark_v3_approval_20260809.yaml"
    az["endpoint_version"] = "endpoint_v6"
    az["caller_version"] = "caller_v4"
    az["split_version"] = "split_v3"
    au["source_commit"] = commit
    au["scope"] = "PHASE1_BENCHMARK_V3_ONLY"
    au["artifact_root"] = ARTIFACT_ROOT

    # add PHASE1-BENCHMARK-V3 phase (idempotent)
    if not any(ph.get("phase_id") == "PHASE1-BENCHMARK-V3" for ph in ac["phase_graph"]):
        ac["phase_graph"].append({
            "phase_id": "PHASE1-BENCHMARK-V3",
            "dependencies": ["BENCHMARK-RESOURCE"],
            "lifecycle_status": "RUNNING",
            "gate_result": "NOT_RUN",
            "execution_authorized": True,
            "training_required": False,
            "candidate_model_training_allowed": False,
            "confirmatory_test_outcome_access_allowed": False,
            "gpu_required": False,
            "authority_epoch": 20,
            "endpoint": "endpoint_v6",
            "caller": "caller_v4",
            "split": "split_v3",
            "route": "BENCHMARK_V3_REBUILD",
        })
    ac["governance_resolution"] = {
        "benchmark_v3_route": "PHASE1_BENCHMARK_V3_REBUILD",
        "epoch19_status": "SUPERSEDED_FOR_SCOPE_SEMANTICS",
        "current_phase": "PHASE1_BENCHMARK_V3",
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "confirmatory_test_outcome_access_allowed": False,
    }
    ig = ac["integrity"]
    ig["detached_bundle_ledger_path"] = "configs/reactflow_delta/authority_epoch_20.bundle.sha256"
    ig["authority_sentinel_path"] = "configs/reactflow_delta/authority_epoch_20.sentinel.yaml"
    ACTIVE.write_text(yaml.safe_dump(ac, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[updated] {ACTIVE.name} -> epoch 20, PHASE1_BENCHMARK_V3")

    # ---- bundle ledger ----
    members = [ACTIVE, APPROVAL, AMENDMENT, ENDPOINT_V6, ENDPOINT_V5, SPLIT_V3,
               CONTRACT_DOC, STAT_DESIGN, SPLIT_POLICY, PLAN_AUDIT]
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
        "authority_epoch": 20,
        "current_authority_state": "PHASE1_BENCHMARK_V3_REBUILD_EPOCH20",
        "current_phase": "PHASE1_BENCHMARK_V3",
        "active_manifest_sha256": sha256(ACTIVE),
        "bundle_ledger_path": "configs/reactflow_delta/authority_epoch_20.bundle.sha256",
        "bundle_ledger_sha256": bundle_sha,
        "created_at": NOW,
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v6",
        "caller_version": "caller_v4",
        "split_version": "split_v3",
        "scope": "PHASE1_BENCHMARK_V3_ONLY",
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "confirmatory_test_outcome_access_allowed": False,
        "source_commit": commit,
        "artifact_root": ARTIFACT_ROOT,
        "epoch19_status": "SUPERSEDED_FOR_SCOPE_SEMANTICS",
        "m0x_status": "ALREADY_TERMINAL_FAIL_PRESERVED",
        "primary_v4_verdict": "STOP_FROZEN_EPOCH16",
    }
    write_once(SENTINEL, yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True))
    print(f"[written] {SENTINEL.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())