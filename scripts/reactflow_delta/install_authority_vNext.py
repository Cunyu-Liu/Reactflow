#!/usr/bin/env python3
"""Install and validate the prospective-v2 `authority vNext` (epoch 21).

Per ReactFlow-Delta prospective-v2 contract 0.3 / 15.1:
  - The owner P0-P3 authorization line authorizes ONLY the minimal governance
    action of writing, validating and activating `authority vNext` as the single
    ACTIVE authority; it does NOT bypass a stale training_allowed=false machine
    authority for P1/P2/P3.
  - This script (a) validates the epoch-21 active_contract schema + owner auth +
    supersession chain + fail-closed gates, (b) computes the detached bundle
    ledger and sentinel, (c) installs them so the handoff can point to ACTIVE
    authority evidence.

Static/outcome-blind: no network, no data, no training, no locked-outcome read.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return value


CONFIG_SUBDIR = "configs/reactflow_delta"
AUTH_TOKEN = "AUTHORIZE_REACTFLOW_DELTA_PROSPECTIVE_V2_P0_P3"
SUPERSEDED_EPOCH = 20


def validate(root: Path) -> dict[str, Any]:
    cfg = root / CONFIG_SUBDIR
    active = _load_yaml(cfg / "active_contract.yaml")
    checks: dict[str, Any] = {}

    # 1. schema + epoch identity
    checks["schema_version"] = active.get("schema_version") == "reactflow_delta.active_contract.v1"
    checks["epoch_is_21"] = active["authority"]["authority_epoch"] == 21
    checks["state_is_active_prospective_v2"] = (
        active["authority"]["current_authority_state"] == "PROSPECTIVE_V2_EPOCH21_ACTIVE"
    )
    checks["runnable_only_p0"] = active["runnable_phases"] == ["P0"]

    # 2. owner authorization line (independent line) present + exact token
    auth = active.get("authorization", {})
    basis = active["active_contract"].get("effective_authority_basis", {})
    checks["owner_auth_line_exact"] = (
        basis.get("owner_authorization_line") == f"OWNER_AUTHORIZATION: {AUTH_TOKEN}"
    )
    checks["owner_auth_status"] = basis.get("owner_authorization_status") == "PROVIDED_EXACT_MATCH"
    checks["auth_scope_p0_p3_only"] = set(auth.get("approval_scope", [])) == {"P0_P3", "AUTHORITY_VNEXT_RECONCILIATION"}

    # 3. supersession: epoch 20 marked historical, preserve unchanged
    supersedes = active["active_contract"].get("supersedes", [])
    checks["supersedes_epoch20"] = any(
        s.get("authority_disposition") == "SUPERSEDED_HISTORICAL_ONLY"
        and "epoch20" in s.get("contract_id", "")
        and s.get("preservation_status") == "PRESERVE_UNCHANGED"
        for s in supersedes
    )

    # 4. fail-closed: training must NOT be unlocked until P1/P2 gates pass
    checks["training_allowed_false"] = active.get("training_allowed") is False
    checks["candidate_training_allowed_false"] = active.get("candidate_model_training_allowed") is False
    checks["confirmatory_access_false"] = active.get("confirmatory_test_outcome_access_allowed") is False
    checks["p2_p3_training_gated"] = all(
        p["phase_id"] in ("P2", "P3") and p["execution_authorized"] is False
        for p in active["phase_graph"]
        if p["phase_id"] in ("P2", "P3")
    )

    # 5. physical-isolation checks NOT_ESTABLISHED => blocks P2/P3
    ag = active.get("authority_gates", {})
    checks["primary_caller_exclusion_not_established"] = ag.get("primary_caller_exclusion") == "NOT_ESTABLISHED"
    checks["primary_locked_outcome_exclusion_not_established"] = (
        ag.get("primary_locked_outcome_exclusion") == "NOT_ESTABLISHED"
    )
    checks["blocks_p2_p3_while_isolation_unknown"] = (
        ag.get("primary_locked_outcome_exclusion") == "NOT_ESTABLISHED"
    )
    # development may continue while confirmatory sufficiency unknown; P4 blocked
    checks["dev_ok_confirmatory_unknown_p4_blocked"] = (
        ag.get("confirmatory_store_availability") == "NOT_ESTABLISHED"
        and ag.get("confirmatory_statistical_sufficiency") == "NOT_ESTABLISHED"
    )

    all_pass = all(checks.values())
    return {
        "schema_version": "reactflow_delta.prospective_v2.authority_vnext_validation.v1",
        "all_pass": all_pass,
        "checks": checks,
        "verdict": "AUTHORITY_VNEXT_VALIDATED" if all_pass else "AUTHORITY_VNEXT_VALIDATION_FAIL",
    }


def install_bundle_and_sentinel(root: Path) -> dict[str, Any]:
    cfg = root / CONFIG_SUBDIR
    active = _load_yaml(cfg / "active_contract.yaml")
    members = [cfg / p[len(CONFIG_SUBDIR) + 1:] if p.startswith(CONFIG_SUBDIR) else root / p
               for p in active["integrity"]["bundle_member_paths"]]
    bundle_path = cfg / "authority_epoch_21.bundle.sha256"
    lines = []
    for m in members:
        if not m.exists():
            raise FileNotFoundError(f"bundle member missing: {m}")
        lines.append(f"{digest(m)}  {m.relative_to(root)}")
    bundle_content = "\n".join(lines) + "\n"
    bundle_path.write_text(bundle_content, encoding="utf-8")
    bundle_ledger_sha = digest(bundle_path)
    active_manifest_sha = digest(cfg / "active_contract.yaml")

    sentinel = {
        "schema_version": "reactflow_delta.authority_epoch_sentinel.v1",
        "authority_epoch": 21,
        "current_authority_state": "PROSPECTIVE_V2_EPOCH21_ACTIVE",
        "current_phase": "P0",
        "active_manifest_sha256": active_manifest_sha,
        "bundle_ledger_path": f"{CONFIG_SUBDIR}/authority_epoch_21.bundle.sha256",
        "bundle_ledger_sha256": bundle_ledger_sha,
        "created_at": "2026-08-13T18:10:00+08:00",
        "emitted_after": "BUNDLE_LEDGER",
        "embedded_self_sha256": "FORBIDDEN",
        "endpoint_version": "endpoint_v7_all_mutant_full_spectrum",
        "split_version": "split_v4_lopo_puzzle",
        "caller_version": "CALLER_FREE_PRIMARY",
        "scope": "PROSPECTIVE_V2_P0_P3",
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "confirmatory_test_outcome_access_allowed": False,
        "epoch20_status": "SUPERSEDED_HISTORICAL_ONLY",
        "owner_authorization": "PROVIDED_EXACT_MATCH",
    }
    sentinel_path = cfg / "authority_epoch_21.sentinel.yaml"
    sentinel_path.write_text(
        yaml.safe_dump(sentinel, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "active_manifest_sha256": active_manifest_sha,
        "bundle_ledger_sha256": bundle_ledger_sha,
        "sentinel_path": str(sentinel_path.relative_to(root)),
        "bundle_path": str(bundle_path.relative_to(root)),
        "n_bundle_members": len(members),
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    v = validate(root)
    if v["all_pass"]:
        installed = install_bundle_and_sentinel(root)
        v["installed"] = installed
        v["verdict"] = "AUTHORITY_VNEXT_VALIDATED_AND_INSTALLED"
    print(yaml.safe_dump(v, sort_keys=False, allow_unicode=True))
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
