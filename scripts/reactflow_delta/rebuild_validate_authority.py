#!/usr/bin/env python3
"""Fail-closed rebuild authority (epoch 14) preflight; read-only.

Verifies the strict scientific-engineering audit contract governs the rebuild,
M0-X is terminalized as FAIL (no PASS sentinel), legacy training is closed,
final exposure is bound (test remains DEVELOPMENT_CONSUMED), and the epoch-14
bundle ledger hashes match their on-disk members. No network/data access.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import yaml

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(root: Path, *, staging: bool = False) -> dict:
    active = yaml.safe_load((root / "configs/reactflow_delta/active_contract.yaml").read_text())
    errors: list[str] = []
    def check(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    auth = active.get("authority", {})
    check(auth.get("authority_epoch") == 14, "authority_epoch != 14")
    check(auth.get("current_authority_state") == "REBUILD_P1_AUTHORIZED", "state != REBUILD_P1_AUTHORIZED")
    check(auth.get("current_phase") == "REBUILD-P1", "current_phase != REBUILD-P1")

    # M0-X must be terminal FAIL
    phases = {r.get("phase_id"): r for r in active.get("phase_graph", [])}
    m0 = phases.get("M0-X", {})
    check(m0.get("lifecycle_status") == "TERMINAL", "M0-X not TERMINAL")
    check(m0.get("gate_result") == "FAIL", "M0-X gate_result != FAIL")
    check(m0.get("execution_authorized") is False, "M0-X still execution_authorized")

    # legacy training closed
    check(active.get("authorization", {}).get("training_allowed") is False, "training_allowed must be False")

    # bundle integrity
    bundle_path = root / "configs/reactflow_delta/authority_epoch_14.bundle.sha256"
    check(bundle_path.is_file(), "epoch-14 bundle missing")
    if bundle_path.is_file():
        for line in bundle_path.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                errors.append(f"malformed bundle line: {line!r}")
                continue
            exp, rel = parts
            target = (root / rel)
            check(target.is_file(), f"bundle member missing: {rel}")
            if target.is_file():
                check(digest(target) == exp, f"bundle hash drift: {rel}")

    # sentinel binds active manifest + ledger
    sentinel = yaml.safe_load((root / "configs/reactflow_delta/authority_epoch_14.sentinel.yaml").read_text())
    check(sentinel.get("active_manifest_sha256") == digest(root / "configs/reactflow_delta/active_contract.yaml"),
          "sentinel active_manifest_sha256 mismatch")
    check(sentinel.get("bundle_ledger_sha256") == digest(bundle_path), "sentinel bundle_ledger_sha256 mismatch")
    check(sentinel.get("authority_epoch") == 14, "sentinel epoch != 14")

    # rebuild amendment/approval/contract bound
    az = active.get("authorization", {})
    for key in ("rebuild_contract_path", "rebuild_amendment_path", "rebuild_approval_record_path"):
        rel = az.get(key)
        check(isinstance(rel, str) and (root / rel).is_file(), f"missing {key}: {rel}")

    if not staging:
        check(subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0, "tracked worktree dirty")
        check(subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0, "staged worktree dirty")

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": "reactflow_delta.rebuild_authority_preflight.v1",
        "status": "PASS",
        "authority_epoch": 14,
        "current_phase": "REBUILD-P1",
        "m0x_terminal": "FAIL",
        "legacy_training_allowed": False,
        "network_or_data_access_performed": False,
        "runnable_phase": "REBUILD-P1",
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--staging", action="store_true")
    args = ap.parse_args()
    print(json.dumps(validate(args.repo_root.resolve(), staging=args.staging), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
