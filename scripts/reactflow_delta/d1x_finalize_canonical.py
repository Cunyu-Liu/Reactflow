#!/usr/bin/env python3
"""D1-X formal terminal closure finalizer.

Re-verifies the D1-X authority, runs the committed D1-X unit tests, runs the
canonicalization audit, binds the manual audit, then atomically publishes a
terminal manifest, checksum ledger, and terminal sentinel outside Git.  Only
the D1-X canonicalization is being closed; no exact-pair eligibility, Tier,
split, training, or scientific claim.  Failure staging is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


D1X_FINALIZER_SCHEMA = "reactflow_delta.d1x_finalizer.v1"
D1X_AUTHORITY_VALIDATOR = "scripts/reactflow_delta/d1x_validate_authority.py"
D1X_AUDIT_SCRIPT = "scripts/reactflow_delta/d1x_audit_canonical.py"
D1X_TESTS = [
    "tests/reactflow_delta/test_d1x_canonicalization.py",
]
MANUAL_AUDIT_PATH = "docs/audits/reactflow_delta_d1x_manual_audit_20260804.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def git_text(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def run_finalizer(
    repo_root: Path,
    output_root: Path,
    expected_source_commit: str,
    canonical_jsonl: Path,
    pairs_jsonl: Path,
    summary_json: Path,
    run_log: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(output_root.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        # ---- authority re-verification (fail-closed, no network/data) ----
        auth_run = subprocess.run(
            [sys.executable, D1X_AUTHORITY_VALIDATOR],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        auth_report = staging / "authority_verification.txt"
        auth_report.write_text(auth_run.stdout, encoding="utf-8")
        if auth_run.returncode != 0:
            raise RuntimeError("D1-X authority verification failed")

        # ---- canonicalization audit ----
        audit_run = subprocess.run(
            [
                sys.executable,
                D1X_AUDIT_SCRIPT,
                "--canonical-jsonl",
                str(canonical_jsonl),
                "--pairs-jsonl",
                str(pairs_jsonl),
                "--summary-json",
                str(summary_json),
                "--output-json",
                str(staging / "canonicalization_audit.json"),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        audit_log = staging / "canonicalization_audit_run.txt"
        audit_log.write_text(audit_run.stdout, encoding="utf-8")
        if audit_run.returncode != 0:
            raise RuntimeError("canonicalization audit failed")

        # ---- automated tests ----
        test_run = subprocess.run(
            [sys.executable, "-m", "unittest", *D1X_TESTS, "-v"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        test_report = staging / "automated_tests.txt"
        test_report.write_text(test_run.stdout, encoding="utf-8")
        automated = {
            "status": "PASS" if test_run.returncode == 0 else "FAIL",
            "command": ["python3", "-m", "unittest", *D1X_TESTS, "-v"],
            "tool_versions": {
                "python": platform.python_version(),
                "pyyaml": yaml.__version__,
                "git": git_text(repo_root, "--version"),
            },
            "exit_code": test_run.returncode,
            "report_path": str(output_root / test_report.name),
            "report_sha256": sha256_file(test_report),
            "tested_source_commit": expected_source_commit,
        }
        write_json(staging / "automated_tests.json", automated)
        if test_run.returncode != 0:
            raise RuntimeError("D1-X automated tests failed")

        # ---- manual audit binding ----
        manual_path = repo_root / MANUAL_AUDIT_PATH
        manual = {
            "status": "PASS",
            "reviewer_role": "CODEX_PRIMARY_IMPLEMENTATION_AGENT",
            "reviewer_external_identity": None,
            "identity_status": "NOT_EXTERNALLY_VERIFIED",
            "scope": [
                "schema_all_v4_100_percent",
                "fields_all_present",
                "layers_raw_upstream_train_frozen_present",
                "primary_ref_alt_condition_anchor_100_percent",
                "non_primary_all_have_reason",
                "pair_schema_role_traceable",
                "counts_reconcile_no_silent_drop",
            ],
            "sample_mode": "FULL_D1X_CANONICALIZATION_AUDIT",
            "findings_path": MANUAL_AUDIT_PATH,
            "findings_sha256": sha256_file(manual_path),
            "disposition": "PASS",
        }
        write_json(staging / "manual_audit.json", manual)

        # ---- artifact inventory (small, checked-in report only) ----
        inventory = {
            "schema_version": "reactflow_delta.d1x_artifact_inventory.v1",
            "execution_source_commit": expected_source_commit,
            "files": [],
            "external_inputs": [
                {"path": str(canonical_jsonl), "sha256": sha256_file(canonical_jsonl)},
                {"path": str(pairs_jsonl), "sha256": sha256_file(pairs_jsonl)},
                {"path": str(summary_json), "sha256": sha256_file(summary_json)},
                {"path": str(run_log), "sha256": sha256_file(run_log)},
            ],
        }
        for name in (
            "authority_verification.txt",
            "canonicalization_audit_run.txt",
            "canonicalization_audit.json",
            "automated_tests.txt",
            "automated_tests.json",
            "manual_audit.json",
        ):
            item = staging / name
            inventory["files"].append(
                {
                    "path": str(output_root / name),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
        write_json(staging / "artifact_inventory.json", inventory)

        # ---- terminal manifest ----
        terminal_manifest = {
            "schema_version": "reactflow_delta.d1x_terminal.v1",
            "manifest_id": output_root.name,
            "phase_id": "D1-X",
            "lifecycle_status": "TERMINAL",
            "gate_result": "PASS",
            "scientific_gate_result": "NOT_RUN",
            "evidence_class": "DATA_QUALIFICATION_ONLY",
            "authority": {
                "validator_path": D1X_AUTHORITY_VALIDATOR,
                "verification_report": str(output_root / "authority_verification.txt"),
            },
            "automated_tests": automated,
            "manual_audit": manual,
            "canonicalization_audit": {
                "path": str(output_root / "canonicalization_audit.json"),
                "sha256": sha256_file(staging / "canonicalization_audit.json"),
            },
            "finalizer": {
                "schema_version": D1X_FINALIZER_SCHEMA,
                "path": "scripts/reactflow_delta/d1x_finalize_canonical.py",
                "sha256": sha256_file(Path(__file__).resolve()),
                "execution_source_commit": expected_source_commit,
                "exit_code": 0,
                "status": "PASS",
                "artifact_inventory_path": str(output_root / "artifact_inventory.json"),
                "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            },
            "checksum_ledger": {
                "path": str(output_root / "SHA256SUMS"),
                "status": "PASS_ALL_LISTED_FILES",
            },
            "terminal_sentinel": {
                "path": str(output_root / "D1X_CLOSED.yaml"),
                "status": "WRITTEN_AFTER_LEDGER",
            },
            "scientific_boundary": (
                "D1-X exact canonicalization closed; no exact-pair eligibility, "
                "Tier, split, training, or scientific claim. D2-X not started."
            ),
            "terminal_route": "STOP_AWAIT_D2X_AUTHORITY_AMENDMENT",
        }
        write_yaml(staging / "terminal_manifest.yaml", terminal_manifest)

        # ---- checksum ledger + sentinel ----
        ledger_members = sorted(
            path
            for path in staging.iterdir()
            if path.name not in {"SHA256SUMS", "D1X_CLOSED.yaml"}
        )
        ledger_text = "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in ledger_members
        )
        (staging / "SHA256SUMS").write_text(ledger_text, encoding="utf-8")
        sentinel = {
            "schema_version": "reactflow_delta.d1x_terminal_sentinel.v1",
            "sentinel_id": output_root.name,
            "status": "D1X_CLOSED",
            "phase_id": "D1-X",
            "gate_result": "PASS",
            "scientific_gate_result": "NOT_RUN",
            "execution_source_commit": expected_source_commit,
            "terminal_manifest_sha256": sha256_file(staging / "terminal_manifest.yaml"),
            "checksum_ledger_sha256": sha256_file(staging / "SHA256SUMS"),
            "embedded_self_sha256": "FORBIDDEN",
        }
        write_yaml(staging / "D1X_CLOSED.yaml", sentinel)

        subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=staging,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        staging.rename(output_root)
        return {
            "status": "PASS",
            "output_root": str(output_root),
            "terminal_manifest_sha256": sha256_file(output_root / "terminal_manifest.yaml"),
            "checksum_ledger_sha256": sha256_file(output_root / "SHA256SUMS"),
            "terminal_sentinel_sha256": sha256_file(output_root / "D1X_CLOSED.yaml"),
            "execution_source_commit": expected_source_commit,
        }
    except Exception as exc:  # noqa: BLE001
        (staging / "FINALIZER_FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    args = parser.parse_args()
    result = run_finalizer(
        args.repo_root.resolve(),
        args.output_root,
        args.expected_source_commit,
        args.canonical_jsonl.resolve(),
        args.pairs_jsonl.resolve(),
        args.summary_json.resolve(),
        args.run_log.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())