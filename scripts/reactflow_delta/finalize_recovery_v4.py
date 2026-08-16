#!/usr/bin/env python3
"""Create an append-only formal Recovery finalization receipt.

The finalizer validates the immutable epoch-1 Git receipt, runs the committed
automated tests, binds the manual audit, rechecks pilot closure and protected
untracked files, then atomically publishes a terminal manifest, checksum ledger,
and sentinel outside Git.  Failure staging directories are preserved.
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

from validate_recovery_v4 import RECEIPT_COMMIT, sha256, validate_receipt


FINALIZER_SCHEMA = "reactflow_delta.recovery_finalizer.v2"
MANUAL_REPORT = "docs/audits/reactflow_delta_v4_recovery_manual_audit_20260803.md"
TEST_PATH = "tests/reactflow_delta/test_recovery_authority_v4.py"
FINALIZER_PATH = "scripts/reactflow_delta/finalize_recovery_v4.py"
VALIDATOR_PATH = "scripts/reactflow_delta/validate_recovery_v4.py"
PILOT_ROOT = Path(
    "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/recovery/"
    "pilot_closure_20260803T200500+0800_r2"
)
PROTECTED_UNTRACKED = {
    "analyze_best_ckpt.py": "a2673d5a7cae7f1b3243840bb204464d2b9d33eb42dbf5ab99250c257863f9d8",
    "analyze_r2.py": "5bca8ae2890e7d5c310bc93f5611a6320b3058cecaf9d89e1376c9c5582a7794",
    "configs/reactflow_delta/epro_lite_v2.yaml.bak": "8690a59c6e368cec63bf92feaaf3883c105f22c4f763204489a1b64c66f05062",
    "docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_3_TrainingAuth_20260731.md": "9a38b220ecb6f324cf8a33b19bd30bfa98ada1112df0db440d9c1bc67ecf4f45",
    "docs/d2r_methodology.md": "ff5cbcc8a52223b6e8dcfb51f2cf1a587ee84e3ab6cd822ae0392c1418eaeaaf",
    "scripts/reactflow_delta/m0r2_gate_assess.py.bak_v1": "5969a318547c528625b4ca0d0cafa7fafcbf67046eac5e3b4738eb81ad3a6257",
}


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


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


def assert_clean_and_protected(root: Path, expected_source_commit: str) -> None:
    if git_text(root, "rev-parse", "HEAD") != expected_source_commit:
        raise RuntimeError("current HEAD does not equal expected finalizer source commit")
    if subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0:
        raise RuntimeError("tracked worktree is dirty")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0:
        raise RuntimeError("staged worktree is dirty")
    status_bytes = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
    )
    observed: set[str] = set()
    for entry in status_bytes.split(b"\0"):
        if not entry:
            continue
        if not entry.startswith(b"?? "):
            raise RuntimeError(
                "tracked/staged status appeared after clean checks: "
                + entry.decode("utf-8", errors="backslashreplace")
            )
        observed.add(entry[3:].decode("utf-8", errors="strict"))
    if observed != set(PROTECTED_UNTRACKED):
        raise RuntimeError(
            f"untracked inventory drift: expected={sorted(PROTECTED_UNTRACKED)}, observed={sorted(observed)}"
        )
    for relative, expected_hash in PROTECTED_UNTRACKED.items():
        if sha256_file(root / relative) != expected_hash:
            raise RuntimeError(f"protected untracked hash drift: {relative}")


def run_finalizer(repo_root: Path, output_root: Path, expected_source_commit: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(output_root.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        assert_clean_and_protected(repo_root, expected_source_commit)
        receipt = validate_receipt(repo_root, RECEIPT_COMMIT)
        write_json(staging / "receipt_validation.json", receipt)

        pilot_check = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=PILOT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (staging / "pilot_checksum_reverification.txt").write_text(
            pilot_check.stdout, encoding="utf-8"
        )
        if pilot_check.returncode != 0:
            raise RuntimeError("pilot checksum reverification failed")

        command = [
            sys.executable,
            "-m",
            "unittest",
            TEST_PATH,
            "-v",
        ]
        test_run = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        test_report = staging / "automated_tests.txt"
        test_report.write_text(test_run.stdout, encoding="utf-8")
        automated = {
            "status": "PASS" if test_run.returncode == 0 else "FAIL",
            "command": command,
            "command_string": " ".join(command),
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
            raise RuntimeError("automated Recovery tests failed")

        manual_path = repo_root / MANUAL_REPORT
        manual = {
            "status": "PASS",
            "reviewer_role": "CODEX_PRIMARY_IMPLEMENTATION_AGENT",
            "reviewer_external_identity": None,
            "identity_status": "NOT_EXTERNALLY_VERIFIED",
            "scope": [
                "frozen_contract_and_superseded_contract_hashes",
                "approval_identity_and_scope_limitations",
                "pilot_closure_manifest_ledger_and_sentinel",
                "historical_git_receipt_parent_diff_ledger_and_sentinel",
                "authorization_denials_and_no_downstream_execution",
                "protected_untracked_inventory",
            ],
            "sample_mode": "FULL_GOVERNANCE_ARTIFACT_SET_PLUS_EXTERNAL_PILOT_LEDGER",
            "findings_count": 3,
            "findings_path": MANUAL_REPORT,
            "findings_sha256": sha256_file(manual_path),
            "disposition": "PASS_AFTER_FORWARD_REPAIR",
        }
        write_json(staging / "manual_audit.json", manual)

        inventory = {
            "schema_version": "reactflow_delta.recovery_artifact_inventory.v2",
            "execution_source_commit": expected_source_commit,
            "files": [],
            "external_inputs": [
                {
                    "path": str(PILOT_ROOT / "closure_manifest.yaml"),
                    "sha256": sha256_file(PILOT_ROOT / "closure_manifest.yaml"),
                },
                {
                    "path": str(PILOT_ROOT / "SHA256SUMS"),
                    "sha256": sha256_file(PILOT_ROOT / "SHA256SUMS"),
                },
                {
                    "path": str(PILOT_ROOT / "DEVELOPMENT_CLOSED"),
                    "sha256": sha256_file(PILOT_ROOT / "DEVELOPMENT_CLOSED"),
                },
            ],
        }
        for name in (
            "receipt_validation.json",
            "pilot_checksum_reverification.txt",
            "automated_tests.txt",
            "automated_tests.json",
            "manual_audit.json",
        ):
            item = staging / name
            inventory["files"].append(
                {"path": str(output_root / name), "bytes": item.stat().st_size, "sha256": sha256_file(item)}
            )
        write_json(staging / "artifact_inventory.json", inventory)

        terminal_manifest = {
            "schema_version": "reactflow_delta.recovery_terminal.v2",
            "manifest_id": output_root.name,
            "phase_id": "RECOVERY_CONTRACT_REWRITE",
            "lifecycle_status": "TERMINAL",
            "gate_result": "PASS",
            "scientific_gate_result": "NOT_RUN",
            "evidence_class": "ENGINEERING_ONLY",
            "parent_receipt": {
                "commit": RECEIPT_COMMIT,
                "validation_path": str(output_root / "receipt_validation.json"),
                "validation_sha256": sha256_file(staging / "receipt_validation.json"),
            },
            "automated_tests": automated,
            "manual_audit": manual,
            "finalizer": {
                "schema_version": FINALIZER_SCHEMA,
                "path": FINALIZER_PATH,
                "sha256": sha256_file(repo_root / FINALIZER_PATH),
                "validator_path": VALIDATOR_PATH,
                "validator_sha256": sha256_file(repo_root / VALIDATOR_PATH),
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
                "path": str(output_root / "RECOVERY_V4_CLOSED.yaml"),
                "status": "WRITTEN_AFTER_LEDGER",
            },
            "permissions": {
                "full_data_recall_allowed": False,
                "training_allowed": False,
                "new_split_allowed": False,
                "confirmatory_test_access_allowed": False,
                "cross_project_export_allowed": False,
                "new_wet_lab_allowed": False,
            },
            "terminal_route": "STOP_AWAIT_D0X_AUTHORITY_AMENDMENT",
        }
        write_yaml(staging / "terminal_manifest.yaml", terminal_manifest)

        ledger_members = sorted(
            path
            for path in staging.iterdir()
            if path.name not in {"SHA256SUMS", "RECOVERY_V4_CLOSED.yaml"}
        )
        ledger_text = "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in ledger_members
        )
        (staging / "SHA256SUMS").write_text(ledger_text, encoding="utf-8")
        sentinel = {
            "schema_version": "reactflow_delta.recovery_terminal_sentinel.v2",
            "sentinel_id": output_root.name,
            "status": "RECOVERY_V4_CLOSED",
            "phase_id": "RECOVERY_CONTRACT_REWRITE",
            "gate_result": "PASS",
            "scientific_gate_result": "NOT_RUN",
            "execution_source_commit": expected_source_commit,
            "terminal_manifest_sha256": sha256_file(staging / "terminal_manifest.yaml"),
            "checksum_ledger_sha256": sha256_file(staging / "SHA256SUMS"),
            "finalizer_sha256": sha256_file(repo_root / FINALIZER_PATH),
            "embedded_self_sha256": "FORBIDDEN",
        }
        write_yaml(staging / "RECOVERY_V4_CLOSED.yaml", sentinel)

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
            "terminal_sentinel_sha256": sha256_file(output_root / "RECOVERY_V4_CLOSED.yaml"),
            "execution_source_commit": expected_source_commit,
        }
    except Exception as exc:
        (staging / "FINALIZER_FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    result = run_finalizer(
        args.repo_root.resolve(),
        args.output_root,
        args.expected_source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
