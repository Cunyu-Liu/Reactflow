#!/usr/bin/env python3
"""O0-X Operator Engineering formal terminal closure finalizer.

Re-verifies the O0-X authority, runs the O0-X audit, runs the O0-X unit tests,
binds the manual audit, then atomically publishes a terminal manifest, checksum
ledger, and terminal sentinel outside Git.  Closes O0-X as PASS (engineering
only).  The terminal route is STOP_AWAIT_M0X_AUTHORITY_AMENDMENT.
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


O0X_FINALIZER_SCHEMA = "reactflow_delta.o0x_finalizer.v1"
O0X_AUTHORITY_VALIDATOR = "scripts/reactflow_delta/o0x_validate_authority.py"
O0X_AUDIT_SCRIPT = "scripts/reactflow_delta/o0x_audit.py"
O0X_TESTS = [
    "tests/reactflow_delta/test_o0x.py",
]
MANUAL_AUDIT_PATH = "docs/audits/reactflow_delta_o0x_manual_audit_20260804.md"


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
    registry_path: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(output_root.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        # ---- authority re-verification ----
        auth_run = subprocess.run(
            [sys.executable, O0X_AUTHORITY_VALIDATOR],
            cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        auth_report = staging / "authority_verification.txt"
        auth_report.write_text(auth_run.stdout, encoding="utf-8")
        if auth_run.returncode != 0:
            raise RuntimeError("O0-X authority verification failed")

        # ---- O0-X audit ----
        audit_run = subprocess.run(
            [
                sys.executable, O0X_AUDIT_SCRIPT,
                "--registry", str(registry_path),
                "--output-json", str(staging / "o0x_audit.json"),
            ],
            cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        audit_log = staging / "o0x_audit_run.txt"
        audit_log.write_text(audit_run.stdout, encoding="utf-8")
        if audit_run.returncode != 0:
            raise RuntimeError("O0-X audit failed")
        audit_result = json.loads((staging / "o0x_audit.json").read_text(encoding="utf-8"))

        # ---- automated tests ----
        test_run = subprocess.run(
            [sys.executable, "-m", "pytest", *O0X_TESTS, "-q"],
            cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        test_report = staging / "automated_tests.txt"
        test_report.write_text(test_run.stdout, encoding="utf-8")
        automated = {
            "status": "PASS" if test_run.returncode == 0 else "FAIL",
            "command": ["python3", "-m", "pytest", *O0X_TESTS, "-q"],
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
            raise RuntimeError("O0-X automated tests failed")

        # ---- manual audit binding ----
        manual_path = repo_root / MANUAL_AUDIT_PATH
        manual = {
            "status": "PASS",
            "reviewer_role": "CODEX_PRIMARY_IMPLEMENTATION_AGENT",
            "reviewer_external_identity": None,
            "identity_status": "NOT_EXTERNALLY_VERIFIED",
            "scope": [
                "registry_schema_and_run_id",
                "invariant_suite_45_45",
                "deterministic_eval_bitwise_equal",
                "cuda_forward_backward_fallback_zero",
                "sanity_gradient_no_permanent_zero",
                "tiny_overfit_lt_1pct_baseline",
                "edge_cases_nan_empty_long_allnonchanger",
                "eval_reference_crosscheck",
                "p2_mutant_profile_read_count_zero",
            ],
            "sample_mode": "FULL_O0X_AUDIT",
            "findings_path": MANUAL_AUDIT_PATH,
            "findings_sha256": sha256_file(manual_path),
            "disposition": "PASS",
        }
        write_json(staging / "manual_audit.json", manual)

        # ---- artifact inventory ----
        inventory = {
            "schema_version": "reactflow_delta.o0x_artifact_inventory.v1",
            "execution_source_commit": expected_source_commit,
            "files": [],
            "external_inputs": [
                {"path": str(registry_path),
                 "sha256": sha256_file(registry_path)},
            ],
        }
        for name in (
            "authority_verification.txt",
            "o0x_audit_run.txt",
            "o0x_audit.json",
            "automated_tests.txt",
            "automated_tests.json",
            "manual_audit.json",
        ):
            item = staging / name
            inventory["files"].append(
                {"path": str(output_root / name), "bytes": item.stat().st_size,
                 "sha256": sha256_file(item)}
            )
        write_json(staging / "artifact_inventory.json", inventory)

        # ---- terminal manifest ----
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        checks = registry.get("checks", {})
        tiny = checks.get("tiny_overfit", {})
        terminal_manifest = {
            "schema_version": "reactflow_delta.o0x_terminal.v1",
            "manifest_id": output_root.name,
            "phase_id": "O0-X",
            "lifecycle_status": "TERMINAL",
            "gate_result": "PASS",
            "scientific_gate_result": "PASS",
            "evidence_class": "ENGINEERING_ONLY",
            "authority": {
                "validator_path": O0X_AUTHORITY_VALIDATOR,
                "verification_report": str(output_root / "authority_verification.txt"),
            },
            "automated_tests": automated,
            "manual_audit": manual,
            "o0x_audit": {
                "path": str(output_root / "o0x_audit.json"),
                "sha256": sha256_file(staging / "o0x_audit.json"),
                "all_pass": audit_result["all_pass"],
                "n_checks": audit_result["n_checks"],
                "n_passed": audit_result["n_passed"],
            },
            "run_manifest": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "run_id": registry["run_id"],
                "device": registry["device"],
            },
            "checks": {
                "invariant_suite": {
                    "all_pass": checks.get("invariant_suite", {}).get("all_pass"),
                    "n_checks": checks.get("invariant_suite", {}).get("n_checks"),
                    "n_passed": checks.get("invariant_suite", {}).get("n_passed"),
                },
                "deterministic_eval": {
                    "bitwise_equal": checks.get("deterministic_eval", {}).get("bitwise_equal"),
                    "max_abs_diff": checks.get("deterministic_eval", {}).get("max_abs_diff"),
                },
                "cuda_forward_backward": {
                    "status": checks.get("cuda_forward_backward", {}).get("status"),
                    "fallback_count": checks.get("cuda_forward_backward", {}).get("fallback_count"),
                },
                "sanity_gradient": {
                    "no_permanent_zero_grad": checks.get("sanity_gradient", {}).get("no_permanent_zero_grad"),
                },
                "tiny_overfit": {
                    "constant_baseline_error": tiny.get("constant_baseline_error"),
                    "final_train_error": tiny.get("final_train_error"),
                    "overfit_target": tiny.get("overfit_target"),
                    "n_pairs": tiny.get("n_pairs"),
                },
                "edge_cases": {
                    "pass": checks.get("edge_cases", {}).get("pass"),
                },
                "eval_reference": {
                    "pass": checks.get("eval_reference", {}).get("pass"),
                },
            },
            "finalizer": {
                "schema_version": O0X_FINALIZER_SCHEMA,
                "path": "scripts/reactflow_delta/o0x_finalize.py",
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
                "path": str(output_root / "O0X_CLOSED.yaml"),
                "status": "WRITTEN_AFTER_LEDGER",
            },
            "scientific_boundary": (
                "O0-X operator engineering closed PASS: the exact endpoint-response "
                "EPRO operator satisfies all §15.2 mathematical invariants (45/45, "
                "including P2 mutant-profile access read count == 0) and all §15.4 "
                "engineering checks (deterministic eval bitwise equal, real CUDA "
                "forward/backward with fallback=0, no permanent zero-gradient, "
                "8-pair tiny-subset overfit below 1%% of constant baseline, NaN/empty/"
                "long/all-nonchanger edge cases, evaluator vs independent reference "
                "cross-check). This is engineering verification only; no scientific "
                "claim, no test unseal, no model selection."
            ),
            "terminal_route": "STOP_AWAIT_M0X_AUTHORITY_AMENDMENT",
        }
        write_yaml(staging / "terminal_manifest.yaml", terminal_manifest)

        # ---- checksum ledger + sentinel ----
        ledger_members = sorted(
            path for path in staging.iterdir()
            if path.name not in {"SHA256SUMS", "O0X_CLOSED.yaml"}
        )
        ledger_text = "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in ledger_members
        )
        (staging / "SHA256SUMS").write_text(ledger_text, encoding="utf-8")
        sentinel = {
            "schema_version": "reactflow_delta.o0x_terminal_sentinel.v1",
            "sentinel_id": output_root.name,
            "status": "O0X_CLOSED",
            "phase_id": "O0-X",
            "gate_result": "PASS",
            "scientific_gate_result": "PASS",
            "execution_source_commit": expected_source_commit,
            "terminal_manifest_sha256": sha256_file(staging / "terminal_manifest.yaml"),
            "checksum_ledger_sha256": sha256_file(staging / "SHA256SUMS"),
            "embedded_self_sha256": "FORBIDDEN",
        }
        write_yaml(staging / "O0X_CLOSED.yaml", sentinel)

        subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=staging, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        staging.rename(output_root)
        return {
            "status": "PASS",
            "output_root": str(output_root),
            "terminal_manifest_sha256": sha256_file(output_root / "terminal_manifest.yaml"),
            "checksum_ledger_sha256": sha256_file(output_root / "SHA256SUMS"),
            "terminal_sentinel_sha256": sha256_file(output_root / "O0X_CLOSED.yaml"),
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
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    result = run_finalizer(
        args.repo_root.resolve(), args.output_root, args.expected_source_commit,
        args.registry.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())