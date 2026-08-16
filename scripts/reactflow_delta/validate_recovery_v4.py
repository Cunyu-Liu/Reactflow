#!/usr/bin/env python3
"""Validate the immutable Recovery authority receipt at its Git commit.

The validator reads blobs from the historical receipt commit, so later active
manifest epochs do not invalidate the evidence.  It never edits the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


RECEIPT_COMMIT = "940697de985dbd7425112babd2d429e81ddde4cb"
RECEIPT_PARENT = "10e52412a1612667993209e56117b1608a084297"
CONTRACT_SHA256 = "631962f88790103aa3383c9ed22de2943f6874455b4fcb587e18eb2a7d277c15"
V3_SHA256 = "3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10"
LEDGER_PATH = "configs/reactflow_delta/authority_epoch_1.bundle.sha256"
SENTINEL_PATH = "configs/reactflow_delta/authority_epoch_1.sentinel.yaml"
ACTIVE_PATH = "configs/reactflow_delta/active_contract.yaml"
RECOVERY_PATH = "manifests/reactflow_delta/recovery_v4_terminal_manifest_20260803.yaml"
EXPECTED_DIFF = {
    ACTIVE_PATH,
    LEDGER_PATH,
    SENTINEL_PATH,
    "docs/approvals/reactflow_delta_v4_approval_20260803.yaml",
    "docs/approvals/reactflow_delta_v4_d0x_approval_context_20260803.md",
    "docs/audits/reactflow_delta_v4_recovery_acceptance_20260803.md",
    RECOVERY_PATH,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DuplicateKeyError(ValueError):
    pass


class StrictLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: StrictLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DuplicateKeyError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def strict_yaml(data: bytes, source: str) -> dict[str, Any]:
    value = yaml.load(data.decode("utf-8"), Loader=StrictLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    return value


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def git_blob(root: Path, commit: str, path: str) -> bytes:
    return _git(root, "show", f"{commit}:{path}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_ledger(data: bytes) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        pieces = line.split("  ", 1)
        if len(pieces) != 2 or not SHA256_RE.fullmatch(pieces[0]) or not pieces[1]:
            raise ValueError(f"malformed ledger line {line_number}")
        rows.append((pieces[0], pieces[1]))
    paths = [path for _, path in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("ledger paths must be unique and lexicographically sorted")
    if LEDGER_PATH in paths or SENTINEL_PATH in paths:
        raise ValueError("ledger self/sentinel inclusion is forbidden")
    return rows


def validate_receipt(root: Path, receipt_commit: str = RECEIPT_COMMIT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def require(condition: bool, check_id: str, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise ValueError(f"{check_id}: {detail}")

    parent_line = _git(root, "rev-list", "--parents", "-n", "1", receipt_commit).decode().strip().split()
    require(len(parent_line) == 2, "RCV-GIT-PARENT-COUNT", str(parent_line))
    require(parent_line[1] == RECEIPT_PARENT, "RCV-GIT-PARENT", parent_line[1])
    diff_paths = set(
        _git(root, "diff", "--name-only", f"{receipt_commit}^", receipt_commit)
        .decode("utf-8")
        .splitlines()
    )
    require(diff_paths == EXPECTED_DIFF, "RCV-GIT-DIFF-ALLOWLIST", repr(sorted(diff_paths)))

    ledger_bytes = git_blob(root, receipt_commit, LEDGER_PATH)
    ledger_rows = parse_ledger(ledger_bytes)
    for expected_hash, path in ledger_rows:
        require(
            sha256(git_blob(root, receipt_commit, path)) == expected_hash,
            "RCV-LEDGER-MEMBER",
            path,
        )

    sentinel_bytes = git_blob(root, receipt_commit, SENTINEL_PATH)
    sentinel = strict_yaml(sentinel_bytes, SENTINEL_PATH)
    active_bytes = git_blob(root, receipt_commit, ACTIVE_PATH)
    active = strict_yaml(active_bytes, ACTIVE_PATH)
    recovery_bytes = git_blob(root, receipt_commit, RECOVERY_PATH)
    strict_yaml(recovery_bytes, RECOVERY_PATH)
    require(sentinel["detached_ledger_sha256"] == sha256(ledger_bytes), "RCV-SENTINEL-LEDGER", LEDGER_PATH)
    require(sentinel["active_manifest_sha256"] == sha256(active_bytes), "RCV-SENTINEL-ACTIVE", ACTIVE_PATH)
    require(sentinel["recovery_terminal_manifest_sha256"] == sha256(recovery_bytes), "RCV-SENTINEL-RECOVERY", RECOVERY_PATH)
    require(active["active_contract"]["sha256"] == CONTRACT_SHA256, "RCV-CONTRACT-HASH", CONTRACT_SHA256)
    authorization = active["authorization"]
    require(authorization["runnable_phases"] == [], "RCV-NO-RUNNABLE-PHASE", repr(authorization["runnable_phases"]))
    for field in (
        "training_allowed",
        "full_data_recall_allowed",
        "new_split_allowed",
        "confirmatory_test_access_allowed",
        "cross_project_export_allowed",
        "new_wet_lab_allowed",
    ):
        require(authorization[field] is False, f"RCV-DENY-{field.upper()}", repr(authorization[field]))
    v3_path = "docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md"
    require(sha256(git_blob(root, receipt_commit, v3_path)) == V3_SHA256, "RCV-V3-PRESERVED", v3_path)
    return {
        "schema_version": "reactflow_delta.recovery_receipt_validation.v1",
        "receipt_commit": receipt_commit,
        "receipt_parent": RECEIPT_PARENT,
        "result": "PASS",
        "checks": checks,
        "limitations": [
            "Receipt integrity PASS does not by itself satisfy V4 phase-manifest automated_tests/manual_audit/finalizer requirements.",
            "User identity assurance remains platform-role-only; no cryptographic signature is asserted.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--receipt-commit", default=RECEIPT_COMMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_receipt(args.repo_root.resolve(), args.receipt_commit)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
