#!/usr/bin/env python3
"""Fail-closed O0-X authority preflight; performs no network or data access.

Mirrors the B0-X validator for the O0-X Operator Engineering authority
epoch 11.  Verifies only O0-X is runnable, D0-X/D1-X/D2-X/PH0-X/B0-X are
PASS, the O0-X amendment/approval hashes are bound and intact, and the
governance Git state is clean.  No test access, no scientific model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


class DuplicateKeyError(ValueError):
    pass


class Loader(yaml.SafeLoader):
    pass


def _mapping(loader: Loader, node, deep: bool = False) -> dict:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DuplicateKeyError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path, *, staging: bool = False) -> dict[str, Any]:
    active_path = root / "configs/reactflow_delta/active_contract.yaml"
    active = load_yaml(active_path)
    errors: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    auth = active.get("authorization", {})
    authority = active.get("authority", {})
    check(authority.get("pointer_role") == "SINGLE_ACTIVE_AUTHORITY", "not single active authority")
    check(authority.get("current_phase") == "O0-X", "current phase is not O0-X")
    check(auth.get("allowed_phases") == ["D0-X", "D1-X", "D2-X", "PH0-X", "B0-X", "O0-X"],
          "allowed phases mismatch")
    check(auth.get("runnable_phases") == ["O0-X"], "runnable phases must equal [O0-X]")
    check(auth.get("new_split_allowed") is False, "new split must be disallowed")
    check(auth.get("confirmatory_test_access_allowed") is False,
          "confirmatory test access must be false")
    check(auth.get("cross_project_export_allowed") is False, "cross-project export must be false")
    check(auth.get("new_wet_lab_allowed") is False, "new wet-lab must be false")
    # O0-X authorizes tiny engineering overfit (8-32 pair) on train-only fixtures.
    check(auth.get("training_allowed") is True, "training (tiny-overfit) must be allowed")
    check(auth.get("blind_test_aggregate_allowed") is True, "blind test aggregate must be allowed")

    bindings = active.get("bindings", {})
    required = {
        "o0x_amendment": "o0x_amendment_sha256",
        "b0x_amendment": "b0x_amendment_sha256",
        "ph0x_amendment": "ph0x_amendment_sha256",
        "d2x_amendment": "d2x_amendment_sha256",
        "d1x_amendment": "d1x_amendment_sha256",
        "d0x_amendment": "d0x_amendment_sha256",
        "source_universe_manifest": "source_universe_manifest_sha256",
        "config": "config_sha256",
        "license_policy": "license_policy_sha256",
        "parser_fixture_manifest": "parser_fixture_manifest_sha256",
        "initial_exposure_ledger": "exposure_ledger_sha256",
        "test_access_ledger": "test_access_ledger_sha256",
    }
    for stem, hash_key in required.items():
        path_key = stem + "_path"
        relative = bindings.get(path_key)
        expected = bindings.get(hash_key)
        check(isinstance(relative, str), f"missing {path_key}")
        check(isinstance(expected, str) and len(expected) == 64, f"missing {hash_key}")
        if isinstance(relative, str) and isinstance(expected, str):
            target = (root / relative).resolve()
            try:
                target.relative_to(root.resolve())
                check(target.is_file(), f"missing bound file {relative}")
                if target.is_file():
                    check(digest(target) == expected, f"hash drift: {relative}")
            except ValueError:
                errors.append(f"bound path escapes repository: {relative}")

    check(bindings.get("o0x_run_id") == "o0x_operator_engineering_20260804_v1", "o0x_run_id mismatch")

    phases = {row.get("phase_id"): row for row in active.get("phase_graph", [])}
    for pid in ("RECOVERY_CONTRACT_REWRITE", "PILOT_CLOSURE", "D0-X", "D1-X", "D2-X", "PH0-X", "B0-X"):
        check(phases.get(pid, {}).get("gate_result") == "PASS", f"{pid} not PASS")
    check(phases.get("O0-X", {}).get("execution_authorized") is True, "O0-X execution not authorized")
    for phase_id, row in phases.items():
        if phase_id != "O0-X":
            check(row.get("execution_authorized") is False, f"unexpected authorized phase: {phase_id}")

    if not staging:
        execution_source = bindings.get("execution_source_commit")
        check(isinstance(execution_source, str) and len(execution_source) == 40,
              "execution_source_commit binding missing")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", execution_source, head], cwd=root
        )
        check(is_ancestor.returncode == 0, "execution_source_commit is not an ancestor of HEAD")
        check(subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0, "tracked worktree dirty")
        check(subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0, "staged worktree dirty")

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": "reactflow_delta.o0x_authority_preflight.v1",
        "status": "PASS",
        "staging_mode": staging,
        "network_or_data_access_performed": False,
        "training_allowed": True,
        "blind_test_inspection_allowed": False,
        "runnable_phase": "O0-X",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--staging", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.repo_root.resolve(), staging=args.staging), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
