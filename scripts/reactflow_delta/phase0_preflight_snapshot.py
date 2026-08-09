#!/usr/bin/env python3
"""Phase 0 Batch 0A: fresh read-only preflight snapshot for benchmark_v3 recovery.

Generates (idempotent, refuses to overwrite):
  artifacts/benchmark_v3/preflight_snapshot.json
  artifacts/benchmark_v3/input_hashes.tsv
  artifacts/benchmark_v3/process_snapshot.json
Writes under the /mnt benchmark_v3 artifact root.
Read-only: does not modify repo tracked files or running processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/cunyuliu/reactflow_delta_goal_20260729")
WT = Path("/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809")
ART = Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/benchmark_v3")

ANCHOR = "887066d63369df3f43deec886beba41326a30dee"

INPUT_PATHS = [
    "configs/reactflow_delta/active_contract.yaml",
    "configs/reactflow_delta/split_v2.yaml",
    "configs/reactflow_delta/endpoint_v5.yaml",
    "configs/reactflow_delta/authority_epoch_19.sentinel.yaml",
    "configs/reactflow_delta/authority_epoch_19.bundle.sha256",
    "data_registry/d0x_v2/asset_disposition_20260807.jsonl",
    "data_registry/d0x_v2/asset_disposition_20260807.tsv",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    # input hashes
    rows = []
    for rel in INPUT_PATHS:
        p = ROOT / rel
        if p.exists():
            h = sha256_bytes(p.read_bytes())
            rows.append(f"{h}\t{rel}\t{os.path.getsize(p)}")
        else:
            rows.append(f"UNKNOWN_NOT_ASSERTED\t{rel}\tMISSING")
    input_hashes = "\n".join(rows) + "\n"
    write_once(ART / "input_hashes.tsv", input_hashes)

    # process snapshot (read-only ps of own user, python/train-ish)
    ps_lines = []
    try:
        r = subprocess.run(
            ["ps", "-u", "cunyuliu", "-o", "pid=,stat=,etime=,cmd="],
            capture_output=True, text=True, timeout=60,
        )
        for ln in r.stdout.splitlines():
            if any(k in ln for k in ("python", "train", "reactflow", "conda", "p3_", "r05_", "rna_")):
                ps_lines.append(ln.strip())
    except Exception as e:  # pragma: no cover
        ps_lines.append(f"PS_ERROR: {e}")
    process_snapshot = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user": "cunyuliu",
        "running_process_count_matching": len(ps_lines),
        "process_lines": ps_lines,
        "note": "Read-only snapshot; no process was terminated, signalled, or modified.",
    }
    write_once(ART / "process_snapshot.json", json.dumps(process_snapshot, indent=2, sort_keys=False) + "\n")

    # preflight snapshot
    snapshot = {
        "schema_version": "reactflow_delta.benchmark_v3.preflight_snapshot.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "PHASE0_BATCH0A",
        "authority_epoch": "19 (current, not yet superseded)",
        "plan_anchor_commit": ANCHOR,
        "live_repo": {"path": str(ROOT), "branch": git(ROOT, "rev-parse", "--abbrev-ref", "HEAD"), "head": git(ROOT, "rev-parse", "HEAD")},
        "worktree": {"path": str(WT), "branch": git(WT, "rev-parse", "--abbrev-ref", "HEAD"), "head": git(WT, "rev-parse", "HEAD")},
        "anchor_match": git(ROOT, "rev-parse", "HEAD") == ANCHOR,
        "artifact_root": str(ART),
        "remote_mutation_attestation": False,
        "confirmatory_outcome_accessed": False,
        "process_interference": False,
        "current_phase": "BENCHMARK-RESOURCE (epoch 19)",
        "training_allowed_current": False,
        "note": "Fresh read-only preflight. Only critical inputs hashed. No model trained, no test outcome opened.",
    }
    write_once(ART / "preflight_snapshot.json", json.dumps(snapshot, indent=2, sort_keys=False) + "\n")
    print(json.dumps({"preflight": str(ART / "preflight_snapshot.json"), "hashes": len(rows), "processes": len(ps_lines)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
