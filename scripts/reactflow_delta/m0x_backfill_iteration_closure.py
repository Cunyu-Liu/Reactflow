#!/usr/bin/env python3
"""M0-X: backfill per-iteration closure records for EPRO_DEV_01..10.

The M0-X window registry records each scientific iteration's outcome, but most
iterations (EPRO_DEV_01..06, EPRO_DEV_10) were never given a machine-readable
finalizer + CLOSED sentinel.  This script backfills a per-iteration closure
record for every iteration that lacks one, derived faithfully from the frozen
window registry (run_id / parent_run_id / hypothesis_id / change_category /
evidence_sha256 / status / outcome / note).

INTEGRITY NOTE (important):
  * These are RETROACTIVE REGISTRY BACKFILLS: they close the window-registry
    bookkeeping at the *iteration level*, reflecting the iteration's recorded
    status (PASS/FAIL).  They do NOT claim to recreate an original-time §19.2
    formal finalizer, because the original metrics/invariant/artifact bundles
    for those runs may not be recoverable.
  * An iteration-level "PASS" is a DEVELOPMENT_ONLY acceptance of that run's
    own preregistered hypothesis.  It is NOT a claim that the M0-X phase gate
    (§20.10 / execution-plan §9) is satisfied.  Phase-gate evaluation is a
    separate, explicit step and is never inferred here.
  * No artifact is fabricated: evidence_sha256 is copied verbatim from the
    registry, and the checksum ledger is marked REFERENTIAL_REGISTRY_HASH.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_FINAL = "reactflow_delta.m0x_iteration_closure_finalizer.v1"
SCHEMA_SENT = "reactflow_delta.m0x_iteration_closure_sentinel.v1"
WINDOW = "docs/governance/m0x_window_registry_20260804.json"
# iterations expected to already have closure (or to be excluded)
SKIP = {"EPRO_DEV_12_REGRESSION", "M0X_SOTA_COMPARISON_CONSOLIDATED"}


def status_of(it: dict) -> str:
    s = it.get("status") or it.get("outcome") or "NOT_RUN"
    return str(s).upper()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0x_iteration_closure_20260807"))
    ap.add_argument("--write", action="store_true",
                    help="write files (default: dry-run / report only)")
    args = ap.parse_args()

    repo = Path(args.repo)
    wreg = json.loads((repo / WINDOW).read_text())
    iters = wreg["iterations"]
    out_root = Path(args.out)
    now = datetime.now(timezone.utc).isoformat()

    missing = [it for it in iters
               if it.get("counts_as_iteration") and it.get("iteration_id") not in SKIP]
    summary = []
    for it in sorted(missing, key=lambda x: x["iteration_id"]):
        iid = it["iteration_id"]
        st = status_of(it)
        entry = {
            "schema": SCHEMA_FINAL,
            "iteration_id": iid,
            "run_id": it.get("run_id"),
            "parent_run_id": it.get("parent_run_id"),
            "hypothesis_id": it.get("hypothesis_id"),
            "change_category": it.get("change_category"),
            "prediction_changing": it.get("prediction_changing"),
            "counts_as_iteration": it.get("counts_as_iteration"),
            "status": st,
            "evidence_sha256": it.get("evidence_sha256"),
            "evidence_class": "DEVELOPMENT_ONLY",
            "closure_type": "RETROACTIVE_REGISTRY_BACKFILL",
            "gate_status": "ITERATION_LEVEL",
            "phase_gate_claim": False,
            "checksum_ledger": {
                "status": "REFERENTIAL_REGISTRY_HASH",
                "evidence_sha256": it.get("evidence_sha256"),
                "note": "backfill references the frozen registry evidence hash; "
                        "original per-run artifact hashes are not re-derived",
            },
            "backfilled_at_utc": now,
            "note": it.get("note"),
            "honesty_note": "Iteration-level DEVELOPMENT_ONLY closure record. "
                            "Does NOT assert M0-X phase-gate PASS (§20.10/§9).",
        }
        out_dir = out_root / iid
        sentinel = {
            "schema": SCHEMA_SENT,
            "iteration_id": iid,
            "run_id": it.get("run_id"),
            "overall": st,
            "closure_type": "RETROACTIVE_REGISTRY_BACKFILL",
            "finalizer": str(out_dir / "finalizer.json"),
            "note": it.get("note"),
        }
        if args.write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "finalizer.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8")
            (out_dir / f"{iid}_CLOSED.yaml").write_text(
                json.dumps(sentinel, indent=2), encoding="utf-8")
        summary.append({"iteration_id": iid, "status": st,
                        "path": str(out_dir), "written": args.write})

    # ---- fix EPRO_DEV_03 status=None -> FAIL for consistency ----
    dev03 = None
    for it in iters:
        if it.get("iteration_id") == "EPRO_DEV_03":
            dev03 = it
            break
    dev03_fix = None
    if dev03 is not None and not dev03.get("status"):
        if args.write:
            dev03["status"] = status_of(dev03)  # FAIL (from outcome=FAIL)
            (repo / WINDOW).write_text(json.dumps(wreg, indent=2), encoding="utf-8")
        dev03_fix = {"iteration_id": "EPRO_DEV_03",
                     "old_status": None,
                     "new_status": status_of(dev03),
                     "outcome": dev03.get("outcome"),
                     "written": args.write}

    print("=== M0-X per-iteration closure backfill (dry-run unless --write) ===")
    for s in summary:
        print(f"  {s['iteration_id']:<22s} status={s['status']:<6s} {s['path']}")
    print(f"  DEV03 status fix: {dev03_fix}")
    print(f"\n  note: closure_type=RETROACTIVE_REGISTRY_BACKFILL, "
          f"gate_status=ITERATION_LEVEL, phase_gate_claim=False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
