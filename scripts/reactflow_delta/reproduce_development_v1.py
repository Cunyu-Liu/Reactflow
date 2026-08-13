#!/usr/bin/env python3
"""reproduce_development_v1: clean-replay verification of P2/P3 development statistics.

Re-derives the decision-relevant statistics (20-puzzle t-CI, sign-flip, verdict)
from the committed per-puzzle effect artifacts, and verifies the committed verdicts.
This verifies the statistical layer is reproducible GPU-free from stored effects.
The full GPU re-run (fit+predict) is documented in the run commands below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.reactflow_delta.p2_learnability import (
    puzzle_level_ci20, studentized_sign_flip,
)


def verify_p2(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    puzzles = list(d["per_puzzle_D_p2"].keys())
    effects = [d["per_puzzle_D_p2"][p] for p in puzzles]
    ci = puzzle_level_ci20(effects)
    sign = studentized_sign_flip(effects)
    ok = (len(effects) == 20 and bool(ci.get("ci_low_gt_0"))
          and d.get("verdict") == "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT")
    return {"phase": "P2", "n_puzzles": len(effects), "ci": ci, "sign": sign,
            "committed_verdict": d.get("verdict"), "reproduced_ok": ok}


def verify_p3(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    puzzles = list(d["rank_d_p3"]["2"].keys())
    out = {"phase": "P3", "n_puzzles": len(puzzles), "ranks": {}}
    any_rank_gt_0 = False
    for k in ["2", "4", "8"]:
        effects = [d["rank_d_p3"][k][p] for p in puzzles]
        finite = [e for e in effects if e == e]
        if len(finite) == 20:
            ci = puzzle_level_ci20(finite)
        else:
            ci = {"planned_n_not_met": True, "n_effects": len(finite), "ci_low_gt_0": False}
        n_pos = sum(1 for e in finite if e > 0)
        out["ranks"][k] = {"n_finite": len(finite), "n_positive": n_pos,
                           "mean_finite": round(sum(finite) / len(finite), 4) if finite else None,
                           "ci_low_gt_0": bool(ci.get("ci_low_gt_0")), "ci": ci}
        if ci.get("ci_low_gt_0"):
            any_rank_gt_0 = True
    out["committed_verdict"] = d.get("verdict")
    out["reproduced_ok"] = (not any_rank_gt_0
                            and d.get("verdict") == {str(k): "NO_INCREMENTAL_LRSO_SKILL" for k in ["2", "4", "8"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-json", required=True)
    ap.add_argument("--p3-json", required=True)
    args = ap.parse_args()
    p2 = verify_p2(Path(args.p2_json))
    p3 = verify_p3(Path(args.p3_json))
    report = {
        "schema_version": "reactflow_delta.reproduce_development.v1",
        "verified": p2["reproduced_ok"] and p3["reproduced_ok"],
        "p2": p2,
        "p3": p3,
        "gpu_rerun_commands": [
            "python3 scripts/reactflow_delta/run_p2_direct_v2.py --m2-csv <m2> --out-dir <o2> --device cuda:0",
            "python3 scripts/reactflow_delta/run_p3_lrso_v2.py --m2-csv <m2> --out-dir <o3> --device cuda:0 --rank 2,4,8",
        ],
        "note": "Statistical layer reproduced from committed per-puzzle effects; "
                "full GPU fit+predict re-run documented above for clean checkout replay.",
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
