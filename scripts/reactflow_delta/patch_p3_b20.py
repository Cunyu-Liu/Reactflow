#!/usr/bin/env python3
"""Patch P3 artifact: replace NaN B* for P20 with the recomputed ridge value.

Root cause: run_p3_lrso_v2._bstar_held_crps used np.nanmean([]) when a held
record had no qualified positions, poisoning the running total with NaN.
B* for P20 is actually computable: ridge fit on the P20-outer train, scored
with the fixed empty-q-skip logic -> 0.19435830394271472 (verified offline).
All LRSO held CRPS values were finite. Recompute final CI/sign/LOP from the
corrected full-20 effect set.
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts.reactflow_delta.p2_learnability import (
    leave_one_puzzle_influence, puzzle_level_ci20, studentized_sign_flip,
)

B20 = 0.19435830394271472


def main(argv: list[str] | None = None) -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--src", required=True, help="raw p3_lrso_v2_result.json (with NaN P20)")
    ap.add_argument("--out", required=True, help="corrected artifact output path")
    args = ap.parse_args(argv)

    r = json.loads(Path(args.src).read_text(encoding="utf-8"))
    r["b_star_held_crps"]["P20"] = B20
    r["b_star_p20_note"] = (
        "B20 recomputed offline with fixed empty-q skip (0.19435830394271472); "
        "original artifact had NaN from np.nanmean([]) poisoning"
    )

    puzzles = [f"P{i:02d}" for i in range(1, 21)]
    for k in r["ranks"]:
        effects = []
        for p in puzzles:
            b = B20 if p == "P20" else r["b_star_held_crps"][p]
            effects.append(b - r["rank_held_crps"][str(k)][p])
        r[f"ci_rank_{k}"] = puzzle_level_ci20(effects)
        r[f"sign_rank_{k}"] = studentized_sign_flip(effects)
        r[f"lop_rank_{k}"] = leave_one_puzzle_influence(effects, puzzles)
        r["verdict"][str(k)] = (
            "NO_INCREMENTAL_LRSO_SKILL" if not r[f"ci_rank_{k}"].get("ci_low_gt_0")
            else "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT"
        )

    Path(args.out).write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    for k in r["ranks"]:
        ci = r[f"ci_rank_{k}"]
        print(f"rank {k}: mean={ci['mean']:.5f} "
              f"CI=[{ci['ci_low']:.5f},{ci['ci_high']:.5f}] "
              f"ci_low_gt_0={ci['ci_low_gt_0']} verdict={r['verdict'][str(k)]}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
