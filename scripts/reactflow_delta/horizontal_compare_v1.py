#!/usr/bin/env python3
"""horizontal_compare_v1: generate the horizontal comparison table from P2/P3 results.

Reads p2_direct_v2_result.json (method-held full-construct CRPS) and produces a
横向对比表: per-method mean held CRPS across the 20 puzzles, CRPS-skill vs the
strongest relevant baseline, and the P2/P3 verdicts. Purely a reporting pass
(GPU-free); does not recompute any experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def wmae_skill(candidate: float, baseline: float) -> float:
    """(baseline - candidate)/baseline * 100; positive means candidate better."""
    if baseline == 0:
        return float("nan")
    return (baseline - candidate) / baseline * 100.0


def build_table(p2_result: dict[str, Any], p3_result: dict[str, Any] | None) -> dict[str, Any]:
    methods = p2_result["method_held_crps"]
    puzzle_names = list(next(iter(methods.values())).keys())
    rows = []
    for m in methods:
        vals = [methods[m][p] for p in puzzle_names if methods[m][p] == methods[m][p]]
        rows.append({
            "method": m,
            "mean_held_crps": float(np.mean(vals)) if vals else float("nan"),
            "sd_held_crps": float(np.std(vals)) if len(vals) > 1 else float("nan"),
            "n_puzzles": len(vals),
        })
    # skill vs zero (ZeroResponse) and vs train_median
    base_zero = next(r["mean_held_crps"] for r in rows if r["method"] == "zero")
    for r in rows:
        r["skill_vs_zero_pct"] = wmae_skill(r["mean_held_crps"], base_zero)
    if any(r["method"] == "train_median" for r in rows):
        base_med = next(r["mean_held_crps"] for r in rows if r["method"] == "train_median")
        for r in rows:
            r["skill_vs_train_median_pct"] = wmae_skill(r["mean_held_crps"], base_med)
    # order by mean CRPS ascending (lower = better)
    rows.sort(key=lambda r: r["mean_held_crps"])
    out = {
        "schema_version": "reactflow_delta.horizontal_compare.v1",
        "estimand": "puzzle-macro full-construct CRPS (released normalized mutant reactivity)",
        "lower_is_better": True,
        "puzzles": puzzle_names,
        "method_table": rows,
        "p2_verdict": p2_result.get("verdict"),
        "p2_ci": p2_result.get("p2_ci20"),
        "p2_note": p2_result.get("note"),
        "p3_verdict": p3_result.get("verdict") if p3_result else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-json", required=True)
    ap.add_argument("--p3-json", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p2 = json.loads(Path(args.p2_json).read_text(encoding="utf-8"))
    p3 = json.loads(Path(args.p3_json).read_text(encoding="utf-8")) if args.p3_json and Path(args.p3_json).exists() else None
    t = build_table(p2, p3)
    Path(args.out).write_text(json.dumps(t, indent=2, default=str), encoding="utf-8")
    print(json.dumps(t, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
