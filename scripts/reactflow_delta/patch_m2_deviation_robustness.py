#!/usr/bin/env python3
"""patch_m2_deviation_robustness — add the LOO robustness field to an existing
M2 deviation report WITHOUT recomputing per_seed / per_position.

The original report (m2_deviation_report.json) was generated before the LOO
sensitivity check (_loo_sensitivity) was added to m2_deviation_report.py.  Rather
than rerun the full 6-permutation report (heavy on the CPU-constrained server),
this driver recomputes only the mu-ensemble design blocks from the existing keyed
predictions and merges in the `robustness` field.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_deviation_report as mdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="keyed_predictions jsonl")
    ap.add_argument("--report", required=True, help="existing m2_deviation_report.json")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--perm-seed", type=int, default=20260812)
    args = ap.parse_args()

    rows = mdr._load_rows(args.pred)
    base, model = mdr._unroll(rows)
    common = [k for k in base if len(model.get(k, {})) == len(mdr.SEEDS)]
    ens = {k: np.mean([model[k][s] for s in mdr.SEEDS], axis=0) for k in common}
    blocks = mdr._design_blocks(base, model, ens, common, W=21)
    if not blocks:
        print("NO BLOCKS — abort", file=sys.stderr)
        return 1

    rob = mdr._loo_sensitivity(blocks, args.n_perm, args.perm_seed)
    rp = Path(args.report)
    report = json.loads(rp.read_text(encoding="utf-8"))
    report["mu_ensemble"]["robustness"] = rob
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== LOO robustness (mu-ensemble) ===")
    print(json.dumps(rob, ensure_ascii=False, indent=2))
    print(f"PATCHED -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
