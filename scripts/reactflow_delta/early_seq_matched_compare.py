#!/usr/bin/env python3
"""early_seq_matched_compare.py — matched early comparison: restrict the plain
residual-MLP M2 predictions to the same fold-designs the residual-MLP+global-seq
run has completed so far, so we can tell whether global-seq actually helps BEFORE
the full 159-fold run finishes (avoid waiting hours to catch a regression).

Writes a filtered MLP keyed-predictions file for the shared completed designs.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", required=True, help="seq fold_progress.json")
    ap.add_argument("--pred-mlp", required=True, help="plain MLP keyed predictions")
    ap.add_argument("--out-mlp-filtered", required=True, help="filtered MLP jsonl to write")
    args = ap.parse_args()

    done = set(json.loads(Path(args.progress).read_text(encoding="utf-8"))
               .get("completed_folds", []))
    keep = set()
    out_lines = []
    for line in Path(args.pred_mlp).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r["pair_id"].split(":")[0]
        if d in done:
            keep.add(d)
            out_lines.append(line)
    Path(args.out_mlp_filtered).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"matched designs={len(done)} (present in MLP preds={len(keep)}) "
          f"rows={len(out_lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
