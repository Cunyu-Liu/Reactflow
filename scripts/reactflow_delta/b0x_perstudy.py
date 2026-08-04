#!/usr/bin/env python3
"""Diagnose per-study / per-parent skill for the B0-X P2 baseline.

Verifies the B0-X FAIL condition "结果由单一 group 驱动" is NOT hit: the
P2 model's skill must be positive in more than one validation study (and,
where possible, across parents).  Also reports the cluster CI and the pooled
skill so the preregistered CI/sensitivity gate can be judged honestly.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_baselines import run_baseline  # noqa: E402
from b0x_evaluate import pooled_skill  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups["train"]
    val = groups["validation"]

    # strongest trivial = wt_only (ridge)
    trivial = run_baseline("wt_only", train, val, device="cpu")
    ref_preds = trivial.predictions

    p2 = run_baseline("p2_paired", train, val, device="cuda",
                      hidden=args.hidden, epochs=args.epochs,
                      lr=args.lr, weight_decay=args.weight_decay)
    assert p2.status == "ok", p2.error

    # pooled skill vs wt_only
    pooled = pooled_skill(val, p2.predictions, ref_preds)
    print(f"pooled skill_wmae vs wt_only: {pooled['skill_wmae']:.6f}")

    # per-study skill
    by_study = defaultdict(list)
    for p in val:
        by_study[p.study].append(p)
    print("\nper-study skill (vs wt_only):")
    for study, ps in by_study.items():
        sk = pooled_skill(ps, p2.predictions, ref_preds)
        print(f"  {study}: n={len(ps)} skill_wmae={sk['skill_wmae']:.6f}")

    # per-parent skill
    by_parent = defaultdict(list)
    for p in val:
        by_parent[p.parent].append(p)
    print("\nper-parent skill (vs wt_only):")
    pos = 0
    for parent, ps in by_parent.items():
        sk = pooled_skill(ps, p2.predictions, ref_preds)
        if sk["skill_wmae"] > 0:
            pos += 1
        print(f"  {parent[:20]}: n={len(ps)} skill_wmae={sk['skill_wmae']:.6f}")
    print(f"\npositive-parent count: {pos}/{len(by_parent)}")

    # per-study beat-trivial (does each study's P2 beat its own-study wt_only skill?)
    print("\nper-study P2 skill vs within-study zero (absolute):")
    for study, ps in by_study.items():
        z = {p.pair_id: np.zeros(len(p.mask), dtype=np.float32) for p in ps}
        sk = pooled_skill(ps, p2.predictions, z)
        print(f"  {study}: n={len(ps)} skill_wmae_vs_zero={sk['skill_wmae']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())