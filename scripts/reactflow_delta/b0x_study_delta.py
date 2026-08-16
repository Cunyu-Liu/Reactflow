#!/usr/bin/env python3
"""Compare validation-study delta characteristics (signal vs noise).

Checks whether CIDGMP's delta is noise-dominated (hence not learnable by any
model) vs TRP4P6, to judge whether the cross-study learnability failure is a
model limitation or a data/validation-split limitation.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0x_data import load_pairs, split_groups  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    args = ap.parse_args()

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups["train"]
    val = groups["validation"]

    by_study = defaultdict(list)
    for p in val:
        by_study[p.study].append(p)

    print("=== validation study delta stats ===")
    for study, ps in by_study.items():
        eligible = [d for p in ps for i, d in enumerate(p.delta) if p.mask[i]]
        arr = np.array(eligible)
        nonnz = np.abs(arr) > 1e-9
        # per-pair delta magnitude (mean |delta| over eligible positions)
        per_pair_mag = []
        for p in ps:
            d = [p.delta[i] for i in range(len(p.mask)) if p.mask[i]]
            if d:
                per_pair_mag.append(np.mean(np.abs(d)))
        print(f"  {study}: n_pairs={len(ps)}")
        print(f"    eligible positions={len(arr)} nonzero={nonnz.mean():.2%}")
        print(f"    delta std={arr.std():.4f} mean_abs={np.abs(arr).mean():.4f} "
              f"q50_abs={np.percentile(np.abs(arr),50):.4f} q90_abs={np.percentile(np.abs(arr),90):.4f}")
        print(f"    per-pair mean|delta|: mean={np.mean(per_pair_mag):.4f} "
              f"std={np.std(per_pair_mag):.4f}")

    # training study delta stats for reference
    print("\n=== training study delta stats ===")
    by_study_tr = defaultdict(list)
    for p in train:
        by_study_tr[p.study].append(p)
    for study, ps in by_study_tr.items():
        arr = np.array([d for p in ps for i, d in enumerate(p.delta) if p.mask[i]])
        print(f"  {study}: n_pairs={len(ps)} delta_std={arr.std():.4f} "
              f"mean_abs={np.abs(arr).mean():.4f} nonzero={np.abs(arr).mean()*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())