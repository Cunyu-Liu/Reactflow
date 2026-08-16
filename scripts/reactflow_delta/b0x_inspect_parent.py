#!/usr/bin/env python3
"""Inspect the outlier parent (6cbaa...) in the B0-X validation split.

Checks whether the extreme per-parent skill is a real learnable signal or a
data artifact (e.g., degenerate delta, near-constant delta, tiny distinct
mutation types, or a leakage/duplicate pattern).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0x_data import load_pairs, split_groups, _finite  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    args = ap.parse_args()

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    target_prefix = "6cbaa39ddf87e795d77b"
    by_parent = defaultdict(list)
    for p in pairs:
        by_parent[p.parent].append(p)

    print("=== validation parents ===")
    for parent, ps in by_parent.items():
        print(f"  {parent[:20]} study={ps[0].study} n={len(ps)}")

    # match by prefix
    full_parent = next((p for p in by_parent if p.startswith(target_prefix)), None)
    if full_parent is None:
        print(f"\n  parent {target_prefix} not found")
        return 0
    ps = by_parent[full_parent]
    print(f"\n=== outlier parent {full_parent[:30]} n={len(ps)} ===")

    # mutation types
    types = Counter((p.ref_allele, p.alt_allele) for p in ps)
    print("  mutation types (ref,alt):", dict(types))
    seq_lens = Counter(len(p.seq) for p in ps)
    print("  sequence lengths:", dict(seq_lens))

    # delta stats
    eligible = [d for p in ps for i, d in enumerate(p.delta) if p.mask[i]]
    nonnz = [d for d in eligible if abs(d) > 1e-9]
    print(f"  eligible positions: {len(eligible)}")
    print(f"  nonzero delta positions: {len(nonnz)} ({len(nonnz)/max(len(eligible),1):.2%})")
    if eligible:
        arr = np.array(eligible)
        print(f"  delta mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  delta quantiles [0,25,50,75,100]: {np.percentile(arr,[0,25,50,75,100])}")

    # mutation positions
    mpos = Counter(p.mutation_pos for p in ps)
    print("  mutation position spread:", len(mpos), "distinct positions")

    # WT reactivity range
    wt_all = [v for p in ps for v in p.wt_reactivity if _finite(v)]
    print(f"  WT reactivity: mean={np.mean(wt_all):.3f} std={np.std(wt_all):.3f}")

    # duplicate-looking pairs (same seq + same mutation)
    seen = Counter((p.seq, p.mutation_pos, p.ref_allele, p.alt_allele) for p in ps)
    dups = {k: v for k, v in seen.items() if v > 1}
    print(f"  duplicate (seq,pos,ref,alt) keys: {len(dups)}")

    # Identify the source accessions
    accs = Counter(p.source for p in ps)
    print("  source accessions:", dict(accs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())