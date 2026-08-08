#!/usr/bin/env python3
"""Phase 3 scheme-3: precompute sparse contact graphs for the 36 unique WT
sequences in the p2 pool, using ViennaRNA partition-function BPP.

Runs in the `editflow` env (ViennaRNA 2.7.2). Writes a small pickle to
--out-cache mapping seq_sha256 -> {seq, edges, weights} where edges are the
sparse top-k base-pair contacts (0-indexed) used by the repaired EPRO operator.

The run script (pc_cng_gpu env) loads this cache; it never calls ViennaRNA.
"""
from __future__ import annotations

import argparse, hashlib, json, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.reactflow_delta.models.epro_v1 import (  # noqa: E402
    build_contact_edges, normalize_edge_weights,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="p2 cache pickle")
    ap.add_argument("--out-cache", required=True, help="output contact cache pickle")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    with open(args.cache, "rb") as fh:
        c = pickle.load(fh)
    ri = c["rec_index"]
    pool = set(c["pool"])

    # collect unique WT sequences in pool (k[0] is full accession; pool is study prefix)
    def _study(sa: str) -> str:
        return (sa or "").split("_")[0]

    seqs = {}
    for k, v in ri.items():
        if _study(k[0]) in pool:
            s = v.get("canonical_sequence")
            if s:
                seqs[s] = seqs.get(s, 0) + 1

    out = {}
    for s, cnt in seqs.items():
        edges, weights = build_contact_edges(s, top_k=args.top_k)
        weights = normalize_edge_weights(edges, weights, len(s), alpha=0.5)
        h = hashlib.sha256(s.upper().replace("T", "U").encode("ascii")).hexdigest()
        out[h] = {
            "seq": s,
            "length": len(s),
            "n_pairs": cnt,
            "n_edges": int(edges.shape[0]),
            "edges": edges,          # (E,2) int64
            "weights": weights,      # (E,) float32
        }
        print(f"  {h[:8]} len={len(s)} edges={edges.shape[0]} pairs={cnt}", flush=True)

    Path(args.out_cache).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_cache, "wb") as fh:
        pickle.dump({"schema": "reactflow_delta.phase3.scheme3.contacts.v1",
                     "top_k": args.top_k,
                     "alpha": 0.5,
                     "n_unique_seqs": len(out),
                     "contacts": out}, fh, protocol=4)
    print(f"WROTE {args.out_cache} ({len(out)} seqs)")


if __name__ == "__main__":
    main()
