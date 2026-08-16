#!/usr/bin/env python3
"""One-time build of a P2 data cache from the v2 canonical records.

Reads the (huge) canonical_records_v2.jsonl exactly once and stores only the
records needed by the P2 learnability experiment:
  - every WT + mutant canonical record referenced by a primary pair in the P2
    pool, and
  - every WT canonical record of the P2 pool studies (for replicate groups).

Writes a single pickle to --out-cache.  This avoids re-scanning the 40GB stream
for every outer fold.
"""
import argparse, json, pickle, collections
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", required=True)
    ap.add_argument("--pairs-jsonl", required=True)
    ap.add_argument("--pool-studies", required=True, help="comma-separated study prefixes to include")
    ap.add_argument("--out-cache", required=True)
    args = ap.parse_args()

    pool = set(s for s in args.pool_studies.split(",") if s)

    # --- load pairs in the pool ---
    pairs = []
    pair_studies = set()
    n_skipped = 0
    with open(args.pairs_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            p = json.loads(line)
            st = p["source_accession"].split("_")[0]
            if st not in pool:
                n_skipped += 1
                continue
            pairs.append(p)
            pair_studies.add(st)

    # set of referenced (accession, profile_index, asset_name)
    referenced = set()
    for p in pairs:
        referenced.add((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        referenced.add((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))

    # --- single pass over canonical stream ---
    rec_index = {}
    wt_count_by_study = collections.Counter()
    total = 0
    with open(args.canonical_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            sa = r.get("source_accession") or ""
            st = sa.split("_")[0]
            key = (sa, r.get("source_profile_index"), r.get("source_asset_name"))
            if key in referenced:
                rec_index[key] = r
                continue
            if r.get("is_wt") and st in pool:
                rec_index[key] = r
                wt_count_by_study[st] += 1

    print(json.dumps({
        "total_stream_records": total,
        "n_pairs_pool": len(pairs),
        "n_pool_studies": len(pool),
        "pair_studies_seen": sorted(pair_studies),
        "n_referenced_or_wt_records_cached": len(rec_index),
        "wt_cached_by_study": dict(wt_count_by_study),
        "n_skipped_pairs": n_skipped,
    }, sort_keys=True, indent=2))

    # sanity: sequence base content of a couple cached WT records
    from collections import Counter as _C
    shown = 0
    for k, r in rec_index.items():
        if r.get("is_wt"):
            print("sample WT record key=", k, "len_seq=", len(r["canonical_sequence"]),
                  "bases=", dict(_C(r["canonical_sequence"])))
            shown += 1
            if shown >= 3:
                break

    Path(args.out_cache).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_cache, "wb") as fh:
        pickle.dump({"rec_index": rec_index, "pairs": pairs, "pool": sorted(pool)}, fh, protocol=4)
    print("WROTE", args.out_cache)

if __name__ == "__main__":
    main()
