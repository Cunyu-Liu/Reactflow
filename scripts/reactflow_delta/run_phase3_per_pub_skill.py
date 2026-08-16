#!/usr/bin/env python3
"""Phase 3 closure -> benchmark/resource route (deliverable 5).

Per-publication decomposition of the Phase 3 scheme-3 negative result. The LOOCV
heldout arrays are tagged with the held-out publication, so (because each publication
is held out in exactly one fold) per-publication skill is exactly per-fold skill.

Shows the candidate (EPRO) vs same-capacity generic skill difference holds uniformly:
no publication gives EPRO a material, cross-seed consistent advantage over generic.
This strengthens the aggregate fail-closed verdict (diff CI low <= 0) with a
publication-level uniformity check.

CPU-only; reads existing heldout .npz.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def _wmae(y, w, pred):
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    denom = w.sum()
    if denom <= 0:
        return None
    return float((w * np.abs(p - y)).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="scheme3 results dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    d = Path(args.dir)
    per_pub = {}  # pub -> {seeds: {variant: skill_diff, ...}}
    n_pairs_pub = {}

    for s in range(args.seeds):
        e = np.load(d / f"heldout_epro_seed{s}.npz")
        g = np.load(d / f"heldout_generic_seed{s}.npz")
        t = np.load(d / f"heldout_trivial_seed{s}.npz")
        # per-publication grouped indices
        pubs = sorted(set(map(str, e["pub"])))
        for pub in pubs:
            idx = np.where(e["pub"] == pub)[0]
            n = len(idx)
            n_pairs_pub[pub] = n_pairs_pub.get(pub, 0) + n
            if n < args.min_pairs:
                continue
            y = e["y"][idx]; w = e["w"][idx]
            sk_epro = None
            sk_gen = None
            wtr = _wmae(y, w, t["pred"][idx])
            if wtr is not None and wtr > 0:
                we = _wmae(y, w, e["pred"][idx])
                wg = _wmae(y, w, g["pred"][idx])
                if we is not None and wg is not None:
                    sk_epro = 1.0 - we / wtr
                    sk_gen = 1.0 - wg / wtr
            if sk_epro is None or sk_gen is None:
                continue
            rec = per_pub.setdefault(pub, {"skills": []})
            rec["skills"].append({"seed": s, "epro": sk_epro, "generic": sk_gen,
                                  "diff": sk_epro - sk_gen})

    rows = []
    for pub, rec in per_pub.items():
        diffs = [x["diff"] for x in rec["skills"]]
        epros = [x["epro"] for x in rec["skills"]]
        gens = [x["generic"] for x in rec["skills"]]
        rows.append({
            "publication": pub,
            "n_pairs": n_pairs_pub.get(pub, 0),
            "n_seeds": len(diffs),
            "epro_mean_skill": float(np.mean(epros)),
            "generic_mean_skill": float(np.mean(gens)),
            "diff_mean": float(np.mean(diffs)),
            "diff_min": float(min(diffs)),
            "diff_max": float(max(diffs)),
            "fraction_seeds_epro_gt_generic": sum(1 for x in diffs if x > 0) / len(diffs),
            "all_seeds_epro_gt_generic": bool(all(x > 0 for x in diffs)),
        })
    rows.sort(key=lambda r: -r["n_pairs"])

    all_diffs = [x["diff"] for r in rows for x in per_pub[r["publication"]]["skills"]]
    all_epro_gt = sum(1 for v in all_diffs if v > 0) / max(len(all_diffs), 1)
    n_pub_any_seed_gt = sum(1 for r in rows if r["fraction_seeds_epro_gt_generic"] > 0)
    n_pub_all_gt = sum(1 for r in rows if r["all_seeds_epro_gt_generic"])
    n_pub_gt_half = sum(1 for r in rows if r["fraction_seeds_epro_gt_generic"] >= 0.5)

    report = {
        "schema": "reactflow_delta.phase3.benchmark_resource.per_pub_skill.v1",
        "run_id": Path(args.out_dir).name,
        "authority_epoch": 18,
        "phase": "PHASE3-BENCHMARK-RESOURCE",
        "question": ("Does EPRO (repaired nonlocal propagation) beat the same-capacity "
                     "generic at ANY held-out publication, with cross-seed consistency?"),
        "note": "Per-publication skill == per-fold skill because LOOCV holds out each pub in exactly one fold.",
        "min_pairs": args.min_pairs,
        "summary": {
            "n_publications_analyzed": len(rows),
            "pooled_fraction_seed_pubs_epro_gt_generic": all_epro_gt,
            "n_pubs_any_seed_epro_gt": n_pub_any_seed_gt,
            "n_pubs_all_seeds_epro_gt": n_pub_all_gt,
            "n_pubs_majority_seeds_epro_gt": n_pub_gt_half,
            "overall_diff_mean": float(np.mean(all_diffs)) if all_diffs else None,
            "overall_diff_min": float(np.min(all_diffs)) if all_diffs else None,
            "overall_diff_max": float(np.max(all_diffs)) if all_diffs else None,
        },
        "per_publication": rows,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_pub_skill.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[per_pub_skill] wrote -> {out/'per_pub_skill.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
