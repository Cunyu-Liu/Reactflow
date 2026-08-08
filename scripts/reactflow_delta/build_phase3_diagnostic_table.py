#!/usr/bin/env python3
"""Phase 3 closure -> benchmark/resource route (deliverable 3).

Consolidate the three diagnostics (deliverable 1: caller reliability / label shift;
deliverable 2: magnitude-vs-noise floor) into a single auditable per-publication
evidence table + merged JSON, forming the main evidence table for a resource /
negative-result paper.

CPU-only; reads the two existing report JSONs, merges per-publication rows.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caller-report", required=True)
    ap.add_argument("--noise-report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    caller = json.loads(Path(args.caller_report).read_text(encoding="utf-8"))
    noise = json.loads(Path(args.noise_report).read_text(encoding="utf-8"))

    caller_pub = caller["publication_label_shift"]["per_publication"]
    noise_pub = noise["per_publication"]
    pubs = sorted(set(caller_pub) | set(noise_pub))

    rows = []
    for pub in pubs:
        c = caller_pub.get(pub, {})
        n = noise_pub.get(pub, {})
        row = {
            "publication": pub,
            "n_pairs": c.get("n", n.get("n", 0)),
            "n_called": c.get("called", 0),
            "n_nocall": c.get("nocall", 0),
            "nocall_rate": (c.get("nocall", 0) / c.get("n", 1)) if c.get("n") else None,
            "changers_rate": c.get("changers_rate"),
            "n_with_noise": n.get("with_noise", 0),
            "pos_total": n.get("pos_total", 0),
            "fraction_pos_below_1x_noise": n.get("fraction_pos_below_1x"),
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["n_pairs"])

    summary = {
        "n_publications": len(pubs),
        "n_pairs_total": sum(r["n_pairs"] for r in rows),
        "n_pairs_nocall_total": sum(r["n_nocall"] for r in rows),
        "pooled_nocall_rate": sum(r["n_nocall"] for r in rows) / sum(r["n_pairs"] for r in rows),
        "pooled_fraction_pos_below_1x_noise": (
            noise["noise_floor"]["fraction_pair_position_below_1x_noise"]),
        "caller_global_icc": caller["caller_reliability"]["global_icc"],
        "overall_feature_mean_abs_smd": caller["feature_domain_shift"]["overall_mean_abs_smd"],
    }

    report = {
        "schema": "reactflow_delta.phase3.benchmark_resource.consolidated.v1",
        "run_id": Path(args.out_dir).name,
        "authority_epoch": 18,
        "phase": "PHASE3-BENCHMARK-RESOURCE",
        "purpose": ("Per-publication evidence table for resource/negative-result paper: "
                    "caller coverage (NO_CALL), label shift (changers rate), and magnitude "
                    "signal-vs-noise (fraction below 1x replicate noise)."),
        "sources": {
            "caller": args.caller_report,
            "noise_floor": args.noise_report,
        },
        "summary": summary,
        "per_publication": rows,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3_diagnostic_table.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # TSV main table
    header = ["publication", "n_pairs", "n_called", "nocall_rate", "changers_rate",
              "n_with_noise", "pos_total", "fraction_pos_below_1x_noise"]
    with (out / "phase3_diagnostic_table.tsv").open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[h] is None else (f"{r[h]:.4f}" if isinstance(r[h], float)
                                                         else str(r[h])) for h in header) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\n[consolidated] wrote -> {out/'phase3_diagnostic_table.json'} and .tsv")


if __name__ == "__main__":
    raise SystemExit(main())
