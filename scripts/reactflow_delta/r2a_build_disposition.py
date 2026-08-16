"""R2A: 1024-asset controlled disposition ledger.

Joins the frozen RMDB release asset manifest (1024 rows) with the D0-X parse
summary (610 failures w/ error categories) and the file-level requalify ledger
to emit a complete 1024-row disposition ledger where every asset has a
non-empty, auditable disposition (never left NOT_SEARCHED/missing-as-zero).

Contract: ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md
R2A (§13.2): "1024/1024无空disposition；missing不为zero；新增publication yield可审计".
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ART = Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta")
ASSET_MANIFEST = Path("/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x/rmdb_release_assets_20260803.jsonl")
PARSE_SUMMARY = ART / "d0x" / "d0x_parse_20260803T0000" / "candidate_summary.json"
REQUALIFY = ART / "d0x" / "d0x_requalify_20260803T0000" / "requalification_ledger.json"
OUT = Path("/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x_v2")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def categorize(error: str) -> str:
    if not error:
        return "PARSE_FAIL_UNKNOWN"
    if "length does not match SEQPOS" in error:
        return "PARSE_FAIL_LENGTH_MISMATCH_REACTIVITY_VS_SEQPOS"
    if "non-numeric value" in error:
        return "PARSE_FAIL_NON_NUMERIC_REACTIVITY"
    if "malformed SEQPOS token" in error:
        return "PARSE_FAIL_MALFORMED_SEQPOS"
    if "missing REACTIVITY rows" in error:
        return "PARSE_FAIL_MISSING_REACTIVITY_ROWS"
    if "annotation token lacks key/value separator" in error:
        return "PARSE_FAIL_ANNOTATION_MISSING_SEPARATOR"
    if "invalid positive profile index" in error:
        return "PARSE_FAIL_INVALID_PROFILE_INDEX"
    return "PARSE_FAIL_OTHER"


def main() -> int:
    # 1. asset manifest -> master 1024 rows
    assets = {}
    for line in ASSET_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        assets[a["source_accession"]] = a
    if len(assets) != 1024:
        print(f"ERROR: asset manifest has {len(assets)} != 1024", file=sys.stderr)
        return 1

    # 2. parse summary -> failure category per accession
    ps = json.loads(PARSE_SUMMARY.read_text())
    fails = {f["source_accession"]: f for f in ps["parse_failures"]}
    if len(fails) != ps["parse_failed_file_count"]:
        print("ERROR: parse_failures count mismatch", file=sys.stderr)
        return 1

    # 3. requalify ledger -> file-level verified/hash
    rq = json.loads(REQUALIFY.read_text())
    rq_by_acc = {r["source_accession"]: r for r in rq["records"]}

    rows = []
    parse_success = 0
    parse_fail = 0
    for acc, a in assets.items():
        file_rec = rq_by_acc.get(acc, {})
        file_verified = bool(file_rec.get("hash_match"))
        if acc in fails:
            category = categorize(fails[acc]["error"])
            disposition = category
            parse_fail += 1
        else:
            disposition = "PARSE_SUCCESS"
            parse_success += 1
        rows.append({
            "source_accession": acc,
            "asset_name": a["asset_name"],
            "asset_id": a["asset_id"],
            "source_group": a["source_group"],
            "release_tag": a["release_tag"],
            "source_id": a["source_id"],
            "license_status": a["license_status"],
            "file_present": bool(file_rec.get("file_present")),
            "file_verified_hash": file_verified,
            "disposition": disposition,
            "parse_error": (fails[acc]["error"] if acc in fails else None),
        })

    # order rows deterministically by accession
    rows.sort(key=lambda r: r["source_accession"])

    # 4. category counts + publication(accession-prefix) yield
    from collections import Counter
    cat_counts = Counter(r["disposition"] for r in rows)
    # rescue/publication yield: distinct source_group (as closest available pub proxy)
    yield_by_group = Counter(
        r["source_group"] for r in rows if r["disposition"] == "PARSE_SUCCESS")

    n_empty = sum(1 for r in rows if not r["disposition"])
    ledger = {
        "schema_version": "reactflow_delta.d0x_v2.asset_disposition.v1",
        "phase_id": "REBUILD-R2A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_manifest_path": "data_registry/d0x/rmdb_release_assets_20260803.jsonl",
        "asset_count": len(rows),
        "parse_success_count": parse_success,
        "parse_fail_count": parse_fail,
        "empty_disposition_count": n_empty,
        "disposition_categories": dict(cat_counts),
        "parse_success_by_source_group": dict(yield_by_group),
        "notes": (
            "disposition is never empty; missing/parse-failure is an explicit "
            "PARSE_FAIL_* category, not imputed zero. source_group used as "
            "publication proxy until R2C group-atom/publication metadata lands."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    # ledger TSV (auditable row-level)
    tsv = OUT / "asset_disposition_20260807.tsv"
    keys = ["source_accession", "asset_name", "asset_id", "source_group",
            "release_tag", "license_status", "file_present",
            "file_verified_hash", "disposition", "parse_error"]
    with tsv.open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    # JSONL rows
    jl = OUT / "asset_disposition_20260807.jsonl"
    with jl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # summary
    (OUT / "asset_disposition_20260807.summary.json").write_text(
        json.dumps(ledger, indent=2))
    # hashes
    sums = "\n".join(f"{sha256(p)}  data_registry/d0x_v2/{p.name}"
                     for p in (tsv, jl))
    (OUT / "SHA256SUMS").write_text(sums + "\n")

    print(json.dumps(ledger, indent=2))
    print(f"rows written: {len(rows)}, empty: {n_empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
