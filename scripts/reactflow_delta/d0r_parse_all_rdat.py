#!/usr/bin/env python3
"""D0-R: parse ALL downloaded RDAT files and extract candidate WT-mutant relations.

Unlike d0r_reaudit_tierA.py (which only processes Tier A non-Ribonanza entries),
this script processes every .rdat file in --download-dir, regardless of registry
classification.  This maximizes pair discovery from the full 1024-entry registry.

Reuses the proven audit_file + build_relations logic from d0r_reaudit_tierA.
Outputs:
  - d0r_all_audit.json       : per-file audit
  - d0r_all_relations.json   : candidate single-mutant relations
  - d0r_all_summary.json     : aggregate counts per file/study/parent
Forward-only: no claims beyond candidate_only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import shared logic from d0r_reaudit_tierA
_SCRIPT_DIR = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "d0r_reaudit_tierA", _SCRIPT_DIR / "d0r_reaudit_tierA.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

audit_file = _mod.audit_file
build_relations = _mod.build_relations


def load_registry_index(registry_path: Path) -> dict[str, dict[str, Any]]:
    """Load registry into {rmdb_id: entry} dict."""
    idx: dict[str, dict[str, Any]] = {}
    with registry_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["rmdb_id"]] = r
    return idx


def classify_entry(rmdb_id: str, rec: dict[str, Any]) -> str:
    """Classify entry as tierA_nonribo, ribonanza, or other."""
    t = " ".join(str(rec.get(k, "")) for k in ("rmdb_id", "name", "description", "comments", "category"))
    if _mod.RIBO_RE.search(t):
        return "ribonanza"
    sigs = [s for s, p in _mod.TIERA_PATS.items() if p.search(t)]
    if sigs:
        return "tierA_nonribo"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--download-dir", type=Path, required=True)
    ap.add_argument("--artifacts-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().isoformat()

    registry_idx = load_registry_index(args.registry)
    print(f"[registry] {len(registry_idx)} entries", file=sys.stderr)

    rdat_files = sorted(args.download_dir.glob("*.rdat"))
    print(f"[scan] {len(rdat_files)} RDAT files in {args.download_dir}", file=sys.stderr)

    audits: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for i, rdat_path in enumerate(rdat_files, 1):
        rmdb_id = rdat_path.stem
        rec = registry_idx.get(rmdb_id, {"rmdb_id": rmdb_id})
        cat = classify_entry(rmdb_id, rec)
        category_counts[cat] += 1

        audit = audit_file(rdat_path)
        audit["rmdb_id"] = rmdb_id
        audit["owner"] = rec.get("owner")
        audit["citation_doi"] = rec.get("citation", {}).get("doi")
        audit["registry_category"] = cat
        audit["construct_count_metadata"] = int(rec.get("construct_count", "0"))
        audits.append(audit)

        if audit.get("parse_status") == "error":
            parse_errors.append({"rmdb_id": rmdb_id, "error": audit["parse_error"]})
            if i % 50 == 0:
                print(f"[progress] {i}/{len(rdat_files)} (errors={len(parse_errors)})", file=sys.stderr)
            continue

        rels = build_relations(audit, rec, rdat_path)
        all_relations.extend(rels)

        if i % 50 == 0 or i == len(rdat_files):
            print(f"[progress] {i}/{len(rdat_files)} files, "
                  f"{len(all_relations)} relations so far, "
                  f"errors={len(parse_errors)}", file=sys.stderr)

    # Aggregate
    study_counts: Counter[tuple] = Counter()
    parent_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    modifier_counts: Counter[str] = Counter()
    for rel in all_relations:
        study_counts[(rel["owner"], rel["citation_doi"])] += 1
        parent_counts[rel["parent_prefix"]] += 1
        owner_counts[rel["owner"]] += 1
        modifier_counts[rel.get("modifier", "unknown")] += 1

    total = len(all_relations)
    summary = {
        "schema_version": "reactflow-delta-d0r-all-parse-summary-v1",
        "generated_at": now,
        "rdat_files_scanned": len(rdat_files),
        "parse_errors": len(parse_errors),
        "registry_categories": dict(category_counts),
        "total_candidate_relations": total,
        "distinct_studies": len(study_counts),
        "distinct_parents": len(parent_counts),
        "distinct_owners": len(owner_counts),
        "top_parents": dict(parent_counts.most_common(30)),
        "top_owners": dict(owner_counts.most_common(20)),
        "modifier_distribution": dict(modifier_counts),
    }

    # Write outputs
    audit_path = args.artifacts_dir / "d0r_all_audit.json"
    relations_path = args.artifacts_dir / "d0r_all_relations.json"
    summary_path = args.artifacts_dir / "d0r_all_summary.json"

    with audit_path.open("w") as fh:
        json.dump({"schema_version": "reactflow-delta-d0r-all-audit-v1",
                   "generated_at": now, "audits": audits}, fh, indent=2)
    with relations_path.open("w") as fh:
        json.dump({"schema_version": "reactflow-delta-d0r-all-relations-v1",
                   "generated_at": now, "relations": all_relations}, fh, indent=2)
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Slim manifest (tracked in git)
    manifest = {
        "schema_version": "reactflow-delta-d0r-all-parse-manifest-v1",
        "generated_at": now,
        "rdat_files_scanned": len(rdat_files),
        "parse_errors": len(parse_errors),
        "total_candidate_relations": total,
        "distinct_studies": len(study_counts),
        "distinct_parents": len(parent_counts),
        "audit_path": str(audit_path),
        "relations_path": str(relations_path),
        "summary_path": str(summary_path),
    }
    with args.manifest.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    print(json.dumps({"total_relations": total, "distinct_parents": len(parent_counts),
                      "distinct_studies": len(study_counts), "parse_errors": len(parse_errors)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
