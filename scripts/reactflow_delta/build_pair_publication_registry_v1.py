#!/usr/bin/env python3
"""Build the pair-level publication registry v1 (benchmark_v3 / Task 1B).

Reads primary_pairs_v2.jsonl (7961 rows) and assigns each eligible exact pair a
stable pair_id = SHA-256 of the canonical JSON of the identity tuple:
    (dataset_version, source_accession, asset_id, wt_profile_id,
     mutant_profile_id, edit_position, ref_base, exact_alt_base, probe,
     condition_id)
using stable field order and stable serialization.

Publication identity is resolved primary = canonical PMID (from the accession
registry) else canonical DOI else UNRESOLVED_PUBLICATION:<stable_id>.  Same-PMID
across studies and multi-PMID within a study are recorded as anomaly tables.
UNRESOLVED publications do NOT count toward the confirmed publication N.

Core logic is importable and testable with an in-memory pair list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DATASET_VERSION = "reactflow_delta.v2"
UNKNOWN_NOT_ASSERTED = "UNKNOWN_NOT_ASSERTED"
UNASSIGNED = "UNASSIGNED"

PUBLICATION_KEYS = [
    "pair_id", "asset_id", "study_id", "source_accession", "citation_raw",
    "pmid", "doi", "publication_id_normalized", "citation_resolution_status",
    "resolution_evidence", "parent_id", "sequence_sha256", "lineage_id",
    "rna_family", "probe", "platform", "condition_id", "license_status",
    "proposed_split_role",
]

RESOLVED = "RESOLVED"
UNRESOLVED_PUBLICATION = "UNRESOLVED_PUBLICATION"
AMBIGUOUS = "AMBIGUOUS"


def _clean(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def canonical_json(obj) -> str:
    """Deterministic JSON serialization (stable key order, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def study_id_of(accession: str) -> str:
    return accession.split("_")[0]


def condition_id_of(condition: Optional[dict]) -> str:
    """Stable condition id from the condition dict (None -> hashed empty)."""
    return sha256_hex(canonical_json(condition if condition is not None else {}))


def probe_of(condition: Optional[dict]) -> str:
    if not condition:
        return UNKNOWN_NOT_ASSERTED
    modifier = condition.get("modifier") or []
    if not modifier:
        return UNKNOWN_NOT_ASSERTED
    return ",".join(sorted(str(m) for m in modifier))


def identity_tuple(pair: dict, asset_id: object, condition_id: str) -> tuple:
    """The stable identity tuple used to derive pair_id (fixed field order)."""
    coordinate = pair.get("coordinate") or {}
    return (
        DATASET_VERSION,
        pair.get("source_accession"),
        asset_id,
        f"{pair.get('source_accession')}:{pair.get('wt_profile_index')}",
        f"{pair.get('source_accession')}:{pair.get('mutant_profile_index')}",
        coordinate.get("offset"),
        pair.get("ref_allele"),
        pair.get("alt_allele"),
        probe_of(pair.get("condition")),
        condition_id,
    )


def pair_id_of(pair: dict, asset_id: object, condition_id: str) -> str:
    return sha256_hex(canonical_json(identity_tuple(pair, asset_id, condition_id)))


def citation_raw_of(registry_entry: Optional[dict]) -> str:
    if not registry_entry:
        return ""
    c = registry_entry.get("citation") or {}
    parts = []
    for k in ("title", "authors", "journal", "year", "pubmed", "doi"):
        v = _clean(c.get(k))
        if v:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def resolve_publication(registry_entry: Optional[dict], study_id: str):
    """Return (publication_id_normalized, citation_resolution_status, pmid, doi, evidence)."""
    entry = registry_entry or {}
    citation = entry.get("citation") or {}
    pubmed = _clean(citation.get("pubmed"))
    doi = _clean(citation.get("doi"))
    if pubmed:
        return f"pmid_{pubmed}", RESOLVED, pubmed, doi, "pubmed from d0r_accession_registry"
    if doi:
        return f"doi_{doi}", RESOLVED, "", doi, "doi from d0r_accession_registry"
    return f"UNRESOLVED_PUBLICATION:{study_id}", UNRESOLVED_PUBLICATION, "", "", \
        "no pubmed/doi in d0r_accession_registry"


# ---------------------------------------------------------------------------
# core (pure, testable) logic
# ---------------------------------------------------------------------------
def build_pair_publication_registry(
    pair_rows: Iterable[dict],
    registry: Optional[Dict[str, dict]] = None,
    asset_meta: Optional[Dict[str, dict]] = None,
    proposed_roles: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Build the pair publication registry from an in-memory pair list.

    registry: {rmdb_id: accession-registry entry}.
    asset_meta: {source_accession: asset disposition row} for asset_id,
                rna_family (source_group), license_status.
    proposed_roles: {study_id: proposed split role}.
    """
    registry = registry or {}
    asset_meta = asset_meta or {}
    proposed_roles = proposed_roles or {}

    rows: List[dict] = []
    for pair in pair_rows:
        accession = pair.get("source_accession")
        asset = asset_meta.get(accession, {})
        asset_id = asset.get("asset_id")
        study_id = study_id_of(accession)
        condition_id = condition_id_of(pair.get("condition"))
        pid = pair_id_of(pair, asset_id, condition_id)

        pub_id, status, pmid, doi, evidence = resolve_publication(
            registry.get(accession), study_id)

        rows.append({
            "pair_id": pid,
            "asset_id": asset_id,
            "study_id": study_id,
            "source_accession": accession,
            "citation_raw": citation_raw_of(registry.get(accession)),
            "pmid": pmid,
            "doi": doi,
            "publication_id_normalized": pub_id,
            "citation_resolution_status": status,
            "resolution_evidence": evidence,
            "parent_id": pair.get("wt_reuse_group") or UNKNOWN_NOT_ASSERTED,
            "sequence_sha256": UNKNOWN_NOT_ASSERTED,  # filled by Task 1C
            "lineage_id": UNKNOWN_NOT_ASSERTED,       # filled by Task 1C
            "rna_family": asset.get("source_group") or UNKNOWN_NOT_ASSERTED,
            "probe": probe_of(pair.get("condition")),
            "platform": UNKNOWN_NOT_ASSERTED,
            "condition_id": condition_id,
            "license_status": asset.get("license_status") or UNKNOWN_NOT_ASSERTED,
            "proposed_split_role": proposed_roles.get(study_id, UNASSIGNED),
        })
    return rows


def publication_anomalies(rows: Iterable[dict]) -> dict:
    """Detect same-PMID-across-studies and multi-PMID-within-study anomalies."""
    pmid_to_studies: Dict[str, set] = defaultdict(set)
    study_to_pmids: Dict[str, set] = defaultdict(set)
    for r in rows:
        pmid = _clean(r.get("pmid"))
        study = r.get("study_id")
        if pmid:
            pmid_to_studies[pmid].add(study)
            study_to_pmids[study].add(pmid)
    same_pmid_across_studies = [
        {"pmid": pmid, "studies": sorted(studies)}
        for pmid, studies in sorted(pmid_to_studies.items())
        if len(studies) > 1
    ]
    multi_pmid_within_study = [
        {"study_id": study, "pmids": sorted(pmids)}
        for study, pmids in sorted(study_to_pmids.items())
        if len(pmids) > 1
    ]
    return {
        "same_pmid_across_studies": same_pmid_across_studies,
        "multi_pmid_within_study": multi_pmid_within_study,
    }


def confirmed_publication_n(rows: Iterable[dict]) -> int:
    """Number of distinct RESOLVED publications carrying a PMID (UNRESOLVED excluded)."""
    pubs = set()
    for r in rows:
        if r.get("citation_resolution_status") == RESOLVED and _clean(r.get("pmid")):
            pubs.add(r["publication_id_normalized"])
    return len(pubs)


def build_ledger(rows: List[dict], anomalies: dict) -> dict:
    resolved = [r for r in rows if r.get("citation_resolution_status") == RESOLVED]
    unresolved = [r for r in rows if r.get("citation_resolution_status") == UNRESOLVED_PUBLICATION]
    distinct_pubs = sorted({r["publication_id_normalized"] for r in rows})
    confirmed_pubs = sorted({r["publication_id_normalized"] for r in rows
                             if r.get("citation_resolution_status") == RESOLVED
                             and _clean(r.get("pmid"))})
    pub_to_studies: Dict[str, set] = defaultdict(set)
    for r in rows:
        pub_to_studies[r["publication_id_normalized"]].add(r["study_id"])
    return {
        "schema_version": "reactflow_delta.pair_publication_registry.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs": len(rows),
        "resolved_pairs": len(resolved),
        "unresolved_pairs": len(unresolved),
        "distinct_publications": distinct_pubs,
        "confirmed_publication_n": len(confirmed_pubs),
        "confirmed_publications": confirmed_pubs,
        "publication_studies": {
            pub: sorted(studies) for pub, studies in sorted(pub_to_studies.items())
        },
        "anomalies": anomalies,
        "note": (
            "confirmed_publication_N counts only RESOLVED publications with a "
            "PMID; UNRESOLVED_PUBLICATION entries never count toward N."
        ),
    }


def write_tsv(rows: List[dict], out_path: Path) -> None:
    with out_path.open("w") as fh:
        fh.write("\t".join(PUBLICATION_KEYS) + "\n")
        for r in rows:
            fh.write("\t".join(_clean(r.get(k)) for k in PUBLICATION_KEYS) + "\n")


def _yaml_scalar(v: object, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(v, bool):
        return f"{pad}{str(v).lower()}"
    if isinstance(v, (int, float)):
        return f"{pad}{v}"
    return f"{pad}{json.dumps(str(v))}"


def _yaml_dump(obj, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_yaml_dump(v, indent + 2))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v, 0).strip()}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(_yaml_dump(item, indent + 2))
            elif isinstance(item, (list,)):
                lines.append(f"{pad}-{_yaml_dump(item, indent + 2)}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item, 0).strip()}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(obj, 0).strip()}"


def write_yaml(ledger: dict, out_path: Path) -> None:
    body = _yaml_dump(ledger, 0)
    out_path.write_text(body + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    here = Path(__file__).resolve().parent
    worktree_root = here.parent.parent
    default_out = worktree_root / "data_registry" / "reactflow_delta"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pairs", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0r_accession_registry.jsonl"))
    parser.add_argument("--asset-disposition", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x_v2/"
        "asset_disposition_20260807.jsonl"))
    parser.add_argument("--split-v2", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/split_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=default_out)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [json.loads(line) for line in args.primary_pairs.open() if line.strip()]
    registry = {}
    for line in args.registry.open():
        if line.strip():
            rec = json.loads(line)
            registry[rec.get("rmdb_id")] = rec
    asset_meta = {}
    for line in args.asset_disposition.open():
        if line.strip():
            rec = json.loads(line)
            asset_meta[rec.get("source_accession")] = rec

    # proposed roles from split_v2 study_roles (best available before split_v3)
    proposed_roles: Dict[str, str] = {}
    if args.split_v2.exists():
        import re
        text = args.split_v2.read_text()
        m = re.search(r"study_roles:\n((?:  [^\n]+\n)+)", text)
        if m:
            for line in m.group(1).splitlines():
                key, _, val = line.strip().partition(":")
                if key:
                    proposed_roles[key] = val.strip()

    rows = build_pair_publication_registry(
        pair_rows, registry=registry, asset_meta=asset_meta,
        proposed_roles=proposed_roles)
    anomalies = publication_anomalies(rows)
    ledger = build_ledger(rows, anomalies)

    tsv_path = args.output_dir / "pair_publication_registry_v1.tsv"
    yaml_path = args.output_dir / "publication_resolution_ledger_v1.yaml"
    write_tsv(rows, tsv_path)
    write_yaml(ledger, yaml_path)

    print(json.dumps({
        "total_pairs": len(rows),
        "confirmed_publication_n": ledger["confirmed_publication_n"],
        "resolved_pairs": ledger["resolved_pairs"],
        "unresolved_pairs": ledger["unresolved_pairs"],
        "same_pmid_across_studies": len(anomalies["same_pmid_across_studies"]),
        "multi_pmid_within_study": len(anomalies["multi_pmid_within_study"]),
        "tsv": str(tsv_path),
        "yaml": str(yaml_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())