#!/usr/bin/env python3
"""Build the asset-level disposition v3 ledger (benchmark_v3 / Task 1A).

Reads the 1024-asset d0x_v2 disposition JSONL and writes exactly one row per
asset (1024 rows) to a TSV plus a JSON schema describing the columns.

Guarantees (contract):
  * one row per asset_id -- uniqueness, no silent drop.  Any asset for which
    no info can be found still gets a row with UNKNOWN_NOT_ASSERTED / NOT_RUN
    values, never dropped.
  * missing/parse-failure is an explicit status (PARSE_FAIL_* /
    UNKNOWN_NOT_ASSERTED), never imputed as 0-reactivity or empty.
  * per-profile LENGTH_MISMATCH rows are counted separately from the asset
    count (11309 is a PROFILE count, never an assets count).
  * asset_sha256 is never fabricated: it is taken from the raw .rdat file only
    if actually present on disk, otherwise UNKNOWN_NOT_ASSERTED.

The 39GB canonical_records_v2.jsonl is streamed (never fully loaded) to compute
n_profiles / n_records.  The core logic is importable and testable with
in-memory asset rows + profile rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# controlled vocabulary
# ---------------------------------------------------------------------------
NOT_RUN = "NOT_RUN"
UNKNOWN_NOT_ASSERTED = "UNKNOWN_NOT_ASSERTED"
NONE = "NONE"

# The d0x_v2 asset disposition JSONL keys.
ASSET_KEYS = [
    "asset_id", "source_url_or_accession", "source_accession", "asset_name",
    "asset_sha256", "parser_version", "asset_parse_status", "n_profiles",
    "n_records", "n_exact_pairs", "profile_failure_rows",
    "unique_failure_reason", "failure_recoverability", "license_status",
    "citation_resolution_status", "notes",
]

# Parse-failure categories that are judged recoverable (re-parseable with a
# tolerable parser / reconstructed source).  Everything else on a failed asset
# is judged UNRECOVERABLE.
RECOVERABLE_FAIL_SUBSTRINGS = (
    "LENGTH_MISMATCH_REACTIVITY_VS_SEQPOS",
    "MISSING_REACTIVITY_ROWS",
    "ANNOTATION_MISSING_SEPARATOR",
    "MALFORMED_SEQPOS",
)

# Citation resolution statuses.
RESOLVED = "RESOLVED"
UNRESOLVED_PUBLICATION = "UNRESOLVED_PUBLICATION"
AMBIGUOUS = "AMBIGUOUS"


def asset_disposition_v3_schema() -> dict:
    """JSON schema (field name, type, description, required) for the v3 TSV."""
    fields = [
        ("asset_id", "integer", "Frozen RMDB asset identifier (unique primary key).", True),
        ("source_url_or_accession", "string", "Source URL or accession of the raw asset.", True),
        ("source_accession", "string", "RMDB source accession (file base name without .rdat).", True),
        ("asset_name", "string", "Asset file name (e.g. <acc>.rdat).", True),
        ("asset_sha256", "string", "SHA-256 of the raw .rdat file if present on disk, else "
                                   "UNKNOWN_NOT_ASSERTED. Never fabricated.", True),
        ("parser_version", "string", "Parser/canonicalization version that produced the disposition.", True),
        ("asset_parse_status", "string", "PARSE_SUCCESS or explicit PARSE_FAIL_* category; never empty.", True),
        ("n_profiles", "string", "Number of distinct profiles for the asset from canonical_records_v2 "
                                 "(NOT_RUN if the 39GB count has not been produced).", True),
        ("n_records", "string", "Number of canonical records for the asset from canonical_records_v2 "
                                "(NOT_RUN if the 39GB count has not been produced).", True),
        ("n_exact_pairs", "string", "Number of primary exact-delta pairs for the asset "
                                    "(NOT_RUN if not counted).", True),
        ("profile_failure_rows", "integer", "Count of LENGTH_MISMATCH PROFILE rows for the asset "
                                            "(a profile count, never an asset count).", True),
        ("unique_failure_reason", "string", "Single most common profile parse failure reason, or the "
                                            "asset's own disposition if PARSE_FAIL, else NONE.", True),
        ("failure_recoverability", "string", "RECOVERABLE / UNRECOVERABLE / NOT_APPLICABLE.", True),
        ("license_status", "string", "License status carried through from d0x_v2 (VERIFIED_CC0_RMDB).", True),
        ("citation_resolution_status", "string", "RESOLVED / UNRESOLVED_PUBLICATION / AMBIGUOUS from "
                                                 "the accession registry.", True),
        ("notes", "string", "Brief human-readable notes.", False),
    ]
    return {
        "schema_version": "reactflow_delta.asset_disposition_v3.v1",
        "generated_by": "scripts/reactflow_delta/build_asset_disposition_v3.py",
        "description": "Asset-level disposition ledger: one row per frozen RMDB asset.",
        "row_key": "asset_id",
        "row_count_expected": 1024,
        "fields": [
            {"name": name, "type": typ, "description": desc, "required": required}
            for (name, typ, desc, required) in fields
        ],
    }


def _study_prefix(accession: str) -> str:
    """Study id = source_accession prefix before the first underscore."""
    return accession.split("_")[0]


def resolve_citation_status(registry_entry: Optional[dict]) -> str:
    """Resolve citation status for an asset from the accession-registry entry.

    The registry is keyed by rmdb_id (one entry per asset accession), so a
    single entry carrying BOTH a pubmed and a doi is the SAME publication and
    is RESOLVED (not ambiguous).  RESOLVED if the entry carries a non-empty
    pubmed or doi; otherwise UNRESOLVED_PUBLICATION.  A missing entry is
    UNRESOLVED_PUBLICATION (never invented).  AMBIGUOUS is reserved for a
    genuine many-to-one publication mapping (kept in the controlled
    vocabulary; not triggered by a single PMID+DOI entry).
    """
    if not registry_entry:
        return UNRESOLVED_PUBLICATION
    citation = registry_entry.get("citation") or {}
    pubmed = _clean(citation.get("pubmed"))
    doi = _clean(citation.get("doi"))
    if pubmed or doi:
        return RESOLVED
    return UNRESOLVED_PUBLICATION


def judge_failure_recoverability(asset_parse_status: str) -> str:
    """Judgment: RECOVERABLE / UNRECOVERABLE / NOT_APPLICABLE."""
    if asset_parse_status == "PARSE_SUCCESS":
        return "NOT_APPLICABLE"
    if any(s in asset_parse_status for s in RECOVERABLE_FAIL_SUBSTRINGS):
        return "RECOVERABLE"
    return "UNRECOVERABLE"


def _clean(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s


def _fmt(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def count_literal(v: object) -> str:
    """Format a count, defaulting to NOT_RUN when the count is absent/None."""
    if v is None:
        return NOT_RUN
    return str(v)


# ---------------------------------------------------------------------------
# core (pure, testable) logic
# ---------------------------------------------------------------------------
def build_asset_disposition_rows(
    asset_rows: Iterable[dict],
    profile_rows: Iterable[dict],
    record_counts: Optional[Dict[str, dict]] = None,
    pair_counts: Optional[Dict[str, int]] = None,
    registry: Optional[Dict[str, dict]] = None,
    sha256_by_accession: Optional[Dict[str, str]] = None,
    parser_version: str = "reactflow_delta.d0x_v2.asset_disposition.v2",
) -> List[dict]:
    """Build the v3 asset rows from in-memory inputs.

    asset_rows : d0x_v2 asset disposition rows (one per asset).
    profile_rows: asset_disposition_v2 PROFILE rows (many per asset, failures).
    record_counts: {accession: {"n_records":int,"n_profiles":int}} or None.
    pair_counts: {accession: int} exact-pair counts or None.
    registry: {rmdb_id: accession-registry entry} or None.
    sha256_by_accession: {accession: sha256} from raw files or None.

    Exactly one row is emitted per asset_id; missing info is UNKNOWN_NOT_ASSERTED
    / NOT_RUN, never dropped.
    """
    record_counts = record_counts or {}
    pair_counts = pair_counts or {}
    registry = registry or {}
    sha256_by_accession = sha256_by_accession or {}

    # group profile rows by accession
    profile_by_acc: Dict[str, List[dict]] = {}
    for pr in profile_rows:
        acc = pr.get("source_accession")
        profile_by_acc.setdefault(acc, []).append(pr)

    rows: List[dict] = []
    for asset in asset_rows:
        acc = asset.get("source_accession")
        asset_parse_status = _fmt(asset.get("disposition")) or UNKNOWN_NOT_ASSERTED

        acc_profiles = profile_by_acc.get(acc, [])
        length_mismatch_rows = [
            p for p in acc_profiles
            if _fmt(p.get("disposition_reason")) == "LENGTH_MISMATCH"
        ]
        profile_failure_rows = len(length_mismatch_rows)

        # most common profile parse-failure reason, else asset's own disposition
        if asset_parse_status == "PARSE_SUCCESS":
            unique_failure_reason = NONE
        else:
            reason_counts = Counter(
                _fmt(p.get("disposition_reason")) for p in acc_profiles
                if _fmt(p.get("disposition_reason"))
            )
            if reason_counts:
                unique_failure_reason = reason_counts.most_common(1)[0][0]
            else:
                unique_failure_reason = asset_parse_status

        cnt = record_counts.get(acc)
        n_profiles = cnt.get("n_profiles") if cnt else None
        n_records = cnt.get("n_records") if cnt else None

        registry_entry = registry.get(acc)

        notes = []
        if asset_parse_status != "PARSE_SUCCESS":
            notes.append(f"parse_fail:{asset_parse_status}")
        if profile_failure_rows:
            notes.append(f"{profile_failure_rows} LENGTH_MISMATCH profile rows")
        if not cnt:
            notes.append("39GB canonical count not yet produced (NOT_RUN)")

        rows.append({
            "asset_id": asset.get("asset_id"),
            "source_url_or_accession": acc,
            "source_accession": acc,
            "asset_name": _fmt(asset.get("asset_name")),
            "asset_sha256": sha256_by_accession.get(acc, UNKNOWN_NOT_ASSERTED),
            "parser_version": parser_version,
            "asset_parse_status": asset_parse_status,
            "n_profiles": count_literal(n_profiles),
            "n_records": count_literal(n_records),
            "n_exact_pairs": count_literal(pair_counts.get(acc)),
            "profile_failure_rows": profile_failure_rows,
            "unique_failure_reason": unique_failure_reason,
            "failure_recoverability": judge_failure_recoverability(asset_parse_status),
            "license_status": _fmt(asset.get("license_status")) or UNKNOWN_NOT_ASSERTED,
            "citation_resolution_status": resolve_citation_status(registry_entry),
            "notes": "; ".join(notes),
        })

    # uniqueness guarantee: no silent drop, no duplicate asset_id
    ids = [r["asset_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("asset_id uniqueness violated: duplicate asset_id present")
    return rows


# ---------------------------------------------------------------------------
# file / streaming helpers (heavy job)
# ---------------------------------------------------------------------------
def stream_count_canonical(
    canonical_path: Path,
    progress_every: int = 1_000_000,
) -> Dict[str, dict]:
    """Stream canonical_records_v2.jsonl and count records/profiles per accession.

    Never loads the file fully; only extracts source_accession and
    source_profile_index per line.  Returns {accession:
    {"n_records": int, "n_profiles": int}}.
    """
    counts: Dict[str, dict] = {}
    nlines = 0
    with canonical_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            acc = rec.get("source_accession")
            if acc is None:
                continue
            entry = counts.setdefault(acc, {"n_records": 0, "n_profiles": 0, "_profiles": set()})
            entry["n_records"] += 1
            entry["_profiles"].add(rec.get("source_profile_index"))
            nlines += 1
            if nlines % progress_every == 0:
                print(f"  streamed {nlines} canonical records", flush=True)
    # collapse profile sets into counts
    out: Dict[str, dict] = {}
    for acc, entry in counts.items():
        out[acc] = {
            "n_records": entry["n_records"],
            "n_profiles": len(entry["_profiles"]),
        }
    return out


def stream_count_pairs(primary_pairs_path: Path) -> Dict[str, int]:
    """Count primary exact pairs per source_accession (small file, direct)."""
    counts: Dict[str, int] = {}
    with primary_pairs_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            acc = rec.get("source_accession")
            if acc is not None:
                counts[acc] = counts.get(acc, 0) + 1
    return counts


def find_and_hash_rdat(accession: str, raw_root: Path) -> str:
    """Search raw RMDB tree for <accession>.rdat and return its sha256.

    Returns UNKNOWN_NOT_ASSERTED if the file is not found (never fabricates).
    """
    target = f"{accession}.rdat"
    for p in raw_root.rglob(target):
        if p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    return UNKNOWN_NOT_ASSERTED


def index_rdat_hashes(raw_root: Path, accessions: Iterable[str]) -> Dict[str, str]:
    """Build sha256 for every raw .rdat file present under raw_root.

    Performs a single tree walk (one rglob) and indexes by file name so we
    never walk the whole tree once per accession.  Files not found on disk are
    UNKNOWN_NOT_ASSERTED (never fabricated).
    """
    wanted = set(accessions)
    target_names = {f"{acc}.rdat" for acc in wanted}
    found: Dict[str, Path] = {}
    for p in raw_root.rglob("*.rdat"):
        if p.name in target_names and p.name not in found:
            found[p.name] = p
    result: Dict[str, str] = {}
    for acc in accessions:
        p = found.get(f"{acc}.rdat")
        if p is not None:
            result[acc] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            result[acc] = UNKNOWN_NOT_ASSERTED
    return result


def load_jsonl_records(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_registry(path: Path) -> Dict[str, dict]:
    reg: Dict[str, dict] = {}
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        reg[rec.get("rmdb_id")] = rec
    return reg


def write_tsv(rows: List[dict], out_path: Path) -> None:
    with out_path.open("w") as fh:
        fh.write("\t".join(ASSET_KEYS) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(k)) for k in ASSET_KEYS) + "\n")


def write_schema(out_path: Path) -> None:
    out_path.write_text(json.dumps(asset_disposition_v3_schema(), indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    here = Path(__file__).resolve().parent
    worktree_root = here.parent.parent
    default_out = worktree_root / "data_registry" / "reactflow_delta"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-disposition", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0x_v2/"
        "asset_disposition_20260807.jsonl"))
    parser.add_argument("--profile-disposition", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/asset_disposition_v2.jsonl"))
    parser.add_argument("--canonical-records", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/canonical_records_v2.jsonl"))
    parser.add_argument("--primary-pairs", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/data_registry/d0r_accession_registry.jsonl"))
    parser.add_argument("--raw-root", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_raw/rmdb"))
    parser.add_argument("--count-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--refresh-counts", action="store_true",
                        help="Stream the 39GB canonical file to rebuild the count cache.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count_cache = args.count_cache or (args.output_dir / "asset_counts_by_accession.json")

    # load identity files
    asset_rows = load_jsonl_records(args.asset_disposition)
    profile_rows = load_jsonl_records(args.profile_disposition)
    registry = load_registry(args.registry)

    if args.refresh_counts:
        print("streaming canonical_records_v2 (39GB) ...", flush=True)
        canonical_counts = stream_count_canonical(args.canonical_records)
        print("counting primary pairs ...", flush=True)
        pair_counts = stream_count_pairs(args.primary_pairs)
        cache = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "canonical_counts": canonical_counts,
            "pair_counts": pair_counts,
        }
        count_cache.write_text(json.dumps(cache) + "\n")
        print(f"wrote count cache: {count_cache}", flush=True)
    else:
        canonical_counts = {}
        pair_counts = {}
        if count_cache.exists():
            cache = json.loads(count_cache.read_text())
            canonical_counts = cache.get("canonical_counts", {})
            pair_counts = cache.get("pair_counts", {})

    # raw-file sha256 (single tree walk; only files present on disk are hashed)
    acc_list = [a.get("source_accession") for a in asset_rows]
    sha256_by_accession = index_rdat_hashes(args.raw_root, acc_list)

    rows = build_asset_disposition_rows(
        asset_rows=asset_rows,
        profile_rows=profile_rows,
        record_counts=canonical_counts,
        pair_counts=pair_counts,
        registry=registry,
        sha256_by_accession=sha256_by_accession,
    )

    tsv_path = args.output_dir / "asset_disposition_v3.tsv"
    schema_path = args.output_dir / "asset_disposition_v3.schema.json"
    write_tsv(rows, tsv_path)
    write_schema(schema_path)

    n_len_mismatch = sum(1 for r in profile_rows
                         if r.get("disposition_reason") == "LENGTH_MISMATCH")
    print(json.dumps({
        "asset_rows": len(rows),
        "unique_asset_ids": len({r["asset_id"] for r in rows}),
        "profile_rows_total": len(profile_rows),
        "profile_length_mismatch_total": n_len_mismatch,
        "count_cache_available": bool(canonical_counts),
        "tsv": str(tsv_path),
        "schema": str(schema_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())