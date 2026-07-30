"""Provenance-first helpers for the ReactFlow-Delta D0 public-data audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
import difflib
import json
import math
from pathlib import Path
import re
from typing import Any

from .manifests import sha256_file
from .schema import SOURCE_REGISTRY_SCHEMA_VERSION, validate_source_registry_record


RAW_MANIFEST_SCHEMA_VERSION = "reactflow-delta-raw-manifest-v1"
RMDB_METADATA_SPECS = (
    (
        "github_releases.json",
        "github-release-index",
        "https://api.github.com/repos/DasLab/rmdb.github.io/releases?per_page=100",
    ),
    (
        "github_main_commit.json",
        "github-main-commit",
        "https://api.github.com/repos/DasLab/rmdb.github.io/commits/main",
    ),
    (
        "github_root_contents.json",
        "github-root-contents",
        "https://api.github.com/repos/DasLab/rmdb.github.io/contents?ref=main",
    ),
    ("rmdb_index.html", "rmdb-index", "https://rmdb.stanford.edu/"),
    ("rdat_specification.html", "rdat-specification", "https://rmdb.stanford.edu/deposit/specs/"),
    ("rmdb_about.html", "rmdb-about-license", "https://rmdb.stanford.edu/about/"),
)

RMDB_CANDIDATE_MANIFEST_SCHEMA_VERSION = "reactflow-delta-rmdb-candidate-manifest-v1"
RMDB_FILENAME_RULES = {
    "m2_named_candidate": re.compile(r"(?:^|_)M2(?:[A-Z0-9_]|$)", re.IGNORECASE),
    "m2r_named_unconfirmed": re.compile(r"M2R", re.IGNORECASE),
    "variant_or_library_named_candidate": re.compile(r"(?:^|_)(?:ETERNA|OK[0-9]|LIB)(?:_|[0-9]|$)", re.IGNORECASE),
    "explicit_rescue_or_compensatory_named_candidate": re.compile(r"rescue|compens", re.IGNORECASE),
}
RDAT_FIXTURE_MANIFEST_SCHEMA_VERSION = "reactflow-delta-rdat-fixture-manifest-v1"
RMDB_FIXTURE_SELECTIONS = {
    "m2_named_candidate": (
        "SPINACH_M2G4_0001.rdat",
        "M2SL5_2A3_0000.rdat",
        "M2SL5_DMS_0000.rdat",
    ),
    "variant_or_library_named_candidate": (
        "ETERNA_TOD_0000.rdat",
        "ETERNA_R42_0004.rdat",
        "ETERNA_R42_0005.rdat",
    ),
}


def build_rmdb_metadata_registry(
    metadata_dir: str | Path,
    *,
    retrieved_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build auditable registry records from one immutable RMDB metadata snapshot.

    This function inventories only official release/index/format/license metadata.
    It does not download RDAT assets, parse constructs, infer pairs, or assign a
    data-feasibility tier.
    """

    directory = Path(metadata_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"RMDB metadata directory is missing: {directory}")
    _validate_retrieved_at(retrieved_at)

    main_commit = _load_json_object(directory / "github_main_commit.json")
    commit_sha = main_commit.get("sha")
    if not isinstance(commit_sha, str) or len(commit_sha) != 40:
        raise ValueError("github_main_commit.json does not contain a 40-character commit SHA")

    releases = _load_json_list(directory / "github_releases.json")
    release_summary = _summarize_releases(releases)
    records: list[dict[str, Any]] = []
    for filename, upstream_id, source_url in RMDB_METADATA_SPECS:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"required RMDB metadata file is missing: {path}")
        record = {
            "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
            "record_id": f"rmdb:{upstream_id}:{commit_sha}",
            "source": "RMDB",
            "source_version": commit_sha,
            "source_url": source_url,
            "publication_doi": "10.1093/bioinformatics/bts554",
            "publication_pmid": None,
            "license": "CC0-1.0",
            "retrieved_at": retrieved_at,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "upstream_id": upstream_id,
            "raw_path": str(path.resolve()),
            "parser_version": "rmdb-metadata-v1",
            "download_status": "downloaded",
            "source_tier": "A",
            "source_type": "rmdb",
            "missing_reasons": {
                "publication_pmid": "not asserted by the frozen RMDB metadata snapshot",
            },
        }
        records.append(validate_source_registry_record(record))

    observed_files = [_file_provenance(path) for path in sorted(directory.iterdir()) if path.is_file()]
    raw_manifest = {
        "schema_version": RAW_MANIFEST_SCHEMA_VERSION,
        "stage": "D0",
        "scope": "RMDB release/index/format/license metadata only; no RDAT asset payloads",
        "retrieved_at": retrieved_at,
        "source": {
            "name": "RMDB",
            "github_repository": "https://github.com/DasLab/rmdb.github.io",
            "main_commit": commit_sha,
            "data_license": "CC0-1.0",
            "publication_doi": "10.1093/bioinformatics/bts554",
        },
        "metadata_directory": str(directory.resolve()),
        "release_summary": release_summary,
        "files": observed_files,
        "scientific_boundary": "Release assets are not constructs or pairs. No pair count, tier decision, normalization, or learned training is authorized by this manifest.",
    }
    return records, raw_manifest


def write_jsonl_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Create a registry JSONL file once; never silently overwrite prior evidence."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in records:
            validated = validate_source_registry_record(record)
            handle.write(json.dumps(validated, sort_keys=True, separators=(",", ":")) + "\n")


def write_json_document(path: str | Path, document: Mapping[str, Any]) -> None:
    """Create a JSON audit artifact once; never silently overwrite prior evidence."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_rmdb_filename_candidate_manifest(release_index_path: str | Path) -> dict[str, Any]:
    """Create a pre-RDAT candidate manifest from immutable release asset metadata.

    Filename matches are deliberately only candidate labels. They must be
    confirmed or rejected by entry metadata and RDAT content before an entry is
    called mutate-and-map, M2-seq, rescue, variant-library, construct, or pair.
    """

    path = Path(release_index_path)
    releases = _load_json_list(path)
    assets = _flatten_release_assets(releases)
    categories = []
    for category, rule in RMDB_FILENAME_RULES.items():
        matches = [asset for asset in assets if rule.search(asset["name"])]
        categories.append(
            {
                "candidate_category": category,
                "classification_basis": "strict filename rule only",
                "rule": rule.pattern,
                "candidate_count": len(matches),
                "candidate_assets": matches,
                "rdat_confirmation_required": True,
            }
        )

    selected_fixtures = []
    asset_by_name = {asset["name"]: asset for asset in assets}
    for category, names in RMDB_FIXTURE_SELECTIONS.items():
        for name in names:
            if name not in asset_by_name:
                raise ValueError(f"frozen fixture selection is absent from release index: {name}")
            selected_fixtures.append(
                {
                    "candidate_category": category,
                    "selection_basis": "smallest fixed set of three strict filename candidates",
                    "rdat_confirmation_required": True,
                    **asset_by_name[name],
                }
            )

    explicit_rescue = next(
        category for category in categories if category["candidate_category"] == "explicit_rescue_or_compensatory_named_candidate"
    )
    return {
        "schema_version": RMDB_CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "stage": "D0",
        "input_release_index": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        },
        "categories": categories,
        "fixture_selection": selected_fixtures,
        "unresolved_absences": [
            {
                "candidate_category": "explicit_rescue_or_compensatory_named_candidate",
                "candidate_count": explicit_rescue["candidate_count"],
                "action": "Do not substitute near-matching names. Search entry metadata and RDAT annotations before declaring rescue availability.",
            }
        ],
        "scientific_boundary": "This is a filename-based discovery manifest only. It contains no confirmed experiment class, construct count, WT-single-mutant pair count, tier decision, or learned training result.",
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _summarize_releases(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for release in releases:
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            raise ValueError("release assets must be a list")
        asset_sizes = []
        for asset in assets:
            if not isinstance(asset, dict) or not isinstance(asset.get("size"), int) or asset["size"] < 0:
                raise ValueError("release asset size must be a non-negative integer")
            asset_sizes.append(asset["size"])
        tag_name = release.get("tag_name")
        if not isinstance(tag_name, str) or not tag_name:
            raise ValueError("release tag_name must be a non-empty string")
        summary.append(
            {
                "tag_name": tag_name,
                "published_at": release.get("published_at"),
                "asset_count": len(assets),
                "asset_bytes": sum(asset_sizes),
                "html_url": release.get("html_url"),
            }
        )
    return summary


def _flatten_release_assets(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for release in releases:
        tag_name = release.get("tag_name")
        if not isinstance(tag_name, str) or not tag_name:
            raise ValueError("release tag_name must be a non-empty string")
        release_assets = release.get("assets", [])
        if not isinstance(release_assets, list):
            raise ValueError("release assets must be a list")
        for asset in release_assets:
            if not isinstance(asset, dict):
                raise ValueError("release asset must be an object")
            name = asset.get("name")
            size = asset.get("size")
            digest = asset.get("digest")
            url = asset.get("browser_download_url")
            if not isinstance(name, str) or not name.endswith(".rdat"):
                raise ValueError("release asset name must end with .rdat")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("release asset size must be a non-negative integer")
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("release asset digest must be a SHA-256 value")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError("release asset browser_download_url must be HTTPS")
            assets.append({"release_tag": tag_name, "name": name, "bytes": size, "upstream_sha256": digest.removeprefix("sha256:"), "browser_download_url": url})
    return assets


def build_rdat_fixture_manifest(candidate_manifest_path: str | Path, fixture_dir: str | Path) -> dict[str, Any]:
    """Verify a fixed RDAT fixture set against release-index SHA-256 values.

    Verification establishes only byte-level fixture provenance. It deliberately
    does not parse RDAT content or infer experiment class, construct identity,
    normalization, or a WT-single-mutant pair.
    """

    candidate_path = Path(candidate_manifest_path)
    candidate_manifest = _load_json_object(candidate_path)
    fixtures = candidate_manifest.get("fixture_selection")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("candidate manifest must contain a non-empty fixture_selection")

    directory = Path(fixture_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"fixture directory is missing: {directory}")
    partials = sorted(directory.glob("*.part"))
    if partials:
        raise ValueError(f"fixture directory contains incomplete downloads: {partials}")

    verified = []
    expected_names = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("fixture selection item must be an object")
        name = fixture.get("name")
        expected_sha256 = fixture.get("upstream_sha256")
        expected_bytes = fixture.get("bytes")
        category = fixture.get("candidate_category")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".rdat"):
            raise ValueError("fixture name must be a plain .rdat filename")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"fixture {name} lacks an upstream SHA-256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"fixture {name} has invalid byte count")
        if not isinstance(category, str) or not category:
            raise ValueError(f"fixture {name} lacks a candidate category")
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"fixture is missing: {path}")
        observed_sha256 = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_sha256 != expected_sha256:
            raise ValueError(f"fixture SHA-256 mismatch: {name}")
        if observed_bytes != expected_bytes:
            raise ValueError(f"fixture byte count mismatch: {name}")
        expected_names.add(name)
        verified.append({"candidate_category": category, "name": name, "path": str(path.resolve()), "bytes": observed_bytes, "sha256": observed_sha256, "status": "verified_against_release_index", "rdat_confirmation_pending": True})

    observed_names = {path.name for path in directory.glob("*.rdat")}
    if observed_names != expected_names:
        raise ValueError("fixture directory .rdat filenames do not exactly match the frozen selection")
    return {
        "schema_version": RDAT_FIXTURE_MANIFEST_SCHEMA_VERSION,
        "stage": "D0",
        "input_candidate_manifest": {"path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path)},
        "fixture_directory": str(directory.resolve()),
        "fixtures": verified,
        "fixture_counts_by_candidate_category": dict(sorted(Counter(item["candidate_category"] for item in verified).items())),
        "scientific_boundary": "All fixtures are byte-verified but await RDAT parsing. No entry has been confirmed as a specific experiment class, construct, or pair.",
    }


def _file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_retrieved_at(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("retrieved_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")

# ============================================================================
# D1 Cleaning pipeline (v3 §6.6, Phase D1 T-D1.2–T-D1.10)
#
# All D1 cleaning functions are added here per v3.0 §15 Phase D1 file map
# ("Modify: src/reactflow/delta/data.py"). Each T-D1 step adds focused,
# tested functions that implement one stage of the v3 §6.6 14-step cleaning
# order.
# ============================================================================

# --- T-D1.2: Condition exact matching (v3 §6.5, §6.6 step 5) ---

# The construct fields that define an experimental condition. WT and mutant
# must be identical on ALL of these (both-null counts as identical, since
# within a single RDAT file the absence of a field means the same experiment).
CONDITION_MATCH_FIELDS = (
    "probe",
    "probe_protocol",
    "temperature",
    "ligand",
    "ligand_concentration",
    "buffer",
    "in_vivo_in_vitro",
)


def match_conditions(
    wt_construct: Mapping[str, Any],
    mut_construct: Mapping[str, Any],
) -> dict[str, Any]:
    """Check exact condition match between WT and mutant constructs (T-D1.2).

    Per v3 §6.5, WT and mutant must have identical experimental conditions
    (probe, temperature, ligand, buffer, etc.) except for the edit itself.
    Both-null counts as a match: within a single RDAT file the absence of a
    condition field means the same experimental context for both profiles.

    Returns a dict with:
      - ``condition_match_fields``: list of field names that matched (for the
        pair schema field of the same name).
      - ``condition_match_status``: ``"exact_match"`` or ``"mismatch"`` (for
        the pair schema field of the same name).
      - ``mismatched_fields``: list of field names that differed (extra
        diagnostic, used by T-D1.10 to assign exclusion reasons).
    """
    matched: list[str] = []
    mismatched: list[str] = []
    for field in CONDITION_MATCH_FIELDS:
        wt_val = wt_construct.get(field)
        mut_val = mut_construct.get(field)
        if wt_val == mut_val:
            matched.append(field)
        else:
            mismatched.append(field)
    status = "exact_match" if not mismatched else "mismatch"
    return {
        "condition_match_fields": matched,
        "condition_match_status": status,
        "mismatched_fields": mismatched,
    }


# --- T-D1.3: Substitution verification (v3 §6.6 step 4, v3.1 §3.1/§3.3) ---

# T (DNA) is normalized to U per v3 §6.6 step 3 before substitution
# verification (v3.1 §3.1: "DNA→RNA T→U 规范后" verify). The verify_substitution
# path normalizes defensively so the comparison is always on RNA coordinates.


def _normalize_rna(seq: str | None) -> str | None:
    """Normalize a sequence to RNA (T → U) for substitution verification.

    Per v3 §6.6 step 3, sequences are T/U normalized before edit verification.
    This helper applies the same normalization defensively so that callers do
    not need to pre-normalize. Returns ``None`` if the input is ``None``.
    """

    if seq is None:
        return None
    return seq.replace("T", "U").replace("t", "u")


def _cigar_from_sequences(wt: str, mut: str) -> str:
    """Build a SAM-like CIGAR string from two sequences.

    For equal-length sequences (the D1 v1 substitution-only scope, v3 §2.3 /
    v3.1 §2.3) this is the trivial ``<N>M``. For unequal-length sequences
    (indel case, deferred in D1 v1 and always excluded) a best-effort global
    alignment CIGAR is produced via :class:`difflib.SequenceMatcher` so the
    ``alignment_cigar`` pair-schema field stays honest and non-empty.
    """

    if len(wt) == len(mut):
        return f"{len(wt)}M"
    sm = difflib.SequenceMatcher(None, wt, mut, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        n = max(i2 - i1, j2 - j1)
        if n == 0:
            continue
        if tag == "equal":
            parts.append(f"{n}M")
        elif tag == "replace":
            parts.append(f"{n}M")
        elif tag == "delete":
            parts.append(f"{n}D")
        elif tag == "insert":
            parts.append(f"{n}I")
    return "".join(parts) or f"{len(wt)}M"


def verify_substitution(
    wt_sequence: str | None,
    mut_sequence: str | None,
) -> dict[str, Any]:
    """Verify a single-substitution edit between WT and mutant sequences (T-D1.3).

    Implements v3.1 §3.1 sequence-based substitution verification: after
    DNA→RNA (T→U) normalization, compares two sequences and detects whether
    they differ by exactly one substitution. Equal length is required for a
    substitution (v3.1 §2.3: WT/mutant 等长); unequal length is an indel and
    is excluded with ``indel_not_substitution`` (indels are deferred in D1 v1).

    Returns a dict with the pair-schema edit fields:
      - ``edit_type``: ``"substitution"`` (equal length) / ``"insertion"`` /
        ``"deletion"`` (unequal length). Always one of EDIT_TYPES.
      - ``edit_count``: number of differing positions (0 for identical / indel
        excluded pairs reports the substitution count, 0 when unverifiable).
      - ``edit_positions``: 0-indexed positions that differ (aligns with the
        0-indexed mask arrays in the pair schema).
      - ``wt_alleles`` / ``mut_alleles``: bases at the differing positions.
      - ``alignment_cigar``: SAM-like CIGAR (non-empty, schema-required).
      - ``is_substitution_single``: True iff exactly one substitution
        (``edit_count == 1`` and equal length).
      - ``exclusion_reason``: a single EXCLUSION_REASONS value when the pair is
        not a single substitution, else ``None``. Values:
          * ``substitution_not_verifiable`` — a sequence is missing/empty;
          * ``indel_not_substitution`` — unequal length (insertion/deletion);
          * ``edit_count_not_one`` — 0 edits (no-edit control) or >1 edits.
    """

    wt_n = _normalize_rna(wt_sequence)
    mut_n = _normalize_rna(mut_sequence)

    # Unverifiable: missing or empty sequence (cannot establish the edit).
    if not wt_n or not mut_n:
        return {
            "edit_type": "substitution",
            "edit_count": 0,
            "edit_positions": [],
            "wt_alleles": [],
            "mut_alleles": [],
            "alignment_cigar": _cigar_from_sequences(wt_n or "", mut_n or ""),
            "is_substitution_single": False,
            "exclusion_reason": "substitution_not_verifiable",
        }

    # Indel: unequal length (deferred in D1 v1, always excluded).
    if len(wt_n) != len(mut_n):
        edit_type = "insertion" if len(mut_n) > len(wt_n) else "deletion"
        return {
            "edit_type": edit_type,
            "edit_count": 0,
            "edit_positions": [],
            "wt_alleles": [],
            "mut_alleles": [],
            "alignment_cigar": _cigar_from_sequences(wt_n, mut_n),
            "is_substitution_single": False,
            "exclusion_reason": "indel_not_substitution",
        }

    # Equal length: collect substitution positions (0-indexed).
    cigar = f"{len(wt_n)}M"
    edit_positions: list[int] = []
    wt_alleles: list[str] = []
    mut_alleles: list[str] = []
    for i, (w, m) in enumerate(zip(wt_n, mut_n)):
        if w != m:
            edit_positions.append(i)
            wt_alleles.append(w)
            mut_alleles.append(m)
    edit_count = len(edit_positions)
    is_single = edit_count == 1
    exclusion = None if is_single else "edit_count_not_one"
    return {
        "edit_type": "substitution",
        "edit_count": edit_count,
        "edit_positions": edit_positions,
        "wt_alleles": wt_alleles,
        "mut_alleles": mut_alleles,
        "alignment_cigar": cigar,
        "is_substitution_single": is_single,
        "exclusion_reason": exclusion,
    }


def verify_annotation_ref(
    encoded_position_1indexed: int,
    encoded_ref: str,
    header_sequence: str | None,
    offset: int = 0,
) -> dict[str, Any]:
    """Verify an annotation-encoded ref base against the header SEQUENCE (T-D1.3).

    Implements v3.1 §3.3 HIV3PR genome-numbering offset fix. Some RDAT files
    (e.g. ``HIV3PR_DMS_*``, 8 files) encode mutation annotations in genome
    numbering rather than construct-local 1-indexed positions: the RDAT
    ``OFFSET`` header gives the genome coordinate of construct position 1, so
    an annotation like ``G8932X`` with ``OFFSET 8931`` refers to construct
    position 1 (index 0). The D0-R audit recorded these as
    ``annotation_ref_mismatch`` (historical evidence, preserved); D1 re-verifies
    the ref using the correct genome-numbering offset so the candidate can be
    re-classified (it still needs per-profile sequence or replicate
    corroboration to upgrade per §3.2 — the offset fix alone does NOT upgrade).

    Verification order (first match wins):
      1. Construct-local 1-indexed: ``index = pos - 1`` (annotation already
         construct-local, the common case).
      2. Genome-numbering offset: ``index = pos - 1 - offset`` (HIV3PR case,
         only attempted when ``offset > 0``).

    Both the encoded ref and the header sequence are T→U normalized before
    comparison (v3.1 §3.1: verify after DNA→RNA normalization).

    Returns a dict with:
      - ``ref_verified``: bool.
      - ``ref_match_index``: ``"construct_local_1indexed"`` /
        ``"genome_numbering_offset"`` / ``None``.
      - ``construct_local_position_1indexed``: the resolved construct-local
        1-indexed position when verified, else ``None``.
      - ``actual_base``: the header base at the resolved index (RNA), or
        ``None`` if out of range.
      - ``exclusion_reason``: ``"annotation_ref_mismatch"`` when not verified,
        else ``None``.
    """

    seq = _normalize_rna(header_sequence)
    ref = _normalize_rna(encoded_ref)

    result: dict[str, Any] = {
        "ref_verified": False,
        "ref_match_index": None,
        "construct_local_position_1indexed": None,
        "actual_base": None,
        "exclusion_reason": "annotation_ref_mismatch",
    }
    if not seq or not ref:
        return result

    def _try_index(idx0: int, match_label: str) -> bool:
        if 0 <= idx0 < len(seq) and seq[idx0] == ref:
            result["ref_verified"] = True
            result["ref_match_index"] = match_label
            result["construct_local_position_1indexed"] = idx0 + 1
            result["actual_base"] = seq[idx0]
            result["exclusion_reason"] = None
            return True
        return False

    # 1. Construct-local 1-indexed (common case).
    if _try_index(encoded_position_1indexed - 1, "construct_local_1indexed"):
        return result

    # 2. Genome-numbering offset (HIV3PR: annotation uses genome coords,
    #    subtract OFFSET to recover the construct-local index).
    if offset and offset > 0:
        if _try_index(encoded_position_1indexed - 1 - offset, "genome_numbering_offset"):
            return result

    # No match: record the actual base at the most relevant in-range index for
    # diagnosis. Prefer the genome-offset index (HIV3PR-relevant, where the
    # annotation actually points), then the construct-local index.
    diag_indices = (
        [encoded_position_1indexed - 1 - offset, encoded_position_1indexed - 1]
        if (offset and offset > 0)
        else [encoded_position_1indexed - 1]
    )
    for diag_idx0 in diag_indices:
        if 0 <= diag_idx0 < len(seq):
            result["actual_base"] = seq[diag_idx0]
            break
    return result


# --- T-D1.4: Alignment masks + comparable-positions check (v3 §6.6 step 6) ---

# Minimum fraction of unedited positions that must be comparable (have valid
# reactivity in both WT and mutant) for a pair to enter the D1 v1 scope
# (v3 §6.5: "至少 60% 未编辑位置可比"; v3.1 §4 exclusion reason
# ``comparable_positions_below_60pct``).
COMPARABLE_MIN_FRACTION = 0.60


def build_position_masks(edit_positions: Iterable[int], length: int) -> dict[str, Any]:
    """Build unchanged/changed position masks from edit positions (T-D1.4).

    Implements v3 §6.6 step 6 ("构建 alignment") for the D1 v1 substitution-only
    scope: given the 0-indexed edit positions (from :func:`verify_substitution`)
    and the alignment length, produce the two complementary 0/1 masks required
    by the pair schema (v3 §6.4):

      - ``unchanged_position_mask``: ``1`` where WT and mutant agree (unedited),
        ``0`` where edited.
      - ``changed_position_mask``: ``1`` where edited, ``0`` where unchanged
        (the exact complement).

    Masks are 0-indexed and align with the 0-indexed ``edit_positions`` and
    with the ``array`` of ``integer enum [0, 1]`` pair-schema fields frozen in
    T-D1.1 (schema.py). For the D1 v1 equal-length substitution scope the
    alignment length equals ``len(wt_sequence) == len(mut_sequence)`` and the
    CIGAR is the trivial ``<N>M`` already produced by
    :func:`verify_substitution`; this function does not re-derive the CIGAR.

    The distance-band masks (``local_mask`` / ``mid_mask`` / ``remote_mask``)
    are *not* produced here: they are distance bins relative to the edit site
    and belong to v3 §6.6 step 13 (physical features), with boundaries frozen
    per v3 §9.3 / T-D2.7 (no test data). They are deferred to a later D1 task.

    Returns a dict with:
      - ``unchanged_position_mask``: list[int] of 0/1, length ``length``.
      - ``changed_position_mask``: list[int] of 0/1, length ``length``.
      - ``unchanged_position_count``: int (number of 1s in unchanged mask).
      - ``changed_position_count``: int (number of 1s in changed mask).
    """

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    positions = list(edit_positions)
    for pos in positions:
        # bool is a subclass of int; reject it so True/False are not silently
        # treated as positions 1/0.
        if isinstance(pos, bool) or not isinstance(pos, int):
            raise ValueError(f"edit position must be a plain int, got {pos!r}")
        if pos < 0 or pos >= length:
            raise ValueError(f"edit position {pos} out of range [0, {length})")
    edited = set(positions)
    unchanged = [0 if i in edited else 1 for i in range(length)]
    changed = [1 if i in edited else 0 for i in range(length)]
    return {
        "unchanged_position_mask": unchanged,
        "changed_position_mask": changed,
        "unchanged_position_count": sum(unchanged),
        "changed_position_count": sum(changed),
    }


def check_comparable_positions(
    unchanged_position_mask: Iterable[int],
    wt_valid_mask: Iterable[int] | None = None,
    mut_valid_mask: Iterable[int] | None = None,
    *,
    min_fraction: float = COMPARABLE_MIN_FRACTION,
) -> dict[str, Any]:
    """Check the >=60% comparable-positions D1 v1 scope rule (T-D1.4).

    Implements v3 §6.5 ("至少 60% 未编辑位置可比") and produces the
    ``comparable_positions_below_60pct`` exclusion reason (v3.1 §4). A position
    is *comparable* when it is unedited (``unchanged_position_mask[i] == 1``)
    AND has valid reactivity in both WT and mutant (``wt_valid_mask[i] == 1``
    and ``mut_valid_mask[i] == 1``). The comparable fraction is
    ``comparable_count / unchanged_count``; the pair enters the D1 v1 scope
    only when this fraction is ``>= min_fraction`` (default 0.60).

    The valid masks come from v3 §6.6 step 9 (missingness/SNR/coverage), which
    runs after this step in the full pipeline. To keep this function callable
    both before and after step 9, ``wt_valid_mask`` / ``mut_valid_mask``
    default to ``None`` meaning "no missingness information available; treat
    every position as valid". Callers that have real missingness data MUST
    pass the step-9 valid masks to get an honest comparable fraction.

    Returns a dict with:
      - ``comparable_position_mask``: list[int] of 0/1 (1 where unedited AND
        valid in both WT and mutant), same length as the unchanged mask.
      - ``comparable_position_count``: int (number of comparable positions).
      - ``unchanged_position_count``: int (denominator: unedited positions).
      - ``comparable_fraction``: float in [0.0, 1.0]. When
        ``unchanged_position_count == 0`` the fraction is ``0.0`` (no unedited
        positions to compare -> not comparable).
      - ``is_comparable``: bool, ``comparable_fraction >= min_fraction``.
      - ``exclusion_reason``: ``"comparable_positions_below_60pct"`` when not
        comparable, else ``None``.
    """

    unchanged = [int(x) for x in unchanged_position_mask]
    n = len(unchanged)
    for v in unchanged:
        if v not in (0, 1):
            raise ValueError(f"unchanged_position_mask entries must be 0/1, got {v!r}")
    if wt_valid_mask is None:
        wt_valid = [1] * n
    else:
        wt_valid = [int(x) for x in wt_valid_mask]
        if len(wt_valid) != n:
            raise ValueError(
                f"wt_valid_mask length {len(wt_valid)} != unchanged mask length {n}"
            )
    if mut_valid_mask is None:
        mut_valid = [1] * n
    else:
        mut_valid = [int(x) for x in mut_valid_mask]
        if len(mut_valid) != n:
            raise ValueError(
                f"mut_valid_mask length {len(mut_valid)} != unchanged mask length {n}"
            )
    for v in wt_valid + mut_valid:
        if v not in (0, 1):
            raise ValueError(f"valid mask entries must be 0/1, got {v!r}")

    comparable = [
        1 if (u == 1 and w == 1 and m == 1) else 0
        for u, w, m in zip(unchanged, wt_valid, mut_valid)
    ]
    unchanged_count = sum(unchanged)
    comparable_count = sum(comparable)
    if unchanged_count > 0:
        fraction = comparable_count / unchanged_count
    else:
        fraction = 0.0
    is_comparable = fraction >= min_fraction
    exclusion = None if is_comparable else "comparable_positions_below_60pct"
    return {
        "comparable_position_mask": comparable,
        "comparable_position_count": comparable_count,
        "unchanged_position_count": unchanged_count,
        "comparable_fraction": fraction,
        "is_comparable": is_comparable,
        "exclusion_reason": exclusion,
    }


# --- T-D1.5: Probe eligibility masks (v3 §6.6 step 7, v3.1 §4 probe_mismatch) ---

# Mapping from a *normalized* chemical-probe name to the frozenset of RNA bases
# it can modify. Used by build_probe_eligibility_mask (construct-level, schema
# field ``probe_eligibility_mask``) and build_probe_eligibility_unchanged_mask
# (pair-level, schema field ``probe_eligibility_unchanged_mask``).
#
# Per v3 §6.6 step 7 ("构建 probe eligibility"), a position is probe-eligible
# iff its RNA base is in the probe's eligible-base set; the pair-level
# eligibility-unchanged mask is then ``1`` where WT and mutant share the same
# eligibility at that position. The §12.1 main endpoint restricts regression
# to unedited + aligned + eligibility-unchanged + valid positions and excludes
# edited / eligibility-changed positions, so these masks gate every downstream
# Δreactivity computation.
#
# Chemistry (verified against RDAT ``ANNOTATION modifier:X`` conventions and
# the published DMS-MaP / SHAPE-MaP literature):
#   - DMS  (dimethyl sulfate)      → N1-A, N3-C        → {A, C}
#   - CMCT (1-cyclohexyl-3-(2-morpholinoethyl)carbodiimide) → N1-G, N3-U → {G, U}
#   - SHAPE-class acylates the ribose 2'-OH, sequence-independent → {A,C,G,U}.
#     Members normalized to "SHAPE": 1M7, NMIA, SHAPE, 2A3 (2A3 = 2-aminopyridine-
#     3-carboxylic acid imidazolide; Marinus et al. 2021 NAR).
#   - nomod / none → control, no probe → empty set (mask all 0).
# Any probe not in this table is treated as "UNKNOWN": the mask is set to
# ``None`` and ``probe_known`` is ``False``. The pipeline does NOT raise —
# T-D1.10 decides how the candidate pair is routed (v3.1 §4 has no
# ``probe_unknown`` exclusion reason; such pairs are typically routed to
# ``parent_lineage_unverified`` or kept as ``candidate_only``).
PROBE_ELIGIBLE_BASES: dict[str, frozenset[str]] = {
    "DMS": frozenset({"A", "C"}),
    "CMCT": frozenset({"G", "U"}),
    "SHAPE": frozenset({"A", "C", "G", "U"}),
    "NOMOD": frozenset(),
    "NONE": frozenset(),
}

# Aliases (case-insensitive) collapsed into a single canonical key. All
# SHAPE-class reagents acylate the 2'-OH universally and share the same
# eligible-base set, so they map to "SHAPE".
_PROBE_ALIASES: dict[str, str] = {
    "1M7": "SHAPE",
    "NMIA": "SHAPE",
    "SHAPE": "SHAPE",
    "2A3": "SHAPE",
    "DMS": "DMS",
    "CMCT": "CMCT",
    "NOMOD": "NOMOD",
    "NONE": "NONE",
}


def normalize_probe(probe: str | None) -> str:
    """Normalize a chemical-probe name (T-D1.5).

    Uppercases, strips whitespace, and collapses the SHAPE-class reagents
    (1M7, NMIA, SHAPE, 2A3) to the single canonical key ``"SHAPE"``. Returns
    ``"UNKNOWN"`` for ``None``/empty input or any probe not present in
    :data:`_PROBE_ALIASES`. The result is always a key of
    :data:`PROBE_ELIGIBLE_BASES` or the literal ``"UNKNOWN"``.
    """
    if probe is None:
        return "UNKNOWN"
    key = probe.strip().upper()
    if not key:
        return "UNKNOWN"
    return _PROBE_ALIASES.get(key, "UNKNOWN")


def build_probe_eligibility_mask(
    sequence: str | None, probe: str | None
) -> dict[str, Any]:
    """Build the construct-level probe-eligibility mask (T-D1.5, v3 §6.6 step 7).

    For each 0-indexed position of ``sequence`` the mask is ``1`` if the RNA
    base at that position belongs to the probe's eligible-base set, else
    ``0``. The sequence is T→U normalized first (v3 §6.6 step 3, via
    :func:`_normalize_rna`) so DNA-typed header sequences are handled
    defensively.

    Returns a dict with:
      - ``mask``: list[int] of 0/1, or ``None`` if the probe is unknown or the
        sequence is ``None``.
      - ``normalized_probe``: the canonical probe key (or ``"UNKNOWN"``).
      - ``probe_known``: ``True`` iff the probe is recognized.
      - ``eligible_base_count``: number of eligible positions (``None`` when
        the mask is ``None``).

    For an unknown probe the function does **not** raise; downstream callers
    (T-D1.10) decide how to route the candidate.
    """
    normalized = normalize_probe(probe)
    probe_known = normalized != "UNKNOWN"
    if not probe_known:
        return {
            "mask": None,
            "normalized_probe": normalized,
            "probe_known": False,
            "eligible_base_count": None,
        }
    eligible_bases = PROBE_ELIGIBLE_BASES[normalized]
    if sequence is None:
        return {
            "mask": None,
            "normalized_probe": normalized,
            "probe_known": True,
            "eligible_base_count": None,
        }
    seq = _normalize_rna(sequence)
    mask = [1 if base in eligible_bases else 0 for base in seq]
    return {
        "mask": mask,
        "normalized_probe": normalized,
        "probe_known": True,
        "eligible_base_count": sum(mask),
    }


def build_probe_eligibility_unchanged_mask(
    wt_sequence: str | None, mut_sequence: str | None, probe: str | None
) -> dict[str, Any]:
    """Build the pair-level probe-eligibility-unchanged mask (T-D1.5).

    Implements v3 §6.6 step 7 at the pair level: for each 0-indexed position
    the mask is ``1`` where WT and mutant have the *same* probe eligibility
    (both eligible or both ineligible), and ``0`` where eligibility changed
    (i.e. an edit toggled eligibility). The mask aligns with the pair-schema
    field ``probe_eligibility_unchanged_mask`` (T-D1.1, schema.py).

    Per v3 §6.5 the WT and mutant share the same probe (enforced by T-D1.2
    ``condition_match_status``); the single ``probe`` argument is therefore
    used for both. Eligibility can only change at edited positions, because an
    unedited position carries the same base (hence the same eligibility) in WT
    and mutant — so the unchanged mask is ``1`` at every unedited position and
    may be ``0`` only at edit positions.

    Returns a dict with:
      - ``mask``: list[int] of 0/1, or ``None`` if the probe is unknown or
        either sequence is ``None``.
      - ``normalized_probe``: canonical probe key (or ``"UNKNOWN"``).
      - ``probe_known``: ``True`` iff the probe is recognized.
      - ``eligibility_changed_count``: number of positions with an eligibility
        change (``None`` when the mask is ``None``).
      - ``eligibility_changed_positions``: sorted 0-indexed positions where
        eligibility changed (empty list, or ``None`` when the mask is
        ``None``).

    This function does **not** emit a pair-level exclusion reason. Eligibility
    change is a *per-position* mask, not a pair-level exclusion; the pair-level
    ``probe_mismatch`` exclusion (v3.1 §4) is assigned by T-D1.10 based on the
    T-D1.2 ``condition_match_status``. A ``ValueError`` is raised on unequal
    sequence lengths as a defensive guard (T-D1.3 ``verify_substitution``
    guarantees equal length for every D1 v1 pair).
    """
    normalized = normalize_probe(probe)
    probe_known = normalized != "UNKNOWN"
    if not probe_known:
        return {
            "mask": None,
            "normalized_probe": normalized,
            "probe_known": False,
            "eligibility_changed_count": None,
            "eligibility_changed_positions": None,
        }
    if wt_sequence is None or mut_sequence is None:
        return {
            "mask": None,
            "normalized_probe": normalized,
            "probe_known": True,
            "eligibility_changed_count": None,
            "eligibility_changed_positions": None,
        }
    wt_n = _normalize_rna(wt_sequence)
    mut_n = _normalize_rna(mut_sequence)
    if len(wt_n) != len(mut_n):
        raise ValueError(
            f"wt_sequence length {len(wt_n)} != mut_sequence length "
            f"{len(mut_n)} (probe-eligibility-unchanged requires equal-length "
            "sequences; T-D1.3 verify_substitution must precede this call for "
            "D1 v1 pairs)"
        )
    eligible_bases = PROBE_ELIGIBLE_BASES[normalized]
    mask: list[int] = []
    changed_positions: list[int] = []
    for i, (wb, mb) in enumerate(zip(wt_n, mut_n)):
        if (wb in eligible_bases) == (mb in eligible_bases):
            mask.append(1)
        else:
            mask.append(0)
            changed_positions.append(i)
    return {
        "mask": mask,
        "normalized_probe": normalized,
        "probe_known": True,
        "eligibility_changed_count": len(changed_positions),
        "eligibility_changed_positions": changed_positions,
    }


# --- T-D1.6: Replicate / no-edit / control identification (v3 §6.6 step 8) ---

# Construct-level identity fields for replicate grouping. Two constructs that
# share all these fields are measurements of the same biological sample under
# the same condition; if they are distinct measurement instances they are
# replicates (v3.1 §3.1 "同一 parent 的 replicate"). ``parent_id`` anchors the
# RNA family/construct origin, ``sequence_normalized`` anchors the molecule,
# and the seven condition fields are the T-D1.2 CONDITION_MATCH_FIELDS (v3
# §6.5 condition exact-match).
REPLICATE_CONSTRUCT_IDENTITY_FIELDS: tuple[str, ...] = (
    "parent_id",
    "sequence_normalized",
) + CONDITION_MATCH_FIELDS

# Pair-level identity fields = construct identity + edit identity. Two pairs
# sharing all these fields are replicate measurements of the same WT→mut
# substitution under the same condition, providing the independent
# corroboration referenced by v3.1 §3.1/§3.2. ``study_id`` is deliberately
# excluded: replicates of the same parent may span studies, and the
# same-study independent-profile/condition path is a separate corroboration
# route judged by T-D1.10.
REPLICATE_PAIR_IDENTITY_FIELDS: tuple[str, ...] = (
    "parent_id",
    "edit_positions",
    "wt_alleles",
    "mut_alleles",
) + CONDITION_MATCH_FIELDS

# No-probe control probe names (v3 §7.3 "noise threshold 由 controls 冻结").
# These are the probes whose eligible-base set is empty in
# PROBE_ELIGIBLE_BASES (NOMOD / NONE); they measure background reactivity
# without chemical probing and serve as no-probe controls.
NO_PROBE_CONTROL_NAMES = frozenset({"NOMOD", "NONE"})


def classify_no_edit_pair(
    wt_sequence: str | None,
    mut_sequence: str | None,
    *,
    edit_count: int | None = None,
) -> dict[str, Any]:
    """Classify a WT/mutant pair as a no-edit control (T-D1.6, v3 §6.6 step 8).

    A no-edit control is a pair where WT and mutant are identical
    (``edit_count == 0``) — a deliberate WT-vs-WT measurement used to
    characterize measurement noise (v3 §7.3 "noise threshold 由 controls
    冻结"; §6.3 no-edit identity). The D1 v1 candidate scope requires
    ``edit_count == 1`` (v3 §6.5), so no-edit controls are never candidate
    pairs; this classifier identifies them among the full profile set so
    T-D1.8 can use them for noise-threshold freezing.

    Classification precedence:
      1. If ``edit_count`` is explicitly ``0`` → no-edit (trusted over
         sequence comparison, which may be unavailable for annotation-only
         candidates).
      2. Else if both sequences are available and equal after T→U
         normalization (v3 §6.6 step 3) → no-edit.
      3. Else → not no-edit.

    Returns ``{is_no_edit, edit_count, determined_from, reason}``.
    """
    if edit_count == 0:
        return {
            "is_no_edit": True,
            "edit_count": 0,
            "determined_from": "edit_count",
            "reason": "edit_count_zero",
        }
    if wt_sequence is not None and mut_sequence is not None:
        if _normalize_rna(wt_sequence) == _normalize_rna(mut_sequence):
            return {
                "is_no_edit": True,
                "edit_count": 0,
                "determined_from": "sequence",
                "reason": "sequences_identical",
            }
    return {
        "is_no_edit": False,
        "edit_count": edit_count,
        "determined_from": None,
        "reason": None,
    }


def classify_control_pair(
    wt_sequence: str | None,
    mut_sequence: str | None,
    probe: str | None,
    *,
    edit_count: int | None = None,
) -> dict[str, Any]:
    """Classify a pair as a control (T-D1.6, v3 §6.6 step 8).

    A pair is a control if it is a no-edit pair (WT==mutant) and/or a no-probe
    pair (probe ∈ {nomod, none}). No-edit pairs characterize measurement noise
    (v3 §7.3); no-probe pairs characterize background reactivity. Controls
    feed T-D1.8 noise-threshold freezing and the ``replicate_control_subset``
    (v3 §9.2, frozen in T-D2.5). The WT/mutant probe is shared per v3 §6.5
    (enforced by T-D1.2), so a single ``probe`` argument is used.

    Returns ``{is_control, control_type, is_no_edit, is_no_probe, reasons}``
    where ``control_type`` is one of ``"no_edit"``, ``"no_probe"``,
    ``"no_edit_and_no_probe"``, or ``None``; ``reasons`` is the (possibly
    empty) list of machine-readable control reasons.
    """
    no_edit = classify_no_edit_pair(wt_sequence, mut_sequence, edit_count=edit_count)
    normalized = normalize_probe(probe)
    is_no_probe = normalized in NO_PROBE_CONTROL_NAMES
    is_no_edit = no_edit["is_no_edit"]
    reasons: list[str] = []
    if is_no_edit:
        reasons.append(no_edit["reason"])
    if is_no_probe:
        reasons.append("no_probe_control")
    if is_no_edit and is_no_probe:
        control_type = "no_edit_and_no_probe"
    elif is_no_edit:
        control_type = "no_edit"
    elif is_no_probe:
        control_type = "no_probe"
    else:
        control_type = None
    return {
        "is_control": control_type is not None,
        "control_type": control_type,
        "is_no_edit": is_no_edit,
        "is_no_probe": is_no_probe,
        "reasons": reasons,
    }


def _identity_key(
    record: Mapping[str, Any], key_fields: tuple[str, ...]
) -> tuple:
    """Build a hashable identity-key tuple from ``key_fields`` of ``record``.

    List values (e.g. ``edit_positions``, ``wt_alleles``) are converted to
    tuples so the key is hashable; missing fields default to ``None``.
    """
    parts: list[Any] = []
    for f in key_fields:
        v = record.get(f)
        if isinstance(v, list):
            v = tuple(v)
        parts.append(v)
    return tuple(parts)


def identify_replicate_groups(
    records: Iterable[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...] = REPLICATE_CONSTRUCT_IDENTITY_FIELDS,
    id_field: str = "construct_id",
) -> dict[str, Any]:
    """Identify replicate groups among records (T-D1.6, v3 §6.6 step 8).

    Groups records by their identity key (``key_fields``). A group with
    ≥2 distinct record ids is a *replicate group*: its members are
    independent measurements of the same biological sample under the same
    condition (v3.1 §3.1 "同一 parent 的 replicate"). These replicate groups
    provide the independent corroboration required for true_pair upgrade
    (v3.1 §3.1/§3.2) and the replicate-based measurement noise for T-D1.8 /
    the frozen differential caller in T-D1.9.

    ``key_fields`` defaults to :data:`REPLICATE_CONSTRUCT_IDENTITY_FIELDS`
    (construct-level); pass :data:`REPLICATE_PAIR_IDENTITY_FIELDS` for
    pair-level replicate identification. Records missing ``id_field`` are
    skipped (recorded in ``skipped``) rather than silently merged.

    Returns:
      - ``replicate_group_count``: number of groups with ≥2 distinct members.
      - ``replicate_record_ids``: set of record ids belonging to a replicate
        group.
      - ``groups``: ``{identity_key_tuple: [record_ids]}`` for every group.
      - ``record_to_group``: ``{record_id: identity_key_tuple}``.
      - ``record_to_replicate_count``: ``{record_id: int}`` — group size for
        replicate-group members, else 0.
      - ``skipped``: list of records missing ``id_field``.
    """
    groups: dict[tuple, list[Any]] = {}
    record_to_group: dict[Any, tuple] = {}
    skipped: list[Mapping[str, Any]] = []
    for rec in records:
        rid = rec.get(id_field)
        if rid is None:
            skipped.append(rec)
            continue
        key = _identity_key(rec, key_fields)
        groups.setdefault(key, []).append(rid)
        record_to_group[rid] = key
    replicate_record_ids: set[Any] = set()
    record_to_replicate_count: dict[Any, int] = {}
    replicate_group_count = 0
    for members in groups.values():
        distinct = list(dict.fromkeys(members))  # dedupe, preserve order
        if len(distinct) >= 2:
            replicate_group_count += 1
            for rid in distinct:
                replicate_record_ids.add(rid)
                record_to_replicate_count[rid] = len(distinct)
        else:
            record_to_replicate_count[distinct[0]] = 0
    return {
        "replicate_group_count": replicate_group_count,
        "replicate_record_ids": replicate_record_ids,
        "groups": groups,
        "record_to_group": record_to_group,
        "record_to_replicate_count": record_to_replicate_count,
        "skipped": skipped,
    }


# =============================================================================
# T-D1.7: raw / upstream / project-normalized reactivity layers
# (v3 §6.6 step 11 — freeze normalization domain; v3 §6.7 prohibitions;
#  v3.1 §4 D1 Gate — normalization_domain_unknown exclusion reason)
# =============================================================================

# Allowed values for a construct's ``normalization_method`` field (v3 §6.3).
#   raw               — no normalization; reactivity_upstream == reactivity_raw
#   2-8_percent       — per-construct 2-8% normalization (top 2-8% mean scaling)
#   boxplot_95th      — per-construct 95th-percentile scaling (boxplot style)
#   upstream_provided — depositors normalized upstream; upstream == raw
#   project_zscore    — cross-study z-score within a frozen normalization domain
#   unknown           — domain could not be frozen → exclusion reason
NORMALIZATION_METHODS = frozenset({
    "raw",
    "2-8_percent",
    "boxplot_95th",
    "upstream_provided",
    "project_zscore",
    "unknown",
})

# Fields that define a frozen normalization domain (v3 §6.6 step 11). Constructs
# in the same domain share study, probe chemistry, probe protocol and in
# vivo/in vitro context; only within a domain may cross-construct
# (project-level) normalization such as z-scoring be applied. The domain is
# *frozen* at step 11, before the train/validation/test split (step 14 / D2);
# per v3.1 §2.2 and v3 §6.7, post-split the z-score stats are re-fitted on
# train+validation members only (test must not feed normalization).
NORMALIZATION_DOMAIN_FIELDS = ("study_id", "probe", "probe_protocol", "in_vivo_in_vitro")
# Domain fields that are nullable: None is a valid domain component (e.g. a
# protocol-agnostic domain), not an "unknown domain" trigger.
NORMALIZATION_DOMAIN_NULLABLE_FIELDS = frozenset({"probe_protocol"})
# Domain fields that must be present and non-empty; missing any → unknown domain.
NORMALIZATION_DOMAIN_REQUIRED_FIELDS = frozenset(
    set(NORMALIZATION_DOMAIN_FIELDS) - NORMALIZATION_DOMAIN_NULLABLE_FIELDS
)

# 2-8% normalization percentile window (RMDB convention): take the mean of the
# non-missing values whose rank falls in [92nd, 98th] percentile and divide
# every non-missing value by it (nearest-rank method).
NORMALIZATION_2_8_LOW_PERCENTILE = 92.0
NORMALIZATION_2_8_HIGH_PERCENTILE = 98.0


def identify_normalization_domain(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the frozen normalization-domain key for a construct record (T-D1.7).

    The domain is the tuple of ``NORMALIZATION_DOMAIN_FIELDS`` values. For
    nullable fields (``probe_protocol``) ``None`` is a valid domain component.
    If any *required* domain field is missing, ``None`` or empty, the domain
    cannot be frozen and the empty tuple ``()`` is returned; callers must record
    exclusion reason ``normalization_domain_unknown`` (v3.1 §4 D1 Gate). Domains
    are frozen at v3 §6.6 step 11, before the split (step 14 / D2).
    """

    key: list[Any] = []
    for field in NORMALIZATION_DOMAIN_FIELDS:
        value = record.get(field)
        if field in NORMALIZATION_DOMAIN_NULLABLE_FIELDS:
            key.append(value)
            continue
        if value is None or value == "":
            return ()
        key.append(value)
    return tuple(key)


def build_normalization_domains(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Group construct records by frozen normalization domain (T-D1.7).

    Returns ``domain_key -> {"construct_ids": [...], "count": int}``. Records
    with an unfrozen domain (any required domain field missing) are collected
    under the empty-tuple key ``()`` so callers can flag them with
    ``normalization_domain_unknown``.
    """

    domains: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in records:
        key = identify_normalization_domain(rec)
        bucket = domains.setdefault(key, {"construct_ids": [], "count": 0})
        cid = rec.get("construct_id")
        if cid is not None:
            bucket["construct_ids"].append(cid)
        bucket["count"] += 1
    return domains


def check_normalization_domain_compatible(
    wt_record: Mapping[str, Any],
    mut_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether a WT/mutant pair shares a frozen normalization domain (T-D1.7).

    A pair may only be project-normalized together when both members fall in the
    same frozen domain. Returns ``{"compatible": bool, "domain": tuple, "reason":
    str}``. ``reason`` is ``""`` when compatible, otherwise one of
    ``"wt_domain_unknown"``, ``"mut_domain_unknown"``, ``"domain_mismatch"``.
    """

    wt_domain = identify_normalization_domain(wt_record)
    mut_domain = identify_normalization_domain(mut_record)
    if not wt_domain:
        return {"compatible": False, "domain": (), "reason": "wt_domain_unknown"}
    if not mut_domain:
        return {"compatible": False, "domain": (), "reason": "mut_domain_unknown"}
    if wt_domain != mut_domain:
        return {"compatible": False, "domain": (), "reason": "domain_mismatch"}
    return {"compatible": True, "domain": wt_domain, "reason": ""}


def normalize_2_8_percent(
    reactivity: Iterable[float | None],
) -> tuple[list[float | None], float | None]:
    """Apply per-construct 2-8% normalization (T-D1.7 upstream layer).

    Collects the non-missing (non-None, finite) values, sorts them, takes the
    mean of those whose nearest-rank percentile falls in [92nd, 98th], and
    divides every non-missing value by that mean. Missing values are preserved
    as ``None`` (v3 §6.7: missing must not be treated as 0). Returns
    ``(normalized, scale_factor)``; ``scale_factor`` is ``None`` (and the input
    is returned unchanged) when there are fewer than 2 non-missing values or the
    window is degenerate.
    """

    values = list(reactivity)
    finite = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if len(finite) < 2:
        return list(values), None
    finite_sorted = sorted(finite)
    n = len(finite_sorted)
    lo_rank = max(1, math.ceil(NORMALIZATION_2_8_LOW_PERCENTILE / 100.0 * n))
    hi_rank = min(n, math.ceil(NORMALIZATION_2_8_HIGH_PERCENTILE / 100.0 * n))
    if hi_rank < lo_rank:
        return list(values), None
    window = finite_sorted[lo_rank - 1 : hi_rank]
    scale_factor = sum(window) / len(window)
    if scale_factor == 0 or not math.isfinite(scale_factor):
        return list(values), None
    normalized = [None if v is None else v / scale_factor for v in values]
    return normalized, scale_factor


def compute_domain_zscore_stats(
    reactivities: Iterable[Iterable[float | None]],
) -> dict[str, Any]:
    """Compute mean/std of non-missing reactivity across a domain (T-D1.7).

    Pools all non-missing (non-None, finite) values across the given per-construct
    reactivity arrays (the domain members) and returns
    ``{"mean": float|None, "std": float|None, "count": int}`` (sample std, ddof=1).
    Per v3.1 §2.2 and v3 §6.7 the caller MUST pass only train+validation
    reactivities after the split (D2); at D1 the domain is frozen over all
    candidates because no split exists yet. Returns ``None`` stats when fewer
    than 2 values are present.
    """

    pooled: list[float] = []
    for arr in reactivities:
        for v in arr:
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                pooled.append(float(v))
    count = len(pooled)
    if count < 2:
        return {"mean": None, "std": None, "count": count}
    mean = sum(pooled) / count
    var = sum((v - mean) ** 2 for v in pooled) / (count - 1)
    std = math.sqrt(var)
    if not math.isfinite(std):
        return {"mean": None, "std": None, "count": count}
    return {"mean": mean, "std": std, "count": count}


def apply_zscore_normalization(
    reactivity: Iterable[float | None],
    mean: float | None,
    std: float | None,
) -> list[float | None]:
    """Apply project-level z-score normalization to one construct (T-D1.7).

    ``(v - mean) / std`` for non-missing values; missing values are preserved as
    ``None`` (v3 §6.7). If ``mean`` or ``std`` is ``None``, non-finite or the std
    is zero, the input is returned unchanged (the project-normalized layer
    cannot be formed and falls back to the upstream layer).
    """

    values = list(reactivity)
    if (
        mean is None
        or std is None
        or std == 0
        or not isinstance(mean, (int, float))
        or not isinstance(std, (int, float))
        or not math.isfinite(mean)
        or not math.isfinite(std)
    ):
        return list(values)
    return [None if v is None else (v - mean) / std for v in values]


def build_reactivity_layers(
    raw_reactivity: Iterable[float | None],
    normalization_method: str | None,
    domain_mean: float | None = None,
    domain_std: float | None = None,
) -> dict[str, Any]:
    """Build the raw / upstream / project-normalized reactivity layers (T-D1.7).

    Returns a dict with keys:
      - ``reactivity_raw``: the input values (list, None preserved)
      - ``reactivity_upstream``: per-construct normalized layer (2-8% when method
        is ``"2-8_percent"``; identical to raw for ``"raw"``/``"upstream_provided"``
        and as a fail-safe for unrecognized methods)
      - ``reactivity_project``: cross-study z-score layer from domain stats
        (identical to upstream when domain stats are unavailable)
      - ``scale_factor``: the 2-8% scale factor (or ``None``)
      - ``normalization_method``: the recorded method (or ``"unknown"``)

    Per v3 §6.7 missing values are never treated as 0 and propagate through
    every layer. Per v3.1 §2.2 domain stats must come from train+validation only
    (enforced by the caller, not here).
    """

    raw_list = list(raw_reactivity)
    method = normalization_method or "unknown"
    scale_factor: float | None = None
    if method == "2-8_percent":
        upstream, scale_factor = normalize_2_8_percent(raw_list)
    else:
        upstream = list(raw_list)
    project = apply_zscore_normalization(upstream, domain_mean, domain_std)
    return {
        "reactivity_raw": raw_list,
        "reactivity_upstream": upstream,
        "reactivity_project": project,
        "scale_factor": scale_factor,
        "normalization_method": method,
    }


# =============================================================================
# T-D1.8: study/probe measurement-noise estimation
# (v3 §6.6 step 10; v3 §7.2 with-replicates, §7.3 without-replicates;
#  v3.1 §2.2 / §4 D1 Gate — noise must not be estimated from test data)
# =============================================================================

# Minimum number of replicates required to estimate replicate-based noise.
NOISE_ESTIMATION_MIN_REPLICATES = 2
# Minimum number of positions with >=2 non-missing replicate values required to
# form a replicate noise estimate.
NOISE_ESTIMATION_MIN_OVERLAP = 2
# Default percentile used when freezing a noise threshold from no-edit controls
# (v3 §7.3: "noise threshold 由 controls 冻结").
CONTROL_NOISE_THRESHOLD_PERCENTILE = 95.0
# Minimum number of pooled |Δreactivity| values from controls required to freeze
# a domain noise threshold.
CONTROL_NOISE_THRESHOLD_MIN_VALUES = 10


def estimate_replicate_noise(
    reactivities: Iterable[Iterable[float | None]],
    min_overlap: int = NOISE_ESTIMATION_MIN_OVERLAP,
) -> dict[str, Any]:
    """Estimate measurement noise from replicate disagreement (T-D1.8, v3 §7.2).

    Given per-replicate reactivity arrays for the same construct, computes the
    per-position sample variance across replicates (at positions with >= 2
    non-missing values), then averages over positions. Returns
    ``{"noise_std": float|None, "noise_variance": float|None, "n_positions": int,
    "n_replicates": int}``. Missing values are excluded (v3 §6.7: missing is not
    0). Stats are ``None`` when fewer than 2 replicates or fewer than
    ``min_overlap`` positions have replicates. Per v3.1 §2.2 the caller must pass
    only train+validation replicates after the split.
    """

    arrays = [list(a) for a in reactivities]
    n_rep = len(arrays)
    if n_rep < NOISE_ESTIMATION_MIN_REPLICATES:
        return {"noise_std": None, "noise_variance": None, "n_positions": 0, "n_replicates": n_rep}
    length = min((len(a) for a in arrays), default=0)
    if length == 0:
        return {"noise_std": None, "noise_variance": None, "n_positions": 0, "n_replicates": n_rep}
    per_pos_vars: list[float] = []
    for i in range(length):
        vals = [
            a[i]
            for a in arrays
            if i < len(a)
            and a[i] is not None
            and isinstance(a[i], (int, float))
            and math.isfinite(a[i])
        ]
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        per_pos_vars.append(var)
    n_positions = len(per_pos_vars)
    if n_positions < min_overlap:
        return {"noise_std": None, "noise_variance": None, "n_positions": n_positions, "n_replicates": n_rep}
    noise_var = sum(per_pos_vars) / n_positions
    if not math.isfinite(noise_var):
        return {"noise_std": None, "noise_variance": None, "n_positions": n_positions, "n_replicates": n_rep}
    return {
        "noise_std": math.sqrt(noise_var),
        "noise_variance": noise_var,
        "n_positions": n_positions,
        "n_replicates": n_rep,
    }


def estimate_error_variance(
    reactivity_error: Iterable[float | None],
) -> float | None:
    """Estimate measurement variance from upstream REACTIVITY_ERROR (T-D1.8, §7.3).

    Per v3 §7.3, when no replicates are available the measurement noise uses the
    upstream error. RDAT ``REACTIVITY_ERROR`` values are per-position standard
    errors; the measurement variance is the mean of squared errors over
    non-missing positions. Returns ``None`` when no non-missing finite errors are
    present. Missing values excluded (v3 §6.7).
    """

    errs = [
        e
        for e in reactivity_error
        if e is not None and isinstance(e, (int, float)) and math.isfinite(e)
    ]
    if not errs:
        return None
    var = sum(e * e for e in errs) / len(errs)
    if not math.isfinite(var):
        return None
    return var


def freeze_control_noise_threshold(
    control_delta_abs_values: Iterable[float],
    percentile: float = CONTROL_NOISE_THRESHOLD_PERCENTILE,
    min_values: int = CONTROL_NOISE_THRESHOLD_MIN_VALUES,
) -> float | None:
    """Freeze a noise threshold from no-edit control |Δreactivity| (T-D1.8, §7.3).

    Per v3 §7.3 the noise threshold is frozen by controls. Returns the given
    percentile (nearest-rank) of the pooled absolute Δreactivity values from
    no-edit control pairs within a normalization domain. Returns ``None`` when
    fewer than ``min_values`` are present (insufficient controls to freeze a
    threshold). Per v3.1 §2.2 the caller must pass only train+validation controls
    after the split. The threshold feeds the frozen differential caller (T-D1.9).
    """

    values = sorted(
        v
        for v in control_delta_abs_values
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v)
    )
    if len(values) < min_values:
        return None
    rank = max(1, math.ceil(percentile / 100.0 * len(values)))
    rank = min(len(values), rank)
    return values[rank - 1]


def estimate_pair_noise(
    wt_replicate_noise: Mapping[str, Any] | None,
    mut_replicate_noise: Mapping[str, Any] | None,
    wt_error_variance: float | None = None,
    mut_error_variance: float | None = None,
) -> dict[str, Any]:
    """Produce per-pair replicate_noise_estimate and measurement_variance (T-D1.8).

    Combines WT and mutant noise estimates into the pair-level fields required by
    the pair schema (v3 §6.4). Member variance prefers replicate noise (v3 §7.2)
    and falls back to upstream error variance (v3 §7.3). The pair
    ``measurement_variance`` is the sum of member variances (variance of
    Δr = r_m − r_w under independent measurement). ``replicate_noise_estimate`` is
    the std of the replicate-based portion only — it is ``None`` when neither
    member has replicate noise. Missing/None member variances are skipped (v3
    §6.7: missing is not 0).

    Returns ``{"replicate_noise_estimate": float|None, "measurement_variance":
    float|None, "wt_variance": float|None, "mut_variance": float|None,
    "source": str}`` where ``source`` ∈ ``{"replicate", "upstream_error",
    "none"}``.
    """

    wt_rep_var = (
        wt_replicate_noise.get("noise_variance") if wt_replicate_noise is not None else None
    )
    mut_rep_var = (
        mut_replicate_noise.get("noise_variance") if mut_replicate_noise is not None else None
    )
    # Member variance: replicate if available, else upstream error (v3 §7.3).
    wt_var = wt_rep_var if wt_rep_var is not None else wt_error_variance
    mut_var = mut_rep_var if mut_rep_var is not None else mut_error_variance
    # Pair measurement_variance = wt_var + mut_var (independent measurement).
    parts = [v for v in (wt_var, mut_var) if v is not None]
    measurement_variance: float | None = sum(parts) if parts else None
    # replicate_noise_estimate: std of the replicate-based portion only.
    rep_parts = [v for v in (wt_rep_var, mut_rep_var) if v is not None]
    replicate_noise_estimate: float | None
    if rep_parts:
        replicate_noise_estimate = math.sqrt(sum(rep_parts))
    else:
        replicate_noise_estimate = None
    if rep_parts:
        source = "replicate"
    elif parts:
        source = "upstream_error"
    else:
        source = "none"
    return {
        "replicate_noise_estimate": replicate_noise_estimate,
        "measurement_variance": measurement_variance,
        "wt_variance": wt_var,
        "mut_variance": mut_var,
        "source": source,
    }


# =============================================================================
# T-D1.9: frozen differential caller
# (v3 §6.6 step 12 — generate Δreactivity; §7.1 Δr = r_m − r_w;
#  §7.2 replicate-aware caller with FDR; §7.3 no-replicate continuous-only)
# =============================================================================

# Default FDR level for the replicate-aware differential caller (v3 §7.2).
DIFFERENTIAL_FDR_ALPHA = 0.05


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (T-D1.9 helper)."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _benjamini_hochberg(
    p_values: list[float], alpha: float
) -> tuple[float | None, set[int]]:
    """Benjamini–Hochberg FDR procedure (T-D1.9 helper, v3 §7.2).

    Returns ``(threshold_p, rejected_indices)`` where ``threshold_p`` is the
    largest sorted p-value satisfying p_(k) ≤ k/m·α (None if none rejected) and
    ``rejected_indices`` are the 0-indexed positions of rejected hypotheses.
    """

    m = len(p_values)
    if m == 0:
        return None, set()
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    last_k = 0
    for k in range(1, m + 1):
        if sorted_p[k - 1] <= (k / m) * alpha:
            last_k = k
    rejected = set(order[:last_k]) if last_k > 0 else set()
    threshold_p = sorted_p[last_k - 1] if last_k > 0 else None
    return threshold_p, rejected


def compute_delta_reactivity(
    wt_reactivity: Iterable[float | None],
    mut_reactivity: Iterable[float | None],
) -> list[float | None]:
    """Per-position Δr = r_m − r_w (T-D1.9, v3 §6.6 step 12, §7.1).

    The continuous experimental response is the primary label (v3 §7.1).
    Positions where either value is missing or non-finite yield ``None`` (v3
    §6.7: missing must not be treated as 0). Length is the minimum of the two
    arrays; equal-length WT/mutant constructs (v3 §6.5) are the D1 upgrade path.
    """

    wt = list(wt_reactivity)
    mut = list(mut_reactivity)
    n = min(len(wt), len(mut))
    delta: list[float | None] = []
    for i in range(n):
        w = wt[i]
        m = mut[i]
        if (
            w is None
            or m is None
            or not isinstance(w, (int, float))
            or not isinstance(m, (int, float))
            or not math.isfinite(w)
            or not math.isfinite(m)
        ):
            delta.append(None)
        else:
            delta.append(m - w)
    return delta


def frozen_differential_call(
    delta_reactivity: Iterable[float | None],
    noise_threshold: float | None = None,
    measurement_variance: float | None = None,
    has_replicates: bool = False,
    fdr_alpha: float = DIFFERENTIAL_FDR_ALPHA,
) -> dict[str, Any]:
    """Frozen differential caller (T-D1.9, v3 §7.2/§7.3, §6.6 step 12).

    Calls significant changers per position using a *frozen* noise threshold
    (T-D1.8, not learned from test — v3.1 §2.2). The threshold is frozen before
    the split (step 14 / D2).

    - With replicates and a frozen ``noise_threshold`` (v3 §7.2): a position is
      significant when ``|Δr| > noise_threshold`` (replicate-aware caller).
    - Without replicates (v3 §7.3): only the continuous Δr is produced; **no**
      significant-changer claim is made (``significant_mask`` is all zeros).
    - With replicates and ``measurement_variance``: per-position z-scores and a
      Benjamini–Hochberg FDR call at ``fdr_alpha`` (v3 §7.2: output differential
      regions and FDR) are also returned as auxiliary metadata.

    Returns ``{"significant_mask": list[int], "significant_count": int,
    "z_scores": list[float|None], "fdr_significant_mask": list[int],
    "fdr_threshold_p": float|None, "caller_status": str}`` where ``caller_status``
    ∈ ``{"replicate_aware", "no_replicate_continuous_only", "no_threshold"}``.
    """

    delta = list(delta_reactivity)
    n = len(delta)
    # --- z-scores (auxiliary, when per-pair measurement_variance available) ---
    z_scores: list[float | None] = [None] * n
    if (
        measurement_variance is not None
        and isinstance(measurement_variance, (int, float))
        and math.isfinite(measurement_variance)
        and measurement_variance > 0
    ):
        std = math.sqrt(measurement_variance)
        for i, v in enumerate(delta):
            if (
                v is not None
                and isinstance(v, (int, float))
                and math.isfinite(v)
            ):
                z_scores[i] = v / std
    # --- frozen-threshold significance (primary caller) ---
    if has_replicates and noise_threshold is not None and isinstance(
        noise_threshold, (int, float)
    ) and math.isfinite(noise_threshold):
        significant_mask = [
            1 if (
                v is not None
                and isinstance(v, (int, float))
                and math.isfinite(v)
                and abs(v) > noise_threshold
            )
            else 0
            for v in delta
        ]
        caller_status = "replicate_aware"
    elif not has_replicates:
        significant_mask = [0] * n
        caller_status = "no_replicate_continuous_only"
    else:
        significant_mask = [0] * n
        caller_status = "no_threshold"
    significant_count = sum(significant_mask)
    # --- BH-FDR (auxiliary, v3 §7.2; only with replicates) ---
    fdr_significant_mask = [0] * n
    fdr_threshold_p: float | None = None
    if has_replicates:
        p_values: list[float] = []
        idx_map: list[int] = []  # positions with finite z
        for i, z in enumerate(z_scores):
            if z is not None and isinstance(z, (int, float)) and math.isfinite(z):
                p_values.append(2.0 * (1.0 - _normal_cdf(abs(z))))
                idx_map.append(i)
        if p_values:
            fdr_threshold_p, rejected = _benjamini_hochberg(p_values, fdr_alpha)
            for local_i in rejected:
                fdr_significant_mask[idx_map[local_i]] = 1
    return {
        "significant_mask": significant_mask,
        "significant_count": significant_count,
        "z_scores": z_scores,
        "fdr_significant_mask": fdr_significant_mask,
        "fdr_threshold_p": fdr_threshold_p,
        "caller_status": caller_status,
    }


def build_pair_delta_reactivity(
    wt_reactivity_raw: Iterable[float | None],
    mut_reactivity_raw: Iterable[float | None],
    wt_reactivity_normalized: Iterable[float | None] | None = None,
    mut_reactivity_normalized: Iterable[float | None] | None = None,
    noise_threshold: float | None = None,
    measurement_variance: float | None = None,
    has_replicates: bool = False,
    fdr_alpha: float = DIFFERENTIAL_FDR_ALPHA,
) -> dict[str, Any]:
    """Build the pair Δreactivity layers + frozen differential call (T-D1.9).

    Computes ``delta_reactivity_raw`` (always) and ``delta_reactivity_normalized``
    (when normalized layers are provided; v3 §6.4 pair schema), then runs the
    frozen differential caller on the normalized layer (preferred) or the raw
    layer. Returns a dict with the pair-schema Δreactivity arrays and the caller
    output. Per v3 §6.7 missing values are propagated, never treated as 0; per
    v3.1 §2.2 the frozen threshold must come from train+validation only.
    """

    delta_raw = compute_delta_reactivity(wt_reactivity_raw, mut_reactivity_raw)
    if wt_reactivity_normalized is not None and mut_reactivity_normalized is not None:
        delta_normalized = compute_delta_reactivity(
            wt_reactivity_normalized, mut_reactivity_normalized
        )
        call_input = delta_normalized
    else:
        delta_normalized = None
        call_input = delta_raw
    call = frozen_differential_call(
        call_input,
        noise_threshold=noise_threshold,
        measurement_variance=measurement_variance,
        has_replicates=has_replicates,
        fdr_alpha=fdr_alpha,
    )
    return {
        "delta_reactivity_raw": delta_raw,
        "delta_reactivity_normalized": delta_normalized,
        "significant_mask": call["significant_mask"],
        "significant_count": call["significant_count"],
        "z_scores": call["z_scores"],
        "fdr_significant_mask": call["fdr_significant_mask"],
        "fdr_threshold_p": call["fdr_threshold_p"],
        "caller_status": call["caller_status"],
    }


# =============================================================================
# T-D1.10: pair quality weight + exclusion reasons + true_pair upgrade
# (v3.1 §3 pair eligibility; v3 §6.4 pair schema; integrates T-D1.1~9).
# =============================================================================

# Exclusion reasons that block true_pair upgrade but NOT primary_eligible
# (v3.1 §3.1). All other reasons in EXCLUSION_REASONS block both.
UPGRADE_BLOCKER_EXCLUSION_REASONS = frozenset({
    "sequence_based_no_independent_corroboration",
})

# Quality-weight factor scaling points (v3.1 §3.2).
QUALITY_SNR_FULL_FACTOR_AT = 10.0
QUALITY_COVERAGE_FULL_FACTOR_AT = 30.0
QUALITY_NO_REPLICATE_FACTOR = 0.8
QUALITY_UNKNOWN_SIGNAL_FACTOR = 0.5


def _clamp01(value: float) -> float:
    """Clamp a numeric value to the closed interval [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def collect_exclusion_reasons(
    *,
    edit_type: str,
    edit_count: int,
    condition_match_status: str,
    substitution_verified: bool,
    has_wt_anchor: bool,
    normalization_domain_compatible: bool,
    parent_lineage_verified: bool,
    in_vivo_in_vitro_mixed: bool,
    comparable_fraction: float | None = None,
    probe_eligible_unchanged: bool | None = None,
    annotation_ref_verified: bool | None = None,
    is_annotation_only: bool = False,
    is_sequence_based: bool = False,
    has_independent_corroboration: bool = True,
) -> list[str]:
    """Collect the frozen exclusion-reason set for a pair (v3.1 §3.1, §4 D1 Gate).

    Reasons are drawn from the frozen ``EXCLUSION_REASONS`` vocabulary. The
    returned list is the sorted unique set so callers always see a stable,
    machine-readable reason vector. Per v3.1 §4 every rejection must carry at
    least one reason; per v3.1 §2.2 the reason logic is frozen (no
    train-time tuning).
    """

    reasons: set[str] = set()

    # edit_type / edit_count (schema.py L668 invariant ties indel → reason).
    if edit_type != "substitution":
        reasons.add("indel_not_substitution")
    elif edit_count != 1:
        reasons.add("edit_count_not_one")

    # WT anchor: a substitution pair needs a wild-type anchor construct.
    if not has_wt_anchor:
        reasons.add("no_wt_anchor")

    # Substitution verifiability: annotation-only pairs cannot have their
    # alt allele verified by sequencing; otherwise flag as not verifiable.
    if not substitution_verified:
        if is_annotation_only:
            reasons.add("annotation_only_alt_not_verifiable")
        else:
            reasons.add("substitution_not_verifiable")

    # Annotation reference mismatch (only when explicitly checked and false).
    if annotation_ref_verified is False:
        reasons.add("annotation_ref_mismatch")

    # Condition mismatch (schema.py L672 invariant ties status → reason).
    if condition_match_status == "mismatch":
        reasons.add("condition_mismatch")

    # Probe eligibility (only when explicitly checked and false).
    if probe_eligible_unchanged is False:
        reasons.add("probe_mismatch")

    # Comparable-position fraction below the 60% minimum (T-D1.5 constant).
    if comparable_fraction is not None and comparable_fraction < COMPARABLE_MIN_FRACTION:
        reasons.add("comparable_positions_below_60pct")

    # Normalization-domain compatibility (T-D1.7).
    if not normalization_domain_compatible:
        reasons.add("normalization_domain_unknown")

    # Parent lineage (T-D1.6).
    if not parent_lineage_verified:
        reasons.add("parent_lineage_unverified")

    # in_vivo / in_vitro mixing within a pair (T-D1.7 domain field).
    if in_vivo_in_vitro_mixed:
        reasons.add("in_vivo_in_vitro_mixed")

    # Sequence-based edit without independent corroboration: a soft blocker
    # for true_pair upgrade but not for primary_eligible (v3.1 §3.1).
    if is_sequence_based and not has_independent_corroboration:
        reasons.add("sequence_based_no_independent_corroboration")

    return sorted(reasons)


def determine_primary_eligible(exclusion_reasons: list[str]) -> bool:
    """Return True iff no exclusion reason blocks primary eligibility.

    Only ``UPGRADE_BLOCKER_EXCLUSION_REASONS`` (corroboration-only) are soft:
    they keep a pair primary-eligible. Any other reason makes the pair
    ineligible. An empty reason list is trivially eligible (v3.1 §3.1).
    """
    return all(
        r in UPGRADE_BLOCKER_EXCLUSION_REASONS for r in exclusion_reasons
    )


def determine_true_pair(
    exclusion_reasons: list[str], primary_eligible: bool
) -> bool:
    """Return True iff the pair is a clean true_pair: eligible and no reasons.

    Per v3.1 §3.3 a true_pair must have zero exclusion reasons (the
    corroboration-only soft blocker still disqualifies true_pair status).
    """
    return primary_eligible and not exclusion_reasons


def compute_pair_quality_weight(
    *,
    comparable_fraction: float | None = None,
    snr: float | None = None,
    coverage_mean: float | None = None,
    missing_fraction: float | None = None,
    has_replicates: bool = False,
) -> dict[str, Any]:
    """Compute the pair quality weight as a product of clamped factors.

    Factors (v3.1 §3.2): comparable fraction, signal-to-noise, coverage,
    replicate presence, and (1 - missing fraction). Each factor is clamped to
    [0, 1]; unknown inputs default to ``QUALITY_UNKNOWN_SIGNAL_FACTOR`` (0.5)
    so that an unknown signal does not silently inflate the weight. The
    no-replicate penalty is ``QUALITY_NO_REPLICATE_FACTOR`` (0.8). The product
    is clamped to [0, 1] (schema.py L648 requires a non-negative number).
    """

    f_comp = _clamp01(comparable_fraction) if comparable_fraction is not None else QUALITY_UNKNOWN_SIGNAL_FACTOR
    f_snr = (
        _clamp01(snr / QUALITY_SNR_FULL_FACTOR_AT)
        if snr is not None
        else QUALITY_UNKNOWN_SIGNAL_FACTOR
    )
    f_cov = (
        _clamp01(coverage_mean / QUALITY_COVERAGE_FULL_FACTOR_AT)
        if coverage_mean is not None
        else QUALITY_UNKNOWN_SIGNAL_FACTOR
    )
    f_rep = 1.0 if has_replicates else QUALITY_NO_REPLICATE_FACTOR
    f_miss = (
        _clamp01(1.0 - missing_fraction)
        if missing_fraction is not None
        else QUALITY_UNKNOWN_SIGNAL_FACTOR
    )

    weight = _clamp01(f_comp * f_snr * f_cov * f_rep * f_miss)
    return {
        "pair_quality_weight": weight,
        "factors": {
            "comparable": f_comp,
            "snr": f_snr,
            "coverage": f_cov,
            "replicate": f_rep,
            "missing": f_miss,
        },
    }


def evaluate_pair_upgrade(
    *,
    edit_type: str,
    edit_count: int,
    condition_match_status: str,
    substitution_verified: bool,
    has_wt_anchor: bool,
    normalization_domain_compatible: bool,
    parent_lineage_verified: bool,
    in_vivo_in_vitro_mixed: bool,
    comparable_fraction: float | None = None,
    probe_eligible_unchanged: bool | None = None,
    annotation_ref_verified: bool | None = None,
    is_annotation_only: bool = False,
    is_sequence_based: bool = False,
    has_independent_corroboration: bool = True,
    snr: float | None = None,
    coverage_mean: float | None = None,
    missing_fraction: float | None = None,
    has_replicates: bool = False,
) -> dict[str, Any]:
    """Top-level pair upgrade evaluator integrating T-D1.1~9 (v3.1 §3, §4).

    Returns the four pair-schema fields tied to D1 eligibility:
    ``exclusion_reasons`` (sorted frozen-vocabulary list),
    ``primary_eligible`` (bool), ``true_pair`` (bool), and
    ``pair_quality_weight`` (number in [0, 1]) plus the per-factor breakdown.
    Callers must still ensure the remaining pair-schema invariants from
    schema.py L668/L672 (edit_type/condition coupling) hold at the construct
    level — this function emits the matching reasons when those inputs are
    passed truthfully.
    """

    reasons = collect_exclusion_reasons(
        edit_type=edit_type,
        edit_count=edit_count,
        condition_match_status=condition_match_status,
        substitution_verified=substitution_verified,
        has_wt_anchor=has_wt_anchor,
        normalization_domain_compatible=normalization_domain_compatible,
        parent_lineage_verified=parent_lineage_verified,
        in_vivo_in_vitro_mixed=in_vivo_in_vitro_mixed,
        comparable_fraction=comparable_fraction,
        probe_eligible_unchanged=probe_eligible_unchanged,
        annotation_ref_verified=annotation_ref_verified,
        is_annotation_only=is_annotation_only,
        is_sequence_based=is_sequence_based,
        has_independent_corroboration=has_independent_corroboration,
    )
    primary_eligible = determine_primary_eligible(reasons)
    true_pair = determine_true_pair(reasons, primary_eligible)
    qw = compute_pair_quality_weight(
        comparable_fraction=comparable_fraction,
        snr=snr,
        coverage_mean=coverage_mean,
        missing_fraction=missing_fraction,
        has_replicates=has_replicates,
    )
    return {
        "exclusion_reasons": reasons,
        "primary_eligible": primary_eligible,
        "true_pair": true_pair,
        "pair_quality_weight": qw["pair_quality_weight"],
        "quality_factors": qw["factors"],
    }
