"""Provenance-first helpers for the ReactFlow-Delta D0 public-data audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
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
