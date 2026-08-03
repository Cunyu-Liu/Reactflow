#!/usr/bin/env python3
"""Freeze the five authoritative RMDB GitHub release asset indexes.

The script fetches metadata only.  It never downloads RDAT payloads and never
overwrites an existing output.  The normalized snapshot is accepted only when
it is a one-to-one match to the frozen 1,024-accession registry seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY = "DasLab/rmdb.github.io"
RELEASE_TAGS = (
    "data-eterna",
    "data-puzzle",
    "data-riboswitches",
    "data-rna-structures",
    "data-general",
)
SHA256_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")
HTTPS_ASSET_PREFIX = (
    "https://github.com/DasLab/rmdb.github.io/releases/download/"
)


class SnapshotError(ValueError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_registry_ids(path: Path) -> set[str]:
    result: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise SnapshotError(f"blank registry line {line_number}")
        row = json.loads(line)
        accession = row.get("rmdb_id")
        if not isinstance(accession, str) or not accession:
            raise SnapshotError(f"invalid rmdb_id at registry line {line_number}")
        if accession in result:
            raise SnapshotError(f"duplicate registry rmdb_id: {accession}")
        result.add(accession)
    if not result:
        raise SnapshotError("empty RMDB accession registry")
    return result


def _fetch_release(tag: str) -> tuple[bytes, dict[str, Any]]:
    url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{tag}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ReactFlow-Delta-D0X-metadata-freezer/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise SnapshotError(f"GitHub API returned HTTP {response.status} for {tag}")
        raw = response.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SnapshotError(f"release payload is not an object: {tag}")
    return raw, payload


def normalize_release_payloads(
    payloads: list[tuple[bytes, dict[str, Any]]],
    *,
    registry_ids: set[str],
    frozen_at: str,
    rmdb_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(payloads) != len(RELEASE_TAGS):
        raise SnapshotError("exactly five release payloads are required")
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_asset_ids: set[int] = set()
    release_summaries: list[dict[str, Any]] = []
    for expected_tag, (raw, payload) in zip(RELEASE_TAGS, payloads):
        if payload.get("tag_name") != expected_tag:
            raise SnapshotError(
                f"release tag mismatch: expected {expected_tag!r}, "
                f"got {payload.get('tag_name')!r}"
            )
        release_id = payload.get("id")
        assets = payload.get("assets")
        if not isinstance(release_id, int) or not isinstance(assets, list) or not assets:
            raise SnapshotError(f"release {expected_tag} lacks id or non-empty assets")
        raw_sha = _sha256_bytes(raw)
        release_summaries.append(
            {
                "tag": expected_tag,
                "release_id": release_id,
                "asset_count": len(assets),
                "api_payload_sha256": raw_sha,
                "published_at": payload.get("published_at"),
                "updated_at": payload.get("updated_at"),
            }
        )
        for asset in assets:
            if not isinstance(asset, dict):
                raise SnapshotError(f"non-object asset in {expected_tag}")
            asset_id = asset.get("id")
            name = asset.get("name")
            size = asset.get("size")
            url = asset.get("browser_download_url")
            digest_match = SHA256_DIGEST.fullmatch(str(asset.get("digest", "")))
            if not isinstance(asset_id, int) or asset_id in seen_asset_ids:
                raise SnapshotError(f"invalid or duplicate asset id in {expected_tag}: {asset_id}")
            if not isinstance(name, str) or not name.endswith(".rdat"):
                raise SnapshotError(f"non-RDAT asset name in {expected_tag}: {name!r}")
            if name in seen_names:
                raise SnapshotError(f"duplicate asset name across releases: {name}")
            if not isinstance(size, int) or size <= 0:
                raise SnapshotError(f"invalid asset size for {name}: {size!r}")
            if not isinstance(url, str) or not url.startswith(HTTPS_ASSET_PREFIX):
                raise SnapshotError(f"non-authoritative asset URL for {name}: {url!r}")
            if digest_match is None:
                raise SnapshotError(f"missing valid upstream SHA-256 digest for {name}")
            seen_asset_ids.add(asset_id)
            seen_names.add(name)
            records.append(
                {
                    "schema_version": "reactflow_delta.rmdb_release_asset.v1",
                    "source_id": "RMDB",
                    "source_accession": name[:-5],
                    "source_group": expected_tag,
                    "rmdb_repository": REPOSITORY,
                    "rmdb_repository_commit": rmdb_commit,
                    "release_id": release_id,
                    "release_tag": expected_tag,
                    "release_api_payload_sha256": raw_sha,
                    "asset_id": asset_id,
                    "asset_name": name,
                    "asset_url": url,
                    "expected_bytes": size,
                    "expected_sha256": digest_match.group(1),
                    "asset_updated_at": asset.get("updated_at"),
                    "license_status": "VERIFIED_CC0_RMDB",
                    "initial_disposition": "NOT_SEARCHED",
                    "frozen_at": frozen_at,
                }
            )
    records.sort(key=lambda item: item["source_accession"])
    asset_accessions = {item["source_accession"] for item in records}
    if asset_accessions != registry_ids:
        missing_assets = sorted(registry_ids - asset_accessions)
        unregistered_assets = sorted(asset_accessions - registry_ids)
        raise SnapshotError(
            "release/registry membership mismatch: "
            f"missing_assets={missing_assets[:10]}, "
            f"unregistered_assets={unregistered_assets[:10]}"
        )
    summary = {
        "schema_version": "reactflow_delta.rmdb_release_snapshot_summary.v1",
        "source_id": "RMDB",
        "repository": REPOSITORY,
        "repository_commit": rmdb_commit,
        "frozen_at": frozen_at,
        "release_tags": list(RELEASE_TAGS),
        "releases": release_summaries,
        "asset_count": len(records),
        "registry_accession_count": len(registry_ids),
        "membership_match": True,
        "initial_disposition": "NOT_SEARCHED",
        "scientific_boundary": (
            "Metadata-only frozen source universe. Asset and construct counts are "
            "not exact Delta pair counts."
        ),
    }
    return records, summary


def _write_create_once(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--rmdb-commit", required=True)
    args = parser.parse_args()
    registry_ids = _load_registry_ids(args.registry)
    payloads = [_fetch_release(tag) for tag in RELEASE_TAGS]
    records, summary = normalize_release_payloads(
        payloads,
        registry_ids=registry_ids,
        frozen_at=args.frozen_at,
        rmdb_commit=args.rmdb_commit,
    )
    jsonl = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in records
    )
    _write_create_once(args.output_jsonl, jsonl)
    summary["asset_index_path"] = str(args.output_jsonl)
    summary["asset_index_sha256"] = _sha256_bytes(jsonl.encode("utf-8"))
    _write_create_once(
        args.output_summary,
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
