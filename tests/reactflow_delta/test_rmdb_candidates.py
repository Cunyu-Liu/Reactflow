from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.delta.data import (
    RMDB_CANDIDATE_MANIFEST_SCHEMA_VERSION,
    RMDB_FIXTURE_SELECTIONS,
    build_rmdb_filename_candidate_manifest,
)


def _asset(name: str) -> dict[str, object]:
    return {
        "name": name,
        "size": 10,
        "digest": "sha256:" + "c" * 64,
        "browser_download_url": f"https://example.org/{name}",
    }


def _write_release_index(path: Path, *, omit: str | None = None) -> None:
    assets = [
        _asset(name)
        for names in RMDB_FIXTURE_SELECTIONS.values()
        for name in names
        if name != omit
    ]
    assets.extend([_asset("HC16M2R_1M7_0001.rdat"), _asset("unrelated.rdat")])
    path.write_text(json.dumps([{"tag_name": "data-general", "assets": assets}]))


def test_candidate_manifest_is_explicitly_filename_only(tmp_path: Path) -> None:
    index = tmp_path / "releases.json"
    _write_release_index(index)
    manifest = build_rmdb_filename_candidate_manifest(index)

    assert manifest["schema_version"] == RMDB_CANDIDATE_MANIFEST_SCHEMA_VERSION
    assert len(manifest["fixture_selection"]) == 6
    assert all(item["rdat_confirmation_required"] for item in manifest["fixture_selection"])
    categories = {item["candidate_category"]: item for item in manifest["categories"]}
    assert categories["explicit_rescue_or_compensatory_named_candidate"]["candidate_count"] == 0
    assert "no confirmed experiment class" in manifest["scientific_boundary"]


def test_missing_fixed_fixture_fails_closed(tmp_path: Path) -> None:
    index = tmp_path / "releases.json"
    _write_release_index(index, omit="M2SL5_DMS_0000.rdat")
    with pytest.raises(ValueError, match="absent from release index"):
        build_rmdb_filename_candidate_manifest(index)


def test_non_digest_release_asset_fails_closed(tmp_path: Path) -> None:
    index = tmp_path / "releases.json"
    _write_release_index(index)
    payload = json.loads(index.read_text())
    payload[0]["assets"][0]["digest"] = None
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        build_rmdb_filename_candidate_manifest(index)
