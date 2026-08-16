from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reactflow.delta.data import RDAT_FIXTURE_MANIFEST_SCHEMA_VERSION, build_rdat_fixture_manifest


def _write_candidate_manifest(path: Path, name: str, payload: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "fixture_selection": [
                    {
                        "candidate_category": "m2_named_candidate",
                        "name": name,
                        "bytes": len(payload),
                        "upstream_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        )
    )


def test_fixture_manifest_requires_exact_verified_fixture_set(tmp_path: Path) -> None:
    payload = b"RDAT_VERSION\t0.34\n"
    candidate = tmp_path / "candidate.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "example.rdat").write_bytes(payload)
    _write_candidate_manifest(candidate, "example.rdat", payload)

    manifest = build_rdat_fixture_manifest(candidate, fixtures)

    assert manifest["schema_version"] == RDAT_FIXTURE_MANIFEST_SCHEMA_VERSION
    assert manifest["fixture_counts_by_candidate_category"] == {"m2_named_candidate": 1}
    assert manifest["fixtures"][0]["status"] == "verified_against_release_index"
    assert manifest["fixtures"][0]["rdat_confirmation_pending"] is True


def test_extra_or_partial_fixture_fails_closed(tmp_path: Path) -> None:
    payload = b"RDAT_VERSION\t0.34\n"
    candidate = tmp_path / "candidate.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "example.rdat").write_bytes(payload)
    _write_candidate_manifest(candidate, "example.rdat", payload)
    (fixtures / "extra.rdat").write_bytes(payload)
    with pytest.raises(ValueError, match="do not exactly match"):
        build_rdat_fixture_manifest(candidate, fixtures)
    (fixtures / "extra.rdat").unlink()
    (fixtures / "example.rdat.part").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete downloads"):
        build_rdat_fixture_manifest(candidate, fixtures)


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "example.rdat").write_bytes(b"actual")
    _write_candidate_manifest(candidate, "example.rdat", b"expected")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_rdat_fixture_manifest(candidate, fixtures)
