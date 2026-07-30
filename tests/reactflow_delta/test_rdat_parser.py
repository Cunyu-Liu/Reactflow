from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reactflow.delta.rdat import RdatParseError, build_rdat_construct_parse_manifest, parse_rdat


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_parser_preserves_nan_as_missing_and_mutation_annotation(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fixture.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34",
                "NAME\tfixture",
                "SEQUENCE\tACGU",
                "OFFSET\t0",
                "ANNOTATION\texperimentType:MutateAndMap\tmodifier:DMS",
                "ANNOTATION_DATA:1\tmutation:WT",
                "ANNOTATION_DATA:2\tmutation:A1X",
                "SEQPOS\tA1\tC2\tG3\tU4",
                "REACTIVITY:1\t0.1\tNaN\t0.3\t0.4",
                "REACTIVITY:2\t0.2\t0.3\t0.4\t0.5",
                "REACTIVITY_ERROR:2\t0.01\t0.01\t0.01\t0.01",
            ]
        ),
    )
    document = parse_rdat(path)
    assert document["profiles"][0]["reactivity"][1] is None
    assert document["profiles"][0]["missing_reactivity_count"] == 1
    assert document["profiles"][1]["annotation"]["mutation"] == ["A1X"]
    assert document["global_annotations"][0]["experimentType"] == ["MutateAndMap"]


def test_parser_fails_closed_on_profile_length_mismatch(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.rdat",
        "\n".join(["RDAT_VERSION\t0.34", "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1"]),
    )
    with pytest.raises(RdatParseError, match="length"):
        parse_rdat(path)


def test_parser_fails_closed_on_unknown_rdat_version(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad-version.rdat",
        "\n".join(["RDAT_VERSION\t0.33", "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2"]),
    )
    with pytest.raises(RdatParseError, match="0.34"):
        parse_rdat(path)


def test_construct_parse_manifest_retains_profile_annotations_without_reactivity(tmp_path: Path) -> None:
    fixture = _write(
        tmp_path / "fixture.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34", "NAME\tfixture", "SEQUENCE\tAC", "OFFSET\t0",
                "ANNOTATION\texperimentType:MutateAndMap\tchemical:MgCl2\tchemical:Na-HEPES",
                "ANNOTATION_DATA:1\tmutation:WT", "ANNOTATION_DATA:2\tmutation:A1X",
                "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2", "REACTIVITY:2\t0.3\tNaN",
            ]
        ),
    )
    manifest_path = tmp_path / "fixtures.json"
    manifest_path.write_text(json.dumps({"schema_version": "reactflow-delta-rdat-fixture-manifest-v1", "fixtures": [{"name": "fixture.rdat", "path": str(fixture), "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "candidate_category": "candidate", "status": "verified_against_release_index"}]}))

    document = build_rdat_construct_parse_manifest(manifest_path)

    parsed_fixture = document["fixtures"][0]
    assert parsed_fixture["global_annotations"][0]["chemical"] == ["MgCl2", "Na-HEPES"]
    assert parsed_fixture["profiles"][1]["annotation"]["mutation"] == ["A1X"]
    assert "reactivity" not in parsed_fixture["profiles"][1]
    assert parsed_fixture["profiles"][1]["missing_reactivity_count"] == 1


def test_construct_parse_manifest_fails_closed_on_fixture_checksum_change(tmp_path: Path) -> None:
    fixture = _write(tmp_path / "fixture.rdat", "\n".join(["RDAT_VERSION\t0.34", "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2"]))
    manifest_path = tmp_path / "fixtures.json"
    manifest_path.write_text(json.dumps({"schema_version": "reactflow-delta-rdat-fixture-manifest-v1", "fixtures": [{"name": "fixture.rdat", "path": str(fixture), "sha256": "0" * 64, "candidate_category": "candidate", "status": "verified_against_release_index"}]}))
    with pytest.raises(RdatParseError, match="checksum"):
        build_rdat_construct_parse_manifest(manifest_path)
