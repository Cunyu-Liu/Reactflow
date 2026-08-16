from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reactflow.delta.pairing import RdatParseError, build_rmdb_construct_audit, classify_mutation_labels
from reactflow.delta.rdat import build_rdat_construct_parse_manifest


def _write_fixture(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "RDAT_VERSION\t0.34", "NAME\tfixture", "SEQUENCE\tACGU", "OFFSET\t0",
                "ANNOTATION\texperimentType:MutateAndMap\tmodifier:DMS",
                "ANNOTATION_DATA:1\tmutation:WT\tsequence:ACGU",
                "ANNOTATION_DATA:2\tmutation:A1G\tsequence:GCGU",
                "ANNOTATION_DATA:3\tmutation:C2X\tsequence:ACGU",
                "ANNOTATION_DATA:4\tmutation:A1G,C2U\tsequence:GCGU",
                "ANNOTATION_DATA:5\tsequence:GCGU",
                "SEQPOS\tA1\tC2\tG3\tU4",
                "REACTIVITY:1\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:2\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:3\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:4\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:5\t0.1\t0.2\t0.3\t0.4",
            ]
        )
    )
    return path


def _fixture_manifest(path: Path, fixture: Path, digest: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "reactflow-delta-rdat-fixture-manifest-v1",
                "fixtures": [{"name": fixture.name, "path": str(fixture), "sha256": digest or hashlib.sha256(fixture.read_bytes()).hexdigest(), "candidate_category": "candidate", "status": "verified_against_release_index"}],
            }
        )
    )
    return path


def test_classify_mutation_labels_distinguishes_unknown_endpoints() -> None:
    assert classify_mutation_labels(["WT"])["mutation_class"] == "explicit_wt"
    assert classify_mutation_labels(["A1G"])["mutation_class"] == "single_exact_endpoint"
    assert classify_mutation_labels(["C2X"])["mutation_class"] == "single_site_endpoint_unknown"
    assert classify_mutation_labels(["A1G,C2U"])["mutation_class"] == "double_exact_endpoint"
    assert classify_mutation_labels([])["mutation_class"] == "mutation_annotation_missing"


def test_construct_audit_counts_profiles_without_promoting_pairs(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "fixture.rdat")
    fixture_manifest = _fixture_manifest(tmp_path / "fixtures.json", fixture)
    parsed_manifest = tmp_path / "parsed.json"
    parsed_manifest.write_text(json.dumps(build_rdat_construct_parse_manifest(fixture_manifest)))

    audit = build_rmdb_construct_audit(fixture_manifest, parsed_manifest)

    assert audit["summary"]["profile_record_count"] == 5
    assert audit["summary"]["explicit_wt_profile_count"] == 1
    assert audit["summary"]["confirmed_single_mutant_profile_count"] == 1
    assert audit["summary"]["single_site_endpoint_unknown_profile_count"] == 1
    assert audit["summary"]["confirmed_double_mutant_profile_count"] == 1
    assert audit["summary"]["same_sequence_replicate_group_count"] == 2
    assert audit["summary"]["pair_eligible_profile_count"] == 0
    assert audit["construct_records"][1]["probe"] == "DMS"
    assert audit["construct_records"][1]["pair_ineligibility_reason"] == "parent lineage is not independently established"


def test_construct_audit_fails_closed_when_fixture_bytes_change(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "fixture.rdat")
    fixture_manifest = _fixture_manifest(tmp_path / "fixtures.json", fixture, digest="0" * 64)
    parsed_manifest = tmp_path / "parsed.json"
    parsed_manifest.write_text(json.dumps({"schema_version": "reactflow-delta-rdat-construct-parse-manifest-v1", "fixtures": [{"name": fixture.name, "sha256": "0" * 64, "profiles": [], "global_annotations": []}]}))
    with pytest.raises(RdatParseError, match="checksum"):
        build_rmdb_construct_audit(fixture_manifest, parsed_manifest)
