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


def test_parser_treats_empty_inf_na_as_missing(tmp_path: Path) -> None:
    # _numeric_values: empty strings, Inf, -Inf, N/A, NA are treated as missing
    # (None), not parse errors. Same handling as NaN.
    path = _write(
        tmp_path / "edge.rdat",
        "\n".join([
            "RDAT_VERSION\t0.34",
            "NAME\tedge",
            "SEQUENCE\tACGU",
            "OFFSET\t0",
            "SEQPOS\tA1\tC2\tG3\tU4",
            "REACTIVITY:1\t0.1\t\tInf\tN/A",
        ]),
    )
    document = parse_rdat(path)
    reactivity = document["profiles"][0]["reactivity"]
    assert reactivity[0] == 0.1
    assert reactivity[1] is None  # empty string
    assert reactivity[2] is None  # Inf
    assert reactivity[3] is None  # N/A
    assert document["profiles"][0]["missing_reactivity_count"] == 3


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
        "\n".join(["RDAT_VERSION\t0.99", "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2"]),
    )
    with pytest.raises(RdatParseError, match="not accepted"):
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


# ---------------------------------------------------------------------------
# D1 §5 parser extensions (v3.1 forward-only)
# ---------------------------------------------------------------------------


class TestVersionAlias:
    """§5.1: VERSION as alias for RDAT_VERSION (TRP4P6 files)."""

    def test_version_aliased_to_rdat_version(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "alias.rdat",
            "\n".join([
                "VERSION\t0.34", "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0",
                "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["RDAT_VERSION"] == "0.34"
        assert document["headers"]["VERSION"] == "0.34"

    def test_version_and_rdat_version_agree(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "both.rdat",
            "\n".join([
                "VERSION\t0.34", "RDAT_VERSION\t0.34",
                "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0",
                "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["RDAT_VERSION"] == "0.34"

    def test_version_and_rdat_version_conflict_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "conflict.rdat",
            "\n".join([
                "VERSION\t0.34", "RDAT_VERSION\t0.4",
                "NAME\tx", "SEQUENCE\tAC", "OFFSET\t0",
                "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2",
            ]),
        )
        with pytest.raises(RdatParseError, match="conflicting"):
            parse_rdat(path)


class TestAcceptedRdatVersions:
    """§5.2: accept RDAT_VERSION 0.4 / 0.22 / 0.24 / 0.33 / 0.32."""

    @pytest.mark.parametrize("version", ["0.4", "0.22", "0.24", "0.33", "0.32"])
    def test_accepted_version_parses(self, tmp_path: Path, version: str) -> None:
        path = _write(
            tmp_path / f"v{version}.rdat",
            "\n".join([
                f"RDAT_VERSION\t{version}", "NAME\tx", "SEQUENCE\tAC",
                "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["RDAT_VERSION"] == version

    def test_version_0_33_now_accepted(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "v033.rdat",
            "\n".join([
                "RDAT_VERSION\t0.33", "NAME\tx", "SEQUENCE\tAC",
                "OFFSET\t0", "SEQPOS\tA1\tC2", "REACTIVITY:1\t0.1\t0.2",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["RDAT_VERSION"] == "0.33"


class TestSpaceSeparatedFields:
    """§5.3: GLYCFN-style files use spaces instead of tabs."""

    def test_space_separated_file_parses(self, tmp_path: Path) -> None:
        # Minimal GLYCFN-style file: spaces instead of tabs everywhere.
        path = _write(
            tmp_path / "glycfn.rdat",
            "\n".join([
                "RDAT_VERSION 0.24",
                "NAME glycine riboswitch, F. nucleatum",
                "SEQUENCE ggaaauaaUCGGAUGAAGAUAUGAGGAGAGA",
                "STRUCTURE .........(((......)))......",
                "OFFSET -17",
                "SEQPOS -16 -15 -14 0 1 2 3 4 5",
                "ANNOTATION\ttemperature:24C\tmodifier:DMS",
                "ANNOTATION_DATA:1 modifier:DMS",
                "ANNOTATION_DATA:2 mutation:WT modifier:DMS",
                "REACTIVITY:1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
                "REACTIVITY:2 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["RDAT_VERSION"] == "0.24"
        assert document["headers"]["NAME"] == "glycine riboswitch, F. nucleatum"
        assert len(document["seqpos"]) == 9
        assert len(document["profiles"]) == 2
        assert document["profiles"][0]["reactivity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # ANNOTATION line still uses tabs → annotation tokens parsed correctly
        assert document["global_annotations"][0]["temperature"] == ["24C"]
        # ANNOTATION_DATA uses spaces → tokens parsed correctly
        assert document["profiles"][0]["annotation"]["modifier"] == ["DMS"]
        assert document["profiles"][1]["annotation"]["mutation"] == ["WT"]

    def test_space_separated_name_with_spaces_preserved(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "name.rdat",
            "\n".join([
                "RDAT_VERSION 0.34",
                "NAME some rna with multiple words",
                "SEQUENCE ACGU",
                "OFFSET 0",
                "SEQPOS A1 C2 G3 U4",
                "REACTIVITY:1 0.1 0.2 0.3 0.4",
            ]),
        )
        document = parse_rdat(path)
        assert document["headers"]["NAME"] == "some rna with multiple words"

    def test_mixed_tab_and_space_lines(self, tmp_path: Path) -> None:
        # ANNOTATION uses tabs, ANNOTATION_DATA uses spaces (real GLYCFN pattern)
        path = _write(
            tmp_path / "mixed.rdat",
            "\n".join([
                "RDAT_VERSION 0.24",
                "NAME\tx",
                "SEQUENCE\tACGU",
                "OFFSET\t0",
                "SEQPOS A1 C2 G3 U4",
                "ANNOTATION\tmodifier:DMS\tchemical:MgCl2:10mM",
                "ANNOTATION_DATA:1 modifier:DMS chemical:glycine:10mM",
                "REACTIVITY:1 0.1 0.2 0.3 0.4",
            ]),
        )
        document = parse_rdat(path)
        assert document["global_annotations"][0]["chemical"] == ["MgCl2:10mM"]
        assert document["profiles"][0]["annotation"]["modifier"] == ["DMS"]
        assert document["profiles"][0]["annotation"]["chemical"] == ["glycine:10mM"]
