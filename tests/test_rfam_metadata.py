"""Tests for Rfam/MMseqs metadata construction."""

import gzip
import json
import subprocess

import pytest

import reactflow.rfam_metadata as rfam_metadata
from reactflow.rfam_metadata import (
    build_metadata_rows,
    build_rfam_metadata,
    extract_rfam_accession,
    read_cache_metadata_records,
    read_rfam_clan_membership,
    resolve_sequence_clusters,
)
from reactflow.splits import read_split_metadata_tsv


def _write_cache(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_rfam_accession_from_source_id():
    assert extract_rfam_accession("RF02271.fa.csv_1") == "RF02271"
    assert extract_rfam_accession("window:RF00002:12") == "RF00002"
    assert extract_rfam_accession("not_rfam") is None


def test_read_rfam_clan_membership_plain_and_gzip(tmp_path):
    plain = tmp_path / "clan_membership.txt"
    plain.write_text("CL00112\tRF00002\nCL00010\tRF00163\n", encoding="utf-8")
    assert read_rfam_clan_membership(plain) == {"RF00002": "CL00112", "RF00163": "CL00010"}

    gz_path = tmp_path / "clan_membership.txt.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("CL00106\tRF00018\n")
    assert read_rfam_clan_membership(gz_path) == {"RF00018": "CL00106"}


def test_build_rfam_metadata_emits_split_compatible_tsv(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(
        cache,
        [
            {"source_id": "RF00002.fa.csv_1", "sequence": "AUGC"},
            {"source_id": "RF99999.fa.csv_1", "sequence": "CCCC"},
            {"source_id": "RF99999.fa.csv_1", "sequence": "GGGG"},
        ],
    )
    clan_membership = tmp_path / "clan_membership.txt"
    clan_membership.write_text("CL00112\tRF00002\n", encoding="utf-8")

    output = tmp_path / "metadata.tsv"
    summary = build_rfam_metadata(
        [cache],
        output,
        clan_membership_path=clan_membership,
        download_rfam=False,
        cluster_method="exact",
    )

    parsed = read_split_metadata_tsv(output)
    assert parsed["RF00002.fa.csv_1"][0] == "CL00112"
    assert parsed["RF99999.fa.csv_1"][0] == "RF99999"
    assert parsed["RF99999.fa.csv_1#2"][0] == "RF99999"
    assert summary.input_records == 3
    assert summary.records_with_rfam_clan == 1
    assert summary.records_with_family_fallback == 2
    assert summary.cluster_method == "exact"
    assert (tmp_path / "metadata.manifest.json").exists()


def test_cluster_component_merges_rfam_groups_when_sequences_match(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(
        cache,
        [
            {"source_id": "RF00002.fa.csv_1", "sequence": "AUGC"},
            {"source_id": "RF00163.fa.csv_1", "sequence": "AUGC"},
        ],
    )
    records = read_cache_metadata_records([cache])
    rows = build_metadata_rows(
        records,
        rfam_to_clan={"RF00002": "CL00112", "RF00163": "CL00010"},
        clusters={record.record_id: "exact:same" for record in records},
    )

    assert rows[0].cluster == rows[1].cluster
    assert rows[0].clan == rows[1].clan
    assert rows[0].clan.startswith("component:")
    assert {row.rfam_group for row in rows} == {"CL00112", "CL00010"}


def test_python_identity_clusters_near_identical_records(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(
        cache,
        [
            {"source_id": "RF00002.fa.csv_1", "sequence": "ACGUACGU"},
            {"source_id": "RF00163.fa.csv_1", "sequence": "ACGUACGA"},
            {"source_id": "RF99999.fa.csv_1", "sequence": "GGGGGGGG"},
        ],
    )
    clan_membership = tmp_path / "clan_membership.txt"
    clan_membership.write_text("CL00112\tRF00002\nCL00010\tRF00163\n", encoding="utf-8")

    summary = build_rfam_metadata(
        [cache],
        tmp_path / "metadata.tsv",
        clan_membership_path=clan_membership,
        download_rfam=False,
        cluster_method="python-identity",
        mmseqs_min_seq_id=0.875,
        mmseqs_coverage=1.0,
        python_identity_max_records=10,
    )
    rows = read_split_metadata_tsv(tmp_path / "metadata.tsv")

    assert summary.cluster_method == "python-identity"
    assert summary.cluster_count == 2
    assert rows["RF00002.fa.csv_1"][0] == rows["RF00163.fa.csv_1"][0]
    assert rows["RF00002.fa.csv_1"][0].startswith("component:")
    assert rows["RF99999.fa.csv_1"][0] == "RF99999"


def test_python_identity_refuses_large_inputs(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(
        cache,
        [
            {"source_id": "a", "sequence": "AAAA"},
            {"source_id": "b", "sequence": "AAAU"},
        ],
    )
    records = read_cache_metadata_records([cache])

    with pytest.raises(ValueError, match="quadratic"):
        resolve_sequence_clusters(
            records,
            method="python-identity",
            python_identity_max_records=1,
        )


def test_auto_mmseqs_failure_falls_back_to_exact_with_error_tail(monkeypatch, tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(cache, [{"source_id": "a", "sequence": "AAAA"}])
    records = read_cache_metadata_records([cache])

    def fail_mmseqs(*args, **kwargs):
        raise subprocess.CalledProcessError(
            9,
            ("mmseqs", "easy-cluster"),
            output="stdout details",
            stderr="stderr details",
        )

    monkeypatch.setattr(rfam_metadata.shutil, "which", lambda _name: "/fake/mmseqs")
    monkeypatch.setattr(rfam_metadata, "_run_mmseqs_clusters", fail_mmseqs)

    clusters, method, command, error = resolve_sequence_clusters(records, method="auto")

    assert method == "exact"
    assert command is None
    assert clusters == {"a": clusters["a"]}
    assert "returncode=9" in error
    assert "stdout details" in error
    assert "stderr details" in error


def test_strict_mmseqs_failure_raises_with_error_tail(monkeypatch, tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(cache, [{"source_id": "a", "sequence": "AAAA"}])
    records = read_cache_metadata_records([cache])

    def fail_mmseqs(*args, **kwargs):
        raise subprocess.CalledProcessError(
            11,
            ("mmseqs", "easy-cluster"),
            output="bad stdout",
            stderr="bad stderr",
        )

    monkeypatch.setattr(rfam_metadata, "_run_mmseqs_clusters", fail_mmseqs)

    with pytest.raises(RuntimeError, match="returncode=11.*bad stdout.*bad stderr"):
        resolve_sequence_clusters(records, method="mmseqs", mmseqs_bin="/fake/mmseqs")


def test_resolve_sequence_clusters_rejects_unknown_method(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_cache(cache, [{"source_id": "a", "sequence": "AAAA"}])
    records = read_cache_metadata_records([cache])

    with pytest.raises(ValueError, match="auto, exact, mmseqs, python-identity"):
        resolve_sequence_clusters(records, method="bad")
