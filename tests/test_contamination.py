"""Unit tests for ``reactflow.contamination`` (C1-1 Task 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.contamination import (
    ContaminationGrouper,
    ContaminationMergeStats,
    UnionFind,
    annotate_records_from_rfam_clan,
    annotate_records_from_split_manifest,
    extract_pdb_chain,
)
from reactflow.data_registry import DataRecord, sequence_checksum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    record_id: str,
    sequence: str = "ACGUACGU",
    source: str = "test",
    source_id: str = "1",
    parent_id=None,
    family=None,
    clan=None,
    sequence_cluster=None,
    probe="none",
    **kwargs,
) -> DataRecord:
    return DataRecord(
        record_id=record_id,
        sequence=sequence,
        checksum=sequence_checksum(sequence),
        source=source,
        source_version="1.0",
        source_id=source_id,
        parent_id=parent_id,
        family=family,
        clan=clan,
        sequence_cluster=sequence_cluster,
        probe=probe,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_singleton(self):
        uf = UnionFind()
        assert uf.find("a") == "a"
        assert uf.num_groups() == 1

    def test_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.num_groups() == 1

    def test_no_union(self):
        uf = UnionFind()
        uf.find("a")
        uf.find("b")
        assert uf.find("a") != uf.find("b")
        assert uf.num_groups() == 2

    def test_transitive(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")
        assert uf.num_groups() == 1

    def test_lexicographic_root(self):
        uf = UnionFind()
        uf.union("c", "a")
        assert uf.find("c") == "a"
        uf.union("b", "c")
        assert uf.find("b") == "a"

    def test_deterministic_order(self):
        # Same unions in different order -> same root
        uf1 = UnionFind()
        uf1.union("b", "c")
        uf1.union("a", "b")
        uf2 = UnionFind()
        uf2.union("a", "b")
        uf2.union("b", "c")
        assert uf1.find("a") == uf2.find("a") == uf1.find("c") == uf2.find("c")

    def test_union_many(self):
        uf = UnionFind()
        uf.union_many(["a", "b", "c", "d"])
        for x in ["a", "b", "c", "d"]:
            for y in ["a", "b", "c", "d"]:
                assert uf.find(x) == uf.find(y)

    def test_union_many_single(self):
        uf = UnionFind()
        uf.union_many(["a"])
        assert uf.find("a") == "a"

    def test_union_many_empty(self):
        uf = UnionFind()
        uf.union_many([])
        assert uf.num_groups() == 0

    def test_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        comps = uf.components()
        assert len(comps) == 2
        roots = sorted(comps.keys())
        assert comps[roots[0]] == ["a", "b"] or comps[roots[0]] == ["c", "d"]

    def test_group_of(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.group_of("a") == uf.group_of("b")


# ---------------------------------------------------------------------------
# PDB chain extraction
# ---------------------------------------------------------------------------

class TestExtractPdbChain:
    def test_dash(self):
        assert extract_pdb_chain("2N7X-2D") == "2N7X-2D"

    def test_underscore(self):
        assert extract_pdb_chain("1EHZ_A") == "1EHZ-A"

    def test_no_chain(self):
        assert extract_pdb_chain("1EHZ") == "1EHZ"

    def test_non_pdb(self):
        assert extract_pdb_chain("RF02271.fa.csv_1") is None

    def test_empty(self):
        assert extract_pdb_chain("") is None


# ---------------------------------------------------------------------------
# ContaminationGrouper
# ---------------------------------------------------------------------------

class TestContaminationGrouper:
    def test_add_record(self):
        g = ContaminationGrouper()
        r = _make_record("a")
        g.add_record(r)
        assert "a" in g.records
        assert g.group_of("a") == "a"

    def test_add_records(self):
        g = ContaminationGrouper()
        records = [_make_record("a"), _make_record("b")]
        g.add_records(records)
        assert len(g.records) == 2

    def test_merge_exact_sequences(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="ACGU"),
            _make_record("c", sequence="AUGG"),
        ])
        g.merge_exact_sequences()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")
        assert g.stats.exact_sequence == 1

    def test_merge_t_u_equivalent(self):
        # T and U normalize to the same checksum
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="ACGT"),
        ])
        g.merge_exact_sequences()
        # T->U normalization happens in DataRecord construction, so both
        # records have the same checksum.
        assert g.same_group("a", "b")

    def test_merge_parent_windows(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", parent_id="parent1"),
            _make_record("b", parent_id="parent1"),
            _make_record("c", parent_id="parent2"),
        ])
        g.merge_parent_windows()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_mmseqs_clusters(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence_cluster="cl1"),
            _make_record("b", sequence_cluster="cl1"),
            _make_record("c", sequence_cluster="cl2"),
        ])
        g.merge_mmseqs_clusters()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_rfam_family(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", family="RF00001"),
            _make_record("b", family="RF00001"),
            _make_record("c", family="RF00002"),
        ])
        g.merge_rfam_family()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_rfam_clan(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", clan="CL00001"),
            _make_record("b", clan="CL00001"),
            _make_record("c", clan="CL00002"),
        ])
        g.merge_rfam_clan()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_pdb_chains(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", source_id="2N7X-2D"),
            _make_record("b", source_id="2N7X-2D"),
            _make_record("c", source_id="1EHZ-A"),
        ])
        g.merge_pdb_chains()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_probing_constructs(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", parent_id="p1", probe="DMS"),
            _make_record("b", parent_id="p1", probe="DMS"),
            _make_record("c", parent_id="p1", probe="2A3"),
        ])
        g.merge_probing_constructs()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_merge_all(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU", parent_id="p1", family="RF00001"),
            _make_record("b", sequence="ACGU", parent_id="p1", family="RF00001"),
            _make_record("c", sequence="GGGG", parent_id="p2", family="RF00002"),
        ])
        g.merge_all()
        assert g.same_group("a", "b")
        assert not g.same_group("a", "c")

    def test_groups(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="ACGU"),
            _make_record("c", sequence="GGGG"),
        ])
        g.merge_exact_sequences()
        groups = g.groups()
        assert len(groups) == 2

    def test_num_groups(self):
        g = ContaminationGrouper()
        # Use different sequences so they don't merge by exact sequence.
        g.add_records([
            _make_record("a", sequence="ACGUACGU"),
            _make_record("b", sequence="GGGGGGGG"),
        ])
        assert g.num_groups() == 2
        g.merge_exact_sequences()
        assert g.num_groups() == 2  # different sequences stay separate

    def test_split_overlap_pass(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="GGGG"),
        ])
        g.merge_exact_sequences()
        overlap = g.split_overlap({"train": ["a"], "test": ["b"]})
        assert overlap == {}

    def test_split_overlap_fail(self):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="ACGU"),
        ])
        g.merge_exact_sequences()
        overlap = g.split_overlap({"train": ["a"], "test": ["b"]})
        assert "train" in overlap
        assert "test" in overlap["train"]

    def test_to_jsonl(self, tmp_path: Path):
        g = ContaminationGrouper()
        g.add_records([
            _make_record("a", sequence="ACGU"),
            _make_record("b", sequence="ACGU"),
            _make_record("c", sequence="GGGG"),
        ])
        g.merge_exact_sequences()
        path = tmp_path / "groups.jsonl"
        n = g.to_jsonl(path)
        assert n == 2
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        # Groups are sorted by group_id
        assert entries[0]["group_id"] < entries[1]["group_id"]

    def test_stats_dict(self):
        g = ContaminationGrouper()
        g.add_records([_make_record("a"), _make_record("b")])
        g.merge_exact_sequences()
        d = g.stats_dict()
        assert d["total_records"] == 2
        assert d["total_groups"] >= 1
        assert "merge_stats" in d


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

class TestAnnotateFromRfamClan:
    def test_annotate(self):
        records = {
            "a": _make_record("a", family="RF00001"),
            "b": _make_record("b", family="RF00002"),
            "c": _make_record("c", family=None),
        }
        family_to_clan = {"RF00001": "CL00001", "RF00002": "CL00001"}
        n = annotate_records_from_rfam_clan(records, family_to_clan)
        assert n == 2
        assert records["a"].clan == "CL00001"
        assert records["b"].clan == "CL00001"
        assert records["c"].family is None

    def test_does_not_overwrite(self):
        records = {
            "a": _make_record("a", family="RF00001", clan="CL00099"),
        }
        annotate_records_from_rfam_clan(records, {"RF00001": "CL00001"})
        # Should not overwrite existing clan
        assert records["a"].clan == "CL00099"


class TestAnnotateFromSplitManifest:
    def test_annotate(self, tmp_path: Path):
        # The split manifest uses source_id as record_id.
        records = {"test:a": _make_record("test:a", source_id="RF00001.fa.csv_1")}
        manifest = {
            "assignments": [
                {
                    "record_id": "RF00001.fa.csv_1",
                    "cluster": "cl1",
                    "clan": "CL00001",
                    "split": "train",
                },
            ]
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        n = annotate_records_from_split_manifest(records, path)
        assert n == 1
        assert records["test:a"].sequence_cluster == "cl1"
        assert records["test:a"].clan == "CL00001"
