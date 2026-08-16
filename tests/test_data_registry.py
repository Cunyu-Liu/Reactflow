"""Unit tests for ``reactflow.data_registry`` (C1-1 Task 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.data_registry import (
    CANONICAL_PAIRS,
    WOBBLE_PAIRS,
    DataRecord,
    DataSourceSpec,
    KNOWN_SOURCES,
    RegistryStats,
    canonicalize_sequence,
    classify_pair,
    detect_pseudoknots,
    iter_jsonl,
    load_cache_file,
    normalize_probe_name,
    normalize_source_name,
    sequence_checksum,
)


# ---------------------------------------------------------------------------
# Sequence canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalizeSequence:
    def test_uppercase(self):
        assert canonicalize_sequence("acgu") == "ACGU"

    def test_t_to_u(self):
        assert canonicalize_sequence("ACGT") == "ACGU"
        assert canonicalize_sequence("tttt") == "UUUU"

    def test_strip_gaps(self):
        assert canonicalize_sequence("A.C-G-U") == "ACGU"
        assert canonicalize_sequence("A..G--") == "AG"

    def test_replace_non_acgu_with_n(self):
        assert canonicalize_sequence("ACGXR") == "ACGNN"
        assert canonicalize_sequence("acg123") == "ACGNNN"

    def test_empty(self):
        assert canonicalize_sequence("") == ""

    def test_preserves_length_when_no_gaps(self):
        s = "ACGUACGU"
        assert len(canonicalize_sequence(s)) == len(s)

    def test_idempotent(self):
        s = "ACGT-.X"
        once = canonicalize_sequence(s)
        twice = canonicalize_sequence(once)
        assert once == twice


class TestSequenceChecksum:
    def test_deterministic(self):
        assert sequence_checksum("ACGU") == sequence_checksum("ACGU")

    def test_t_u_equivalent(self):
        assert sequence_checksum("ACGT") == sequence_checksum("ACGU")

    def test_case_insensitive(self):
        assert sequence_checksum("acgu") == sequence_checksum("ACGU")

    def test_gap_independent(self):
        assert sequence_checksum("A.C-G-U") == sequence_checksum("ACGU")

    def test_returns_hex(self):
        h = sequence_checksum("ACGU")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Probe / source normalization
# ---------------------------------------------------------------------------

class TestNormalizeProbeName:
    def test_lowercase(self):
        assert normalize_probe_name("dms") == "DMS"
        assert normalize_probe_name("2a3") == "2A3"

    def test_none(self):
        assert normalize_probe_name(None) == "none"

    def test_empty(self):
        assert normalize_probe_name("") == "none"

    def test_spaces(self):
        assert normalize_probe_name(" DMS ") == "DMS"

    def test_unknown_passthrough(self):
        assert normalize_probe_name("CMCT") == "CMCT"


class TestNormalizeSourceName:
    def test_aliases(self):
        assert normalize_source_name("efold") == "efold_train"
        assert normalize_source_name("PDB") == "PDB"
        assert normalize_source_name("archiveii") == "ArchiveII"

    def test_none(self):
        assert normalize_source_name(None) == "unknown"

    def test_unknown_passthrough(self):
        assert normalize_source_name("NewSource") == "NewSource"


# ---------------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------------

class TestClassifyPair:
    def test_canonical(self):
        assert classify_pair("A", "U") == "AU"
        assert classify_pair("U", "A") == "UA"
        assert classify_pair("G", "C") == "GC"
        assert classify_pair("C", "G") == "CG"

    def test_wobble(self):
        assert classify_pair("G", "U") == "GU"
        assert classify_pair("U", "G") == "UG"

    def test_noncanonical(self):
        assert classify_pair("A", "A") == "XX"
        assert classify_pair("A", "C") == "XX"
        assert classify_pair("G", "G") == "XX"

    def test_t_normalized(self):
        assert classify_pair("A", "T") == "AU"
        assert classify_pair("T", "A") == "UA"

    def test_n_is_noncanonical(self):
        assert classify_pair("N", "U") == "XX"
        assert classify_pair("A", "N") == "XX"

    def test_lowercase(self):
        assert classify_pair("a", "u") == "AU"


# ---------------------------------------------------------------------------
# Pseudoknot detection
# ---------------------------------------------------------------------------

class TestDetectPseudoknots:
    def test_no_crossings(self):
        # Nested hairpin: (0,9), (1,8) — no crossing
        pairs = [(0, 9), (1, 8)]
        assert detect_pseudoknots(pairs) == ()

    def test_crossing(self):
        # (0,5) and (2,7) cross: 0 < 2 < 5 < 7
        pairs = [(0, 5), (2, 7)]
        result = detect_pseudoknots(pairs)
        assert (0, 5) in result
        assert (2, 7) in result

    def test_single_pair_no_crossing(self):
        assert detect_pseudoknots([(0, 5)]) == ()

    def test_empty(self):
        assert detect_pseudoknots([]) == ()

    def test_self_pair_ignored(self):
        # Self-pairs are filtered; the remaining single pair has no crossing.
        assert detect_pseudoknots([(3, 3), (0, 5)]) == ()

    def test_unordered_input_normalized(self):
        # Input (5, 0) should be normalized to (0, 5) and cross with (2, 7)
        pairs = [(5, 0), (2, 7)]
        result = detect_pseudoknots(pairs)
        assert (0, 5) in result
        assert (2, 7) in result

    def test_multiple_crossings(self):
        # Three mutually crossing pairs
        pairs = [(0, 5), (2, 7), (4, 9)]
        result = detect_pseudoknots(pairs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# DataRecord
# ---------------------------------------------------------------------------

class TestDataRecordConstruction:
    def _make_record(self, **kwargs):
        defaults = dict(
            record_id="test:1",
            sequence="ACGUACGU",
            checksum=sequence_checksum("ACGUACGU"),
            source="test",
            source_version="1.0",
            source_id="1",
        )
        defaults.update(kwargs)
        return DataRecord(**defaults)

    def test_basic_construction(self):
        r = self._make_record()
        assert r.record_id == "test:1"
        assert r.sequence == "ACGUACGU"
        assert r.length() == 8

    def test_frozen(self):
        r = self._make_record()
        with pytest.raises(Exception):
            r.record_id = "other"  # type: ignore

    def test_default_optionals(self):
        r = self._make_record()
        assert r.pairs == ()
        assert r.pair_types == ()
        assert r.parent_id is None
        assert r.reactivity is None

    def test_has_reactivity_false_by_default(self):
        r = self._make_record()
        assert not r.has_reactivity()

    def test_has_reactivity_true(self):
        r = self._make_record(
            reactivity=(0.1, 0.2, 0.3),
            reactivity_source="real_profile",
        )
        assert r.has_reactivity()
        assert r.has_real_profile()

    def test_has_reactivity_proxy(self):
        r = self._make_record(
            reactivity=(0.1, 0.2),
            reactivity_source="structure_forward_proxy",
        )
        assert r.has_reactivity()
        assert not r.has_real_profile()

    def test_has_pseudoknot(self):
        r = self._make_record(
            pairs=((0, 5), (2, 7)),
            pseudoknot_pairs=((0, 5), (2, 7)),
        )
        assert r.has_pseudoknot()

    def test_pair_counts(self):
        r = self._make_record(
            sequence="AUGCAUGC",
            pairs=((0, 7), (1, 6), (2, 3)),
            pair_types=("AU", "UA", "GC"),
        )
        assert r.canonical_pair_count() == 3
        assert r.wobble_pair_count() == 0
        assert r.noncanonical_pair_count() == 0


class TestDataRecordSerialization:
    def test_round_trip(self):
        r = DataRecord(
            record_id="test:1",
            sequence="ACGU",
            checksum=sequence_checksum("ACGU"),
            source="test",
            source_version="1.0",
            source_id="1",
            pairs=((0, 3),),
            pair_types=("AU",),
            pseudoknot_pairs=(),
            reactivity=(0.1, 0.2, 0.3, 0.4),
            reactivity_source="real_profile",
            probe="DMS",
            family="RF00001",
            quality_flags=("windowed",),
        )
        d = r.to_dict()
        assert d["record_id"] == "test:1"
        assert d["pairs"] == [[0, 3]]
        assert d["reactivity"] == [0.1, 0.2, 0.3, 0.4]
        r2 = DataRecord.from_dict(d)
        assert r2 == r

    def test_json_serializable(self):
        r = DataRecord(
            record_id="test:1",
            sequence="ACGU",
            checksum=sequence_checksum("ACGU"),
            source="test",
            source_version="1.0",
            source_id="1",
        )
        d = r.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        r2 = DataRecord.from_dict(d2)
        assert r2 == r


class TestFromCacheRow:
    def test_basic(self):
        row = {
            "sequence": "ACGUACGU",
            "source_id": "test1",
            "pairs": [[0, 7], [1, 6]],
            "probe": "DMS",
            "reactivity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "reactivity_source": "real_profile",
            "length_bucket": "len_le_64",
            "family": None,
        }
        r = DataRecord.from_cache_row(row, source="PDB")
        assert r.source == "PDB"
        assert r.source_id == "test1"
        assert r.sequence == "ACGUACGU"
        assert r.record_id == "PDB:test1"
        assert r.pairs == ((0, 7), (1, 6))
        # seq[0]='A', seq[7]='U' -> AU; seq[1]='C', seq[6]='G' -> CG
        assert r.pair_types == ("AU", "CG")
        assert r.probe == "DMS"
        assert r.has_real_profile()
        assert "no_family" in r.quality_flags

    def test_windowed(self):
        row = {
            "sequence": "ACGUACGU",
            "source_id": "ENSG00000004399.13:0-8",
            "pairs": [],
            "probe": "DMS",
            "reactivity": [0.1] * 8,
            "reactivity_source": "real_profile",
            "length_bucket": "len_le_64",
            "family": None,
            "window": {"start": 0, "end": 8, "index": 0, "parent_length": 100},
        }
        r = DataRecord.from_cache_row(row, source="human_mRNA")
        assert r.parent_id == "ENSG00000004399.13"
        assert r.parent_coordinates == (0, 8)
        assert r.parent_length == 100
        assert r.window_index == 0
        assert "windowed" in r.quality_flags
        assert r.record_id == "human_mRNA:ENSG00000004399.13:0-8:w0"

    def test_t_normalized(self):
        row = {
            "sequence": "ACGTACGT",
            "source_id": "x",
            "pairs": [],
            "probe": "none",
            "reactivity": [],
            "reactivity_source": "none",
            "length_bucket": "len_le_64",
            "family": None,
        }
        r = DataRecord.from_cache_row(row, source="test")
        assert r.sequence == "ACGUACGU"
        assert r.checksum == sequence_checksum("ACGUACGU")

    def test_rfam_extraction(self):
        row = {
            "sequence": "ACGU",
            "source_id": "RF02271.fa.csv_1",
            "pairs": [],
            "probe": "2A3",
            "reactivity": [0.1, 0.2, 0.3, 0.4],
            "reactivity_source": "structure_forward_proxy",
            "length_bucket": "len_le_64",
            "family": None,
        }
        r = DataRecord.from_cache_row(row, source="efold_train")
        assert r.family == "RF02271"
        assert "proxy_reactivity" in r.quality_flags

    def test_self_pair_filtered(self):
        row = {
            "sequence": "ACGU",
            "source_id": "x",
            "pairs": [[1, 1], [0, 3]],
            "probe": "none",
            "reactivity": [],
            "reactivity_source": "none",
            "length_bucket": "len_le_64",
            "family": None,
        }
        r = DataRecord.from_cache_row(row, source="test")
        assert (1, 1) not in r.pairs
        assert (0, 3) in r.pairs

    def test_out_of_range_filtered(self):
        row = {
            "sequence": "ACGU",
            "source_id": "x",
            "pairs": [[0, 100], [0, 3]],
            "probe": "none",
            "reactivity": [],
            "reactivity_source": "none",
            "length_bucket": "len_le_64",
            "family": None,
        }
        r = DataRecord.from_cache_row(row, source="test")
        assert (0, 100) not in r.pairs
        assert (0, 3) in r.pairs


# ---------------------------------------------------------------------------
# RegistryStats
# ---------------------------------------------------------------------------

class TestRegistryStats:
    def test_add(self):
        stats = RegistryStats()
        r = DataRecord(
            record_id="test:1",
            sequence="AUGCAUGC",
            checksum=sequence_checksum("AUGCAUGC"),
            source="PDB",
            source_version="1.0",
            source_id="1",
            pairs=((0, 7), (1, 6)),
            pair_types=("AU", "UA"),
            reactivity=(0.1,) * 8,
            reactivity_source="real_profile",
            family="RF00001",
            length_bucket="len_le_64",
        )
        stats.add(r)
        assert stats.total_records == 1
        assert stats.by_source["PDB"] == 1
        assert stats.with_real_profile == 1
        assert stats.with_family == 1
        assert stats.total_pairs == 2
        assert stats.total_canonical_pairs == 2


# ---------------------------------------------------------------------------
# Known sources
# ---------------------------------------------------------------------------

class TestKnownSources:
    def test_eleven_sources(self):
        # 6 cached + 5 registered-but-not-downloaded (Rfam, Ribonanza,
        # Ribonanza2, bpRNA, RNAStrAlign) per spec lines 248-251.
        assert len(KNOWN_SOURCES) == 11

    def test_unique_names(self):
        names = [s.name for s in KNOWN_SOURCES]
        assert len(names) == len(set(names))

    def test_has_efold_train(self):
        assert any(s.name == "efold_train" for s in KNOWN_SOURCES)

    def test_has_pdb(self):
        assert any(s.name == "PDB" for s in KNOWN_SOURCES)

    def test_has_rfam(self):
        assert any(s.name == "Rfam" for s in KNOWN_SOURCES)

    def test_has_ribonanza(self):
        assert any(s.name == "Ribonanza" for s in KNOWN_SOURCES)

    def test_has_ribonanza2(self):
        assert any(s.name == "Ribonanza2" for s in KNOWN_SOURCES)

    def test_has_bprna(self):
        assert any(s.name == "bpRNA" for s in KNOWN_SOURCES)

    def test_has_rnastralign(self):
        assert any(s.name == "RNAStrAlign" for s in KNOWN_SOURCES)

    def test_cached_sources_marked_downloaded(self):
        cached = [s for s in KNOWN_SOURCES if s.downloaded]
        # 6 cached sources should be marked as downloaded
        assert len(cached) == 6
        for s in cached:
            assert s.name in {"efold_train", "PDB", "ArchiveII", "viral", "lncRNA", "human_mRNA"}

    def test_not_downloaded_sources_have_upstream_url(self):
        for s in KNOWN_SOURCES:
            if not s.downloaded:
                assert s.upstream_url is not None, f"{s.name} missing upstream_url"
                assert s.upstream_url.startswith("http"), f"{s.name} invalid URL"


# ---------------------------------------------------------------------------
# iter_jsonl / load_cache_file
# ---------------------------------------------------------------------------

class TestIterJsonl:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n\n{"c": 3}\n')
        rows = list(iter_jsonl(path))
        assert len(rows) == 3
        assert rows[0] == {"a": 1}
        assert rows[2] == {"c": 3}


class TestLoadCacheFile:
    def test_load(self, tmp_path: Path):
        path = tmp_path / "cache.jsonl"
        path.write_text(json.dumps({
            "sequence": "ACGU",
            "source_id": "x",
            "pairs": [[0, 3]],
            "probe": "DMS",
            "reactivity": [0.1, 0.2, 0.3, 0.4],
            "reactivity_source": "real_profile",
            "length_bucket": "len_le_64",
            "family": None,
        }) + "\n")
        records = list(load_cache_file(path, source="test"))
        assert len(records) == 1
        assert records[0].source == "test"
        assert records[0].pairs == ((0, 3),)

    def test_limit(self, tmp_path: Path):
        path = tmp_path / "cache.jsonl"
        row = json.dumps({
            "sequence": "ACGU",
            "source_id": "x",
            "pairs": [],
            "probe": "none",
            "reactivity": [],
            "reactivity_source": "none",
            "length_bucket": "len_le_64",
            "family": None,
        })
        path.write_text(row + "\n" + row + "\n" + row + "\n")
        records = list(load_cache_file(path, source="test", limit=2))
        assert len(records) == 2
