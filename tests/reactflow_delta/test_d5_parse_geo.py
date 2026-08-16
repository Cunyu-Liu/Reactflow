#!/usr/bin/env python3
"""Unit tests for d5_parse_geo.py.

Tests use tiny synthetic gzipped files to verify parsing logic without
requiring the real (multi-GB) GEO datasets. Run with:

    cd /home/cunyuliu/reactflow_delta_goal_20260729
    /home/cunyuliu/miniconda3/envs/editflow/bin/python -m pytest \
        tests/reactflow_delta/test_d5_parse_geo.py -v
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "reactflow_delta"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import d5_parse_geo as d5  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic gzipped files mimicking real GEO data
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_gse114002_dir(tmp_path: Path) -> Path:
    """Create a tiny GSE114002-like directory with one CSV.gz file."""
    raw_dir = tmp_path / "GSE114002"
    raw_dir.mkdir(parents=True)
    # Build a CSV with header + 3 rows (one with non-ACGT, one valid, one empty rl)
    csv_content = (
        ",utr,0,1,2,3,4,5,6,7,8,9,10,11,12,13,"
        "total_reads,total,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,rl\n"
        "1,ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC,0.1,0.1,0.1,0.1,"
        "0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,100.0,0.001,"
        "0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,5.5\n"
        "2,ACGTNCGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC,0.1,0.1,0.1,0.1,"
        "0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,100.0,0.001,"
        "0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,3.3\n"
        "3,TTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTT,0.1,0.1,0.1,0.1,"
        "0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,50.0,0.001,"
        "0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,\n"
    )
    out_file = raw_dir / "GSM3130435_egfp_unmod_1.csv.gz"
    with gzip.open(out_file, "wt", encoding="utf-8") as fh:
        fh.write(csv_content)
    # Also write a manifest.json for provenance testing
    manifest = {
        "provider": "GEO",
        "accession": "GSE114002",
        "source_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114002/suppl/",
        "retrieved_at_utc": "2026-07-28T05:21:09+00:00",
        "files": [
            {"name": "GSM3130435_egfp_unmod_1.csv.gz", "sha256": "abc123",
             "bytes": 100, "downloaded": True, "expected_bytes": 100}
        ],
        "skipped": [],
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path  # return raw_dir parent


@pytest.fixture
def tmp_gse145046_dir(tmp_path: Path) -> Path:
    """Create a tiny GSE145046-like directory with two condition TXT.gz files."""
    raw_dir = tmp_path / "GSE145046"
    raw_dir.mkdir(parents=True)
    # File 1: Monosome rep1
    content1 = "ACGTACGTAC\t100\t10.0\nTTTTAAAACC\t50\t5.0\nACGTACGTAC\t80\t8.0\n"
    with gzip.open(raw_dir / "GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz", "wt") as fh:
        fh.write(content1)
    # File 2: Polysome rep1
    content2 = "ACGTACGTAC\t200\t20.0\nTTTTAAAACC\t10\t1.0\nGGGGCCCCAT\t30\t3.0\n"
    with gzip.open(raw_dir / "GSM4305124_3_read_count_In_vivo_Polysome_rep1.txt.gz", "wt") as fh:
        fh.write(content2)
    # File 3: Half_life_2h rep1
    content3 = "ACGTACGTAC\t90\t9.0\nTTTTAAAACC\t60\t6.0\n"
    with gzip.open(raw_dir / "GSM4305139_18_read_count_In_vivo_Half_life_2h_rep1.txt.gz", "wt") as fh:
        fh.write(content3)
    # File 4: Half_life_5h rep1
    content4 = "ACGTACGTAC\t110\t11.0\nTTTTAAAACC\t40\t4.0\n"
    with gzip.open(raw_dir / "GSM4305140_19_read_count_In_vivo_Half_life_5h_rep1.txt.gz", "wt") as fh:
        fh.write(content4)
    # Manifest
    manifest = {
        "provider": "GEO",
        "accession": "GSE145046",
        "source_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145046/suppl/",
        "retrieved_at_utc": "2026-07-28T00:49:28+00:00",
        "files": [],
        "skipped": [],
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: GSE114002 parsing
# ---------------------------------------------------------------------------

class TestGSE114002Parsing:
    def test_parse_valid_row(self, tmp_gse114002_dir: Path):
        """Valid ACGT row with rl is parsed correctly."""
        path = tmp_gse114002_dir / "GSE114002" / "GSM3130435_egfp_unmod_1.csv.gz"
        records = list(d5.parse_gse114002_file(
            path, "random_50mer", "unmodified", 1, "eGFP"
        ))
        # Row 1 is valid ACGT, row 2 has N (filtered), row 3 has empty rl (filtered)
        assert len(records) == 1
        rec = records[0]
        assert rec["utr"] == "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"
        assert rec["rl"] == 5.5
        assert rec["library"] == "random_50mer"
        assert rec["rna_chemistry"] == "unmodified"
        assert rec["replicate"] == 1
        assert rec["cds"] == "eGFP"
        assert rec["total_reads"] == 100.0
        assert rec["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz"

    def test_non_acgt_filtered(self, tmp_gse114002_dir: Path):
        """Sequences with non-ACGT characters (N, etc.) are skipped."""
        path = tmp_gse114002_dir / "GSE114002" / "GSM3130435_egfp_unmod_1.csv.gz"
        records = list(d5.parse_gse114002_file(
            path, "random_50mer", "unmodified", 1, "eGFP"
        ))
        # The N-containing sequence should not appear
        for rec in records:
            assert "N" not in rec["utr"]

    def test_empty_rl_filtered(self, tmp_gse114002_dir: Path):
        """Rows with empty rl are skipped."""
        path = tmp_gse114002_dir / "GSE114002" / "GSM3130435_egfp_unmod_1.csv.gz"
        records = list(d5.parse_gse114002_file(
            path, "random_50mer", "unmodified", 1, "eGFP"
        ))
        assert len(records) == 1  # only the first row survives

    def test_corrupted_gzip_graceful(self, tmp_path: Path):
        """Trailing gzip corruption is handled gracefully — valid records recovered."""
        raw_dir = tmp_path / "GSE114002"
        raw_dir.mkdir(parents=True)
        # Write a valid CSV, gzip it, then append garbage bytes to corrupt the tail
        csv_content = (
            ",utr,0,1,rl\n"
            "1,ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC,0.1,0.1,5.5\n"
            "2,TTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTT,0.1,0.1,3.3\n"
        )
        import gzip as gz
        with gz.open(raw_dir / "test.csv.gz", "wt") as fh:
            fh.write(csv_content)
        # Append garbage to corrupt the gzip stream tail
        with open(raw_dir / "test.csv.gz", "ab") as fh:
            fh.write(b"\x00" * 100 + b"GARBAGE_NOT_VALID_GZIP")
        records = list(d5.parse_gse114002_file(
            raw_dir / "test.csv.gz", "random_50mer", "unmodified", 1, "eGFP"
        ))
        # Should recover at least the first record before corruption
        assert len(records) >= 1

    def test_full_parse_gse114002(self, tmp_gse114002_dir: Path):
        """Full parse_gse114002 writes records file and returns stats."""
        out_dir = tmp_gse114002_dir / "out"
        out_dir.mkdir()
        stats = d5.parse_gse114002(tmp_gse114002_dir, out_dir)
        assert stats["accession"] == "GSE114002"
        assert stats["total_records"] == 1
        assert stats["distinct_utrs"] == 1
        assert "GSM3130435_egfp_unmod_1.csv.gz" in stats["per_file"]
        # Verify records file
        records_path = out_dir / "d5_gse114002_records.jsonl"
        assert records_path.exists()
        lines = records_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["rl"] == 5.5


# ---------------------------------------------------------------------------
# Tests: GSE145046 parsing
# ---------------------------------------------------------------------------

class TestGSE145046Parsing:
    def test_parse_single_file(self, tmp_gse145046_dir: Path):
        """Parse one TXT.gz file, extract condition and records."""
        path = tmp_gse145046_dir / "GSE145046" / "GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz"
        condition_key, category, replicate, records = d5.parse_gse145046_file(path)
        assert condition_key == "In_vivo_Monosome"
        assert category == "translation"
        assert replicate == 1
        assert len(records) == 3
        assert records[0]["seq"] == "ACGTACGTAC"
        assert records[0]["read_count"] == 100

    def test_condition_extraction_from_filename(self):
        """Condition regex correctly extracts condition from various filename patterns."""
        test_cases = [
            ("GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz", "In_vivo_Monosome"),
            ("GSM4305139_18_read_count_In_vivo_Half_life_2h_rep1.txt.gz", "In_vivo_Half_life_2h"),
            ("GSM4305143_22_read_count_In_vitro_Half_life_0min.txt.gz", "In_vitro_Half_life_0min"),
            ("GSM4546416_read_count_Randomly_synthesized_oligos_rep2.txt.gz", "Randomly_synthesized_oligos"),
        ]
        for filename, expected in test_cases:
            m = d5.GSE145046_CONDITION_RE.search(filename)
            assert m is not None, f"Regex failed for {filename}"
            assert m.group(1) == expected, f"Expected {expected}, got {m.group(1)} for {filename}"

    def test_replicate_extraction(self):
        """Replicate number is extracted from filename suffix."""
        import re
        m = re.search(r"_rep(\d+)\.txt\.gz$", "GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz")
        assert m is not None
        assert int(m.group(1)) == 1

    def test_merge_by_sequence(self, tmp_gse145046_dir: Path):
        """Merging across condition files groups records by sequence."""
        data_dir = tmp_gse145046_dir / "GSE145046"
        file_records = []
        for path in sorted(data_dir.glob("GSM*_read_count_*.txt.gz")):
            ck, cat, rep, records = d5.parse_gse145046_file(path)
            file_records.append((ck, cat, rep, records))
        merged = d5.merge_gse145046_conditions(file_records)
        # ACGTACGTAC appears in Monosome (2 entries: 100, 80), Polysome (200), Half_life_2h (90), Half_life_5h (110)
        assert "ACGTACGTAC" in merged
        seq_conds = merged["ACGTACGTAC"]
        assert "In_vivo_Monosome" in seq_conds
        assert "In_vivo_Polysome" in seq_conds
        assert "In_vivo_Half_life_2h" in seq_conds
        assert "In_vivo_Half_life_5h" in seq_conds
        # Monosome has 2 entries (rep1 only but duplicate seq in same file)
        assert len(seq_conds["In_vivo_Monosome"]) == 2

    def test_derived_labels_te(self):
        """Translation efficiency is computed correctly."""
        conditions = {
            "In_vivo_Monosome": [(1, 100, 10.0)],
            "In_vivo_Polysome": [(1, 300, 30.0)],
            "In_vivo_Half_life_2h": [(1, 90, 9.0)],
            "In_vivo_Half_life_5h": [(1, 110, 11.0)],
        }
        labels = d5.compute_derived_labels("ACGT", conditions)
        # TE = 300 / (100 + 300) = 0.75
        assert abs(labels["te_estimate"] - 0.75) < 1e-6
        # Stability = 110 / (90 + 110) = 0.55
        assert abs(labels["stability_estimate"] - 0.55) < 1e-6
        # Total reads = 100 + 300 + 90 + 110 = 600
        assert labels["total_reads_sum"] == 600.0

    def test_derived_labels_missing_conditions(self):
        """Missing conditions yield None for derived labels."""
        conditions = {
            "In_vivo_Monosome": [(1, 100, 10.0)],
            # No Polysome, no Half_life
        }
        labels = d5.compute_derived_labels("ACGT", conditions)
        assert labels["te_estimate"] is None
        assert labels["stability_estimate"] is None
        assert labels["total_reads_sum"] == 100.0

    def test_full_parse_gse145046(self, tmp_gse145046_dir: Path):
        """Full parse_gse145046 writes records file and returns stats."""
        out_dir = tmp_gse145046_dir / "out"
        out_dir.mkdir()
        stats = d5.parse_gse145046(tmp_gse145046_dir, out_dir)
        assert stats["accession"] == "GSE145046"
        assert stats["total_files_parsed"] == 4
        assert stats["total_records"] > 0
        # Verify records file
        records_path = out_dir / "d5_gse145046_records.jsonl"
        assert records_path.exists()
        lines = records_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == stats["total_records"]
        # Each record should have derived_labels
        for line in lines:
            rec = json.loads(line)
            assert "seq" in rec
            assert "conditions" in rec
            assert "derived_labels" in rec
            assert "n_conditions" in rec


# ---------------------------------------------------------------------------
# Tests: manifest and summary
# ---------------------------------------------------------------------------

class TestManifestAndSummary:
    def test_load_source_manifest_found(self, tmp_gse114002_dir: Path):
        """load_source_manifest reads existing manifest.json."""
        manifest = d5.load_source_manifest(tmp_gse114002_dir, "GSE114002")
        assert manifest["accession"] == "GSE114002"
        assert manifest["source_url"].startswith("https://ftp.ncbi.nlm.nih.gov")
        assert manifest["manifest_found"] is not False

    def test_load_source_manifest_missing(self, tmp_path: Path):
        """load_source_manifest returns not-found dict when manifest missing."""
        manifest = d5.load_source_manifest(tmp_path, "GSE999999")
        assert manifest["manifest_found"] is False

    def test_build_manifest(self, tmp_gse114002_dir: Path):
        """build_manifest combines both dataset manifests."""
        manifest = d5.build_manifest(tmp_gse114002_dir, tmp_gse114002_dir / "out")
        assert manifest["schema_version"] == "reactflow-delta-d5-geo-manifest-v1"
        assert "GSE114002" in manifest["datasets"]
        assert "GSE145046" in manifest["datasets"]
        assert manifest["datasets"]["GSE114002"]["source_url"].startswith("https://")


# ---------------------------------------------------------------------------
# Tests: CLI main
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_gse114002_only(self, tmp_gse114002_dir: Path, monkeypatch):
        """main() with --accession GSE114002 parses only GSE114002."""
        out_dir = tmp_gse114002_dir / "cli_out"
        rc = d5.main([
            "--raw-dir", str(tmp_gse114002_dir),
            "--out-dir", str(out_dir),
            "--accession", "GSE114002",
        ])
        assert rc == 0
        assert (out_dir / "d5_gse114002_records.jsonl").exists()
        assert (out_dir / "d5_geo_manifest.json").exists()
        assert (out_dir / "d5_geo_summary.json").exists()
        # GSE145046 should NOT be parsed
        assert not (out_dir / "d5_gse145046_records.jsonl").exists()
        summary = json.loads((out_dir / "d5_geo_summary.json").read_text())
        assert "GSE114002" in summary["datasets"]
        assert "GSE145046" not in summary["datasets"]

    def test_main_all(self, tmp_gse145046_dir: Path, monkeypatch):
        """main() with --accession all parses both datasets."""
        # Also need GSE114002 dir for full 'all' run
        gse114002_dir = tmp_gse145046_dir / "GSE114002"
        gse114002_dir.mkdir(parents=True)
        # Write minimal GSE114002 file
        csv_content = (
            ",utr,0,1,rl\n1,ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC,"
            "0.1,0.1,5.5\n"
        )
        with gzip.open(gse114002_dir / "GSM3130435_egfp_unmod_1.csv.gz", "wt") as fh:
            fh.write(csv_content)
        (gse114002_dir / "manifest.json").write_text(json.dumps({
            "accession": "GSE114002", "source_url": "https://example.com",
            "files": [], "skipped": []
        }))

        out_dir = tmp_gse145046_dir / "cli_out"
        rc = d5.main([
            "--raw-dir", str(tmp_gse145046_dir),
            "--out-dir", str(out_dir),
            "--accession", "all",
        ])
        assert rc == 0
        assert (out_dir / "d5_gse114002_records.jsonl").exists()
        assert (out_dir / "d5_gse145046_records.jsonl").exists()
