"""Unit tests for D2-X split/exposure logic (outcome-blind, no data access)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reactflow_delta.d2x_split import build_split, _study_of, _norm_identity


def _rec(study, seq, sa):
    return {
        "source_accession": sa,
        "canonical_sequence": seq,
        "parent_lineage_evidence": {
            "parent_sequence_sha256": "sha_" + seq.replace("U", "U"),
            "design_group": sa,
        },
    }


def _pair(study, file, mi):
    return {
        "source_accession": f"{study}_{file}",
        "file_sha256": f"f_{study}_{file}",
        "mutant_profile_index": mi,
    }


class StudyOfTest(unittest.TestCase):
    def test_study_prefix(self):
        self.assertEqual(_study_of("16SFWJ_1M7_0001"), "16SFWJ")
        self.assertEqual(_study_of("ADD140_RSQ_0001"), "ADD140")


class NormIdentityTest(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(_norm_identity("AAAA", "AAAA"), 1.0)

    def test_different_length(self):
        self.assertEqual(_norm_identity("AAAA", "AAA"), 0.0)

    def test_mismatch(self):
        self.assertLess(_norm_identity("AAAA", "AAAC"), 1.0)


class BuildSplitTest(unittest.TestCase):
    def _run(self, tmp: Path):
        # test study 16SFWJ (1 seq), validation CIDGMP (1 seq), train rest
        records = [
            _rec("16SFWJ", "AAAA" * 10, "16SFWJ_1M7_0001"),
            _rec("CIDGMP", "CCCC" * 10, "CIDGMP_SHP_0002"),
            _rec("ADD140", "GGGG" * 10, "ADD140_1M7_0001"),
            _rec("RNAPZ5", "UUUU" * 10, "RNAPZ5_1M7_0002"),
        ]
        pairs = (
            [_pair("16SFWJ", "1M7_0001", i) for i in range(1, 105)]
            + [_pair("CIDGMP", "SHP_0002", i) for i in range(1, 5)]
            + [_pair("ADD140", "1M7_0001", i) for i in range(1, 4)]
            + [_pair("RNAPZ5", "1M7_0002", i) for i in range(1, 4)]
        )
        return build_split(records, pairs, tmp)

    def test_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run(Path(d))
            split = out["split_manifest"]
            self.assertEqual(split["assignment"]["16SFWJ"], "test")
            self.assertEqual(split["assignment"]["CIDGMP"], "validation")
            self.assertEqual(split["assignment"]["ADD140"], "train")
            self.assertEqual(split["assignment"]["RNAPZ5"], "train")

    def test_overlap_zero(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run(Path(d))
            self.assertTrue(out["exposure_audit"]["overlap_zero"])
            self.assertTrue(out["exposure_audit"]["near_duplicate"]["leakage_near_dup"])

    def test_test_sealed(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run(Path(d))
            self.assertEqual(out["test_seal"]["seal_status"], "SEALED")
            self.assertEqual(out["test_seal"]["test_studies"], ["16SFWJ"])

    def test_ledger_no_access(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run(Path(d))
            self.assertTrue(out["test_access_ledger"]["append_only"])
            self.assertFalse(any(e["sample_level_labels_read"] for e in out["test_access_ledger"]["entries"]))


if __name__ == "__main__":
    unittest.main()
