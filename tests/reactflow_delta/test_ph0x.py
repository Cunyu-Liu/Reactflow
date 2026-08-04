"""Unit tests for PH0-X identifiability/reliability logic (no data access)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# scripts use sibling imports (e.g. "from ph0x_caller import ..."), so the
# scripts directory must be on sys.path for direct module imports.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scripts.reactflow_delta import ph0x_caller
from scripts.reactflow_delta import ph0x_permutation
from scripts.reactflow_delta import ph0x_blind_certificate


def _rec(study, seq, mpos, wt_react, mut_react, data_role="PRIMARY_EXACT_DELTA",
         probe=("1M7",), wt_error=None, mut_error=None):
    return {
        "source_accession": f"{study}_1M7_0001",
        "canonical_sequence": seq,
        "data_role": data_role,
        "probe": probe,
        "temperature": ["24C"],
        "wt_anchor_reactivity": wt_react,
        "wt_anchor_error": wt_error or [0.1] * len(wt_react),
        "mutation_coordinate_system": {"sequence_index_0_based": mpos},
        "reactivity_layers": {
            "raw": {"error": mut_error or [0.1] * len(mut_react)},
            "train_frozen": {"reactivity": mut_react},
            "position_mask": [1] * len(mut_react),
        },
    }


def _split(assign):
    return {
        "assignment": assign,
        "outcome_blind": True,
        "pair_counts": {"train": 1, "validation": 1, "test": 1},
    }


class MaxClusterTest(unittest.TestCase):
    def test_window_limits_to_15(self):
        # 40 consecutive signal positions must not merge into a whole-sequence
        # cluster; the window is bounded by CLUSTER_WINDOW.
        w = [1.0] * 40
        el = [1] * 40
        self.assertLessEqual(ph0x_caller._max_cluster(w, el), 15.0 + 1e-9)

    def test_window_ignores_ineligible(self):
        w = [1.0] * 5
        el = [0, 1, 1, 1, 0]
        self.assertAlmostEqual(ph0x_caller._max_cluster(w, el), 3.0 ** 0.5)

    def test_span_returns_winning_run(self):
        w = [0.0, 1.0, 1.0, 1.0, 0.0]
        el = [1, 1, 1, 1, 1]
        T, run = ph0x_caller._max_cluster_span(w, el)
        self.assertAlmostEqual(T, 3.0 ** 0.5)
        self.assertEqual(run, [0, 1, 2, 3])


class NoiseModelTest(unittest.TestCase):
    def test_study_median_fallback_when_no_per_pos(self):
        seq = "A" * 20
        recs = [
            _rec("STUD", seq, 5, [0.0] * 20, [0.0] * 20),
            _rec("STUD", seq, 6, [0.0] * 20, [0.0] * 20),
        ]
        nm = ph0x_caller.NoiseModel(recs)
        # no per-position error (mut_error defaults 0.1) -> finite noise
        self.assertIsNotNone(nm.noise_std("STUD", ("1M7",)))


class CallerTest(unittest.TestCase):
    def test_frozen_call_emits_tier_counts(self):
        seq = "A" * 30
        recs = []
        for i in range(4):
            recs.append(_rec("TRAIN", seq, i, [0.0] * 30, [5.0] * 30))
        recs.append(_rec("VAL", seq, 2, [0.0] * 30, [5.0] * 30))
        recs.append(_rec("TEST", seq, 3, [0.0] * 30, [5.0] * 30))
        split = _split({"TRAIN": "train", "VAL": "validation", "TEST": "test"})
        nm = ph0x_caller.NoiseModel(recs)
        out = ph0x_caller.frozen_call(recs, split, nm)
        self.assertIn("tier_changers", out)
        self.assertIn("tier_b_conditions", out)
        self.assertEqual(out["caller"]["name"], "frozen_replicate_aware_max_cluster")


class PermutationTest(unittest.TestCase):
    def test_statistic_counts_mutation_in_cluster(self):
        cells = [
            {"study": "S", "weights": [0.0, 1.0, 1.0, 1.0, 0.0],
             "mask": [1, 1, 1, 1, 1], "mpos": 2},
            {"study": "S", "weights": [0.0, 0.0, 0.0, 0.0, 0.0],
             "mask": [1, 1, 1, 1, 1], "mpos": 4},
        ]
        # null95 below 1.73 (sqrt of 3) -> first cell counted (mutation 2 in [1,2,3])
        self.assertEqual(ph0x_permutation._statistic(cells, 1.5), 1)

    def test_p_value(self):
        # _p_value uses the module permutation count N_PERM
        n = ph0x_permutation.N_PERM
        self.assertEqual(ph0x_permutation._p_value([1, 2, 3], 3), (1 + 1) / (n + 1))
        self.assertEqual(ph0x_permutation._p_value([1, 2, 3], 0), (3 + 1) / (n + 1))


class BlindCertificateTest(unittest.TestCase):
    def test_build_certificate(self):
        caller = {"tier_changers": {"test": 25}, "tier_pairs": {"test": 100}}
        seal = Path(__file__).parent / "blank_seal.txt"
        seal.write_text("seal", encoding="utf-8")
        ledger = Path(__file__).parent / "blank_ledger.txt"
        ledger.write_text("ledger", encoding="utf-8")
        try:
            cert = ph0x_blind_certificate.build_certificate([], {}, seal, ledger, caller)
            self.assertEqual(cert["certificate_status"], "PASS_AGGREGATE_VIABILITY")
            self.assertEqual(cert["aggregate_changer_count"], 25)
            self.assertTrue(cert["test_changers_ge_20"])
            self.assertTrue(cert["test_split_is_sealed"])
            self.assertTrue(cert["disclosure"].startswith("aggregate-only"))
        finally:
            seal.unlink()
            ledger.unlink()


if __name__ == "__main__":
    unittest.main()