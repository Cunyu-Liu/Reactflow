#!/usr/bin/env python3
"""Unit tests for B0-X data, baselines, and evaluator (contract §20.8).

Synthetic, self-contained: no network, no GPU, no real data.  Covers the
eligible-mask delta computation, capacity-ladder baseline fits/predictions,
and the frozen evaluator metrics (Skill, WMAE, cluster CI, group-aware
permutation, learning curve).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0x_data import Pair, compute_delta_and_mask, load_pairs, split_groups  # noqa: E402
from b0x_baselines import (  # noqa: E402
    REGISTRY,
    EditOnlyBaseline,
    MutationTypeMeanBaseline,
    P2PairedBaseline,
    TrainMeanBaseline,
    TreeBaseline,
    WTOnlyBaseline,
    ZeroBaseline,
    run_baseline,
)
from b0x_evaluate import (  # noqa: E402
    cluster_ci,
    group_aware_permutation,
    learning_curve,
    per_pair_loss,
    pooled_skill,
)
from b0x_validate_authority import validate  # noqa: E402


def make_pair(pair_id="s1_r1", study="s1", split="train", parent="p1",
              seq="ACGUACGUACGUACGUACGUACGUACGUAC", mutation_pos=5,
              ref="A", alt="G", wt=None, mutant=None, mask=None):
    if wt is None:
        wt = [0.1 * (i % 5) for i in range(len(seq))]
    if mutant is None:
        mutant = [0.1 * (i % 5) + 0.05 for i in range(len(seq))]
    if mask is None:
        mask = [1] * len(seq)
    delta = [float(m - w) for m, w in zip(mutant, wt)]
    return Pair(
        pair_id=pair_id, study=study, split=split, parent=parent, seq=seq,
        mutation_pos=mutation_pos, ref_allele=ref, alt_allele=alt,
        wt_reactivity=wt, mutant_reactivity=mutant, mask=mask, delta=delta,
        n_eligible=sum(mask), source=study,
    )


class TestData(unittest.TestCase):
    def test_compute_delta_and_mask(self):
        r = {
            "reactivity_layers": {
                "train_frozen": {"reactivity": [0.5, 0.6, None, 0.8]},
                "position_mask": [1, 1, 0, 1],
            },
            "wt_anchor_reactivity": [0.3, 0.4, 0.5, 0.6],
        }
        delta, mask = compute_delta_and_mask(r)
        self.assertEqual(len(delta), 4)
        self.assertAlmostEqual(delta[0], 0.2)
        self.assertIsNone(delta[2])
        self.assertEqual(mask, [1, 1, 0, 1])

    def test_load_pairs_primary_exact_delta_only(self):
        records = [
            {
                "data_role": "PRIMARY_EXACT_DELTA", "source_accession": "s1_r1",
                "source_profile_index": 0, "canonical_sequence": "ACGUACGUACGU",
                "ref_allele": "A", "alt_allele": "G",
                "mutation_coordinate_system": {"sequence_index_0_based": 2},
                "parent_lineage_evidence": {"parent_sequence_sha256": "p1"},
                "wt_anchor_reactivity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                "reactivity_layers": {
                    "train_frozen": {"reactivity": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
                    "position_mask": [1] * 12,
                },
            },
            {
                "data_role": "OTHER_ROLE", "source_accession": "s2_r2",
                "source_profile_index": 0, "canonical_sequence": "ACGUACGUACGU",
                "ref_allele": "A", "alt_allele": "U",
                "mutation_coordinate_system": {"sequence_index_0_based": 3},
                "parent_lineage_evidence": {"parent_sequence_sha256": "p2"},
                "wt_anchor_reactivity": [0.1] * 12,
                "reactivity_layers": {"train_frozen": {"reactivity": [0.2] * 12}, "position_mask": [1] * 12},
            },
        ]
        manifest = {"assignment": {"s1": "train", "s2": "validation"}}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            canon = td / "canon.jsonl"
            canon.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            split = td / "split.json"
            split.write_text(json.dumps(manifest), encoding="utf-8")
            pairs = load_pairs(canon, split, splits={"train", "validation"})
            self.assertEqual(len(pairs), 1)  # only PRIMARY_EXACT_DELTA
            self.assertEqual(pairs[0].study, "s1")
            self.assertEqual(pairs[0].split, "train")
            self.assertEqual(pairs[0].mutation_pos, 2)
            self.assertEqual(pairs[0].delta[0], 0.1)
            groups = split_groups(pairs)
            self.assertEqual(len(groups["train"]), 1)


class TestTrivialBaselines(unittest.TestCase):
    def setUp(self):
        self.train = [
            make_pair("t1", "s1", "train", "p1", ref="A", alt="G", mutation_pos=2),
            make_pair("t2", "s1", "train", "p1", ref="A", alt="G", mutation_pos=4),
            make_pair("t3", "s2", "train", "p2", ref="C", alt="U", mutation_pos=6),
        ]
        self.val = [make_pair("v1", "s3", "validation", "p3", ref="A", alt="G", mutation_pos=3)]

    def test_zero(self):
        b = ZeroBaseline()
        b.fit(self.train)
        pred = b.predict(self.val[0])
        self.assertTrue(np.all(pred == 0))

    def test_train_mean(self):
        b = TrainMeanBaseline()
        b.fit(self.train)
        self.assertAlmostEqual(b.mean, 0.05)
        pred = b.predict(self.val[0])
        self.assertTrue(np.allclose(pred, 0.05))

    def test_mutation_type_mean(self):
        b = MutationTypeMeanBaseline()
        b.fit(self.train)
        self.assertIn("A>G", b.means)
        pred = b.predict(self.val[0])
        self.assertTrue(np.allclose(pred, b.means["A>G"]))

    def test_edit_only(self):
        b = EditOnlyBaseline()
        b.fit(self.train)
        pred = b.predict(self.val[0])
        self.assertEqual(pred[3], b.edit_val)
        self.assertEqual(pred[0], 0.0)

    def test_ridge_wt_only(self):
        b = WTOnlyBaseline(alpha=1.0)
        b.fit(self.train)
        pred = b.predict(self.val[0])
        self.assertEqual(len(pred), len(self.val[0].mask))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_tree(self):
        if shutil.which("sklearn") is None and "sklearn" not in sys.modules:
            try:
                import sklearn  # noqa: F401
            except ImportError:
                self.skipTest("sklearn not available")
        b = TreeBaseline()
        b.fit(self.train)
        pred = b.predict(self.val[0])
        self.assertEqual(len(pred), len(self.val[0].mask))

    def test_p2_paired_cpu_fit(self):
        b = P2PairedBaseline(device="cpu", hidden=8, epochs=2, batch_size=8)
        b.fit(self.train)
        pred = b.predict(self.val[0])
        self.assertEqual(len(pred), len(self.val[0].mask))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_pair_scale_robust(self):
        # Different WT reactivity scales should yield different per-pair scales.
        from b0x_baselines import _pair_scale
        seq = "ACGUACGUACGUACGUACGUACGUACGUAC"
        small = make_pair("s1", "s1", "train", "p1", seq=seq,
                          wt=[0.1] * len(seq), mutant=[0.2] * len(seq))
        large = make_pair("s2", "s2", "train", "p2", seq=seq,
                          wt=[100.0] * len(seq), mutant=[200.0] * len(seq))
        self.assertGreater(_pair_scale(large), _pair_scale(small) * 100)

    def test_p2_paired_scale_invariance(self):
        # A model trained on a small-scale study should predict a large-scale
        # study's delta at the right magnitude (scale-invariant denormalization).
        b = P2PairedBaseline(device="cpu", hidden=12, epochs=40, batch_size=16, seed=0)
        seq = "ACGUACGUACGUACGUACGUACGUACGUAC"
        wt = [0.1 * (i % 5) for i in range(len(seq))]
        small = [make_pair("t1", "s1", "train", "p1", seq=seq, ref="A", alt="G",
                           mutation_pos=3, wt=wt, mutant=[w + 0.05 for w in wt])]
        b.fit(small)
        # large-scale validation pair with same relative pattern
        wt_large = [100.0 * (i % 5) for i in range(len(seq))]
        val_large = make_pair("v1", "s2", "validation", "p2", seq=seq, ref="A", alt="G",
                              mutation_pos=3, wt=wt_large,
                              mutant=[w + 50.0 for w in wt_large])
        pred = b.predict(val_large)
        # denormalized prediction should be on the large (raw) scale
        self.assertGreater(np.abs(pred).max(), 1.0)

    def test_run_baseline_registry(self):
        res = run_baseline("zero", self.train, self.val, device="cpu")
        self.assertEqual(res.status, "ok")
        self.assertIn("zero", REGISTRY)


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.pairs = [
            make_pair("v1", "s1", "validation", "p1", ref="A", alt="G", mutation_pos=2),
            make_pair("v2", "s2", "validation", "p2", ref="C", alt="U", mutation_pos=4),
            make_pair("v3", "s3", "validation", "p3", ref="A", alt="G", mutation_pos=6),
        ]
        self.zero_preds = {p.pair_id: np.zeros(len(p.mask), dtype=np.float32) for p in self.pairs}
        self.good_preds = {p.pair_id: np.array(p.delta, dtype=np.float32) for p in self.pairs}

    def test_pooled_skill_perfect_is_1(self):
        sk = pooled_skill(self.pairs, self.good_preds, self.zero_preds)
        self.assertAlmostEqual(sk["skill_mae"], 1.0, places=3)
        self.assertAlmostEqual(sk["skill_wmae"], 1.0, places=3)

    def test_pooled_skill_zero_vs_zero_is_0(self):
        sk = pooled_skill(self.pairs, self.zero_preds, self.zero_preds)
        self.assertAlmostEqual(sk["skill_mae"], 0.0, places=6)

    def test_per_pair_loss(self):
        loss = per_pair_loss(self.pairs[0], self.good_preds[self.pairs[0].pair_id])
        self.assertGreater(loss["n"], 0)
        self.assertAlmostEqual(loss["mae"], 0.0, places=6)

    def test_cluster_ci(self):
        ci = cluster_ci(self.pairs, self.good_preds, self.zero_preds, n_boot=50, seed=1)
        self.assertAlmostEqual(ci["point"], 1.0, places=2)
        self.assertEqual(ci["n_studies"], 3)

    def test_group_aware_permutation(self):
        # Pairs sharing one (study, parent) block so the within-block shuffle
        # actually breaks prediction-target alignment.  Give distinct delta
        # vectors so the null skill is strictly below the real skill.
        def mk(pid, offset):
            n = 12
            seq = "ACGU" * 3
            wt = [0.1 * (i % 5) for i in range(n)]
            mutant = [w + offset for w in wt]
            return make_pair(pid, "s1", "validation", "p1", ref="A", alt="G",
                             mutation_pos=2, wt=wt, mutant=mutant, seq=seq)
        block = [mk("b1", 0.05), mk("b2", 0.10), mk("b3", 0.15)]
        good = {p.pair_id: np.array(p.delta, dtype=np.float32) for p in block}
        zero = {p.pair_id: np.zeros(len(p.mask), dtype=np.float32) for p in block}
        perm = group_aware_permutation(block, "p2", good, zero, n_perm=20, seed=1)
        self.assertTrue(perm["pass_real_gt_null"])
        self.assertLessEqual(perm["p_value"], 1.0)

    def test_learning_curve(self):
        lc = learning_curve(self.pairs, self.pairs, self.good_preds, self.zero_preds,
                            fractions=(0.5, 1.0), seed=1)
        self.assertIn("frac_0.5", lc)
        self.assertIn("frac_1.0", lc)


class TestAuthorityValidator(unittest.TestCase):
    def test_validate_requires_git_repo(self):
        # On a bare temp dir without governance files, the validator must raise
        # (fail-closed) rather than silently passing.
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(Exception):
                validate(Path(td), staging=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)