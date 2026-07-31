"""Tests for ``reactflow.delta.baselines`` (B0 baselines).

Covers:
  * Alt marginalization: 3 mutant seqs, correct base substitution, ref validation.
  * Sequence-to-array index mapping.
  * Non-learned baselines (zero, mean, distance, edit-only, nearest, local-release)
    on synthetic pairs.
  * Thermo baseline scaffolding (RNAfold) with mocked folder.
  * Learned baseline scaffolding (Siamese / generic paired) with tiny torch model.
  * Parameter counting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.baselines import (
    BASELINE_REGISTRY,
    DistanceDecayBaseline,
    EditOnlyBaseline,
    EternaFoldBaseline,
    GenericPairedBaseline,
    LocalReleaseBaseline,
    MutationTypeMeanBaseline,
    NearestTrainBaseline,
    RNAfoldBaseline,
    RNAplfoldBaseline,
    SiameseBaseline,
    StaticReactivityBaseline,
    ZeroChangeBaseline,
    alt_candidates,
    build_mutant_sequences,
    count_parameters,
    map_seq_array_to_delta,
)
from reactflow.delta.evaluate import PairRecord, build_endpoint_mask, to_float_array


# ---------------------------------------------------------------------------
# Synthetic PairRecord helpers
# ---------------------------------------------------------------------------


_UNSET = object()


def _rec(
    pair_id: str = "p1",
    parent: str = "parentA",
    study: str = "doiA",
    delta: list | None = None,
    edit_arr_idx: int = 2,
    edit_pos_1idx: int = 3,
    ref: str = "G",
    weight: float = 1.0,
    seq_positions: list | None = None,
    wt_sequence: str | None = None,
    wt_reactivity=_UNSET,
    wt_features: dict | None = None,
) -> PairRecord:
    n = len(delta) if delta is not None else 6
    if delta is None:
        delta = [0.1, -0.2, 0.5, 0.0, 0.3, -0.1]
    if seq_positions is None:
        seq_positions = [float(i + 1) for i in range(n)]
    if wt_reactivity is _UNSET:
        wt_reactivity = [0.1] * n
    else:
        # wt_reactivity may be explicitly None.
        pass
    d = to_float_array(delta)
    wt = to_float_array(wt_reactivity) if wt_reactivity is not None else None
    mut_list = (
        [v + dv for v, dv in zip(wt_reactivity, delta)]
        if wt_reactivity is not None
        else [0.1 + dv for dv in delta]
    )
    mut = to_float_array(mut_list)
    mask = build_endpoint_mask(d, wt if wt is not None else to_float_array([0.1] * n), mut, edit_arr_idx)
    return PairRecord(
        pair_id=pair_id,
        parent=parent,
        study=study,
        rdat_path=f"/fake/{pair_id}.rdat",
        wt_profile_index=1,
        mutant_profile_index=2,
        edit_arr_idx=edit_arr_idx,
        edit_pos_1indexed=edit_pos_1idx,
        encoded_ref=ref,
        aligned_length=n,
        delta_true=d,
        endpoint_mask=mask,
        pair_quality_weight=weight,
        seq_positions=np.array(seq_positions, dtype=float),
        wt_sequence=wt_sequence,
        wt_reactivity=wt,
        wt_features=wt_features,
    )


# ---------------------------------------------------------------------------
# Alt marginalization
# ---------------------------------------------------------------------------


class TestAltMarginalization:
    def test_alt_candidates_excludes_ref(self):
        assert set(alt_candidates("A")) == {"C", "G", "U"}
        assert set(alt_candidates("C")) == {"A", "G", "U"}
        assert set(alt_candidates("G")) == {"A", "C", "U"}
        assert set(alt_candidates("U")) == {"A", "C", "G"}
        assert len(alt_candidates("A")) == 3

    def test_alt_candidates_dna_T_normalized(self):
        # T should be treated as U.
        assert set(alt_candidates("T")) == {"A", "C", "G"}

    def test_build_mutant_sequences(self):
        wt = "ACGUACGU"
        muts = build_mutant_sequences(wt, edit_pos_1indexed=3, ref_base="G")
        assert len(muts) == 3
        # Position 3 (0-indexed 2) should differ from G in each mutant.
        for m in muts:
            assert m[2] != "G"
            assert m[2] in {"A", "C", "U"}
            # All other positions identical to WT.
            for i in range(len(wt)):
                if i != 2:
                    assert m[i] == wt[i]

    def test_build_mutant_sequences_lowercase(self):
        wt = "acguacgu"
        muts = build_mutant_sequences(wt, edit_pos_1indexed=1, ref_base="A")
        assert len(muts) == 3
        for m in muts:
            assert m[0] != "A"  # normalized to upper
            assert m[0] in {"C", "G", "U"}

    def test_build_mutant_sequences_ref_mismatch_raises(self):
        wt = "ACGU"
        with pytest.raises(ValueError, match="WT base at position 2 is 'C'"):
            build_mutant_sequences(wt, edit_pos_1indexed=2, ref_base="G")

    def test_build_mutant_sequences_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            build_mutant_sequences("ACGU", edit_pos_1indexed=99, ref_base="A")


class TestMapSeqArrayToDelta:
    def test_identity_mapping(self):
        # seq_positions = [1, 2, 3, 4] -> arr idx i = seq pos i+1 -> identity
        rec = _rec(
            delta=[0.1, 0.2, 0.3, 0.4],
            edit_arr_idx=0,
            seq_positions=[1.0, 2.0, 3.0, 4.0],
        )
        seq_arr = np.array([10.0, 20.0, 30.0, 40.0])
        out = map_seq_array_to_delta(seq_arr, rec)
        assert out.tolist() == [10.0, 20.0, 30.0, 40.0]

    def test_offset_mapping(self):
        # seq_positions = [3, 4, 5] -> arr idx 0 reads seq_arr[2]
        rec = _rec(
            delta=[0.1, 0.2, 0.3],
            edit_arr_idx=0,
            seq_positions=[3.0, 4.0, 5.0],
        )
        seq_arr = np.array([0.0, 0.0, 10.0, 20.0, 30.0])
        out = map_seq_array_to_delta(seq_arr, rec)
        assert out.tolist() == [10.0, 20.0, 30.0]

    def test_nan_positions_become_zero(self):
        rec = _rec(
            delta=[0.1, 0.2, 0.3],
            edit_arr_idx=0,
            seq_positions=[1.0, float("nan"), 3.0],
        )
        seq_arr = np.array([10.0, 20.0, 30.0])
        out = map_seq_array_to_delta(seq_arr, rec)
        assert out.tolist() == [10.0, 0.0, 30.0]


# ---------------------------------------------------------------------------
# Non-learned baselines
# ---------------------------------------------------------------------------


class TestZeroChangeBaseline:
    def test_predicts_zeros(self):
        b = ZeroChangeBaseline()
        b.fit([_rec()])
        rec = _rec(delta=[0.1, 0.2, 0.3], edit_arr_idx=0)
        pred = b.predict(rec)
        assert pred.tolist() == [0.0, 0.0, 0.0]
        assert pred.shape == (3,)

    def test_fit_is_noop(self):
        b = ZeroChangeBaseline()
        b.fit([])  # no crash
        assert b.predict(_rec(delta=[0.1])).shape == (1,)


class TestMutationTypeMeanBaseline:
    def test_mean_profile(self):
        train = [
            _rec(pair_id="t1", delta=[0.1, 0.2, 0.3, 0.4], edit_arr_idx=2),
            _rec(pair_id="t2", delta=[0.3, 0.4, 0.5, 0.6], edit_arr_idx=2),
        ]
        b = MutationTypeMeanBaseline()
        b.fit(train)
        # edit pos excluded; mean of [0.1,0.3] , [0.2,0.4], [_,_], [0.4,0.6]
        # = [0.2, 0.3, 0, 0.5]  (edit idx 2 -> 0 in mean profile)
        pred = b.predict(_rec(delta=[0.0, 0.0, 0.0, 0.0], edit_arr_idx=2))
        assert pred[0] == pytest.approx(0.2)
        assert pred[1] == pytest.approx(0.3)
        assert pred[2] == pytest.approx(0.0)  # edit pos never seen in train mask
        assert pred[3] == pytest.approx(0.5)

    def test_different_lengths(self):
        train = [
            _rec(pair_id="t1", delta=[0.1, 0.2, 0.3], edit_arr_idx=1),
            _rec(pair_id="t2", delta=[0.4, 0.5], edit_arr_idx=0),
        ]
        b = MutationTypeMeanBaseline()
        b.fit(train)
        pred = b.predict(_rec(delta=[0.0, 0.0, 0.0], edit_arr_idx=1))
        assert pred.shape == (3,)


class TestDistanceDecayBaseline:
    def test_predicts_decaying_from_edit(self):
        train = [_rec(pair_id="t1", delta=[0.5, 0.4, 0.3, 0.2, 0.1],
                       edit_arr_idx=2, edit_pos_1idx=3,
                       seq_positions=[1.0, 2.0, 3.0, 4.0, 5.0])]
        b = DistanceDecayBaseline()
        b.fit(train)
        rec = _rec(delta=[0.0] * 5, edit_arr_idx=2, edit_pos_1idx=3,
                    seq_positions=[1.0, 2.0, 3.0, 4.0, 5.0])
        pred = b.predict(rec)
        assert pred.shape == (5,)
        # All finite, and the peak should be at the edit position.
        assert np.all(np.isfinite(pred))
        assert np.argmax(pred) == 2  # edit arr idx


class TestEditOnlyBaseline:
    def test_only_edit_position_nonzero(self):
        train = [_rec(pair_id="t1", delta=[0.0, 0.0, 0.5, 0.0, 0.0], edit_arr_idx=2)]
        b = EditOnlyBaseline()
        b.fit(train)
        rec = _rec(delta=[0.0] * 5, edit_arr_idx=2)
        pred = b.predict(rec)
        assert pred[2] != 0.0
        assert pred[0] == 0.0 and pred[1] == 0.0 and pred[3] == 0.0 and pred[4] == 0.0


class TestNearestTrainBaseline:
    def test_copies_nearest_train_delta(self):
        train = [
            _rec(pair_id="t1", parent="A", delta=[0.1, 0.2, 0.9, 0.3, 0.4],
                 edit_arr_idx=2, wt_features={"bpp_paired_prob": 0.9, "n_contacts": 1}),
            _rec(pair_id="t2", parent="B", delta=[0.5, 0.5, 0.5, 0.5, 0.5],
                 edit_arr_idx=2, wt_features={"bpp_paired_prob": 0.1, "n_contacts": 0}),
        ]
        b = NearestTrainBaseline()
        b.fit(train)
        # Query similar to t1.
        rec = _rec(pair_id="q", parent="A", delta=[0.0] * 5, edit_arr_idx=2,
                    wt_features={"bpp_paired_prob": 0.85, "n_contacts": 1})
        pred = b.predict(rec)
        # Should copy t1's delta (same parent + close features).
        assert pred[0] == pytest.approx(0.1)

    def test_empty_train_returns_zeros(self):
        b = NearestTrainBaseline()
        b.fit([])
        rec = _rec(delta=[0.0] * 3, edit_arr_idx=0)
        assert b.predict(rec).tolist() == [0.0, 0.0, 0.0]


class TestLocalReleaseBaseline:
    def test_peak_at_edit(self):
        train = [_rec(pair_id="t1", delta=[0.1, 0.2, 0.5, 0.2, 0.1],
                       edit_arr_idx=2, edit_pos_1idx=3,
                       seq_positions=[1.0, 2.0, 3.0, 4.0, 5.0],
                       wt_features={"bpp_paired_prob": 0.9})]
        b = LocalReleaseBaseline()
        b.fit(train)
        rec = _rec(delta=[0.0] * 5, edit_arr_idx=2, edit_pos_1idx=3,
                    seq_positions=[1.0, 2.0, 3.0, 4.0, 5.0],
                    wt_features={"bpp_paired_prob": 0.9})
        pred = b.predict(rec)
        assert np.argmax(pred) == 2  # peak at edit
        # Decays away from edit.
        assert pred[0] < pred[2]
        assert pred[4] < pred[2]


# ---------------------------------------------------------------------------
# Thermo baselines (with mocked folder)
# ---------------------------------------------------------------------------


class TestRNAfoldBaseline:
    def test_predict_with_mocked_folder(self):
        wt = "GCGCGCGCGC"  # len 10
        rec = _rec(
            pair_id="t",
            delta=[0.0] * 10,
            edit_arr_idx=2,
            edit_pos_1idx=3,
            ref="G",
            seq_positions=[float(i + 1) for i in range(10)],
            wt_sequence=wt,
        )
        b = RNAfoldBaseline()

        # Mock _fold_seq to return a deterministic unpaired_prob.
        def fake_fold(seq):
            n = len(seq)
            # Mutant sequences have higher unpaired prob at position 2 (0-indexed).
            up = np.full(n, 0.5)
            if seq[2] != "G":  # mutant
                up[2] = 0.9
            return {"unpaired_prob": up}

        with patch.object(b, "_fold_seq", side_effect=fake_fold):
            pred = b.predict(rec)
        # Delta unpaired at pos 2 = 0.9 - 0.5 = 0.4 averaged over 3 alts.
        assert pred.shape == (10,)
        # Position 2 (arr idx 2) = seq pos 3 = 0.4
        assert pred[2] == pytest.approx(0.4, abs=1e-6)
        # Other positions = 0
        assert pred[0] == pytest.approx(0.0, abs=1e-6)

    def test_no_wt_sequence_returns_zeros(self):
        rec = _rec(delta=[0.0] * 4, edit_arr_idx=0, wt_sequence=None)
        b = RNAfoldBaseline()
        assert b.predict(rec).tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_fit_is_noop(self):
        b = RNAfoldBaseline()
        b.fit([_rec()])  # should not crash; thermo baselines have no fit state
        assert not hasattr(b, "_model") or b._model is None


class TestEternaFoldBaseline:
    def test_predict_with_mocked_cli(self):
        wt = "GCGCGCGCGC"
        rec = _rec(
            pair_id="t",
            delta=[0.0] * 10,
            edit_arr_idx=2,
            edit_pos_1idx=3,
            ref="G",
            seq_positions=[float(i + 1) for i in range(10)],
            wt_sequence=wt,
        )
        b = EternaFoldBaseline()

        def fake_fold(seq):
            n = len(seq)
            up = np.zeros(n)
            if seq[2] != "G":
                up[2] = 1.0
            return {"unpaired_prob": up}

        with patch.object(b, "_fold_seq", side_effect=fake_fold):
            pred = b.predict(rec)
        assert pred[2] == pytest.approx(1.0, abs=1e-6)
        assert pred[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Learned baselines (torch, skipped if unavailable)
# ---------------------------------------------------------------------------

_has_torch = True
try:
    import torch  # noqa: F401
except ImportError:
    _has_torch = False

_skip_no_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


@_skip_no_torch
class TestSiameseBaseline:
    def test_fit_and_predict_tiny(self):
        # Tiny synthetic train set: 2 parents, 4 pairs each.
        train = []
        for i in range(4):
            train.append(_rec(
                pair_id=f"tA{i}", parent="A", study="doiA",
                delta=[0.1 * (i + 1), 0.2, 0.3, 0.4, 0.5, 0.6],
                edit_arr_idx=2, edit_pos_1idx=3, ref="G",
                wt_sequence="GCGCGCGCGCGC",
                seq_positions=[float(j + 1) for j in range(12)],
            ))
        for i in range(4):
            train.append(_rec(
                pair_id=f"tB{i}", parent="B", study="doiA",
                delta=[0.1, 0.2, 0.3 * (i + 1), 0.4, 0.5, 0.6],
                edit_arr_idx=2, edit_pos_1idx=3, ref="G",
                wt_sequence="GCGCGCGCGCGC",
                seq_positions=[float(j + 1) for j in range(12)],
            ))
        b = SiameseBaseline(epochs=1, batch_size=4, device="cpu", seed=0)
        b.fit(train)
        assert b._model is not None
        rec = _rec(
            pair_id="q", parent="A", study="doiA",
            delta=[0.0] * 12, edit_arr_idx=2, edit_pos_1idx=3, ref="G",
            wt_sequence="GCGCGCGCGCGC",
            seq_positions=[float(j + 1) for j in range(12)],
        )
        pred = b.predict(rec)
        assert pred.shape == (12,)
        assert np.all(np.isfinite(pred))


@_skip_no_torch
class TestGenericPairedBaseline:
    def test_fit_and_predict_tiny(self):
        train = []
        for i in range(4):
            train.append(_rec(
                pair_id=f"t{i}", parent="A", study="doiA",
                delta=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                edit_arr_idx=2, edit_pos_1idx=3, ref="G",
                wt_sequence="GCGCGCGCGCGC",
                seq_positions=[float(j + 1) for j in range(12)],
            ))
        b = GenericPairedBaseline(epochs=1, batch_size=4, device="cpu", seed=0)
        b.fit(train)
        assert b._model is not None
        rec = _rec(
            pair_id="q", parent="A", study="doiA",
            delta=[0.0] * 12, edit_arr_idx=2, edit_pos_1idx=3, ref="G",
            wt_sequence="GCGCGCGCGCGC",
            seq_positions=[float(j + 1) for j in range(12)],
        )
        pred = b.predict(rec)
        assert pred.shape == (12,)
        assert np.all(np.isfinite(pred))


@_skip_no_torch
class TestStaticReactivityBaseline:
    def test_fit_predict_tiny(self):
        train = []
        for i in range(4):
            train.append(_rec(
                pair_id=f"t{i}", parent="A", study="doiA",
                delta=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                edit_arr_idx=2, edit_pos_1idx=3, ref="G",
                wt_sequence="GCGCGCGCGCGC",
                wt_reactivity=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
                seq_positions=[float(j + 1) for j in range(12)],
            ))
        b = StaticReactivityBaseline(epochs=1, batch_size=4, device="cpu", seed=0)
        b.fit(train)
        assert b._model is not None
        rec = _rec(
            pair_id="q", parent="A", study="doiA",
            delta=[0.0] * 12, edit_arr_idx=2, edit_pos_1idx=3, ref="G",
            wt_sequence="GCGCGCGCGCGC",
            wt_reactivity=[0.1] * 12,
            seq_positions=[float(j + 1) for j in range(12)],
        )
        pred = b.predict(rec)
        assert pred.shape == (12,)
        assert np.all(np.isfinite(pred))

    def test_no_wt_reactivity_skips(self):
        # Without wt_reactivity, fit produces no model.
        train = [_rec(pair_id="t", delta=[0.1] * 6, edit_arr_idx=0,
                      wt_sequence="GCGCGC", wt_reactivity=None)]
        b = StaticReactivityBaseline(epochs=1, device="cpu")
        b.fit(train)
        assert b._model is None


# ---------------------------------------------------------------------------
# Registry + parameter counting
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_names_present(self):
        expected = {
            "zero_change", "mutation_type_mean", "distance_decay", "edit_only",
            "nearest_train", "local_release", "rnafold", "rnaplfold", "eternafold",
            "static_reactivity", "siamese_matched", "generic_paired_matched",
        }
        assert set(BASELINE_REGISTRY.keys()) == expected

    def test_count_parameters_non_learned(self):
        b = ZeroChangeBaseline()
        assert count_parameters(b) == 0

    def test_count_parameters_learned(self):
        if not _has_torch:
            pytest.skip("torch not installed")
        b = SiameseBaseline(epochs=0, device="cpu")
        # Build a model manually so count_parameters can see it.
        b._max_len = 12
        b._model = b._build_model(12)
        n = count_parameters(b)
        assert n > 0
