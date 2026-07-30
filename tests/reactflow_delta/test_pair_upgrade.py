"""Tests for T-D1.10 pair quality weight + exclusion reasons + true_pair
upgrade (v3.1 §3 pair eligibility; v3 §6.4 pair schema; integrates T-D1.1~9).

Hand-computed fixtures: see ``HandComputation`` class docstrings for the
arithmetic of every quality-weight case, and ``ExclusionMatrix`` for the
frozen reason-vocabulary coverage matrix.
"""

from __future__ import annotations

import pytest

from reactflow.delta.data import (
    COMPARABLE_MIN_FRACTION,
    QUALITY_COVERAGE_FULL_FACTOR_AT,
    QUALITY_NO_REPLICATE_FACTOR,
    QUALITY_SNR_FULL_FACTOR_AT,
    QUALITY_UNKNOWN_SIGNAL_FACTOR,
    UPGRADE_BLOCKER_EXCLUSION_REASONS,
    collect_exclusion_reasons,
    compute_pair_quality_weight,
    determine_primary_eligible,
    determine_true_pair,
    evaluate_pair_upgrade,
)
from reactflow.delta.schema import EXCLUSION_REASONS


# -- kwargs baseline: a fully clean substitution pair ------------------------

CLEAN_KWARGS = dict(
    edit_type="substitution",
    edit_count=1,
    condition_match_status="exact_match",
    substitution_verified=True,
    has_wt_anchor=True,
    normalization_domain_compatible=True,
    parent_lineage_verified=True,
    in_vivo_in_vitro_mixed=False,
)


# =============================================================================
# collect_exclusion_reasons: frozen vocabulary coverage matrix
# =============================================================================


class TestCollectExclusionReasons:
    """One test per reason in the frozen EXCLUSION_REASONS vocabulary (13)."""

    def test_clean_pair_has_no_reasons(self) -> None:
        assert collect_exclusion_reasons(**CLEAN_KWARGS) == []

    def test_indel_not_substitution(self) -> None:
        # schema.py L668 invariant: non-substitution edit_type → this reason.
        kw = {**CLEAN_KWARGS, "edit_type": "deletion"}
        assert collect_exclusion_reasons(**kw) == ["indel_not_substitution"]

    def test_edit_count_not_one(self) -> None:
        kw = {**CLEAN_KWARGS, "edit_count": 2}
        assert collect_exclusion_reasons(**kw) == ["edit_count_not_one"]

    def test_no_wt_anchor(self) -> None:
        kw = {**CLEAN_KWARGS, "has_wt_anchor": False}
        assert collect_exclusion_reasons(**kw) == ["no_wt_anchor"]

    def test_substitution_not_verifiable(self) -> None:
        kw = {**CLEAN_KWARGS, "substitution_verified": False, "is_annotation_only": False}
        assert collect_exclusion_reasons(**kw) == ["substitution_not_verifiable"]

    def test_annotation_only_alt_not_verifiable(self) -> None:
        kw = {**CLEAN_KWARGS, "substitution_verified": False, "is_annotation_only": True}
        assert collect_exclusion_reasons(**kw) == ["annotation_only_alt_not_verifiable"]

    def test_annotation_ref_mismatch(self) -> None:
        kw = {**CLEAN_KWARGS, "annotation_ref_verified": False}
        assert collect_exclusion_reasons(**kw) == ["annotation_ref_mismatch"]

    def test_annotation_ref_verified_true_is_silent(self) -> None:
        # True must NOT trigger a reason (only an explicit False does).
        kw = {**CLEAN_KWARGS, "annotation_ref_verified": True}
        assert collect_exclusion_reasons(**kw) == []

    def test_condition_mismatch(self) -> None:
        # schema.py L672 invariant: condition_match_status='mismatch' → reason.
        kw = {**CLEAN_KWARGS, "condition_match_status": "mismatch"}
        assert collect_exclusion_reasons(**kw) == ["condition_mismatch"]

    def test_probe_mismatch(self) -> None:
        kw = {**CLEAN_KWARGS, "probe_eligible_unchanged": False}
        assert collect_exclusion_reasons(**kw) == ["probe_mismatch"]

    def test_probe_eligible_unchanged_true_is_silent(self) -> None:
        kw = {**CLEAN_KWARGS, "probe_eligible_unchanged": True}
        assert collect_exclusion_reasons(**kw) == []

    def test_comparable_positions_below_60pct(self) -> None:
        kw = {**CLEAN_KWARGS, "comparable_fraction": COMPARABLE_MIN_FRACTION - 0.01}
        assert collect_exclusion_reasons(**kw) == ["comparable_positions_below_60pct"]

    def test_comparable_fraction_at_min_is_clean(self) -> None:
        # Boundary: exactly COMPARABLE_MIN_FRACTION (0.60) is allowed.
        kw = {**CLEAN_KWARGS, "comparable_fraction": COMPARABLE_MIN_FRACTION}
        assert collect_exclusion_reasons(**kw) == []

    def test_normalization_domain_unknown(self) -> None:
        kw = {**CLEAN_KWARGS, "normalization_domain_compatible": False}
        assert collect_exclusion_reasons(**kw) == ["normalization_domain_unknown"]

    def test_parent_lineage_unverified(self) -> None:
        kw = {**CLEAN_KWARGS, "parent_lineage_verified": False}
        assert collect_exclusion_reasons(**kw) == ["parent_lineage_unverified"]

    def test_in_vivo_in_vitro_mixed(self) -> None:
        kw = {**CLEAN_KWARGS, "in_vivo_in_vitro_mixed": True}
        assert collect_exclusion_reasons(**kw) == ["in_vivo_in_vitro_mixed"]

    def test_sequence_based_no_independent_corroboration(self) -> None:
        kw = {
            **CLEAN_KWARGS,
            "is_sequence_based": True,
            "has_independent_corroboration": False,
        }
        assert collect_exclusion_reasons(**kw) == [
            "sequence_based_no_independent_corroboration"
        ]

    def test_sequence_based_with_corroboration_is_clean(self) -> None:
        kw = {
            **CLEAN_KWARGS,
            "is_sequence_based": True,
            "has_independent_corroboration": True,
        }
        assert collect_exclusion_reasons(**kw) == []

    def test_vocabulary_coverage(self) -> None:
        """Every emitted reason must be in the frozen EXCLUSION_REASONS set."""
        # Trigger each reason and confirm membership; this guards against
        # typos and silent drift between data.py and schema.py.
        cases = [
            {"edit_type": "deletion"},
            {"edit_count": 2},
            {"has_wt_anchor": False},
            {"substitution_verified": False, "is_annotation_only": False},
            {"substitution_verified": False, "is_annotation_only": True},
            {"annotation_ref_verified": False},
            {"condition_match_status": "mismatch"},
            {"probe_eligible_unchanged": False},
            {"comparable_fraction": 0.5},
            {"normalization_domain_compatible": False},
            {"parent_lineage_verified": False},
            {"in_vivo_in_vitro_mixed": True},
            {"is_sequence_based": True, "has_independent_corroboration": False},
        ]
        for case in cases:
            kw = {**CLEAN_KWARGS, **case}
            for reason in collect_exclusion_reasons(**kw):
                assert reason in EXCLUSION_REASONS, reason

    def test_multiple_reasons_sorted_unique(self) -> None:
        # Trigger 4 distinct reasons at once; result must be sorted & deduped.
        kw = {
            **CLEAN_KWARGS,
            "has_wt_anchor": False,
            "parent_lineage_verified": False,
            "normalization_domain_compatible": False,
            "in_vivo_in_vitro_mixed": True,
        }
        result = collect_exclusion_reasons(**kw)
        assert result == sorted(result)
        assert len(result) == len(set(result))
        assert set(result) == {
            "no_wt_anchor",
            "normalization_domain_unknown",
            "parent_lineage_unverified",
            "in_vivo_in_vitro_mixed",
        }


# =============================================================================
# determine_primary_eligible / determine_true_pair
# =============================================================================


class TestEligibility:
    def test_empty_reasons_is_eligible_and_true(self) -> None:
        assert determine_primary_eligible([]) is True
        assert determine_true_pair([], True) is True

    def test_soft_blocker_keeps_primary_blocks_true(self) -> None:
        # corroboration-only: primary_eligible stays True, true_pair False.
        reasons = ["sequence_based_no_independent_corroboration"]
        assert determine_primary_eligible(reasons) is True
        assert determine_true_pair(reasons, True) is False

    def test_hard_blocker_blocks_both(self) -> None:
        reasons = ["condition_mismatch"]
        assert determine_primary_eligible(reasons) is False
        assert determine_true_pair(reasons, False) is False

    def test_mixed_soft_and_hard_blocks_both(self) -> None:
        reasons = [
            "condition_mismatch",
            "sequence_based_no_independent_corroboration",
        ]
        assert determine_primary_eligible(reasons) is False
        assert determine_true_pair(reasons, False) is False

    def test_upgrade_blocker_vocabulary(self) -> None:
        # Sanity: the soft-blocker set is a subset of EXCLUSION_REASONS.
        assert UPGRADE_BLOCKER_EXCLUSION_REASONS <= EXCLUSION_REASONS


# =============================================================================
# compute_pair_quality_weight: hand-computed factor arithmetic
# =============================================================================


class TestQualityWeight:
    """Hand-computed fixtures (v3.1 §3.2).

    weight = f_comp * f_snr * f_cov * f_rep * f_miss
      f_comp = clamp(comparable_fraction, 0, 1)        ; None → 0.5
      f_snr  = clamp(snr / 10, 0, 1)                   ; None → 0.5
      f_cov  = clamp(coverage_mean / 30, 0, 1)         ; None → 0.5
      f_rep  = 1.0 if has_replicates else 0.8
      f_miss = clamp(1 - missing_fraction, 0, 1)       ; None → 0.5
    """

    def test_full_quality_weight_is_one(self) -> None:
        # 1.0 * 1.0 * 1.0 * 1.0 * 1.0 = 1.0
        result = compute_pair_quality_weight(
            comparable_fraction=1.0,
            snr=QUALITY_SNR_FULL_FACTOR_AT,        # 10 → 1.0
            coverage_mean=QUALITY_COVERAGE_FULL_FACTOR_AT,  # 30 → 1.0
            missing_fraction=0.0,                  # 1 - 0 = 1.0
            has_replicates=True,                   # 1.0
        )
        assert result["pair_quality_weight"] == pytest.approx(1.0)
        assert result["factors"] == {
            "comparable": 1.0,
            "snr": 1.0,
            "coverage": 1.0,
            "replicate": 1.0,
            "missing": 1.0,
        }

    def test_no_replicate_penalty(self) -> None:
        # 1.0 * 1.0 * 1.0 * 0.8 * 1.0 = 0.8
        result = compute_pair_quality_weight(
            comparable_fraction=1.0,
            snr=QUALITY_SNR_FULL_FACTOR_AT,
            coverage_mean=QUALITY_COVERAGE_FULL_FACTOR_AT,
            missing_fraction=0.0,
            has_replicates=False,
        )
        assert result["pair_quality_weight"] == pytest.approx(QUALITY_NO_REPLICATE_FACTOR)
        assert result["factors"]["replicate"] == QUALITY_NO_REPLICATE_FACTOR

    def test_unknown_snr_defaults_to_half(self) -> None:
        # 1.0 * 0.5 * 1.0 * 1.0 * 1.0 = 0.5
        result = compute_pair_quality_weight(
            comparable_fraction=1.0,
            snr=None,
            coverage_mean=QUALITY_COVERAGE_FULL_FACTOR_AT,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["pair_quality_weight"] == pytest.approx(QUALITY_UNKNOWN_SIGNAL_FACTOR)
        assert result["factors"]["snr"] == QUALITY_UNKNOWN_SIGNAL_FACTOR

    def test_all_unknown_defaults_to_half_power_five(self) -> None:
        # 0.5 * 0.5 * 0.5 * 0.8 * 0.5 = 0.05  (no-rep default)
        result = compute_pair_quality_weight()
        expected = (
            QUALITY_UNKNOWN_SIGNAL_FACTOR ** 4 * QUALITY_NO_REPLICATE_FACTOR
        )
        assert result["pair_quality_weight"] == pytest.approx(expected)
        # 0.5^4 * 0.8 = 0.0625 * 0.8 = 0.05
        assert result["pair_quality_weight"] == pytest.approx(0.05)

    def test_mixed_factors_hand_computed(self) -> None:
        # comp=0.6, snr=5, cov=15, miss=0.2, no-rep
        # 0.6 * 0.5 * 0.5 * 0.8 * 0.8 = 0.096
        result = compute_pair_quality_weight(
            comparable_fraction=0.6,
            snr=5.0,
            coverage_mean=15.0,
            missing_fraction=0.2,
            has_replicates=False,
        )
        assert result["pair_quality_weight"] == pytest.approx(0.096)
        assert result["factors"]["comparable"] == pytest.approx(0.6)
        assert result["factors"]["snr"] == pytest.approx(0.5)
        assert result["factors"]["coverage"] == pytest.approx(0.5)
        assert result["factors"]["replicate"] == pytest.approx(0.8)
        assert result["factors"]["missing"] == pytest.approx(0.8)

    def test_clamping_snr_above_full(self) -> None:
        # snr=100 → 100/10 = 10 → clamped to 1.0
        result = compute_pair_quality_weight(
            comparable_fraction=1.0,
            snr=100.0,
            coverage_mean=1.0,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["factors"]["snr"] == 1.0

    def test_clamping_negative_comparable(self) -> None:
        # comparable_fraction=-0.5 → clamped to 0.0 → weight 0.0
        result = compute_pair_quality_weight(
            comparable_fraction=-0.5,
            snr=10.0,
            coverage_mean=30.0,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["pair_quality_weight"] == 0.0
        assert result["factors"]["comparable"] == 0.0

    def test_clamping_missing_fraction_above_one(self) -> None:
        # missing_fraction=2.0 → 1-2 = -1 → clamped to 0.0
        result = compute_pair_quality_weight(
            comparable_fraction=1.0,
            snr=10.0,
            coverage_mean=30.0,
            missing_fraction=2.0,
            has_replicates=True,
        )
        assert result["factors"]["missing"] == 0.0
        assert result["pair_quality_weight"] == 0.0

    def test_weight_always_non_negative(self) -> None:
        # schema.py L648 invariant: pair_quality_weight must be >= 0.
        for comp in (-1.0, 0.0, 0.5, 1.0, 2.0):
            for snr in (None, 0.0, 5.0, 10.0, 50.0):
                for cov in (None, 0.0, 15.0, 30.0, 90.0):
                    for miss in (None, 0.0, 0.5, 1.0, 2.0):
                        for rep in (True, False):
                            w = compute_pair_quality_weight(
                                comparable_fraction=comp,
                                snr=snr,
                                coverage_mean=cov,
                                missing_fraction=miss,
                                has_replicates=rep,
                            )["pair_quality_weight"]
                            assert 0.0 <= w <= 1.0


# =============================================================================
# evaluate_pair_upgrade: integration of reasons + eligibility + weight
# =============================================================================


class TestEvaluatePairUpgrade:
    def test_clean_pair(self) -> None:
        # Full-quality clean pair: no reasons, primary_eligible, true_pair.
        result = evaluate_pair_upgrade(
            **CLEAN_KWARGS,
            comparable_fraction=1.0,
            snr=QUALITY_SNR_FULL_FACTOR_AT,
            coverage_mean=QUALITY_COVERAGE_FULL_FACTOR_AT,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["exclusion_reasons"] == []
        assert result["primary_eligible"] is True
        assert result["true_pair"] is True
        assert result["pair_quality_weight"] == pytest.approx(1.0)
        assert "quality_factors" in result

    def test_condition_mismatch_blocks_both(self) -> None:
        # hand-computed: reasons=[condition_mismatch], primary=False, true=False
        result = evaluate_pair_upgrade(
            **{**CLEAN_KWARGS, "condition_match_status": "mismatch"},
        )
        assert result["exclusion_reasons"] == ["condition_mismatch"]
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False

    def test_sequence_based_no_corroboration_is_soft_blocker(self) -> None:
        # reasons=[sequence_based_no_independent_corroboration]
        # primary_eligible=True (soft), true_pair=False
        result = evaluate_pair_upgrade(
            **{
                **CLEAN_KWARGS,
                "is_sequence_based": True,
                "has_independent_corroboration": False,
            }
        )
        assert result["exclusion_reasons"] == [
            "sequence_based_no_independent_corroboration"
        ]
        assert result["primary_eligible"] is True
        assert result["true_pair"] is False

    def test_indel_blocks_both(self) -> None:
        # hand-computed: reasons=[indel_not_substitution], primary=False, true=False
        result = evaluate_pair_upgrade(
            **{**CLEAN_KWARGS, "edit_type": "insertion"},
        )
        assert result["exclusion_reasons"] == ["indel_not_substitution"]
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False

    def test_quality_weight_independent_of_reasons(self) -> None:
        # A pair with a hard blocker still gets a quality weight computed
        # from the same numeric inputs (weight is not zeroed by exclusion).
        result = evaluate_pair_upgrade(
            **{**CLEAN_KWARGS, "has_wt_anchor": False},
            comparable_fraction=1.0,
            snr=QUALITY_SNR_FULL_FACTOR_AT,
            coverage_mean=QUALITY_COVERAGE_FULL_FACTOR_AT,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert "no_wt_anchor" in result["exclusion_reasons"]
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False
        # weight is still computed from the full-quality numeric inputs.
        assert result["pair_quality_weight"] == pytest.approx(1.0)

    def test_return_keys_match_pair_schema(self) -> None:
        # The four pair-schema D1 fields + quality_factors breakdown.
        result = evaluate_pair_upgrade(**CLEAN_KWARGS)
        assert set(result.keys()) == {
            "exclusion_reasons",
            "primary_eligible",
            "true_pair",
            "pair_quality_weight",
            "quality_factors",
        }

    def test_indel_plus_condition_mismatch_emits_both(self) -> None:
        # schema.py invariants L668 + L672 both apply; both reasons emitted.
        result = evaluate_pair_upgrade(
            **{
                **CLEAN_KWARGS,
                "edit_type": "deletion",
                "condition_match_status": "mismatch",
            }
        )
        assert set(result["exclusion_reasons"]) == {
            "indel_not_substitution",
            "condition_mismatch",
        }
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False
