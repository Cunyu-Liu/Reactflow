"""T-D1.11 hand-computed fixtures: end-to-end D1 pipeline integration tests.

Each fixture exercises the full D1 cleanup pipeline (T-D1.1~10) on a small,
hand-verifiable synthetic input and asserts hand-computed expected values at
every stage. These fixtures are the proof artifact for the v3.1 §4 D1 Gate
bullet "fixtures 100% 通过".

D1 Gate coverage (v3.1 §4):
  - missing 不作 0       → Fixture 2 (None propagated to delta, not 0)
  - noise 不用 test 估计  → Fixture 6 (threshold frozen from controls only)
  - normalization 不最小化 pair difference → Fixture 3 (per-construct scale)
  - 每个 exclusion 有 machine-readable reason → Fixtures 2,4,5 + vocab check
  - Tier gate 不被降低   → Fixture 4 (soft blocker keeps primary, blocks true_pair)

Hand arithmetic is documented in each fixture's docstring.
"""

from __future__ import annotations

import math

import pytest

from reactflow.delta.data import (
    COMPARABLE_MIN_FRACTION,
    CONTROL_NOISE_THRESHOLD_PERCENTILE,
    build_pair_delta_reactivity,
    build_reactivity_layers,
    compute_domain_zscore_stats,
    estimate_pair_noise,
    estimate_replicate_noise,
    evaluate_pair_upgrade,
    freeze_control_noise_threshold,
    normalize_2_8_percent,
)
from reactflow.delta.schema import EXCLUSION_REASONS


# =============================================================================
# Fixture 1: Clean true_pair, full-quality, end-to-end (v3.1 §3.3)
# =============================================================================
#
# WT raw = [1.0, 2.0, 3.0, 4.0];  Mut raw = [1.0, 2.0, 1.0, 4.0]  (edit pos 2)
#
# 2-8% norm WT: finite=[1,2,3,4], n=4, lo=ceil(0.92*4)=4, hi=ceil(0.98*4)=4
#   window=[4.0], scale=4.0 → upstream=[0.25, 0.50, 0.75, 1.00]
# 2-8% norm Mut: finite=[1,1,2,4], n=4, lo=4, hi=4
#   window=[4.0], scale=4.0 → upstream=[0.25, 0.50, 0.25, 1.00]
#
# delta_raw       = [0, 0, -2.0, 0]
# delta_upstream  = [0, 0, -0.5, 0]
#
# Noise threshold (frozen from controls) = 0.30; has_replicates=True
#   |delta_upstream| = [0, 0, 0.5, 0]; 0.5 > 0.30 → sig at pos 2
#   significant_mask = [0, 0, 1, 0]; caller_status = "replicate_aware"
#
# Quality: comp=1.0, snr=10, cov=30, miss=0, rep=True → weight = 1.0
# Exclusion: clean → []; primary_eligible=True; true_pair=True


class TestFixture1CleanTruePair:
    WT_RAW = [1.0, 2.0, 3.0, 4.0]
    MUT_RAW = [1.0, 2.0, 1.0, 4.0]
    NOISE_THRESHOLD = 0.30

    def test_wt_2_8_normalization_handcomputed(self) -> None:
        upstream, scale = normalize_2_8_percent(self.WT_RAW)
        assert scale == pytest.approx(4.0)
        assert upstream == pytest.approx([0.25, 0.50, 0.75, 1.00])

    def test_mut_2_8_normalization_handcomputed(self) -> None:
        upstream, scale = normalize_2_8_percent(self.MUT_RAW)
        assert scale == pytest.approx(4.0)
        assert upstream == pytest.approx([0.25, 0.50, 0.25, 1.00])

    def test_reactivity_layers_raw_upstream_project(self) -> None:
        # No domain stats → project layer falls back to upstream (T-D1.7).
        wt_layers = build_reactivity_layers(self.WT_RAW, "2-8_percent")
        mut_layers = build_reactivity_layers(self.MUT_RAW, "2-8_percent")
        assert wt_layers["reactivity_raw"] == self.WT_RAW
        assert wt_layers["reactivity_upstream"] == pytest.approx([0.25, 0.50, 0.75, 1.00])
        assert wt_layers["reactivity_project"] == wt_layers["reactivity_upstream"]
        assert wt_layers["scale_factor"] == pytest.approx(4.0)
        assert mut_layers["reactivity_upstream"] == pytest.approx([0.25, 0.50, 0.25, 1.00])

    def test_pair_delta_reactivity_handcomputed(self) -> None:
        wt_layers = build_reactivity_layers(self.WT_RAW, "2-8_percent")
        mut_layers = build_reactivity_layers(self.MUT_RAW, "2-8_percent")
        result = build_pair_delta_reactivity(
            wt_reactivity_raw=self.WT_RAW,
            mut_reactivity_raw=self.MUT_RAW,
            wt_reactivity_normalized=wt_layers["reactivity_upstream"],
            mut_reactivity_normalized=mut_layers["reactivity_upstream"],
            noise_threshold=self.NOISE_THRESHOLD,
            has_replicates=True,
        )
        assert result["delta_reactivity_raw"] == pytest.approx([0.0, 0.0, -2.0, 0.0])
        assert result["delta_reactivity_normalized"] == pytest.approx([0.0, 0.0, -0.5, 0.0])
        assert result["significant_mask"] == [0, 0, 1, 0]
        assert result["significant_count"] == 1
        assert result["caller_status"] == "replicate_aware"

    def test_evaluate_pair_upgrade_clean_true_pair(self) -> None:
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
            comparable_fraction=1.0,
            snr=10.0,
            coverage_mean=30.0,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["exclusion_reasons"] == []
        assert result["primary_eligible"] is True
        assert result["true_pair"] is True
        assert result["pair_quality_weight"] == pytest.approx(1.0)


# =============================================================================
# Fixture 2: Missing propagated + comparable < 60% (v3 §6.7: missing 不作 0)
# =============================================================================
#
# WT raw = [1.0, 2.0, None, 4.0];  Mut raw = [1.0, 2.0, 1.0, None]
#
# delta_raw = [0, 0, None, None]  ← None propagated, NOT 0 or -2.0/-3.0
#
# 2-8% norm WT: finite=[1,2,4], n=3, lo=ceil(0.92*3)=3, hi=ceil(0.98*3)=3
#   window=[4.0], scale=4.0 → upstream=[0.25, 0.50, None, 1.00]
# 2-8% norm Mut: finite=[1,1,2], n=3, lo=3, hi=3
#   window=[2.0], scale=2.0 → upstream=[0.50, 1.00, 0.50, None]
#
# delta_upstream = [-0.25, -0.50, None, None]  ← None still propagated
#
# comparable positions (both non-None) = 2/4 = 0.50 < 0.60 → exclusion reason
# missing_fraction = 2/4 = 0.50
#
# Exclusion: [comparable_positions_below_60pct]; primary_eligible=False; true_pair=False
# Weight: 0.5 * 0.5 * 0.5 * 0.8 * 0.5 = 0.05


class TestFixture2MissingPropagated:
    WT_RAW = [1.0, 2.0, None, 4.0]
    MUT_RAW = [1.0, 2.0, 1.0, None]

    def test_delta_raw_propagates_none(self) -> None:
        result = build_pair_delta_reactivity(self.WT_RAW, self.MUT_RAW)
        assert result["delta_reactivity_raw"] == [0.0, 0.0, None, None]
        # Critical: None is NOT 0.0 (v3 §6.7).
        assert result["delta_reactivity_raw"][2] is None
        assert result["delta_reactivity_raw"][3] is None

    def test_upstream_normalization_preserves_none(self) -> None:
        wt_upstream, wt_scale = normalize_2_8_percent(self.WT_RAW)
        mut_upstream, mut_scale = normalize_2_8_percent(self.MUT_RAW)
        assert wt_scale == pytest.approx(4.0)
        assert mut_scale == pytest.approx(2.0)
        assert wt_upstream == pytest.approx([0.25, 0.50, None, 1.00])
        assert mut_upstream == pytest.approx([0.50, 1.00, 0.50, None])
        # None preserved through normalization.
        assert wt_upstream[2] is None
        assert mut_upstream[3] is None

    def test_delta_normalized_propagates_none(self) -> None:
        wt_upstream, _ = normalize_2_8_percent(self.WT_RAW)
        mut_upstream, _ = normalize_2_8_percent(self.MUT_RAW)
        result = build_pair_delta_reactivity(
            self.WT_RAW, self.MUT_RAW,
            wt_reactivity_normalized=wt_upstream,
            mut_reactivity_normalized=mut_upstream,
            noise_threshold=0.10, has_replicates=True,
        )
        delta_norm = result["delta_reactivity_normalized"]
        # delta = r_m - r_w: [0.50-0.25, 1.00-0.50, None, None] = [0.25, 0.50, None, None]
        # Compare element-by-element: None must be preserved (v3 §6.7).
        assert delta_norm[0] == pytest.approx(0.25)
        assert delta_norm[1] == pytest.approx(0.50)
        assert delta_norm[2] is None
        assert delta_norm[3] is None

    def test_comparable_below_60pct_blocks_upgrade(self) -> None:
        # 2 of 4 positions comparable → 0.50 < COMPARABLE_MIN_FRACTION (0.60).
        comparable_fraction = 0.50
        missing_fraction = 0.50
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
            comparable_fraction=comparable_fraction,
            missing_fraction=missing_fraction,
            has_replicates=False,
        )
        assert result["exclusion_reasons"] == ["comparable_positions_below_60pct"]
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False
        # Weight: 0.5 * 0.5 * 0.5 * 0.8 * 0.5 = 0.05
        assert result["pair_quality_weight"] == pytest.approx(0.05)


# =============================================================================
# Fixture 3: Normalization independence (v3 §6.7: 不最小化 pair difference)
# =============================================================================
#
# WT raw = [1.0, 2.0, 4.0];  Mut raw = [1.0, 2.0, 8.0]
#
# 2-8% norm WT: finite=[1,2,4], n=3, lo=3, hi=3, window=[4.0], scale=4.0
#   upstream=[0.25, 0.50, 1.00]
# 2-8% norm Mut: finite=[1,2,8], n=3, lo=3, hi=3, window=[8.0], scale=8.0
#   upstream=[0.125, 0.25, 1.00]
#
# Key: scale factors (4.0, 8.0) are independently determined by each
# construct's own 92-98th percentile. They are NOT jointly optimized to
# minimize |delta_upstream|. A minimum-difference scaling would rescale WT
# by 2.0 to match Mut's shape (both → [0.25, 0.5, 1.0]) giving delta=0; the
# frozen 2-8% rule does not do this.
#
# delta_raw      = [0, 0, 4.0]
# delta_upstream = [-0.125, -0.25, 0.0]  ← NOT [0, 0, 0] (would be min-diff)


class TestFixture3NormalizationIndependence:
    WT_RAW = [1.0, 2.0, 4.0]
    MUT_RAW = [1.0, 2.0, 8.0]

    def test_scale_factors_independently_determined(self) -> None:
        _, wt_scale = normalize_2_8_percent(self.WT_RAW)
        _, mut_scale = normalize_2_8_percent(self.MUT_RAW)
        assert wt_scale == pytest.approx(4.0)
        assert mut_scale == pytest.approx(8.0)
        # Different scales prove per-construct (not pair-level) normalization.

    def test_delta_upstream_not_minimized(self) -> None:
        wt_upstream, _ = normalize_2_8_percent(self.WT_RAW)
        mut_upstream, _ = normalize_2_8_percent(self.MUT_RAW)
        # delta = r_m - r_w (mut minus wt)
        delta_upstream = [
            (m - w) if (w is not None and m is not None) else None
            for w, m in zip(wt_upstream, mut_upstream)
        ]
        # If normalization minimized pair difference, delta would be [0,0,0].
        # Instead it is [-0.125, -0.25, 0.0] — the frozen 2-8% rule's output.
        assert delta_upstream == pytest.approx([-0.125, -0.25, 0.0])
        assert delta_upstream[0] != 0.0  # non-zero proves no min-diff scaling

    def test_pair_delta_not_minimized(self) -> None:
        wt_upstream, _ = normalize_2_8_percent(self.WT_RAW)
        mut_upstream, _ = normalize_2_8_percent(self.MUT_RAW)
        result = build_pair_delta_reactivity(
            self.WT_RAW, self.MUT_RAW,
            wt_reactivity_normalized=wt_upstream,
            mut_reactivity_normalized=mut_upstream,
            noise_threshold=0.5, has_replicates=True,
        )
        assert result["delta_reactivity_raw"] == pytest.approx([0.0, 0.0, 4.0])
        assert result["delta_reactivity_normalized"] == pytest.approx([-0.125, -0.25, 0.0])


# =============================================================================
# Fixture 4: Soft-blocked sequence-based (v3.1 §3.1: corroboration-only)
# =============================================================================
#
# Same reactivity as Fixture 1, but is_sequence_based=True and
# has_independent_corroboration=False.
#
# Exclusion: [sequence_based_no_independent_corroboration]
# primary_eligible = True  (soft blocker, v3.1 §3.1)
# true_pair = False         (any reason disqualifies true_pair, v3.1 §3.3)
#
# This proves the Tier gate is not lowered: a sequence-based pair without
# independent corroboration is NOT counted as true_pair, even though it stays
# primary-eligible for downstream review.


class TestFixture4SoftBlockedSequenceBased:
    def test_sequence_based_no_corroboration_soft_blocks_true_pair(self) -> None:
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
            is_sequence_based=True,
            has_independent_corroboration=False,
        )
        assert result["exclusion_reasons"] == [
            "sequence_based_no_independent_corroboration"
        ]
        assert result["primary_eligible"] is True
        assert result["true_pair"] is False

    def test_sequence_based_with_corroboration_is_true_pair(self) -> None:
        # With independent corroboration, the soft blocker clears.
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
            is_sequence_based=True,
            has_independent_corroboration=True,
            comparable_fraction=1.0,
            snr=10.0,
            coverage_mean=30.0,
            missing_fraction=0.0,
            has_replicates=True,
        )
        assert result["exclusion_reasons"] == []
        assert result["primary_eligible"] is True
        assert result["true_pair"] is True


# =============================================================================
# Fixture 5: Hard-blocked condition mismatch (schema.py L672 invariant)
# =============================================================================
#
# condition_match_status="mismatch" → exclusion reason condition_mismatch.
# primary_eligible=False; true_pair=False.


class TestFixture5HardBlockedConditionMismatch:
    def test_condition_mismatch_blocks_both(self) -> None:
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="mismatch",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
        )
        assert result["exclusion_reasons"] == ["condition_mismatch"]
        assert result["primary_eligible"] is False
        assert result["true_pair"] is False


# =============================================================================
# Fixture 6: Control-frozen noise threshold (v3.1 §4: noise 不用 test 估计)
# =============================================================================
#
# 12 control |Δreactivity| values (no-edit controls, train+validation only):
#   [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
# 95th percentile nearest-rank: rank = ceil(0.95 * 12) = ceil(11.4) = 12
#   → threshold = values[11] = 0.12
#
# The caller then uses 0.12 as the frozen noise threshold. Test data never
# enters this computation — only control |delta| values are passed in.
#
# WT replicate noise (2 replicates):
#   rep1 = [1.0, 2.0, 3.0, 4.0];  rep2 = [1.1, 1.9, 3.1, 3.9]
#   per-pos var (ddof=1):
#     pos0: [1.0,1.1] mean=1.05 var=(0.0025+0.0025)/1=0.005
#     pos1: [2.0,1.9] mean=1.95 var=0.005
#     pos2: [3.0,3.1] mean=3.05 var=0.005
#     pos3: [4.0,3.9] mean=3.95 var=0.005
#   noise_variance = 0.005; noise_std = sqrt(0.005) ≈ 0.07071


class TestFixture6ControlFrozenNoise:
    CONTROL_DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                      0.07, 0.08, 0.09, 0.10, 0.11, 0.12]

    def test_threshold_frozen_at_95th_percentile(self) -> None:
        # rank = ceil(0.95 * 12) = 12 → value = 0.12
        threshold = freeze_control_noise_threshold(self.CONTROL_DELTAS)
        assert threshold == pytest.approx(0.12)

    def test_threshold_uses_only_control_values(self) -> None:
        # The function signature accepts ONLY control |delta| values — there
        # is no parameter for test data. This is the API-level guarantee that
        # noise is not estimated from test (v3.1 §2.2 / §4).
        threshold = freeze_control_noise_threshold(
            self.CONTROL_DELTAS,
            percentile=CONTROL_NOISE_THRESHOLD_PERCENTILE,
        )
        assert threshold == pytest.approx(0.12)

    def test_insufficient_controls_returns_none(self) -> None:
        # Fewer than min_values (10) → None (cannot freeze threshold).
        assert freeze_control_noise_threshold([0.01, 0.02, 0.03]) is None

    def test_replicate_noise_handcomputed(self) -> None:
        rep1 = [1.0, 2.0, 3.0, 4.0]
        rep2 = [1.1, 1.9, 3.1, 3.9]
        result = estimate_replicate_noise([rep1, rep2])
        assert result["n_replicates"] == 2
        assert result["n_positions"] == 4
        assert result["noise_variance"] == pytest.approx(0.005)
        assert result["noise_std"] == pytest.approx(math.sqrt(0.005))

    def test_pair_noise_combines_wt_mut(self) -> None:
        # WT replicate noise variance = 0.005; Mut error variance = 0.002
        # pair measurement_variance = 0.005 + 0.002 = 0.007
        # pair replicate_noise_estimate = sqrt(0.005) (WT only; Mut has no rep)
        wt_rep_noise = estimate_replicate_noise(
            [[1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.1, 3.9]]
        )
        pair = estimate_pair_noise(
            wt_replicate_noise=wt_rep_noise,
            mut_replicate_noise=None,
            wt_error_variance=None,
            mut_error_variance=0.002,
        )
        assert pair["measurement_variance"] == pytest.approx(0.007)
        assert pair["replicate_noise_estimate"] == pytest.approx(math.sqrt(0.005))

    def test_frozen_threshold_feeds_differential_caller(self) -> None:
        # End-to-end: threshold from controls → caller → significant_mask.
        threshold = freeze_control_noise_threshold(self.CONTROL_DELTAS)
        assert threshold == pytest.approx(0.12)
        # delta_upstream from Fixture 1 = [0, 0, -0.5, 0]
        # |delta| = [0, 0, 0.5, 0]; 0.5 > 0.12 → significant at pos 2
        result = build_pair_delta_reactivity(
            wt_reactivity_raw=[1.0, 2.0, 3.0, 4.0],
            mut_reactivity_raw=[1.0, 2.0, 1.0, 4.0],
            wt_reactivity_normalized=[0.25, 0.50, 0.75, 1.00],
            mut_reactivity_normalized=[0.25, 0.50, 0.25, 1.00],
            noise_threshold=threshold,
            has_replicates=True,
        )
        assert result["significant_mask"] == [0, 0, 1, 0]
        assert result["caller_status"] == "replicate_aware"


# =============================================================================
# Fixture 7: Domain z-score normalization (T-D1.7 project layer)
# =============================================================================
#
# Domain members: WT=[1.0, 2.0, 4.0], Mut=[1.0, 2.0, 8.0]
# Pooled non-missing: [1, 2, 4, 1, 2, 8], n=6
#   mean = (1+2+4+1+2+8)/6 = 18/6 = 3.0
#   var  = ((1-3)^2 + (2-3)^2 + (4-3)^2 + (1-3)^2 + (2-3)^2 + (8-3)^2) / 5
#        = (4 + 1 + 1 + 4 + 1 + 25) / 5 = 36/5 = 7.2
#   std  = sqrt(7.2) ≈ 2.68328
#
# WT project (from WT upstream [0.25, 0.50, 1.00]):
#   (0.25 - 3.0)/2.68328 ≈ -1.02533
#   (0.50 - 3.0)/2.68328 ≈ -0.93185
#   (1.00 - 3.0)/2.68328 ≈ -0.74536


class TestFixture7DomainZscore:
    WT_UPSTREAM = [0.25, 0.50, 1.00]
    MUT_UPSTREAM = [0.125, 0.25, 1.00]

    def test_domain_zscore_stats_handcomputed(self) -> None:
        stats = compute_domain_zscore_stats([self.WT_UPSTREAM, self.MUT_UPSTREAM])
        # pooled = [0.25, 0.50, 1.00, 0.125, 0.25, 1.00], n=6
        pooled = [0.25, 0.50, 1.00, 0.125, 0.25, 1.00]
        mean = sum(pooled) / 6
        var = sum((v - mean) ** 2 for v in pooled) / 5
        std = math.sqrt(var)
        assert stats["count"] == 6
        assert stats["mean"] == pytest.approx(mean)
        assert stats["std"] == pytest.approx(std)

    def test_project_layer_applies_zscore(self) -> None:
        stats = compute_domain_zscore_stats([self.WT_UPSTREAM, self.MUT_UPSTREAM])
        wt_layers = build_reactivity_layers(
            self.WT_UPSTREAM,  # raw = upstream (method="raw" passthrough)
            "raw",
            domain_mean=stats["mean"],
            domain_std=stats["std"],
        )
        # project = (v - mean) / std for each non-missing v
        expected = [
            (v - stats["mean"]) / stats["std"] for v in self.WT_UPSTREAM
        ]
        assert wt_layers["reactivity_project"] == pytest.approx(expected)
        # raw == upstream == input (method="raw" passthrough)
        assert wt_layers["reactivity_raw"] == self.WT_UPSTREAM
        assert wt_layers["reactivity_upstream"] == self.WT_UPSTREAM


# =============================================================================
# D1 Gate invariant checks (v3.1 §4)
# =============================================================================


class TestD1GateInvariants:
    """Programmatic checks for the v3.1 §4 D1 Gate bullets."""

    def test_every_exclusion_reason_in_frozen_vocabulary(self) -> None:
        # The 13-reason vocabulary from schema.py must match exactly what
        # collect_exclusion_reasons can emit (v3.1 §4).
        from reactflow.delta.data import collect_exclusion_reasons

        emitted: set[str] = set()
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
        base = dict(
            edit_type="substitution", edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True, has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True, in_vivo_in_vitro_mixed=False,
        )
        for case in cases:
            kw = {**base, **case}
            emitted.update(collect_exclusion_reasons(**kw))
        # Every emitted reason is in the frozen vocabulary.
        assert emitted <= EXCLUSION_REASONS
        # Every frozen reason is reachable (no dead vocabulary entries).
        assert emitted == EXCLUSION_REASONS

    def test_no_training_flag_in_upgrade_output(self) -> None:
        # v3.1 §4: "不自动进入训练" — evaluate_pair_upgrade must NOT emit any
        # training-authorization flag. training_allowed stays False externally.
        result = evaluate_pair_upgrade(
            edit_type="substitution",
            edit_count=1,
            condition_match_status="exact_match",
            substitution_verified=True,
            has_wt_anchor=True,
            normalization_domain_compatible=True,
            parent_lineage_verified=True,
            in_vivo_in_vitro_mixed=False,
        )
        assert "training_allowed" not in result
        assert "training_enabled" not in result
        assert "auto_train" not in result

    def test_true_pair_honest_only_when_no_reasons(self) -> None:
        # v3.1 §4: "Tier gate 不被降低" — true_pair is True ONLY when the
        # exclusion_reasons list is empty. Any reason (including soft blockers)
        # disqualifies true_pair status.
        from reactflow.delta.data import determine_true_pair

        # Empty reasons + eligible → true_pair
        assert determine_true_pair([], True) is True
        # Soft blocker only → NOT true_pair
        assert determine_true_pair(
            ["sequence_based_no_independent_corroboration"], True
        ) is False
        # Hard blocker → NOT true_pair
        assert determine_true_pair(["condition_mismatch"], False) is False

    def test_missing_not_treated_as_zero(self) -> None:
        # v3.1 §4 / v3 §6.7: "missing 不作 0" — None in reactivity must
        # propagate to delta_reactivity as None, never as 0.0.
        result = build_pair_delta_reactivity(
            [1.0, None, 3.0], [None, 2.0, 3.0]
        )
        delta = result["delta_reactivity_raw"]
        assert delta == [None, None, 0.0]
        # Explicitly: None is not 0.0
        assert delta[0] is None
        assert delta[1] is None

    def test_normalization_not_pair_minimizing(self) -> None:
        # v3.1 §4 / v3 §6.7: "normalization 不最小化 pair difference" — the
        # 2-8% scale factor is per-construct, not jointly optimized.
        wt_scale = normalize_2_8_percent([1.0, 2.0, 4.0])[1]
        mut_scale = normalize_2_8_percent([1.0, 2.0, 8.0])[1]
        # Different scales (4.0 vs 8.0) → independent per-construct.
        assert wt_scale != mut_scale
        assert wt_scale == pytest.approx(4.0)
        assert mut_scale == pytest.approx(8.0)
