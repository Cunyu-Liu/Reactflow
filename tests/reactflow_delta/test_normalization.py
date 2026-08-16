"""Tests for T-D1.7 raw / upstream / project-normalized reactivity layers
and normalization-domain freezing (v3 §6.6 step 11, v3.1 §4 D1 Gate).
"""

from __future__ import annotations

import math

import pytest

from reactflow.delta.data import (
    NORMALIZATION_2_8_HIGH_PERCENTILE,
    NORMALIZATION_2_8_LOW_PERCENTILE,
    NORMALIZATION_DOMAIN_FIELDS,
    NORMALIZATION_DOMAIN_NULLABLE_FIELDS,
    NORMALIZATION_DOMAIN_REQUIRED_FIELDS,
    NORMALIZATION_METHODS,
    apply_zscore_normalization,
    build_normalization_domains,
    build_reactivity_layers,
    check_normalization_domain_compatible,
    compute_domain_zscore_stats,
    identify_normalization_domain,
    normalize_2_8_percent,
)
from reactflow.delta.schema import EXCLUSION_REASONS


# -- constants / schema contract ---------------------------------------------


class TestNormalizationConstants:
    def test_methods_present(self) -> None:
        for method in ("raw", "2-8_percent", "boxplot_95th", "upstream_provided",
                       "project_zscore", "unknown"):
            assert method in NORMALIZATION_METHODS

    def test_domain_fields_order(self) -> None:
        assert NORMALIZATION_DOMAIN_FIELDS == (
            "study_id", "probe", "probe_protocol", "in_vivo_in_vitro",
        )

    def test_nullable_only_probe_protocol(self) -> None:
        assert NORMALIZATION_DOMAIN_NULLABLE_FIELDS == frozenset({"probe_protocol"})
        assert NORMALIZATION_DOMAIN_REQUIRED_FIELDS == frozenset(
            {"study_id", "probe", "in_vivo_in_vitro"}
        )

    def test_percentile_window_is_2_to_8(self) -> None:
        assert NORMALIZATION_2_8_LOW_PERCENTILE == 92.0
        assert NORMALIZATION_2_8_HIGH_PERCENTILE == 98.0

    def test_exclusion_reason_registered(self) -> None:
        assert "normalization_domain_unknown" in EXCLUSION_REASONS


# -- identify_normalization_domain -------------------------------------------


def _construct(**overrides) -> dict:
    base = {
        "construct_id": "c1",
        "study_id": "STD1",
        "probe": "DMS",
        "probe_protocol": None,
        "in_vivo_in_vitro": "in_vitro",
    }
    base.update(overrides)
    return base


class TestIdentifyNormalizationDomain:
    def test_full_domain(self) -> None:
        key = identify_normalization_domain(_construct())
        assert key == ("STD1", "DMS", None, "in_vitro")

    def test_nullable_probe_protocol_none_is_valid(self) -> None:
        key = identify_normalization_domain(_construct(probe_protocol=None))
        assert key == ("STD1", "DMS", None, "in_vitro")

    def test_nullable_probe_protocol_value_is_valid(self) -> None:
        key = identify_normalization_domain(_construct(probe_protocol="legacy"))
        assert key == ("STD1", "DMS", "legacy", "in_vitro")

    @pytest.mark.parametrize("missing_field", ["study_id", "probe", "in_vivo_in_vitro"])
    def test_required_field_missing_returns_empty(self, missing_field: str) -> None:
        rec = _construct()
        rec[missing_field] = None
        assert identify_normalization_domain(rec) == ()

    @pytest.mark.parametrize("missing_field", ["study_id", "probe", "in_vivo_in_vitro"])
    def test_required_field_empty_string_returns_empty(self, missing_field: str) -> None:
        rec = _construct()
        rec[missing_field] = ""
        assert identify_normalization_domain(rec) == ()

    def test_distinct_domains_distinct_keys(self) -> None:
        a = identify_normalization_domain(_construct(study_id="STD1"))
        b = identify_normalization_domain(_construct(study_id="STD2"))
        assert a != b
        assert a == ("STD1", "DMS", None, "in_vitro")
        assert b == ("STD2", "DMS", None, "in_vitro")


# -- build_normalization_domains ---------------------------------------------


class TestBuildNormalizationDomains:
    def test_groups_by_domain(self) -> None:
        records = [
            _construct(construct_id="c1", study_id="STD1"),
            _construct(construct_id="c2", study_id="STD1"),
            _construct(construct_id="c3", study_id="STD2"),
        ]
        domains = build_normalization_domains(records)
        assert domains[("STD1", "DMS", None, "in_vitro")]["count"] == 2
        assert domains[("STD1", "DMS", None, "in_vitro")]["construct_ids"] == ["c1", "c2"]
        assert domains[("STD2", "DMS", None, "in_vitro")]["count"] == 1
        assert domains[("STD2", "DMS", None, "in_vitro")]["construct_ids"] == ["c3"]

    def test_unknown_domain_collected_under_empty_key(self) -> None:
        records = [
            _construct(construct_id="c1"),
            _construct(construct_id="c2", probe=None),  # required field None → unknown
        ]
        domains = build_normalization_domains(records)
        assert () in domains
        assert domains[()]["count"] == 1
        assert domains[()]["construct_ids"] == ["c2"]

    def test_empty_input(self) -> None:
        assert build_normalization_domains([]) == {}


# -- check_normalization_domain_compatible -----------------------------------


class TestCheckNormalizationDomainCompatible:
    def test_compatible_same_domain(self) -> None:
        wt = _construct(construct_id="wt")
        mut = _construct(construct_id="mut")
        result = check_normalization_domain_compatible(wt, mut)
        assert result["compatible"] is True
        assert result["domain"] == ("STD1", "DMS", None, "in_vitro")
        assert result["reason"] == ""

    def test_wt_domain_unknown(self) -> None:
        wt = _construct(construct_id="wt", study_id=None)
        mut = _construct(construct_id="mut")
        result = check_normalization_domain_compatible(wt, mut)
        assert result["compatible"] is False
        assert result["domain"] == ()
        assert result["reason"] == "wt_domain_unknown"

    def test_mut_domain_unknown(self) -> None:
        wt = _construct(construct_id="wt")
        mut = _construct(construct_id="mut", probe=None)
        result = check_normalization_domain_compatible(wt, mut)
        assert result["compatible"] is False
        assert result["reason"] == "mut_domain_unknown"

    def test_domain_mismatch(self) -> None:
        wt = _construct(construct_id="wt", study_id="STD1")
        mut = _construct(construct_id="mut", study_id="STD2")
        result = check_normalization_domain_compatible(wt, mut)
        assert result["compatible"] is False
        assert result["reason"] == "domain_mismatch"

    def test_nullable_none_vs_value_is_mismatch(self) -> None:
        wt = _construct(construct_id="wt", probe_protocol=None)
        mut = _construct(construct_id="mut", probe_protocol="legacy")
        result = check_normalization_domain_compatible(wt, mut)
        assert result["compatible"] is False
        assert result["reason"] == "domain_mismatch"


# -- normalize_2_8_percent ---------------------------------------------------


class TestNormalize28Percent:
    def test_known_values_n100(self) -> None:
        values = [float(i) for i in range(1, 101)]  # 1..100
        normalized, scale = normalize_2_8_percent(values)
        # nearest-rank 92nd..98th of 100 = ranks 92..98 = values 92..98, mean=95.0
        assert scale == pytest.approx(95.0)
        assert normalized[0] == pytest.approx(1.0 / 95.0)
        assert normalized[99] == pytest.approx(100.0 / 95.0)

    def test_known_values_n5_takes_top(self) -> None:
        # n=5: lo_rank=ceil(4.6)=5, hi_rank=ceil(4.9)=5 → window=[max]=[50.0]
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        normalized, scale = normalize_2_8_percent(values)
        assert scale == pytest.approx(50.0)
        assert normalized == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])

    def test_preserves_none(self) -> None:
        values = [10.0, None, 30.0, 40.0, 50.0]
        normalized, scale = normalize_2_8_percent(values)
        assert scale == pytest.approx(50.0)
        assert normalized[1] is None
        assert normalized[0] == pytest.approx(0.2)
        assert normalized[4] == pytest.approx(1.0)

    def test_too_few_values_returns_none_scale(self) -> None:
        normalized, scale = normalize_2_8_percent([5.0])
        assert scale is None
        assert normalized == [5.0]

    def test_all_missing_returns_none_scale(self) -> None:
        normalized, scale = normalize_2_8_percent([None, None, None])
        assert scale is None
        assert normalized == [None, None, None]

    def test_two_values(self) -> None:
        # n=2: lo_rank=ceil(1.84)=2, hi_rank=ceil(1.96)=2 → window=[max]
        values = [2.0, 8.0]
        normalized, scale = normalize_2_8_percent(values)
        assert scale == pytest.approx(8.0)
        assert normalized == pytest.approx([0.25, 1.0])

    def test_ignores_non_finite(self) -> None:
        values = [10.0, float("nan"), float("inf"), 40.0, 50.0]
        normalized, scale = normalize_2_8_percent(values)
        # finite = [10,40,50], n=3: lo=ceil(2.76)=3, hi=ceil(2.94)=3 → [50]
        assert scale == pytest.approx(50.0)
        assert normalized[0] == pytest.approx(0.2)
        # nan/inf preserved as-is (not None, since they were in input)
        assert math.isnan(normalized[1])
        assert normalized[2] == float("inf")

    def test_does_not_mutate_input(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        snapshot = list(values)
        normalize_2_8_percent(values)
        assert values == snapshot

    def test_zero_scale_returns_none(self) -> None:
        # All zeros → window mean 0 → scale None, input unchanged
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        normalized, scale = normalize_2_8_percent(values)
        assert scale is None
        assert normalized == [0.0, 0.0, 0.0, 0.0, 0.0]


# -- compute_domain_zscore_stats ---------------------------------------------


class TestComputeDomainZscoreStats:
    def test_known_stats(self) -> None:
        # pooled = [1,2,3,4], mean=2.5, sample std=sqrt(5/3)
        arrs = [[1.0, 2.0], [3.0, 4.0]]
        stats = compute_domain_zscore_stats(arrs)
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["std"] == pytest.approx(math.sqrt(5.0 / 3.0))

    def test_pools_across_constructs_ignoring_none(self) -> None:
        arrs = [[1.0, None, 3.0], [None, 4.0]]
        stats = compute_domain_zscore_stats(arrs)
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx((1.0 + 3.0 + 4.0) / 3.0)

    def test_too_few_returns_none_stats(self) -> None:
        stats = compute_domain_zscore_stats([[5.0]])
        assert stats["mean"] is None
        assert stats["std"] is None
        assert stats["count"] == 1

    def test_all_missing(self) -> None:
        stats = compute_domain_zscore_stats([[None, None], [None]])
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_ignores_non_finite(self) -> None:
        arrs = [[1.0, float("nan"), float("inf")], [2.0, 3.0]]
        stats = compute_domain_zscore_stats(arrs)
        assert stats["count"] == 3  # only 1,2,3


# -- apply_zscore_normalization ----------------------------------------------


class TestApplyZscoreNormalization:
    def test_known_zscore(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        mean = 2.5
        std = math.sqrt(5.0 / 3.0)
        result = apply_zscore_normalization(values, mean, std)
        assert result[0] == pytest.approx((1.0 - 2.5) / std)
        assert result[3] == pytest.approx((4.0 - 2.5) / std)

    def test_preserves_none(self) -> None:
        values = [1.0, None, 3.0]
        result = apply_zscore_normalization(values, 2.0, 1.0)
        assert result[1] is None
        assert result[0] == pytest.approx(-1.0)

    def test_none_mean_returns_input(self) -> None:
        values = [1.0, 2.0]
        assert apply_zscore_normalization(values, None, 1.0) == [1.0, 2.0]

    def test_none_std_returns_input(self) -> None:
        values = [1.0, 2.0]
        assert apply_zscore_normalization(values, 0.0, None) == [1.0, 2.0]

    def test_zero_std_returns_input(self) -> None:
        values = [1.0, 2.0]
        assert apply_zscore_normalization(values, 1.0, 0.0) == [1.0, 2.0]


# -- build_reactivity_layers -------------------------------------------------


class TestBuildReactivityLayers:
    def test_raw_method_no_upstream_no_project(self) -> None:
        layers = build_reactivity_layers([1.0, 2.0, 3.0], "raw")
        assert layers["reactivity_raw"] == [1.0, 2.0, 3.0]
        assert layers["reactivity_upstream"] == [1.0, 2.0, 3.0]
        assert layers["reactivity_project"] == [1.0, 2.0, 3.0]
        assert layers["scale_factor"] is None
        assert layers["normalization_method"] == "raw"

    def test_2_8_percent_method_applies_upstream(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        layers = build_reactivity_layers(values, "2-8_percent")
        assert layers["scale_factor"] == pytest.approx(50.0)
        assert layers["reactivity_upstream"] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
        # project falls back to upstream when no domain stats
        assert layers["reactivity_project"] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
        assert layers["reactivity_raw"] == values

    def test_project_zscore_with_domain_stats(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        # upstream via 2-8% → [0.2,0.4,0.6,0.8,1.0]; then z-score with mean=0.6,std=...
        layers = build_reactivity_layers(
            values, "2-8_percent", domain_mean=0.6, domain_std=0.31622776601683794
        )
        # project should differ from upstream
        assert layers["reactivity_project"][0] == pytest.approx((0.2 - 0.6) / 0.31622776601683794)
        assert layers["reactivity_upstream"][0] == pytest.approx(0.2)

    def test_unknown_method_falls_back_to_raw(self) -> None:
        layers = build_reactivity_layers([1.0, None, 3.0], None)
        assert layers["normalization_method"] == "unknown"
        assert layers["reactivity_upstream"] == [1.0, None, 3.0]
        assert layers["reactivity_project"] == [1.0, None, 3.0]

    def test_missing_preserved_through_layers(self) -> None:
        values = [10.0, None, 30.0, 40.0, 50.0]
        layers = build_reactivity_layers(values, "2-8_percent")
        assert layers["reactivity_raw"][1] is None
        assert layers["reactivity_upstream"][1] is None
        assert layers["reactivity_project"][1] is None

    def test_upstream_provided_method(self) -> None:
        values = [0.1, 0.2, 0.3]
        layers = build_reactivity_layers(values, "upstream_provided")
        assert layers["reactivity_upstream"] == [0.1, 0.2, 0.3]
        assert layers["scale_factor"] is None
