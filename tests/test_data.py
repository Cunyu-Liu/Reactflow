import csv
import math

import pytest

from reactflow.data import (
    PUBLIC_DATASETS,
    RibonanzaProfile,
    coerce_float,
    effective_mask,
    feature_engineering_report,
    inspect_ribonanza2_h5,
    inverse_error_weights,
    normalize_probe_name,
    normalize_profile,
    probe_base_mask,
    read_ribonanza_csv,
    validate_profile,
)


def test_public_dataset_sources_have_verifiable_urls():
    sources = {source.name: source for source in PUBLIC_DATASETS}
    assert set(sources) >= {
        "Ribonanza2 Training Data",
        "Stanford Ribonanza RNA Folding",
        "RNAndria / eFold Dryad Dataset",
        "RibonanzaNet2 Kaggle Model",
    }
    assert sources["RNAndria / eFold Dryad Dataset"].url == "https://doi.org/10.5061/dryad.79cnp5j95"
    assert sources["RibonanzaNet2 Kaggle Model"].url.endswith("/PyTorch/alpha/1")
    for source in PUBLIC_DATASETS:
        assert source.url.startswith("https://")
        assert source.expected_schema


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", math.nan),
        ("nan", math.nan),
        (None, math.nan),
        ("1.25", 1.25),
        (3, 3.0),
    ],
)
def test_coerce_float_handles_missing_and_numeric_values(raw, expected):
    result = coerce_float(raw)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


def test_probe_normalization_and_base_mask_are_probe_specific():
    assert normalize_probe_name("DMS_MaP") == "DMS"
    assert normalize_probe_name("shape") == "2A3"
    assert probe_base_mask("ACGU", "DMS") == (True, True, False, False)
    assert probe_base_mask("ACGU", "2A3") == (True, True, True, True)
    with pytest.raises(ValueError, match="Unsupported probe"):
        normalize_probe_name("CMCT")


def test_validate_profile_reports_completeness_and_quality_gates():
    profile = RibonanzaProfile(
        sequence="ACGU",
        probe="2A3",
        reactivity=(0.1, math.nan, -0.2, 5.0),
        error=(0.01, math.nan, 0.2, 0.5),
        reads=50,
        snr=0.5,
    )

    report = validate_profile(profile)

    assert report.sequence_length == 4
    assert report.valid_positions == 3
    assert report.missing_positions == 1
    assert report.negative_positions == 1
    assert not report.reads_pass
    assert not report.snr_pass
    assert not report.passed
    assert "reads=50" in ";".join(report.messages)


def test_validate_profile_rejects_bad_lengths_and_invalid_bases():
    profile = RibonanzaProfile(sequence="ACGN", probe="DMS", reactivity=(0.1, 0.2))

    report = validate_profile(profile)

    assert not report.passed
    assert any("invalid RNA bases" in message for message in report.messages)
    assert any("reactivity length" in message for message in report.messages)


def test_normalize_profile_supports_p90_minmax_and_zscore():
    values = (0.0, 1.0, 2.0, math.nan, -1.0)

    p90 = normalize_profile(values, method="p90")
    minmax = normalize_profile(values, method="minmax")
    zscore = normalize_profile(values, method="zscore", clip_negative=False)

    assert math.isnan(p90[3])
    assert p90[4] == 0.0
    assert minmax[0] == pytest.approx(1 / 3)
    assert sum(v for v in zscore if math.isfinite(v)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="method"):
        normalize_profile(values, method="bad")
    with pytest.raises(ValueError, match="no finite"):
        normalize_profile((math.nan,))


def test_normalize_profile_handles_constant_and_nonpositive_scales():
    assert normalize_profile((0.0, 0.0), method="p90") == (0.0, 0.0)
    assert normalize_profile((2.0, 2.0), method="minmax") == (0.0, 0.0)


def test_effective_mask_and_inverse_error_weights_respect_probe_and_missing_values():
    profile = RibonanzaProfile(
        sequence="ACGU",
        probe="DMS",
        reactivity=(0.2, math.nan, 0.3, 0.4),
        error=(0.1, 0.2, 0.3, math.nan),
    )

    mask = effective_mask(profile)
    weights = inverse_error_weights(profile.error, mask)

    assert mask == (True, False, False, False)
    assert weights == pytest.approx((100.0, 0.0, 0.0, 0.0))
    assert inverse_error_weights(None, mask) == (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="lengths differ"):
        inverse_error_weights((0.1,), mask)


def test_read_ribonanza_csv_streams_profiles(tmp_path):
    path = tmp_path / "mini.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sequence_id",
                "sequence",
                "experiment_type",
                "reads",
                "signal_to_noise",
                "reactivity_0001",
                "reactivity_0002",
                "reactivity_0003",
                "reactivity_error_0001",
                "reactivity_error_0002",
                "reactivity_error_0003",
            ]
        )
        writer.writerow(["s1", "ACG", "DMS_MaP", "120", "2.5", "0.1", "", "0.3", "0.01", "", "0.03"])

    profile = next(read_ribonanza_csv(path))

    assert profile.sequence_id == "s1"
    assert profile.sequence == "ACG"
    assert profile.probe == "DMS"
    assert profile.reads == 120.0
    assert math.isnan(profile.reactivity[1])
    assert profile.error == pytest.approx((0.01, math.nan, 0.03), nan_ok=True)


def test_read_ribonanza_csv_limit_stops_stream(tmp_path):
    path = tmp_path / "mini_limit.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence", "experiment_type", "reactivity_0001"])
        writer.writerow(["A", "DMS_MaP", "0.1"])
        writer.writerow(["C", "DMS_MaP", "0.2"])

    profiles = list(read_ribonanza_csv(path, limit=1))

    assert len(profiles) == 1
    assert profiles[0].sequence == "A"


def test_read_ribonanza_csv_rejects_files_without_reactivity_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("sequence,experiment_type\nACG,DMS_MaP\n")

    with pytest.raises(ValueError, match="reactivity"):
        list(read_ribonanza_csv(path))


def test_feature_engineering_report_documents_preprocessing_features():
    profile = RibonanzaProfile(sequence="ACGU", probe="2A3", reactivity=(0.2, 0.4, math.nan, 1.0), sequence_id="x")

    report = feature_engineering_report(profile)

    assert report["sequence_id"] == "x"
    assert report["base_counts"] == {"A": 1, "C": 1, "G": 1, "U": 1}
    assert report["gc_fraction"] == pytest.approx(0.5)
    assert report["effective_positions"] == 3


def test_inspect_h5_reports_missing_optional_dependency_or_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("reads", data=[1.0])

    with pytest.raises(ValueError, match="missing required datasets"):
        inspect_ribonanza2_h5(path)
