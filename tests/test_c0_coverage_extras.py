"""Targeted branch-coverage tests for C0 modules.

These tests exercise validation and edge-case branches in
``reactflow.inference``, ``reactflow.c0_evaluate``, ``reactflow.probing``
and ``reactflow.protocol`` that are not reached by the existing functional
tests.  They are intentionally small and do not re-test behaviour already
covered elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.c0_evaluate import (
    aggregate_structure_records,
    code_sha256,
    frozen_feature_provenance,
    read_decoder_manifest,
    shifted_pair_counts,
    structure_record_metrics,
)
from reactflow.inference import (
    DecoderConfig,
    InferenceConfig,
    InferenceMode,
    MatchingPolicy,
    pair_vs_unpaired_log_odds,
    predict_structure,
)
from reactflow.probing import (
    ProfilePrediction,
    aggregate_full_profiles,
    fit_probe_calibration,
)
from reactflow.protocol import (
    MMSEQS_COMPONENT_HOLDOUT,
    MMSEQS_COMPONENT_TEST,
    normalize_tier_label,
    stable_subset,
)
from reactflow.checkpoint import write_training_checkpoint
from reactflow.synthetic import make_dataset
from reactflow.train import TrainConfig, train_pilot


# ---------------------------------------------------------------------------
# inference.py
# ---------------------------------------------------------------------------


def test_pair_log_odds_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="temperature and epsilon"):
        pair_vs_unpaired_log_odds([[0.0]], [0.0], epsilon=0.0)


def test_pair_log_odds_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        pair_vs_unpaired_log_odds([[0.0, 0.0], [0.0, 0.0]], [0.0])


def test_decode_calibrated_marginal_rejects_unknown_policy():
    class _Unknown:
        pass

    decoder = DecoderConfig(matching_policy=MatchingPolicy.NESTED_DP)
    # Mutate the enum to force the fallthrough raise path.
    object.__setattr__(
        decoder,
        "matching_policy",
        _Unknown(),
    )
    with pytest.raises(ValueError, match="unsupported matching policy"):
        # Use a small valid matrix; the function should reach the raise.
        predict_structure  # noqa: F841  (sanity reference)
        from reactflow.inference import decode_calibrated_marginal

        decode_calibrated_marginal(
            "GGGAAACCC",
            [[0.0] * 9 for _ in range(9)],
            [0.1] * 9,
            decoder,
        )


def test_predict_structure_accepts_training_checkpoint_directly(tmp_path):
    """Cover the ``isinstance(value, TrainingCheckpoint)`` short-circuit."""

    samples = make_dataset(count=1, stem=4, loop=4, seed=11)
    config = TrainConfig(epochs=1, seed=13)
    result = train_pilot(samples=samples, config=config)
    path = tmp_path / "checkpoint.json"
    write_training_checkpoint(path, config=config, result=result, metadata={"fixture": True})

    from reactflow.checkpoint import read_training_checkpoint

    restored = read_training_checkpoint(path)
    direct = predict_structure(
        restored,
        samples[0].sequence,
        inference_config=InferenceConfig(mode=InferenceMode.LEGACY_DIRECT),
    )
    via_path = predict_structure(
        path,
        samples[0].sequence,
        inference_config=InferenceConfig(mode=InferenceMode.LEGACY_DIRECT),
    )
    assert direct.structure == via_path.structure
    assert direct.provenance["checkpoint_path"] is None
    assert direct.provenance["checkpoint_sha256"] is None
    assert via_path.provenance["checkpoint_sha256"] is not None


def test_predict_structure_rejects_empty_sequence(tmp_path):
    samples = make_dataset(count=1, stem=4, loop=4, seed=17)
    config = TrainConfig(epochs=1, seed=19)
    result = train_pilot(samples=samples, config=config)
    path = tmp_path / "checkpoint.json"
    write_training_checkpoint(path, config=config, result=result, metadata={"fixture": True})
    with pytest.raises(ValueError, match="sequence must be non-empty"):
        predict_structure(
            path,
            "",
            inference_config=InferenceConfig(mode=InferenceMode.LEGACY_DIRECT),
        )


# ---------------------------------------------------------------------------
# c0_evaluate.py
# ---------------------------------------------------------------------------


def test_shifted_pair_counts_handles_predictions_without_matches():
    size = 6
    predicted = [[0.0] * size for _ in range(size)]
    target = [[0.0] * size for _ in range(size)]
    # Predicted pair (0,5); target pair (1,3) - too far away for tolerance=1.
    predicted[0][5] = predicted[5][0] = 1.0
    target[1][3] = target[3][1] = 1.0
    counts = shifted_pair_counts(predicted, target, tolerance=1)
    assert counts == {"tp": 0, "fp": 1, "fn": 1}


def test_aggregate_structure_records_returns_count_zero_for_empty_input():
    assert aggregate_structure_records([]) == {"count": 0}


def test_frozen_feature_provenance_handles_none_path():
    result = frozen_feature_provenance(None)
    assert result == {
        "present": False,
        "path": None,
        "manifest_path": None,
        "manifest_sha256": None,
    }


def test_frozen_feature_provenance_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="frozen feature directory does not exist"):
        frozen_feature_provenance(tmp_path / "does_not_exist")


def test_frozen_feature_provenance_finds_legacy_manifest(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    # No sharded_manifest.json; fall back to manifest.json.
    legacy = root / "manifest.json"
    legacy.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    result = frozen_feature_provenance(root)
    assert result["present"] is True
    assert result["manifest_path"].endswith("manifest.json")
    assert len(result["manifest_sha256"]) == 64


def test_frozen_feature_provenance_falls_back_to_provenance_json(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    provenance = root / "provenance.json"
    provenance.write_text(json.dumps({"alt": True}), encoding="utf-8")
    result = frozen_feature_provenance(root)
    assert result["manifest_path"].endswith("provenance.json")


def test_frozen_feature_provenance_raises_when_no_manifest_present(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    (root / "data.bin").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="no auditable top-level manifest"):
        frozen_feature_provenance(root)


def _write_decoder_manifest(
    path: Path,
    *,
    schema_version: int = 1,
    checkpoint_sha256: str = "deadbeef",
    fitted_split: str = "validation",
    code_sha256_value: str | None = "ignored",
) -> None:
    payload = {
        "schema_version": schema_version,
        "checkpoint_sha256": checkpoint_sha256,
        "fitted_split": fitted_split,
    }
    if code_sha256_value is not None:
        payload["code_sha256"] = code_sha256_value
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint_path_with_known_hash(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    return checkpoint


def test_read_decoder_manifest_rejects_wrong_schema(tmp_path):
    manifest = tmp_path / "decoder.json"
    _write_decoder_manifest(manifest, schema_version=2)
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    with pytest.raises(ValueError, match="unsupported decoder manifest schema"):
        read_decoder_manifest(manifest, checkpoint_path=checkpoint)


def test_read_decoder_manifest_rejects_checkpoint_hash_mismatch(tmp_path):
    manifest = tmp_path / "decoder.json"
    _write_decoder_manifest(manifest, checkpoint_sha256="00" * 32)
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    with pytest.raises(ValueError, match="decoder manifest checkpoint hash mismatch"):
        read_decoder_manifest(manifest, checkpoint_path=checkpoint)


def test_read_decoder_manifest_rejects_wrong_fitted_split(tmp_path):
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    manifest = tmp_path / "decoder.json"
    _write_decoder_manifest(
        manifest,
        checkpoint_sha256=code_sha256(),  # not used; we override below
        fitted_split="test",
    )
    # Patch the manifest to use the actual checkpoint hash but test split.
    actual_hash = read_decoder_manifest.__globals__["sha256_path"](checkpoint)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = actual_hash
    payload["fitted_split"] = "test"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not fitted on validation"):
        read_decoder_manifest(manifest, checkpoint_path=checkpoint)


def test_read_decoder_manifest_rejects_missing_code_hash(tmp_path):
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    manifest = tmp_path / "decoder.json"
    _write_decoder_manifest(manifest, code_sha256_value=None)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = read_decoder_manifest.__globals__["sha256_path"](checkpoint)
    payload["fitted_split"] = "validation"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing code hash"):
        read_decoder_manifest(manifest, checkpoint_path=checkpoint)


def test_read_decoder_manifest_rejects_code_hash_mismatch(tmp_path):
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    manifest = tmp_path / "decoder.json"
    _write_decoder_manifest(manifest, checkpoint_sha256="x", code_sha256_value="0" * 64)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = read_decoder_manifest.__globals__["sha256_path"](checkpoint)
    payload["fitted_split"] = "validation"
    payload["code_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decoder manifest code hash mismatch"):
        read_decoder_manifest(manifest, checkpoint_path=checkpoint)


def test_read_decoder_manifest_accepts_valid_payload(tmp_path):
    checkpoint = _checkpoint_path_with_known_hash(tmp_path)
    manifest = tmp_path / "decoder.json"
    actual_hash = read_decoder_manifest.__globals__["sha256_path"](checkpoint)
    _write_decoder_manifest(manifest, checkpoint_sha256=actual_hash, code_sha256_value=code_sha256())
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = actual_hash
    payload["code_sha256"] = code_sha256()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = read_decoder_manifest(manifest, checkpoint_path=checkpoint)
    assert result["checkpoint_sha256"] == actual_hash


# ---------------------------------------------------------------------------
# protocol.py
# ---------------------------------------------------------------------------


def test_normalize_tier_label_rejects_empty_input():
    with pytest.raises(ValueError, match="tier label must be non-empty"):
        normalize_tier_label("")


def test_normalize_tier_label_strips_whitespace():
    assert normalize_tier_label("  in_clan  ") == MMSEQS_COMPONENT_TEST
    assert normalize_tier_label("\tnovel_clan\n") == MMSEQS_COMPONENT_HOLDOUT


def test_stable_subset_rejects_negative_count():
    with pytest.raises(ValueError, match="count must be non-negative"):
        stable_subset([], -1, source_id=lambda x: x, sequence=lambda x: "A")


def test_stable_subset_returns_empty_for_zero_count():
    rows = [{"id": "a", "sequence": "AUGC"}]
    assert stable_subset(rows, 0, source_id=lambda r: r["id"], sequence=lambda r: r["sequence"]) == ()


# ---------------------------------------------------------------------------
# probing.py
# ---------------------------------------------------------------------------


def _profile(source_id, predicted, target, *, source="real_profile", probe="DMS", snr=None, quality=None):
    return ProfilePrediction(
        source_id=source_id,
        probe=probe,
        predicted=predicted,
        target=target,
        weights=[1.0] * len(target),
        reactivity_source=source,
        length=len(target),
        snr=snr,
        quality=quality,
    )


def test_probe_calibration_rejects_mismatched_profile_lengths():
    with pytest.raises(ValueError, match="predicted, target and weights must have the same length"):
        fit_probe_calibration(
            [_profile("a", [0.0, 1.0], [1.0])],
            split="validation",
        )


def test_aggregate_full_profiles_skips_records_with_no_valid_positions():
    # All NaN predicted values -> _valid returns empty -> record skipped.
    records = [
        ProfilePrediction(
            source_id="nan",
            probe="DMS",
            predicted=[float("nan"), float("nan")],
            target=[1.0, 2.0],
            weights=[1.0, 1.0],
            reactivity_source="real_profile",
            length=2,
        )
    ]
    result = aggregate_full_profiles(records, {})
    # No valid positions, so calibrated/raw metrics are None and counts are zero.
    assert result["main"]["valid_position_count"] == 0
    assert result["main"]["raw_mae"] is None
    assert result["main"]["calibrated_mae"] is None


def test_aggregate_full_profiles_handles_records_with_fewer_than_three_valid_points():
    # Only 2 valid positions -> profile correlation skipped, pooled metrics still computed.
    records = [_profile("a", [0.0, 1.0], [1.0, 3.0])]
    result = aggregate_full_profiles(records, fit_probe_calibration(records, split="validation"))
    assert result["main"]["valid_position_count"] == 2
    assert result["main"]["correlation_profile_count"] == 0
    assert result["main"]["profile_macro_pearson"] is None
    assert result["main"]["calibrated_mae"] is not None


def test_aggregate_full_profiles_counts_unknown_reactivity_source_as_excluded():
    records = [
        ProfilePrediction(
            source_id="other",
            probe="DMS",
            predicted=[0.0],
            target=[1.0],
            weights=[1.0],
            reactivity_source="some_other_source",
            length=1,
        )
    ]
    result = aggregate_full_profiles(records, {})
    assert result["excluded_unknown_source_count"] == 1
    assert result["main"]["profile_count"] == 0
    assert result["diagnostic_proxy"]["profile_count"] == 0


def test_aggregate_full_profiles_length_buckets_cover_large_transcripts():
    # Length > 1024 should hit the ``len_gt_1024`` bucket.
    big = _profile("big", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    big = ProfilePrediction(
        source_id=big.source_id,
        probe=big.probe,
        predicted=big.predicted,
        target=big.target,
        weights=big.weights,
        reactivity_source=big.reactivity_source,
        length=2048,
    )
    result = aggregate_full_profiles([big], {})
    assert "len_gt_1024" in result["strata"]["length"]


def test_aggregate_full_profiles_snr_bucket_splits_at_one():
    low_snr = _profile("low", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], snr=0.5)
    high_snr = _profile("high", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], snr=2.0)
    result = aggregate_full_profiles([low_snr, high_snr], {})
    assert "snr_lt_1" in result["strata"]["snr"]
    assert "snr_ge_1" in result["strata"]["snr"]


def test_structure_metrics_with_no_target_pairs_returns_null_ratio():
    size = 4
    predicted = [[0.0] * size for _ in range(size)]
    target = [[0.0] * size for _ in range(size)]
    # No pairs anywhere.
    metrics = structure_record_metrics(predicted, target)
    assert metrics["pair_count_ratio"] is None
    assert metrics["predicted_pair_count"] == 0
    assert metrics["target_pair_count"] == 0
