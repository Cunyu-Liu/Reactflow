from pathlib import Path

import pytest

from reactflow.checkpoint import write_training_checkpoint
from reactflow.constraints import project_greedy_matching
from reactflow.inference import (
    DecoderConfig,
    InferenceConfig,
    InferenceMode,
    MatchingPolicy,
    decode_calibrated_marginal,
    pair_vs_unpaired_log_odds,
    predict_structure,
)
from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
from reactflow.sampling import sample_structure
from reactflow.synthetic import make_dataset
from reactflow.train import TrainConfig, build_features, train_pilot


def _checkpoint(tmp_path: Path):
    samples = make_dataset(count=2, stem=4, loop=4, seed=3)
    config = TrainConfig(epochs=1, seed=5)
    result = train_pilot(samples=samples, config=config)
    path = tmp_path / "checkpoint.json"
    write_training_checkpoint(path, config=config, result=result, metadata={"fixture": True})
    return path, samples[0], result, config


def test_default_inference_is_not_legacy_endpoint():
    assert InferenceConfig().mode is InferenceMode.CALIBRATED_MARGINAL


def test_legacy_direct_exactly_matches_historical_projection(tmp_path):
    path, sample, result, config = _checkpoint(tmp_path)
    model = PairwiseDenoiser(result.parameters, min_loop=config.min_loop)
    features = build_features(sample.sequence, 1.0, [0 for _ in sample.sequence])
    soft = marginal_pair_matrix(model.forward(sample.sequence, features).marginals)
    expected = project_greedy_matching(
        sample.sequence,
        soft,
        min_loop=config.min_loop,
        allow_wobble=True,
        allow_pseudoknot=True,
        min_score=1e-6,
    )
    actual = predict_structure(
        path,
        sample.sequence,
        inference_config=InferenceConfig(mode=InferenceMode.LEGACY_DIRECT),
        decoder_config=DecoderConfig(min_loop=config.min_loop),
    )
    assert actual.structure == expected
    assert actual.provenance["legacy_endpoint_path"] is True


def test_ctmc_is_deterministic_legal_and_uses_dynamic_feature_builder(tmp_path):
    path, sample, _result, config = _checkpoint(tmp_path)
    inference = InferenceConfig(mode=InferenceMode.CTMC_SAMPLE, num_steps=4, num_samples=3, seed=7)
    first = predict_structure(path, sample.sequence, inference_config=inference)
    second = predict_structure(path, sample.sequence, inference_config=inference)
    assert first.validation.valid
    assert len(first.ensemble) == 3
    assert [item.partner_classes for item in first.ensemble] == [item.partner_classes for item in second.ensemble]
    assert first.provenance["legacy_endpoint_path"] is False

    calls = []
    model = PairwiseDenoiser(_result.parameters, min_loop=config.min_loop)

    def builder(sequence, t, states):
        calls.append((t, tuple(states)))
        return build_features(sequence, t, states)

    sample_structure(model, sample.sequence, num_steps=4, seed=1, feature_builder=builder)
    assert len(calls) == 5
    assert [call[0] for call in calls] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_pair_log_odds_prefers_pair_over_high_null_probability():
    pair = ((0.0, 0.8), (0.8, 0.0))
    low_null = pair_vs_unpaired_log_odds(pair, (0.1, 0.1))
    high_null = pair_vs_unpaired_log_odds(pair, (0.9, 0.9))
    assert low_null[0][1] > 0.0
    assert high_null[0][1] < low_null[0][1]


def test_decoder_threshold_and_matching_policy_control_null_choice():
    sequence = "GGGAAACCC"
    size = len(sequence)
    pair = [[0.0 for _ in range(size)] for _ in range(size)]
    pair[0][8] = pair[8][0] = 0.8
    unpaired = [0.1 for _ in range(size)]
    accepted = decode_calibrated_marginal(
        sequence,
        pair,
        unpaired,
        DecoderConfig(threshold=0.0, matching_policy=MatchingPolicy.NESTED_DP),
    )
    rejected = decode_calibrated_marginal(
        sequence,
        pair,
        unpaired,
        DecoderConfig(threshold=100.0, matching_policy=MatchingPolicy.PSEUDOKNOT_ALLOWED_GREEDY),
    )
    assert accepted[0][8] == 1
    assert sum(sum(row) for row in rejected) == 0


def test_manifest_parameters_are_validated():
    with pytest.raises(ValueError, match="temperature"):
        DecoderConfig(temperature=0.0)
    with pytest.raises(ValueError, match="num_steps"):
        InferenceConfig(num_steps=0)
