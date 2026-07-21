import builtins
from dataclasses import replace
import importlib.util
import json
import math
import random

import pytest

from reactflow.features import FrozenFeatureLookup, load_frozen_features
from reactflow.frozen import (
    ARRAY_SINGLE,
    FrozenFeatureProvenance,
    FrozenFeatureRecord,
    default_schema,
    write_frozen_shard,
)
from reactflow.npio import NdArray
from reactflow.data import parse_efold_record
from reactflow.synthetic import make_dataset
from reactflow.train import (
    FEATURE_SIZE,
    EpochRecord,
    TrainConfig,
    TrainingResult,
    build_efold_sample_cache,
    build_features,
    bucket_samples_by_length,
    iter_efold_record_windows,
    load_efold_samples,
    order_samples_by_length_bucket,
    order_samples_family_balanced,
    order_samples_for_training,
    partner_classes_from_pair_matrix,
    read_sample_cache,
    sample_from_cache_obj,
    sample_from_efold_record,
    sample_to_cache_obj,
    train_pilot_torch,
    window_efold_record,
    _assert_finite_training_scalar,
    _predicted_reactivity,
    _reactivity_coefficients,
    _reactivity_logit_gradient,
    train_pilot,
)
from reactflow.reactivity import ReactivityForwardOperator


def _frozen_lookup_for(samples, d_single=5, *, seed=0, skip=()):
    """Build a deterministic frozen lookup keyed by sample sequence.

    Sequences whose index is in ``skip`` are omitted so the missing-record
    zero-fallback path inside ``train_pilot`` is exercised.
    """

    rng = random.Random(seed)
    by_sequence = {}
    for index, sample in enumerate(samples):
        if index in skip:
            continue
        seq = sample.sequence.upper()
        by_sequence[seq] = tuple(
            tuple(rng.random() for _ in range(d_single)) for _ in seq
        )
    return FrozenFeatureLookup(d_single=d_single, by_sequence=by_sequence)


def _frozen_provenance(d_single):
    return FrozenFeatureProvenance(
        model_name="RibonanzaNet2",
        model_version="alpha-v1",
        weights_sha256="",
        produced_by="pytest",
        date="2026-07-10",
        schema=default_schema(d_single=d_single),
        notes="train prefetch fixture",
    )


def _single_array(length, d_single):
    return NdArray.from_nested([[float(i + j) for j in range(d_single)] for i in range(length)], kind="float32")


def test_training_finite_guard_rejects_nan_with_location():
    assert _assert_finite_training_scalar(1.25, "loss", epoch=2, sample_index=3) == 1.25

    with pytest.raises(FloatingPointError, match="non-finite training value: loss=nan epoch=2 sample_index=3"):
        _assert_finite_training_scalar(math.nan, "loss", epoch=2, sample_index=3)


def test_build_features_encodes_base_time_and_partner_state():
    sequence = "ACGU"
    noised = [0, 3, 0, 2]  # position1 paired->index2, position3 paired->index1
    features = build_features(sequence, 0.4, noised)

    assert len(features) == 4
    assert all(len(row) == FEATURE_SIZE for row in features)
    # base one-hot for position 0 (A)
    assert features[0][:4] == (1.0, 0.0, 0.0, 0.0)
    # flow time channel
    assert features[0][4] == 0.4
    # position 0 is unpaired -> unpaired flag on, paired flag off, rel 0
    assert features[0][5] == 1.0
    assert features[0][6] == 0.0
    assert features[0][7] == 0.0
    # position 1 paired to index 2 -> paired flag, rel = (2 - 1)/4
    assert features[1][6] == 1.0
    assert features[1][7] == pytest.approx((2 - 1) / 4)


def test_build_features_rejects_length_mismatch():
    with pytest.raises(ValueError, match="noised_classes length"):
        build_features("ACGU", 0.5, [0, 0])


def test_partner_classes_from_pair_matrix_and_efold_conversion():
    record = parse_efold_record(
        {
            "sequence": "GGGAAACCC",
            "structure": [[0, 8], [1, 7], [2, 6]],
            "shape": [0.1, 0.2, 0.3, None, 0.5, 0.6, 0.7, 0.8, 0.9],
            "id": "hairpin",
        }
    )

    sample = sample_from_efold_record(record)

    assert sample.sequence == "GGGAAACCC"
    assert sample.partner_classes == (9, 8, 7, 0, 0, 0, 3, 2, 1)
    assert partner_classes_from_pair_matrix(sample.pair_matrix) == sample.partner_classes
    assert sample.probe == "2A3"
    assert sample.weights[3] == 0.0  # missing shape position
    assert sample.reactivity[0] == pytest.approx(0.1)


def test_sample_from_efold_record_structure_only_uses_forward_proxy():
    record = parse_efold_record({"sequence": "GGGAAACCC", "structure": [[0, 8], [1, 7], [2, 6]]})

    sample = sample_from_efold_record(record)

    assert len(sample.reactivity) == len(sample.sequence)
    assert all(weight == 1.0 for weight in sample.weights)
    # Structure-only records still provide DFM targets; callers can set
    # lambda_react=0 to avoid treating the forward proxy as experimental signal.
    assert sample.partner_classes[0] == 9


def test_load_efold_samples_filters_illegal_records(tmp_path):
    path = tmp_path / "efold.json"
    path.write_text(
        """
        {
          "ok": {"sequence": "GGGAAACCC", "structure": [[0, 8], [1, 7], [2, 6]]},
          "bad": {"sequence": "AAAAAAAAA", "structure": [[0, 8]]}
        }
        """,
        encoding="utf-8",
    )

    samples = load_efold_samples([path], max_length=20)

    assert len(samples) == 1
    assert samples[0].sequence == "GGGAAACCC"


def test_build_efold_sample_cache_round_trips_jsonl(tmp_path):
    path = tmp_path / "efold.json"
    path.write_text(
        """
        {
          "ok_shape": {
            "sequence": "GGGAAACCC",
            "structure": [[0, 8], [1, 7], [2, 6]],
            "shape": [0.1, null, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
          },
          "too_long": {
            "sequence": "GGGAAACCCGGGAAACCC",
            "structure": [[0, 8]]
          },
          "bad_pair": {
            "sequence": "AAAAAAAAA",
            "structure": [[0, 8]]
          },
          "ok_structure_only": {
            "sequence": "GCAAAAGC",
            "structure": [[0, 7], [1, 6]]
          }
        }
        """,
        encoding="utf-8",
    )
    cache = tmp_path / "cache.jsonl"

    summary = build_efold_sample_cache([path], cache, max_length=12)
    samples = read_sample_cache(cache)
    loaded = load_efold_samples([cache])

    assert summary.accepted == 2
    assert summary.skipped_length == 1
    assert summary.skipped_illegal == 1
    assert summary.with_reactivity == 1
    assert len(samples) == 2
    assert len(loaded) == 2
    assert loaded[0].partner_classes == samples[0].partner_classes
    assert loaded[0].weights[1] == 0.0


def test_efold_windowing_keeps_local_pairs_and_metadata():
    record = parse_efold_record(
        {
            "id": "long",
            "sequence": "GGGAAACCCGGGAAACCC",
            "structure": [[0, 8], [1, 7], [2, 6], [9, 17], [10, 16], [11, 15]],
            "shape": [float(i) for i in range(18)],
        }
    )

    window, metadata = window_efold_record(record, start=9, end=18, index=1)
    sample = sample_from_efold_record(window)

    assert window.record_id == "long:9-18"
    assert metadata == {"index": 1, "start": 9, "end": 18, "parent_length": 18}
    assert window.sequence == "GGGAAACCC"
    assert window.pairs == ((0, 8), (1, 7), (2, 6))
    assert sample.partner_classes == (9, 8, 7, 0, 0, 0, 3, 2, 1)
    assert sample.reactivity[0] == pytest.approx(9.0)


def test_build_cache_windows_long_records_and_tracks_buckets(tmp_path):
    path = tmp_path / "efold_long.json"
    path.write_text(
        json.dumps(
            {
                "long": {
                    "sequence": "GGGAAACCCGGGAAACCC",
                    "structure": [[0, 8], [1, 7], [2, 6], [9, 17], [10, 16], [11, 15]],
                }
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "windows.jsonl"

    summary = build_efold_sample_cache(
        [path],
        cache,
        max_length=9,
        window_size=9,
        window_stride=9,
        length_bucket_boundaries=(9,),
    )
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    loaded = load_efold_samples([path], max_length=9, window_size=9, window_stride=9)

    assert summary.scanned == 1
    assert summary.windowed_records == 1
    assert summary.windows_emitted == 2
    assert summary.accepted == 2
    assert summary.length_buckets == {"len_le_9": 2}
    assert [row["window"]["start"] for row in rows] == [0, 9]
    assert all(row["length_bucket"] == "len_le_9" for row in rows)
    assert len(read_sample_cache(cache)) == 2
    assert len(loaded) == 2
    assert {sample.sequence for sample in loaded} == {"GGGAAACCC"}


def test_length_bucket_helpers_group_and_order_samples():
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    reordered = (samples[1], samples[0], samples[2])

    buckets = bucket_samples_by_length(reordered, (8, 9))
    ordered = order_samples_by_length_bucket(reordered, (8, 9))

    assert sum(len(items) for items in buckets.values()) == 3
    assert [len(sample.sequence) for sample in ordered] == sorted(len(sample.sequence) for sample in reordered)


def test_family_balanced_order_interleaves_groups_within_length_bucket():
    base = make_dataset(count=5, stem=4, loop=4, probe="2A3", seed=3)
    samples = (
        replace(base[0], family="A", source_id="a0"),
        replace(base[1], family="A", source_id="a1"),
        replace(base[2], family="A", source_id="a2"),
        replace(base[3], family="B", source_id="b0"),
        replace(base[4], family="B", source_id="b1"),
    )

    ordered = order_samples_family_balanced(samples, (64,))

    assert [sample.source_id for sample in ordered] == ["a0", "b0", "a1", "b1", "a2"]
    assert order_samples_for_training(samples, (64,), family_balanced=True) == ordered
    assert order_samples_for_training(samples, (64,), family_balanced=False) == samples


def test_cache_round_trip_preserves_family_cluster_metadata():
    sample = replace(make_dataset(count=1)[0], source_id="src0", family="famA", cluster="clu1")
    obj = sample_to_cache_obj(sample, reactivity_source="structure_forward_proxy")
    restored = sample_from_cache_obj(obj)

    assert obj["source_id"] == "src0"
    assert obj["family"] == "famA"
    assert obj["cluster"] == "clu1"
    assert restored.source_id == "src0"
    assert restored.family == "famA"
    assert restored.cluster == "clu1"


def test_reactivity_coefficients_and_prediction():
    operator = ReactivityForwardOperator()
    sequence = "ACGU"
    a_values, c_values = _reactivity_coefficients(operator, sequence, "2A3")

    assert len(a_values) == 4
    assert len(c_values) == 4
    # r_hat = a*q + c ; with q = 1 gives a + c
    marginals = [[1.0] + [0.0] * len(sequence) for _ in sequence]
    predicted = _predicted_reactivity(marginals, a_values, c_values)
    for value, a, c in zip(predicted, a_values, c_values):
        assert value == pytest.approx(a + c)


def test_reactivity_logit_gradient_matches_finite_difference():
    # Two positions, K=3 classes; verify d ell_mag / d logit via finite diff.
    from reactflow.dfm import softmax
    from reactflow.reactivity import fit_weighted_affine_calibration, weighted_mse

    logits = [[0.5, -0.2, 0.1], [0.0, 0.3, -0.4]]
    a_values = (1.3, 0.8)
    c_values = (0.05, 0.02)
    target = (0.7, 0.4)
    weights = (1.0, 1.0)
    lambda_react = 1.0

    marginals = [softmax(row) for row in logits]
    predicted = _predicted_reactivity(marginals, a_values, c_values)
    alpha, gamma = fit_weighted_affine_calibration(predicted, target, weights)
    analytic = _reactivity_logit_gradient(marginals, predicted, target, weights, a_values, alpha, gamma, lambda_react)

    epsilon = 1e-6

    # Calibration is held fixed at the base point (block-coordinate minimization),
    # so the finite difference recomputes the magnitude with frozen (alpha, gamma).
    def frozen_mag(rows):
        m = [softmax(row) for row in rows]
        p = _predicted_reactivity(m, a_values, c_values)
        cal = tuple(alpha * v + gamma for v in p)
        return weighted_mse(cal, target, weights)

    for i in range(2):
        for k in range(3):
            plus = [list(row) for row in logits]
            minus = [list(row) for row in logits]
            plus[i][k] += epsilon
            minus[i][k] -= epsilon
            numeric = lambda_react * (frozen_mag(plus) - frozen_mag(minus)) / (2 * epsilon)
            assert analytic[i][k] == pytest.approx(numeric, abs=1e-6)


def test_train_pilot_decreases_loss_and_avoids_collapse():
    result = train_pilot(config=TrainConfig(epochs=30, seed=0))

    assert isinstance(result, TrainingResult)
    history = result.history
    assert len(history) == 30
    assert isinstance(history[0], EpochRecord)

    # Total loss and reactivity magnitude both decrease over training.
    assert history[-1].total < history[0].total
    assert history[-1].react_magnitude < history[0].react_magnitude

    # Non-collapse guard: F1 never drops to zero (a marginal-only degenerate
    # solution would predict no pairs and score F1 = 0).
    assert min(record.mean_f1 for record in history) > 0.0


def test_train_pilot_is_deterministic():
    a = train_pilot(config=TrainConfig(epochs=12, seed=0))
    b = train_pilot(config=TrainConfig(epochs=12, seed=0))

    for ra, rb in zip(a.history, b.history):
        assert ra.total == pytest.approx(rb.total, abs=1e-15)
        assert ra.mean_f1 == pytest.approx(rb.mean_f1, abs=1e-15)


def test_train_pilot_rejects_empty_dataset():
    with pytest.raises(ValueError, match="at least one sample"):
        train_pilot(samples=[])


def test_train_pilot_accepts_custom_samples():
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    result = train_pilot(samples=samples, config=TrainConfig(epochs=5, seed=1))

    assert len(result.history) == 5
    assert result.parameters.feature_size == FEATURE_SIZE


def test_train_pilot_writes_detailed_profile(tmp_path):
    samples = make_dataset(count=2, stem=4, loop=4, probe="2A3", seed=2)
    profile = tmp_path / "train_profile.jsonl"

    result = train_pilot(samples=samples, config=TrainConfig(epochs=1, seed=1, profile_path=str(profile)))
    summary_path = profile.with_suffix(".summary.json")
    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result.profile_summary == summary
    assert profile.exists()
    assert summary_path.exists()
    assert {"model_forward", "model_backward", "epoch_total"} <= set(summary["phases"])
    assert summary["slowest_phase"]["phase"] == summary["phases_by_total_seconds"][0]["phase"]
    assert summary["slowest_step_phase"]["phase"] == summary["step_phases_by_total_seconds"][0]["phase"]
    assert summary["slowest_step_phase"]["phase"] != "epoch_total"
    assert any(event["phase"] == "projection_f1" and event["length"] == len(samples[0].sequence) for event in events)


def test_train_pilot_thermo_branch_runs_and_reports_thermo():
    # lambda_thermo > 0 activates the Turner semi-supervision branch (both the
    # MC prior precomputation and the thermo logit gradient), which the C3/C4
    # defaults leave dormant.
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    result = train_pilot(
        samples=samples,
        config=TrainConfig(epochs=4, seed=0, lambda_thermo=0.5, thermo_samples=16),
    )

    assert len(result.history) == 4
    assert result.history[-1].thermo > 0.0
    assert result.adapter_parameters is None


def test_train_pilot_thermo_kl_mode_runs():
    samples = make_dataset(count=2, stem=4, loop=4, probe="2A3", seed=3)
    result = train_pilot(
        samples=samples,
        config=TrainConfig(
            epochs=3, seed=0, lambda_thermo=0.3, thermo_mode="kl", thermo_samples=16
        ),
    )
    assert result.history[-1].thermo >= 0.0


def test_train_pilot_adapter_requires_frozen_lookup():
    with pytest.raises(ValueError, match="requires a frozen feature lookup"):
        train_pilot(config=TrainConfig(epochs=1, adapter_dim=2))


def test_train_pilot_adapter_expands_feature_size_and_returns_parameters():
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    frozen = _frozen_lookup_for(samples, d_single=5, seed=11)

    result = train_pilot(
        samples=samples,
        config=TrainConfig(epochs=5, seed=0, adapter_dim=3, adapter_lr=0.1),
        frozen=frozen,
    )

    assert isinstance(result, TrainingResult)
    assert result.parameters.feature_size == FEATURE_SIZE + 3
    assert result.adapter_parameters is not None
    assert result.adapter_parameters.d_adapter == 3
    assert result.adapter_parameters.d_single == 5
    # The adapter actually trained: bias starts at zero and must have moved.
    assert any(abs(b) > 0.0 for b in result.adapter_parameters.bias)


def test_train_pilot_prefetches_lazy_frozen_batches(tmp_path):
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    records = [
        FrozenFeatureRecord(
            f"r{index}",
            sample.sequence,
            {ARRAY_SINGLE: _single_array(len(sample.sequence), d_single=3)},
        )
        for index, sample in enumerate(samples)
    ]
    write_frozen_shard(tmp_path / "multi" / "shard_00000", records[:2], _frozen_provenance(3))
    write_frozen_shard(tmp_path / "multi" / "shard_00001", records[2:], _frozen_provenance(3))
    frozen = load_frozen_features(tmp_path / "multi")
    profile = tmp_path / "prefetch_profile.jsonl"

    train_pilot(
        samples=samples,
        config=TrainConfig(
            epochs=1,
            seed=0,
            adapter_dim=2,
            batch_size=2,
            lambda_react=0.0,
            profile_path=str(profile),
        ),
        frozen=frozen,
    )

    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    prefetch_events = [event for event in events if event["phase"] == "frozen_batch_prefetch"]
    assert prefetch_events
    assert any(event["length"] > 0 for event in prefetch_events)


def test_train_pilot_adapter_is_deterministic():
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    frozen = _frozen_lookup_for(samples, d_single=5, seed=11)
    cfg = TrainConfig(epochs=4, seed=0, adapter_dim=2)

    a = train_pilot(samples=samples, config=cfg, frozen=frozen)
    b = train_pilot(samples=samples, config=cfg, frozen=frozen)

    for ra, rb in zip(a.history, b.history):
        assert ra.total == pytest.approx(rb.total, abs=1e-15)
    assert a.adapter_parameters.weight == b.adapter_parameters.weight
    assert a.adapter_parameters.bias == b.adapter_parameters.bias


def test_train_pilot_adapter_handles_missing_frozen_records():
    # One sample has no frozen record -> zero-vector fallback keeps the feature
    # dimensionality constant and training still completes deterministically.
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    frozen = _frozen_lookup_for(samples, d_single=5, seed=11, skip=(1,))

    result = train_pilot(
        samples=samples,
        config=TrainConfig(epochs=3, seed=0, adapter_dim=2),
        frozen=frozen,
    )

    assert result.parameters.feature_size == FEATURE_SIZE + 2
    assert result.adapter_parameters is not None


def test_train_pilot_torch_import_is_lazy(monkeypatch):
    samples = make_dataset(count=1, stem=4, loop=4, probe="2A3", seed=2)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="requires optional dependency PyTorch"):
        train_pilot_torch(samples=samples, config=TrainConfig(epochs=1, seed=0))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch backend is not installed")
def test_train_pilot_torch_runs_when_available():
    samples = make_dataset(count=2, stem=4, loop=4, probe="2A3", seed=2)

    result = train_pilot_torch(samples=samples, config=TrainConfig(epochs=2, seed=0))

    assert isinstance(result, TrainingResult)
    assert len(result.history) == 2
    assert result.parameters.feature_size == FEATURE_SIZE


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch backend is not installed")
def test_train_pilot_torch_rejects_non_finite_parameters():
    samples = make_dataset(count=2, stem=4, loop=4, probe="2A3", seed=2)

    with pytest.raises(FloatingPointError, match="non-finite training tensor: parameter:"):
        train_pilot_torch(samples=samples, config=TrainConfig(epochs=1, seed=0, learning_rate=math.inf))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch backend is not installed")
def test_train_pilot_torch_prefetches_lazy_frozen_batches(tmp_path):
    samples = make_dataset(count=3, stem=4, loop=4, probe="2A3", seed=2)
    records = [
        FrozenFeatureRecord(
            f"torch_r{index}",
            sample.sequence,
            {ARRAY_SINGLE: _single_array(len(sample.sequence), d_single=3)},
        )
        for index, sample in enumerate(samples)
    ]
    write_frozen_shard(tmp_path / "multi" / "shard_00000", records[:2], _frozen_provenance(3))
    write_frozen_shard(tmp_path / "multi" / "shard_00001", records[2:], _frozen_provenance(3))
    frozen = load_frozen_features(tmp_path / "multi")
    profile = tmp_path / "torch_prefetch_profile.jsonl"

    result = train_pilot_torch(
        samples=samples,
        config=TrainConfig(
            epochs=1,
            seed=0,
            adapter_dim=2,
            batch_size=2,
            lambda_react=0.0,
            profile_path=str(profile),
        ),
        frozen=frozen,
        device="cpu",
    )

    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    prefetch_events = [event for event in events if event["phase"] == "frozen_batch_prefetch"]
    assert result.adapter_parameters is not None
    assert prefetch_events
    assert any(event["length"] > 0 for event in prefetch_events)


def test_lambda_calib_zero_is_bit_for_bit_identical():
    samples = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=1)
    base = train_pilot(samples=samples, config=TrainConfig(epochs=5, seed=0, lambda_calib=0.0))
    # Explicit zero with non-trivial beta/tau must still be a no-op.
    same = train_pilot(
        samples=samples,
        config=TrainConfig(epochs=5, seed=0, lambda_calib=0.0, calib_beta=0.7, calib_tau_squared=0.2),
    )
    assert [r.total for r in base.history] == [r.total for r in same.history]
    assert all(r.calib == 0.0 for r in base.history)


def test_lambda_calib_positive_changes_trajectory_and_reduces_calib_nll():
    samples = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=1)
    off = train_pilot(samples=samples, config=TrainConfig(epochs=8, seed=0, lambda_calib=0.0))
    on = train_pilot(
        samples=samples,
        config=TrainConfig(epochs=8, seed=0, lambda_calib=0.5, calib_beta=1.0, calib_tau_squared=0.1),
    )
    # Turning the term on must actually change the optimization trajectory.
    assert [r.total for r in off.history] != [r.total for r in on.history]
    # The calibration NLL is monitored and should not increase over training.
    assert on.history[-1].calib <= on.history[0].calib + 1e-9


def test_lambda_contact_zero_is_bit_for_bit_identical():
    samples = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=1)
    base = train_pilot(samples=samples, config=TrainConfig(epochs=5, seed=0, lambda_contact=0.0))
    same = train_pilot(
        samples=samples,
        config=TrainConfig(
            epochs=5,
            seed=0,
            lambda_contact=0.0,
            contact_negative_weight=0.9,
            contact_long_range_min_distance=2,
            contact_long_range_weight=5.0,
        ),
    )

    assert [r.total for r in base.history] == [r.total for r in same.history]
    assert all(r.contact == 0.0 for r in base.history)


def test_lambda_contact_positive_changes_trajectory_and_records_loss():
    samples = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=1)
    off = train_pilot(samples=samples, config=TrainConfig(epochs=6, seed=0, lambda_contact=0.0))
    on = train_pilot(
        samples=samples,
        config=TrainConfig(
            epochs=6,
            seed=0,
            lambda_contact=0.2,
            contact_negative_weight=0.25,
            contact_long_range_min_distance=2,
            contact_long_range_weight=3.0,
        ),
    )

    assert [r.total for r in off.history] != [r.total for r in on.history]
    assert on.history[0].contact > 0.0
