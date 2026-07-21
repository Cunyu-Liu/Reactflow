"""Tests for the C5.3 frozen-encoder warm-start bridge (:mod:`reactflow.features`).

The adapter is a plain affine map ``a_i = W h_i + b`` with hand-written backprop.
Its parameter gradient is verified against central finite differences (the same
gate used for the denoiser), and every feature-assembly branch (no adapter /
matched frozen row / missing-record zero fallback) is exercised so the module
clears the >=90% coverage gate on its own.
"""

import math

import pytest

from reactflow.features import (
    AdapterGradients,
    AdapterParameters,
    FeatureAdapter,
    FrozenFeatureLookup,
    adapter_sgd_update,
    build_augmented_features,
    load_frozen_features,
    split_feature_gradient,
    zero_single_rows,
)
from reactflow.frozen import (
    ARRAY_SINGLE,
    FrozenFeatureProvenance,
    FrozenFeatureRecord,
    default_schema,
    write_frozen_shard,
)
from reactflow.npio import NdArray


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _single(rows):
    """Build a ``len(rows) x d`` frozen single-representation array."""

    return NdArray.from_nested([list(map(float, row)) for row in rows], kind="float32")


def _provenance(d_single):
    return FrozenFeatureProvenance(
        model_name="RibonanzaNet2",
        model_version="alpha-v1",
        weights_sha256="",
        produced_by="pytest",
        date="2026-07-07",
        schema=default_schema(d_single=d_single),
        notes="unit-test fixture",
    )


# --------------------------------------------------------------------------- #
# AdapterParameters
# --------------------------------------------------------------------------- #
def test_random_init_is_deterministic_and_seeds_zero_bias():
    a = AdapterParameters.random_init(6, 3, seed=7)
    b = AdapterParameters.random_init(6, 3, seed=7)

    assert a.weight == b.weight
    assert a.bias == b.bias == [0.0, 0.0, 0.0]
    assert a.d_adapter == 3
    assert a.d_single == 6
    # default Xavier-style bound |w| <= 1/sqrt(d_single)
    bound = 1.0 / math.sqrt(6)
    assert all(abs(w) <= bound for row in a.weight for w in row)


def test_random_init_respects_custom_scale_and_validates():
    scaled = AdapterParameters.random_init(4, 2, seed=1, scale=0.0)
    assert scaled.weight == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="must be positive"):
        AdapterParameters.random_init(0, 2)
    with pytest.raises(ValueError, match="must be positive"):
        AdapterParameters.random_init(4, 0)


def test_d_single_is_zero_for_empty_weight():
    params = AdapterParameters(weight=[], bias=[])
    assert params.d_single == 0
    assert params.d_adapter == 0


# --------------------------------------------------------------------------- #
# FeatureAdapter.forward
# --------------------------------------------------------------------------- #
def test_forward_matches_manual_affine_map():
    params = AdapterParameters(weight=[[1.0, 2.0], [-1.0, 0.5]], bias=[0.1, -0.2])
    adapter = FeatureAdapter(params)
    rows = [[1.0, 3.0], [2.0, -1.0]]

    out = adapter.forward(rows)

    assert out[0] == pytest.approx([1.0 * 1 + 2.0 * 3 + 0.1, -1.0 * 1 + 0.5 * 3 - 0.2])
    assert out[1] == pytest.approx([1.0 * 2 + 2.0 * -1 + 0.1, -1.0 * 2 + 0.5 * -1 - 0.2])


def test_forward_rejects_wrong_input_dimension():
    adapter = FeatureAdapter(AdapterParameters.random_init(3, 2, seed=0))
    with pytest.raises(ValueError, match="wrong dimension"):
        adapter.forward([[1.0, 2.0]])  # expects 3


# --------------------------------------------------------------------------- #
# FeatureAdapter.backward
# --------------------------------------------------------------------------- #
def test_backward_matches_finite_difference():
    """dL/dW and dL/db must match central finite differences.

    Downstream loss ``L = sum_i sum_p (sin(a_ip) + a_ip^2)`` gives a non-trivial
    ``g_ip = dL/da_ip = cos(a_ip) + 2 a_ip`` that depends on the adapter output,
    so the chain rule is genuinely exercised (not a constant upstream gradient).
    """

    params = AdapterParameters.random_init(4, 3, seed=5)
    adapter = FeatureAdapter(params)
    single_rows = [[0.3, -0.7, 1.1, 0.2], [-0.5, 0.9, -0.1, 0.4]]

    def loss_from_outputs(outputs):
        return sum(math.sin(a) + a * a for row in outputs for a in row)

    outputs = adapter.forward(single_rows)
    grad_output = [[math.cos(a) + 2.0 * a for a in row] for row in outputs]
    grads = adapter.backward(single_rows, grad_output)

    assert isinstance(grads, AdapterGradients)

    epsilon = 1e-6
    max_rel = 0.0
    d_adapter, d_single = params.d_adapter, params.d_single

    for p in range(d_adapter):
        for f in range(d_single):
            original = params.weight[p][f]
            params.weight[p][f] = original + epsilon
            plus = loss_from_outputs(adapter.forward(single_rows))
            params.weight[p][f] = original - epsilon
            minus = loss_from_outputs(adapter.forward(single_rows))
            params.weight[p][f] = original
            numeric = (plus - minus) / (2 * epsilon)
            denom = max(1.0, abs(numeric), abs(grads.weight[p][f]))
            max_rel = max(max_rel, abs(numeric - grads.weight[p][f]) / denom)

    for p in range(d_adapter):
        original = params.bias[p]
        params.bias[p] = original + epsilon
        plus = loss_from_outputs(adapter.forward(single_rows))
        params.bias[p] = original - epsilon
        minus = loss_from_outputs(adapter.forward(single_rows))
        params.bias[p] = original
        numeric = (plus - minus) / (2 * epsilon)
        denom = max(1.0, abs(numeric), abs(grads.bias[p]))
        max_rel = max(max_rel, abs(numeric - grads.bias[p]) / denom)

    assert max_rel < 1e-5


def test_backward_validates_shapes():
    adapter = FeatureAdapter(AdapterParameters.random_init(3, 2, seed=0))
    rows = [[1.0, 2.0, 3.0]]
    with pytest.raises(ValueError, match="grad_output length"):
        adapter.backward(rows, [])
    with pytest.raises(ValueError, match="grad_output row has wrong dimension"):
        adapter.backward(rows, [[1.0]])  # expects 2
    with pytest.raises(ValueError, match="frozen row has wrong dimension"):
        adapter.backward([[1.0, 2.0]], [[1.0, 2.0]])  # frozen row expects 3


# --------------------------------------------------------------------------- #
# adapter_sgd_update
# --------------------------------------------------------------------------- #
def test_adapter_sgd_update_steps_against_gradient():
    params = AdapterParameters(weight=[[1.0, 2.0]], bias=[0.5])
    grads = AdapterGradients(weight=[[0.5, -1.0]], bias=[2.0])

    adapter_sgd_update(params, grads, learning_rate=0.1)

    assert params.weight == [pytest.approx([1.0 - 0.05, 2.0 + 0.1])]
    assert params.bias == [pytest.approx(0.5 - 0.2)]

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        adapter_sgd_update(params, grads, learning_rate=0.0)


# --------------------------------------------------------------------------- #
# FrozenFeatureLookup
# --------------------------------------------------------------------------- #
def test_lookup_is_case_insensitive():
    lookup = FrozenFeatureLookup(d_single=2, by_sequence={"ACGU": ((1.0, 2.0),) * 4})
    assert lookup.has("acgu")
    assert lookup.single_rows("acgu") == ((1.0, 2.0),) * 4
    assert lookup.single_rows("GGGG") is None
    assert not lookup.has("gggg")


# --------------------------------------------------------------------------- #
# load_frozen_features (disk roundtrip + guards)
# --------------------------------------------------------------------------- #
def test_load_frozen_features_roundtrip(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single([[1, 2], [3, 4], [5, 6], [7, 8]])}),
        FrozenFeatureRecord("b", "GGGG", {ARRAY_SINGLE: _single([[0, 0], [1, 1], [2, 2], [3, 3]])}),
    ]
    write_frozen_shard(tmp_path / "shard", records, _provenance(d_single=2))

    lookup = load_frozen_features(tmp_path / "shard")

    assert lookup.d_single == 2
    assert lookup.has("ACGU") and lookup.has("GGGG")
    assert lookup.single_rows("ACGU")[0] == (1.0, 2.0)
    assert lookup.single_rows("GGGG")[3] == (3.0, 3.0)


def test_load_frozen_features_can_skip_verification(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single([[1, 2]] * 4)})]
    write_frozen_shard(tmp_path / "shard", records, _provenance(d_single=2))
    blob = bytearray((tmp_path / "shard" / "features.npz").read_bytes())
    blob[-1] ^= 0xFF
    (tmp_path / "shard" / "features.npz").write_bytes(bytes(blob))

    lookup = load_frozen_features(tmp_path / "shard", verify=False)
    assert lookup.has("ACGU")


def test_load_frozen_features_rejects_inconsistent_d_single(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single([[1, 2]] * 4)}),
        FrozenFeatureRecord("b", "GGGG", {ARRAY_SINGLE: _single([[1, 2, 3]] * 4)}),
    ]
    write_frozen_shard(tmp_path / "shard", records, _provenance(d_single=2))
    with pytest.raises(ValueError, match="inconsistent d_single"):
        load_frozen_features(tmp_path / "shard")


def test_load_frozen_features_rejects_duplicate_sequence(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single([[1, 2]] * 4)}),
        FrozenFeatureRecord("b", "ACGU", {ARRAY_SINGLE: _single([[3, 4]] * 4)}),
    ]
    write_frozen_shard(tmp_path / "shard", records, _provenance(d_single=2))
    with pytest.raises(ValueError, match="duplicate sequence"):
        load_frozen_features(tmp_path / "shard")


def test_load_frozen_features_rejects_empty_shard(tmp_path):
    write_frozen_shard(tmp_path / "shard", [], _provenance(d_single=2))
    with pytest.raises(ValueError, match="no records"):
        load_frozen_features(tmp_path / "shard")


def test_load_frozen_features_lazily_indexes_sharded_directory(tmp_path):
    records_a = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single([[1, 2]] * 4)}),
        FrozenFeatureRecord("dup_a", "GGGG", {ARRAY_SINGLE: _single([[3, 4]] * 4)}),
    ]
    records_b = [
        FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[5, 6]] * 4)}),
        # Exact duplicate sequence in a later shard should not make sequence-only
        # frozen alignment ambiguous for full-scale windowed exports.
        FrozenFeatureRecord("dup_b", "GGGG", {ARRAY_SINGLE: _single([[7, 8]] * 4)}),
    ]
    write_frozen_shard(tmp_path / "multi" / "shard_00000", records_a, _provenance(d_single=2))
    write_frozen_shard(tmp_path / "multi" / "shard_00001", records_b, _provenance(d_single=2))

    lookup = load_frozen_features(tmp_path / "multi")

    assert len(lookup) == 4
    assert lookup.by_sequence == {}
    assert lookup.has("ACGU")
    assert lookup.has("CCCC")
    assert lookup.single_rows("ACGU")[0] == (1.0, 2.0)
    assert lookup.single_rows("CCCC")[0] == (5.0, 6.0)
    # First occurrence wins for exact duplicate sequences.
    assert lookup.single_rows("GGGG")[0] == (3.0, 4.0)


def test_sharded_lookup_targets_single_record_before_caching_siblings(tmp_path):
    records = [
        FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)}),
        FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[2, 0]] * 4)}),
    ]
    write_frozen_shard(tmp_path / "multi" / "shard_00000", records, _provenance(d_single=2))
    write_frozen_shard(
        tmp_path / "multi" / "shard_00001",
        [FrozenFeatureRecord("c", "GGGG", {ARRAY_SINGLE: _single([[3, 0]] * 4)})],
        _provenance(d_single=2),
    )
    lookup = load_frozen_features(tmp_path / "multi")

    assert lookup.row_by_sequence == {"AAAA": 0, "CCCC": 1, "GGGG": 0}
    assert lookup.single_rows("AAAA")[0] == (1.0, 0.0)
    shard_path = next(iter(lookup._loaded_shards))
    assert list(lookup._loaded_shards[shard_path]) == ["AAAA"]
    assert lookup.single_rows("CCCC")[0] == (2.0, 0.0)
    assert list(lookup._loaded_shards[shard_path]) == ["AAAA", "CCCC"]


def test_sharded_lookup_prefetches_batch_members_by_shard(tmp_path):
    records = [
        FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)}),
        FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[2, 0]] * 4)}),
    ]
    write_frozen_shard(tmp_path / "multi" / "shard_00000", records, _provenance(d_single=2))
    write_frozen_shard(
        tmp_path / "multi" / "shard_00001",
        [FrozenFeatureRecord("c", "GGGG", {ARRAY_SINGLE: _single([[3, 0]] * 4)})],
        _provenance(d_single=2),
    )
    lookup = load_frozen_features(tmp_path / "multi")

    cached = lookup.prefetch(["CCCC", "missing", "AAAA", "CCCC"])

    assert cached == 2
    shard_path = next(path for path in lookup._loaded_shards if path.name == "shard_00000")
    assert list(lookup._loaded_shards[shard_path]) == ["CCCC", "AAAA"]
    assert lookup.single_rows("AAAA")[0] == (1.0, 0.0)
    assert lookup.single_rows("CCCC")[0] == (2.0, 0.0)
    assert len(lookup._loaded_shards) == 1


def test_sharded_lookup_prefetch_respects_lru_capacity_in_batch_order(tmp_path):
    for index, sequence in enumerate(("AAAA", "CCCC", "GGGG")):
        write_frozen_shard(
            tmp_path / "multi" / f"shard_{index:05d}",
            [FrozenFeatureRecord(sequence.lower(), sequence, {ARRAY_SINGLE: _single([[float(index), 0.0]] * 4)})],
            _provenance(d_single=2),
        )
    lookup = load_frozen_features(tmp_path / "multi", max_loaded_shards=2)

    cached = lookup.prefetch(["AAAA", "CCCC", "GGGG"])

    assert cached == 2
    assert [path.name for path in lookup._loaded_shards] == ["shard_00000", "shard_00001"]
    assert lookup.single_rows("AAAA")[0] == (0.0, 0.0)
    assert lookup.single_rows("GGGG")[0] == (2.0, 0.0)
    assert [path.name for path in lookup._loaded_shards] == ["shard_00000", "shard_00002"]


def test_sharded_lookup_missing_sequence_returns_none_without_loading(tmp_path):
    lookup = FrozenFeatureLookup(
        d_single=2,
        by_sequence={},
        shard_by_sequence={},
        row_by_sequence={},
    )

    assert lookup.single_rows("AAAA") is None
    assert lookup._loaded_shards == {}


def test_sharded_lookup_fallback_without_row_metadata(tmp_path):
    records = [
        FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)}),
        FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[2, 0]] * 4)}),
    ]
    shard_dir = tmp_path / "multi" / "shard_00000"
    write_frozen_shard(shard_dir, records, _provenance(d_single=2))
    lookup = FrozenFeatureLookup(
        d_single=2,
        by_sequence={},
        shard_by_sequence={"AAAA": shard_dir, "CCCC": shard_dir},
        row_by_sequence=None,
    )

    assert lookup.single_rows("CCCC")[0] == (2.0, 0.0)
    assert list(lookup._loaded_shards[shard_dir]) == ["AAAA", "CCCC"]


def test_sharded_lookup_rejects_targeted_dimension_mismatch(tmp_path):
    shard_dir = tmp_path / "multi" / "shard_00000"
    write_frozen_shard(
        shard_dir,
        [FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0, 9]] * 4)})],
        _provenance(d_single=3),
    )
    lookup = FrozenFeatureLookup(
        d_single=2,
        by_sequence={},
        shard_by_sequence={"AAAA": shard_dir},
        row_by_sequence={"AAAA": 0},
    )

    with pytest.raises(ValueError, match="incompatible with d_single"):
        lookup.single_rows("AAAA")


def test_load_frozen_features_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no frozen shard"):
        load_frozen_features(tmp_path)


def test_load_frozen_features_rejects_inconsistent_sharded_d_single(tmp_path):
    write_frozen_shard(
        tmp_path / "multi" / "shard_00000",
        [FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)})],
        _provenance(d_single=2),
    )
    write_frozen_shard(
        tmp_path / "multi" / "shard_00001",
        [FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[1, 0, 0]] * 4)})],
        _provenance(d_single=3),
    )

    with pytest.raises(ValueError, match="inconsistent d_single"):
        load_frozen_features(tmp_path / "multi")


def test_sharded_lookup_keeps_bounded_lru_cache(tmp_path):
    write_frozen_shard(
        tmp_path / "multi" / "shard_00000",
        [FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)})],
        _provenance(d_single=2),
    )
    write_frozen_shard(
        tmp_path / "multi" / "shard_00001",
        [FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[2, 0]] * 4)})],
        _provenance(d_single=2),
    )
    write_frozen_shard(
        tmp_path / "multi" / "shard_00002",
        [FrozenFeatureRecord("c", "GGGG", {ARRAY_SINGLE: _single([[3, 0]] * 4)})],
        _provenance(d_single=2),
    )

    lookup = load_frozen_features(tmp_path / "multi", max_loaded_shards=2)

    assert lookup.single_rows("AAAA")[0] == (1.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00000"]
    assert lookup.single_rows("CCCC")[0] == (2.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00000", "shard_00001"]
    # Touch shard_00000 so shard_00001 becomes the least recently used entry.
    assert lookup.single_rows("AAAA")[0] == (1.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00001", "shard_00000"]
    assert lookup.single_rows("GGGG")[0] == (3.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00000", "shard_00002"]


def test_sharded_lookup_can_reproduce_single_active_shard_mode(tmp_path):
    write_frozen_shard(
        tmp_path / "multi" / "shard_00000",
        [FrozenFeatureRecord("a", "AAAA", {ARRAY_SINGLE: _single([[1, 0]] * 4)})],
        _provenance(d_single=2),
    )
    write_frozen_shard(
        tmp_path / "multi" / "shard_00001",
        [FrozenFeatureRecord("b", "CCCC", {ARRAY_SINGLE: _single([[2, 0]] * 4)})],
        _provenance(d_single=2),
    )

    lookup = load_frozen_features(tmp_path / "multi", max_loaded_shards=1)

    assert lookup.single_rows("AAAA")[0] == (1.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00000"]
    assert lookup.single_rows("CCCC")[0] == (2.0, 0.0)
    assert list(path.name for path in lookup._loaded_shards) == ["shard_00001"]


# --------------------------------------------------------------------------- #
# zero_single_rows / build_augmented_features / split_feature_gradient
# --------------------------------------------------------------------------- #
def test_zero_single_rows_shape_and_values():
    rows = zero_single_rows(3, 2)
    assert rows == ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def test_build_augmented_features_no_adapter_is_identity():
    base = [(1.0, 2.0), (3.0, 4.0)]
    augmented, used = build_augmented_features(base, None, None)
    assert augmented == ((1.0, 2.0), (3.0, 4.0))
    assert used is None


def test_build_augmented_features_concatenates_adapter_output():
    params = AdapterParameters(weight=[[1.0, 0.0]], bias=[0.5])
    adapter = FeatureAdapter(params)
    base = [(1.0, 2.0), (3.0, 4.0)]
    single_rows = [[2.0, 9.0], [5.0, 9.0]]

    augmented, used = build_augmented_features(base, adapter, single_rows)

    # trailing slot = W h + b = h[0] + 0.5
    assert augmented[0] == pytest.approx((1.0, 2.0, 2.5))
    assert augmented[1] == pytest.approx((3.0, 4.0, 5.5))
    assert used == ((2.0, 9.0), (5.0, 9.0))


def test_build_augmented_features_missing_record_uses_zero_fallback():
    params = AdapterParameters(weight=[[1.0, 3.0]], bias=[0.7])
    adapter = FeatureAdapter(params)
    base = [(1.0, 2.0), (3.0, 4.0)]

    augmented, used = build_augmented_features(base, adapter, None)

    # zero frozen input -> adapter contributes only its bias
    assert augmented[0] == pytest.approx((1.0, 2.0, 0.7))
    assert augmented[1] == pytest.approx((3.0, 4.0, 0.7))
    assert used == ((0.0, 0.0), (0.0, 0.0))


def test_build_augmented_features_rejects_length_mismatch():
    adapter = FeatureAdapter(AdapterParameters.random_init(2, 1, seed=0))
    with pytest.raises(ValueError, match="single_rows length must match"):
        build_augmented_features([(1.0, 2.0)], adapter, [[1.0, 2.0], [3.0, 4.0]])


def test_split_feature_gradient_takes_trailing_block():
    grad_features = [[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]]
    tail = split_feature_gradient(grad_features, base_size=2)
    assert tail == [[0.3, 0.4], [3.0, 4.0]]
