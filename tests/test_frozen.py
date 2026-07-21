import json

import pytest

from reactflow.frozen import (
    ARRAY_PAIR,
    ARRAY_REACT,
    ARRAY_SINGLE,
    FrozenFeatureProvenance,
    FrozenFeatureRecord,
    default_schema,
    read_frozen_single_arrays,
    read_frozen_single_array,
    read_frozen_shard,
    write_frozen_shard,
)
from reactflow.npio import NdArray


def _single(length: int, dim: int = 4) -> NdArray:
    return NdArray.from_nested([[float(i * dim + j) for j in range(dim)] for i in range(length)], kind="float32")


def _pair(length: int, dim: int = 2) -> NdArray:
    return NdArray.from_nested(
        [[[float(k) for k in range(dim)] for _ in range(length)] for _ in range(length)],
        kind="float32",
    )


def _react(length: int, probes: int = 2) -> NdArray:
    return NdArray.from_nested([[0.1 * i for _ in range(probes)] for i in range(length)], kind="float32")


def _provenance(**overrides) -> FrozenFeatureProvenance:
    base = dict(
        model_name="RibonanzaNet2",
        model_version="alpha-v1",
        weights_sha256="",
        produced_by="pytest",
        date="2026-07-07",
        schema=default_schema(d_single=4, d_pair=2, n_probe=2),
        notes="unit-test fixture",
    )
    base.update(overrides)
    return FrozenFeatureProvenance(**base)


def test_default_schema_includes_only_supplied_arrays():
    single_only = default_schema(d_single=384)
    assert set(single_only) == {ARRAY_SINGLE}
    full = default_schema(d_single=384, d_pair=128, n_probe=2)
    assert set(full) == {ARRAY_SINGLE, ARRAY_PAIR, ARRAY_REACT}
    assert full[ARRAY_SINGLE]["axes"] == ["L", 384]
    assert full[ARRAY_PAIR]["axes"] == ["L", "L", 128]


def test_record_requires_single_and_validates_shapes():
    with pytest.raises(ValueError, match="missing required array"):
        FrozenFeatureRecord(record_id="r", sequence="ACGU", arrays={})
    with pytest.raises(ValueError, match="'single' shape"):
        FrozenFeatureRecord(record_id="r", sequence="ACGU", arrays={ARRAY_SINGLE: _single(3)})
    with pytest.raises(ValueError, match="'pair' shape"):
        FrozenFeatureRecord(
            record_id="r",
            sequence="ACGU",
            arrays={ARRAY_SINGLE: _single(4), ARRAY_PAIR: _pair(3)},
        )
    with pytest.raises(ValueError, match="'react_logits' shape"):
        FrozenFeatureRecord(
            record_id="r",
            sequence="ACGU",
            arrays={ARRAY_SINGLE: _single(4), ARRAY_REACT: _react(3)},
        )


def test_record_accessors():
    record = FrozenFeatureRecord(
        record_id="r",
        sequence="ACGUA",
        arrays={ARRAY_SINGLE: _single(5), ARRAY_PAIR: _pair(5), ARRAY_REACT: _react(5)},
        family="CL00001",
    )
    assert record.length == 5
    assert record.d_single == 4
    assert record.single().shape == (5, 4)
    assert record.pair().shape == (5, 5, 2)
    assert record.react_logits().shape == (5, 2)


def test_write_then_read_roundtrip(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4), ARRAY_PAIR: _pair(4), ARRAY_REACT: _react(4)}, family="CL1"),
        FrozenFeatureRecord("b", "ACGUACGU", {ARRAY_SINGLE: _single(8), ARRAY_PAIR: _pair(8), ARRAY_REACT: _react(8)}, family="CL2"),
    ]
    finalized = write_frozen_shard(tmp_path / "shard", records, _provenance())
    assert finalized.record_count == 2
    assert finalized.content_sha256  # populated on write

    shard = read_frozen_shard(tmp_path / "shard")
    assert shard.provenance.model_name == "RibonanzaNet2"
    assert [r.record_id for r in shard.records] == ["a", "b"]
    by_id = shard.by_id()
    assert by_id["a"].family == "CL1"
    assert by_id["b"].length == 8
    # numeric content survives the round trip
    assert by_id["a"].single().row(0) == records[0].single().row(0)


def test_write_is_byte_deterministic(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)})]
    prov = _provenance(schema=default_schema(d_single=4))
    write_frozen_shard(tmp_path / "s1", records, prov)
    write_frozen_shard(tmp_path / "s2", records, prov)
    h1 = json.loads((tmp_path / "s1" / "provenance.json").read_text())["content_sha256"]
    h2 = json.loads((tmp_path / "s2" / "provenance.json").read_text())["content_sha256"]
    assert h1 == h2
    assert (tmp_path / "s1" / "features.npz").read_bytes() == (tmp_path / "s2" / "features.npz").read_bytes()


def test_features_only_shard_omits_optional_arrays(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)})]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))
    shard = read_frozen_shard(tmp_path / "shard")
    rec = shard.records[0]
    assert rec.pair() is None
    assert rec.react_logits() is None


def test_read_detects_content_hash_mismatch(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)})]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))
    features = tmp_path / "shard" / "features.npz"
    blob = bytearray(features.read_bytes())
    blob[-1] ^= 0xFF
    features.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_frozen_shard(tmp_path / "shard")


def test_read_can_skip_verification(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)})]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))
    features = tmp_path / "shard" / "features.npz"
    blob = bytearray(features.read_bytes())
    blob[-1] ^= 0xFF
    features.write_bytes(bytes(blob))
    # verify=False skips the hash check; the array still parses structurally
    shard = read_frozen_shard(tmp_path / "shard", verify=False)
    assert len(shard.records) == 1


def test_read_frozen_single_array_targets_one_record(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)}),
        FrozenFeatureRecord("b", "GGGG", {ARRAY_SINGLE: NdArray.from_nested([[9.0, 8.0, 7.0, 6.0]] * 4, kind="float32")}),
    ]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))

    single = read_frozen_single_array(tmp_path / "shard", 1)

    assert single.shape == (4, 4)
    assert single.row(0) == (9.0, 8.0, 7.0, 6.0)


def test_read_frozen_single_arrays_targets_multiple_records(tmp_path):
    records = [
        FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)}),
        FrozenFeatureRecord("b", "GGGG", {ARRAY_SINGLE: NdArray.from_nested([[9.0, 8.0, 7.0, 6.0]] * 4, kind="float32")}),
        FrozenFeatureRecord("c", "CCCC", {ARRAY_SINGLE: NdArray.from_nested([[5.0, 4.0, 3.0, 2.0]] * 4, kind="float32")}),
    ]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))

    singles = read_frozen_single_arrays(tmp_path / "shard", [2, 1, 2])

    assert set(singles) == {1, 2}
    assert singles[1].row(0) == (9.0, 8.0, 7.0, 6.0)
    assert singles[2].row(0) == (5.0, 4.0, 3.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        read_frozen_single_arrays(tmp_path / "shard", [-1])


def test_read_frozen_single_array_validates_row_and_hash(tmp_path):
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4)})]
    write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))
    with pytest.raises(ValueError, match="row must be non-negative"):
        read_frozen_single_array(tmp_path / "shard", -1)

    features = tmp_path / "shard" / "features.npz"
    blob = bytearray(features.read_bytes())
    blob[-1] ^= 0xFF
    features.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_frozen_single_array(tmp_path / "shard", 0)


def test_write_rejects_array_absent_from_schema(tmp_path):
    # schema declares only 'single' but the record carries a 'pair' tensor
    records = [FrozenFeatureRecord("a", "ACGU", {ARRAY_SINGLE: _single(4), ARRAY_PAIR: _pair(4)})]
    with pytest.raises(ValueError, match="not declared in schema"):
        write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))


def test_by_id_rejects_duplicate_ids(tmp_path):
    records = [
        FrozenFeatureRecord("dup", "ACGU", {ARRAY_SINGLE: _single(4)}),
        FrozenFeatureRecord("dup", "ACGU", {ARRAY_SINGLE: _single(4)}),
    ]
    finalized = write_frozen_shard(tmp_path / "shard", records, _provenance(schema=default_schema(d_single=4)))
    assert finalized.record_count == 2
    shard = read_frozen_shard(tmp_path / "shard")
    with pytest.raises(ValueError, match="duplicate record id"):
        shard.by_id()


def test_provenance_from_json_requires_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        FrozenFeatureProvenance.from_json_obj({"model_name": "x"})


def test_provenance_json_obj_roundtrip():
    prov = _provenance(weights_sha256="abc123", content_sha256="deadbeef", record_count=5)
    obj = prov.to_json_obj()
    restored = FrozenFeatureProvenance.from_json_obj(obj)
    assert restored.weights_sha256 == "abc123"
    assert restored.content_sha256 == "deadbeef"
    assert restored.record_count == 5
    assert set(restored.schema) == {ARRAY_SINGLE, ARRAY_PAIR, ARRAY_REACT}
