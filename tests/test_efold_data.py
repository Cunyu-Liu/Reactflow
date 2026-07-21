"""Tests for the eFold/RNAndria JSON reader in reactflow.data."""

import json
import math

import pytest

from reactflow.constraints import matrix_to_pairs, validate_pair_matrix
from reactflow.data import (
    EfoldRecord,
    efold_pair_matrix,
    parse_efold_record,
    read_efold_json,
)


def test_parse_efold_record_reads_sequence_pairs_and_shape():
    record = parse_efold_record(
        {
            "sequence": "gggaaaccc",
            "structure": [[0, 8], [1, 7], [2, 6]],
            "shape": [0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "family": "CL00051",
            "id": "rec1",
        }
    )

    assert record.sequence == "GGGAAACCC"
    assert record.pairs == ((0, 8), (1, 7), (2, 6))
    assert record.family == "CL00051"
    assert record.record_id == "rec1"
    assert record.shape is not None
    assert math.isnan(record.shape[2])


def test_parse_efold_record_orders_pairs_and_supports_one_based():
    record = parse_efold_record(
        {"sequence": "GGGAAACCC", "structure": [[8, 0], [7, 1]]},
        one_based=False,
    )
    assert record.pairs == ((0, 8), (1, 7))

    one_based = parse_efold_record(
        {"sequence": "GGGAAACCC", "structure": [[1, 9], [2, 8]]},
        one_based=True,
    )
    assert one_based.pairs == ((0, 8), (1, 7))


def test_parse_efold_record_accepts_pairs_and_dms_aliases():
    record = parse_efold_record(
        {"sequence": "GGAACC", "pairs": [[0, 5]], "dms": [0.1, -1000.0, 0.3, 0.4, 0.5, 0.6]}
    )
    assert record.pairs == ((0, 5),)
    assert record.reactivity_probe == "DMS"
    assert record.shape is not None
    assert record.shape[0] == pytest.approx(0.1)
    assert math.isnan(record.shape[1])
    assert record.shape[2:] == pytest.approx((0.3, 0.4, 0.5, 0.6))


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"structure": [[0, 1]]}, "sequence"),
        ({"sequence": "ACGX", "structure": []}, "invalid RNA bases"),
        ({"sequence": "ACGU"}, "structure"),
        ({"sequence": "ACGU", "structure": [[0, 1, 2]]}, "exactly two"),
        ({"sequence": "ACGU", "structure": [[0, 9]]}, "out of range"),
        ({"sequence": "ACGU", "structure": [[1, 1]]}, "self-pair"),
        ({"sequence": "ACGU", "structure": [[0, 1]], "shape": [0.1, 0.2]}, "shape length"),
        ({"sequence": "ACGU", "structure": "notalist"}, "list of"),
        ({"sequence": "ACGU", "structure": [[0, 1]], "shape": "x"}, "list of floats"),
    ],
)
def test_parse_efold_record_rejects_malformed_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        parse_efold_record(entry)


def test_efold_pair_matrix_round_trips_and_is_legal_nested():
    record = parse_efold_record({"sequence": "GGGAAACCC", "structure": [[0, 8], [1, 7], [2, 6]]})
    matrix = efold_pair_matrix(record)

    assert matrix_to_pairs(matrix) == ((0, 8), (1, 7), (2, 6))
    result = validate_pair_matrix(record.sequence, matrix, allow_pseudoknot=False)
    assert result.valid
    assert result.pair_count == 3


def test_read_efold_json_supports_list_layout(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(
        json.dumps(
            [
                {"sequence": "GGAACC", "structure": [[0, 5]], "id": "a"},
                {"sequence": "GGGAAACCC", "structure": [[0, 8], [1, 7]], "id": "b"},
            ]
        ),
        encoding="utf-8",
    )

    records = list(read_efold_json(path))

    assert [r.record_id for r in records] == ["a", "b"]
    assert records[1].pairs == ((0, 8), (1, 7))


def test_read_efold_json_supports_mapping_layout_and_injects_key(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps({"seqA": {"sequence": "GGAACC", "structure": [[0, 5]]}}),
        encoding="utf-8",
    )

    records = list(read_efold_json(path))

    assert len(records) == 1
    assert records[0].record_id == "seqA"


def test_read_efold_json_limit_and_bad_payload(tmp_path):
    path = tmp_path / "limit.json"
    path.write_text(
        json.dumps(
            [
                {"sequence": "GGAACC", "structure": [[0, 5]]},
                {"sequence": "GGAACC", "structure": [[0, 5]]},
            ]
        ),
        encoding="utf-8",
    )
    assert len(list(read_efold_json(path, limit=1))) == 1

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(ValueError, match="list of records or a mapping"):
        list(read_efold_json(bad))

    non_object = tmp_path / "nonobj.json"
    non_object.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        list(read_efold_json(non_object))


def test_efold_record_is_hashable_dataclass():
    record = EfoldRecord(sequence="ACGU", pairs=((0, 3),))
    assert record.pairs == ((0, 3),)
    assert record.shape is None
