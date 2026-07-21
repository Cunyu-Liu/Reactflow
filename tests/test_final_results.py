import json
import math

from reactflow.final_results import EXPECTED_RESULT_TIERS, result_file_ready, validate_final_result_file, validate_final_result_rows


def _valid_rows():
    rows = []
    for tier in EXPECTED_RESULT_TIERS:
        rows.append(
            {
                "count": 3,
                "mean_f1": 0.4,
                "mean_mcc": 0.3,
                "micro_f1": 0.35,
                "micro_mcc": 0.25,
                "status": "ok",
                "tier": tier,
            }
        )
    return rows


def test_final_result_rows_require_all_tiers_and_metrics():
    result = validate_final_result_rows(_valid_rows())

    assert result.ready is True
    assert result.state == "ready"
    assert "ok_metric_rows=7" in result.detail


def test_final_result_rows_reject_running_rows():
    rows = [{"status": "running_or_pending_json", "tier": "", "count": None}]

    result = validate_final_result_rows(rows)

    assert result.ready is False
    assert result.state == "invalid"
    assert "non_ok_rows=1" in result.detail


def test_final_result_rows_reject_non_finite_metrics():
    rows = _valid_rows()
    rows[0]["mean_f1"] = math.inf

    result = validate_final_result_rows(rows)

    assert result.ready is False
    assert result.state == "invalid"
    assert "invalid_rows=1" in result.detail


def test_final_result_file_states(tmp_path):
    missing = tmp_path / "missing.json"
    assert validate_final_result_file(missing).state == "missing"
    assert result_file_ready(missing) is False

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_rows()), encoding="utf-8")
    assert validate_final_result_file(valid).ready is True
    assert result_file_ready(valid) is True

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    result = validate_final_result_file(invalid)
    assert result.state == "invalid"
    assert "invalid JSON" in result.detail
