from __future__ import annotations

import pytest

from reactflow.delta.availability import RIBONANZA_COMPETITION_DATA_URL, build_ribonanza_availability_report
from reactflow.delta.schema import validate_source_registry_record


def test_ribonanza_unavailability_is_unknown_not_zero() -> None:
    document = build_ribonanza_availability_report(
        retrieved_at="2026-07-30T13:29:31+08:00",
        kaggle_cli_path=None,
        kaggle_config_present=False,
        endpoint_probes=[{"url": RIBONANZA_COMPETITION_DATA_URL, "outcome": "http_error", "http_status": 404}],
    )

    assert document["data_access_outcome"] == "unavailable_in_current_environment"
    assert document["same_condition_single_edit_pair_count"] is None
    assert document["environment"]["kaggle_config_contents_read"] is False
    assert validate_source_registry_record(document["source_registry_record"])["download_status"] == "unavailable"


def test_ribonanza_report_rejects_missing_probes() -> None:
    with pytest.raises(ValueError, match="endpoint_probes"):
        build_ribonanza_availability_report(
            retrieved_at="2026-07-30T13:29:31+08:00",
            kaggle_cli_path=None,
            kaggle_config_present=False,
            endpoint_probes=[],
        )
