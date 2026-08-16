"""Auditable source-availability records for D0 data that cannot be acquired."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import SOURCE_REGISTRY_SCHEMA_VERSION, validate_source_registry_record


RIBONANZA_COMPETITION_DATA_URL = "https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data"
RIBONANZA_TRAIN_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/competitions/download/stanford-ribonanza-rna-folding/train_data.csv"
RIBONANZA_AVAILABILITY_SCHEMA_VERSION = "reactflow-delta-ribonanza-availability-v1"


def probe_http_head(url: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    """Record an HTTP HEAD outcome without downloading a data payload."""

    request = Request(url, method="HEAD", headers={"User-Agent": "ReactFlow-Delta-D0-audit/1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return {"url": url, "outcome": "http_response", "http_status": response.status}
    except HTTPError as exc:
        return {"url": url, "outcome": "http_error", "http_status": exc.code}
    except URLError as exc:
        return {"url": url, "outcome": "network_error", "http_status": None, "error_type": type(exc.reason).__name__}


def build_ribonanza_availability_report(
    *,
    retrieved_at: str,
    kaggle_cli_path: str | None,
    kaggle_config_present: bool,
    endpoint_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a D0 record whose unknown pair count cannot be read as zero."""

    _validate_iso_timestamp(retrieved_at)
    if not isinstance(kaggle_config_present, bool):
        raise ValueError("kaggle_config_present must be boolean")
    if kaggle_cli_path is not None and (not isinstance(kaggle_cli_path, str) or not kaggle_cli_path):
        raise ValueError("kaggle_cli_path must be a non-empty string or null")
    if not isinstance(endpoint_probes, list) or not endpoint_probes:
        raise ValueError("endpoint_probes must be a non-empty list")
    for probe in endpoint_probes:
        if not isinstance(probe, dict) or not isinstance(probe.get("url"), str) or not isinstance(probe.get("outcome"), str):
            raise ValueError("each endpoint probe requires URL and outcome")

    record = validate_source_registry_record(
        {
            "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
            "record_id": "ribonanza:stanford-ribonanza-rna-folding:unavailable-current-environment",
            "source": "Ribonanza",
            "source_version": None,
            "source_url": RIBONANZA_COMPETITION_DATA_URL,
            "publication_doi": None,
            "publication_pmid": None,
            "license": None,
            "retrieved_at": retrieved_at,
            "sha256": None,
            "bytes": None,
            "upstream_id": "stanford-ribonanza-rna-folding",
            "raw_path": None,
            "parser_version": None,
            "download_status": "unavailable",
            "source_tier": "B",
            "source_type": "ribonanza",
            "missing_reasons": {
                "source_version": "no immutable Ribonanza data file was acquired in this environment",
                "publication_doi": "no publication DOI was asserted by the frozen competition reference used for this availability audit",
                "publication_pmid": "no PMID was asserted by the frozen competition reference used for this availability audit",
                "license": "Kaggle data terms/license were not retrievable without an acquired competition data artifact",
                "sha256": "no raw Ribonanza file was acquired",
                "bytes": "no raw Ribonanza file was acquired",
                "raw_path": "no raw Ribonanza file was acquired",
                "parser_version": "no raw Ribonanza file was acquired to parse",
            },
        }
    )
    return {
        "schema_version": RIBONANZA_AVAILABILITY_SCHEMA_VERSION,
        "stage": "D0",
        "retrieved_at": retrieved_at,
        "source_registry_record": record,
        "environment": {"kaggle_cli_path": kaggle_cli_path, "kaggle_config_present": kaggle_config_present, "kaggle_config_contents_read": False},
        "endpoint_probes": endpoint_probes,
        "data_access_outcome": "unavailable_in_current_environment",
        "same_condition_single_edit_pair_count": None,
        "pair_count_missing_reason": "No immutable Ribonanza raw table was acquired, so batch, condition, parent lineage, edit endpoint, and duplicate identity cannot be audited.",
        "scientific_boundary": "This report records unavailability in the current environment. It does not imply that Ribonanza globally lacks eligible pairs and it does not convert unknown into zero.",
    }


def _validate_iso_timestamp(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("retrieved_at must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("retrieved_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("retrieved_at must include timezone")
