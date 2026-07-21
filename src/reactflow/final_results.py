"""Final-result table contract shared by queue and readiness audits.

Full-scale ReactFlow runs finish by writing JSON tables from
``scripts/summarize_ablation_results.py``.  Several operational checks need to
agree on what "finished" means, so this module centralizes the schema contract:
each final result file must be a non-empty list of ``status='ok'`` metric rows,
cover all expected evaluation tiers, and contain finite F1/MCC metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence


FINAL_RESULT_FILES = (
    "warm_rfam_current_exact_results.json",
    "contact_rfam_current_exact_results.json",
    "mmseqs_final_results.json",
)
EXPECTED_RESULT_TIERS = (
    "archiveII",
    "human_mRNA",
    "in_clan",
    "lncRNA",
    "novel_clan",
    "PDB",
    "viral",
)


class FinalResultValidation(NamedTuple):
    """Validation state for one final-result JSON table.

    Formula: ``ready`` is true exactly when ``state == 'ready'``; ``detail``
    records the sufficient statistic tuple
    ``(rows, ok_metric_rows, invalid_rows, non_ok_rows, tiers)`` used by audits.
    Complexity: O(1) storage.
    """

    state: str
    detail: str

    @property
    def ready(self) -> bool:
        """Return whether the result table satisfies the final contract.

        Formula: ``ready = 1[state = 'ready']``.  Complexity: O(1).
        """

        return self.state == "ready"


def _is_finite_metric(value: object) -> bool:
    """Return whether ``value`` is a finite numeric metric.

    Formula: accept iff ``value`` is an ``int`` or ``float`` but not ``bool``,
    and ``isfinite(float(value))``.  Complexity: O(1).
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_final_result_rows(rows: object, *, expected_tiers: Sequence[str] = EXPECTED_RESULT_TIERS) -> FinalResultValidation:
    """Validate final-result rows already loaded from JSON.

    Formula: for row list ``R``, accept iff every element is a mapping with
    ``status='ok'``, positive integer ``count``, non-empty ``tier``, finite
    ``mean_f1/micro_f1/mean_mcc/micro_mcc``, and
    ``{tier(r): r in R_ok}`` covers ``expected_tiers``.  Complexity: O(|R| + T),
    where T is the number of expected tiers.
    """

    if not isinstance(rows, list) or not rows:
        return FinalResultValidation("invalid", "expected non-empty JSON list")

    invalid_rows = 0
    non_ok_rows = 0
    ok_metric_rows = 0
    tiers = set()
    statuses = {}
    for item in rows:
        if not isinstance(item, Mapping):
            invalid_rows += 1
            continue
        status = str(item.get("status", ""))
        statuses[status] = statuses.get(status, 0) + 1
        if status != "ok":
            non_ok_rows += 1
            continue
        metric_fields_ok = (
            isinstance(item.get("tier"), str)
            and item.get("tier")
            and isinstance(item.get("count"), int)
            and not isinstance(item.get("count"), bool)
            and int(item["count"]) > 0
            and _is_finite_metric(item.get("mean_f1"))
            and _is_finite_metric(item.get("micro_f1"))
            and _is_finite_metric(item.get("mean_mcc"))
            and _is_finite_metric(item.get("micro_mcc"))
        )
        if metric_fields_ok:
            ok_metric_rows += 1
            tiers.add(str(item["tier"]))
        else:
            invalid_rows += 1

    missing_tiers = sorted(set(expected_tiers) - tiers)
    if invalid_rows or non_ok_rows or not ok_metric_rows or missing_tiers:
        detail = (
            f"rows={len(rows)}; ok_metric_rows={ok_metric_rows}; invalid_rows={invalid_rows}; "
            f"non_ok_rows={non_ok_rows}; statuses={statuses}; missing_tiers={missing_tiers}"
        )
        return FinalResultValidation("invalid", detail)
    return FinalResultValidation("ready", f"rows={len(rows)}; ok_metric_rows={ok_metric_rows}; tiers={sorted(tiers)}")


def validate_final_result_file(path: Path, *, missing_detail: str = "missing or empty; waiting for watcher") -> FinalResultValidation:
    """Validate one final-result JSON file.

    Formula: if ``path`` is absent or empty, return ``missing``; otherwise parse
    JSON rows ``R`` and return :func:`validate_final_result_rows(R)`.  JSON parse
    errors are ``invalid`` because a malformed non-empty file is stronger
    evidence of a failed producer than a still-pending watcher.  Complexity:
    O(file bytes + |R|).
    """

    if not path.exists() or path.stat().st_size == 0:
        return FinalResultValidation("missing", missing_detail)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return FinalResultValidation("invalid", f"invalid JSON: {exc}")
    return validate_final_result_rows(rows)


def result_file_ready(path: Path) -> bool:
    """Return whether one final-result file is complete and valid.

    Formula: ``ready(path) = 1[validate_final_result_file(path).state =
    'ready']``.  Complexity: O(file bytes + rows).
    """

    return validate_final_result_file(path).ready
