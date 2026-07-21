import importlib.util
import json
from pathlib import Path
import sys

import pytest

from reactflow.protocol import (
    MMSEQS_COMPONENT_HOLDOUT,
    MMSEQS_COMPONENT_TEST,
    normalize_tier_label,
    stable_subset,
)
from reactflow.c0_evaluate import aggregate_structure_records, structure_record_metrics


def _load_merger():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_efold_baseline.py"
    spec = importlib.util.spec_from_file_location("merge_efold_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _progress(tmp_path: Path, tier: str, count: int = 2) -> Path:
    gold = tmp_path / f"{tier}.gold.jsonl"
    pred = tmp_path / f"{tier}.pred.jsonl"
    gold.write_text("{}\n" * count, encoding="utf-8")
    pred.write_text("{}\n" * count, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "tiers": {
            tier: {
                "count": count,
                "gold_count": count,
                "matched_count": count,
                "missing_count": 0,
                "extra_prediction_count": 0,
                "duplicate_gold_count": 0,
                "sequence_mismatch_count": 0,
                "gold": str(gold),
                "predictions": str(pred),
                "mean_f1": 0.2,
                "mean_mcc": 0.19,
            }
        },
    }
    path = tmp_path / f"{tier}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_legacy_tiers_are_read_as_component_semantics():
    assert normalize_tier_label("in_clan") == MMSEQS_COMPONENT_TEST
    assert normalize_tier_label("novel_clan") == MMSEQS_COMPONENT_HOLDOUT
    assert normalize_tier_label("PDB") == "PDB"


def test_stable_subset_is_input_order_independent():
    rows = [
        {"id": "b", "sequence": "GGCC"},
        {"id": "a", "sequence": "AUGC"},
        {"id": "c", "sequence": "UUAA"},
    ]
    first = stable_subset(rows, 2, source_id=lambda row: row["id"], sequence=lambda row: row["sequence"])
    second = stable_subset(list(reversed(rows)), 2, source_id=lambda row: row["id"], sequence=lambda row: row["sequence"])
    assert first == second


def test_full_count_merge_renames_tiers_and_records_hashes(tmp_path):
    module = _load_merger()
    result = module.merge_baselines(_progress(tmp_path, "in_clan"), _progress(tmp_path, "novel_clan"))
    assert set(result["tiers"]) == {MMSEQS_COMPONENT_TEST, MMSEQS_COMPONENT_HOLDOUT}
    assert result["legacy_aliases"]["novel_clan"] == MMSEQS_COMPONENT_HOLDOUT
    assert len(result["tiers"][MMSEQS_COMPONENT_TEST]["provenance"]["gold_sha256"]) == 64


def test_merge_rejects_partial_or_mismatched_artifact(tmp_path):
    module = _load_merger()
    path = _progress(tmp_path, "in_clan")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tiers"]["in_clan"]["matched_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not full-count"):
        module.merge_baselines(path, _progress(tmp_path, "novel_clan"))


def test_structure_metrics_report_short_medium_long_and_pooled_pair_ratio():
    size = 30
    target = [[0.0] * size for _ in range(size)]
    predicted = [[0.0] * size for _ in range(size)]
    for matrix in (target, predicted):
        matrix[0][5] = matrix[5][0] = 1.0
        matrix[1][16] = matrix[16][1] = 1.0
    target[2][28] = target[28][2] = 1.0
    metrics = structure_record_metrics(predicted, target)
    assert metrics["distance_bins"]["short"]["f1"] == 1.0
    assert metrics["distance_bins"]["medium"]["f1"] == 1.0
    assert metrics["distance_bins"]["long"]["recall"] == 0.0
    summary = aggregate_structure_records(
        [{"metrics": metrics, "legal": True, "runtime_seconds": 1.0}]
    )
    assert summary["pair_count_ratio"] == pytest.approx(2 / 3)
    assert summary["distance_bins"]["long"]["f1"] == 0.0
