import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_cross_family_metrics.py"
    spec = importlib.util.spec_from_file_location("audit_cross_family_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_row(run_id, tier, mean_f1, micro_f1=None, mean_mcc=None, count=10):
    return {
        "count": count,
        "mean_f1": mean_f1,
        "mean_mcc": mean_f1 if mean_mcc is None else mean_mcc,
        "micro_f1": mean_f1 if micro_f1 is None else micro_f1,
        "micro_mcc": mean_f1 if mean_mcc is None else mean_mcc,
        "run_id": run_id,
        "status": "ok",
        "tier": tier,
    }


def _write_rows(path: Path, rows) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_cross_family_audit_warns_low_but_parseable_novel_metric(tmp_path):
    mod = _load_module()
    path = tmp_path / "current_queue_status.json"
    _write_rows(
        path,
        [
            _metric_row("RF-A1", "in_clan", 0.0394, micro_f1=0.0358, mean_mcc=0.0384, count=25805),
            _metric_row("RF-A1", "novel_clan", 0.0624, micro_f1=0.0506, mean_mcc=0.0591, count=49588),
        ],
    )

    result = mod.run_audit(path, min_novel_mean_f1=0.15, max_generalization_gap=0.10)

    assert result["summary"]["cross_family_healthy"] is True
    assert result["summary"]["cross_family_claim_ready"] is False
    assert result["summary"]["best_run_id"] == "RF-A1"
    assert result["rows"][0]["status"] == "warn"
    assert "novel_mean_f1=0.0624" in result["rows"][0]["detail"]


def test_cross_family_audit_passes_claim_ready_metric_and_writes_markdown(tmp_path):
    mod = _load_module()
    path = tmp_path / "mmseqs_final_results.json"
    out = tmp_path / "cross_family.md"
    _write_rows(
        path,
        [
            _metric_row("RF-M1", "in_clan", 0.2200, micro_f1=0.2100, mean_mcc=0.2000, count=16606),
            _metric_row("RF-M1", "novel_clan", 0.1800, micro_f1=0.1700, mean_mcc=0.1600, count=46147),
        ],
    )

    result = mod.run_audit(path, min_novel_mean_f1=0.15, max_generalization_gap=0.10)
    mod.write_markdown(result, out)

    assert result["summary"]["cross_family_claim_ready"] is True
    assert result["rows"][0]["status"] == "pass"
    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Cross-Family Metric Audit" in text
    assert "gap=0.0400" in text


def test_cross_family_audit_fails_invalid_metric_fields(tmp_path):
    mod = _load_module()
    path = tmp_path / "bad.json"
    rows = [
        _metric_row("RF-bad", "in_clan", 0.2),
        _metric_row("RF-bad", "novel_clan", None),
    ]
    _write_rows(path, rows)

    result = mod.run_audit(path, min_novel_mean_f1=0.15, max_generalization_gap=0.10)

    assert result["summary"]["cross_family_healthy"] is False
    assert result["rows"][0]["status"] == "fail"
    assert result["rows"][0]["detail"] == "invalid finite metric/count fields"


def test_cross_family_audit_keeps_active_run_pending_visible(tmp_path):
    mod = _load_module()
    path = tmp_path / "queue.json"
    _write_rows(
        path,
        [
            {
                "run_id": "RF-active",
                "status": "running_or_pending_json",
                "tier": "",
            }
        ],
    )

    result = mod.run_audit(path, min_novel_mean_f1=0.15, max_generalization_gap=0.10)

    assert result["summary"]["cross_family_healthy"] is True
    assert result["rows"][0]["status"] == "warn"
    assert result["rows"][0]["item"] == "run:RF-active:cross_family_pending"
