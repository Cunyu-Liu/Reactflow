import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_goal_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_goal_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_docs(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        """
## 科学问题与定制化模型路线
```mermaid
graph TD
```
## 公共数据源
## 数据预处理契约
## SOTA / 竞品表
## 环境
可复现
""",
        encoding="utf-8",
    )
    (root / "docs/data_governance.md").write_text(
        """
https://doi.org/10.5061/dryad.79cnp5j95
https://www.kaggle.com/models/shujun717/ribonanzanet2/PyTorch/alpha/1
Preprocessing Contract
MMseqs
""",
        encoding="utf-8",
    )


def _write_base_audits(full_run_root: Path) -> None:
    _write_json(
        full_run_root / "algorithm_doc_audit.json",
        {"summary": {"strict_ready": True, "placeholder_bodies": 0}},
    )
    _write_json(
        full_run_root / "runtime_health_audit.json",
        {"summary": {"healthy": True, "counts": {"pass": 3, "warn": 0, "fail": 0}}},
    )
    _write_json(
        full_run_root / "system_resource_audit.json",
        {"summary": {"resource_healthy": True, "counts": {"pass": 4, "warn": 0, "fail": 0}}},
    )
    _write_json(
        full_run_root / "queue_progress_audit.json",
        {"summary": {"progress_healthy": True, "counts": {"pass": 3, "warn": 0, "fail": 0}}},
    )
    _write_json(
        full_run_root / "cross_family_metric_audit.json",
        {
            "summary": {
                "best_generalization_gap": -0.02,
                "best_novel_mean_f1": 0.0624,
                "best_run_id": "RF-A1",
                "counts": {"pass": 1, "warn": 0, "fail": 0},
                "cross_family_claim_ready": True,
                "cross_family_healthy": True,
            }
        },
    )
    _write_json(
        full_run_root / "profile_bottleneck_audit.json",
        {"summary": {"bottleneck_healthy": True, "counts": {"pass": 2, "warn": 1, "fail": 0}}},
    )
    _write_json(
        full_run_root / "final_queue_audit.json",
        {"summary": {"final_queue_healthy": True, "final_results_ready": False, "counts": {"pass": 4, "warn": 3, "fail": 0}}},
    )
    _write_json(
        full_run_root / "paper_artifact_audit.json",
        {"summary": {"counts": {"pass": 5, "warn": 1, "fail": 0}}},
    )
    _write_json(
        full_run_root / "reproducibility_manifest.json",
        {"file_count": 3, "files": [{"path": "README.md"}], "audit_summaries": {}},
    )
    _write_json(
        full_run_root / "coverage_audit.json",
        {"passed": True, "percent_covered": 90.53, "threshold": 90.0},
    )


def _write_final_result_rows(path: Path, audit) -> None:
    rows = []
    for tier in audit.EXPECTED_RESULT_TIERS:
        rows.append(
            {
                "artifact": "runs/RF-test",
                "checkpoint_present": True,
                "count": 3,
                "mean_f1": 0.4,
                "mean_mcc": 0.3,
                "micro_f1": 0.35,
                "micro_mcc": 0.25,
                "profile_seconds": 10.0,
                "progress_fraction": None,
                "run_id": "RF-test",
                "samples_per_second": 1.0,
                "slowest_phase": "model_forward",
                "status": "ok",
                "stderr_size": 0,
                "stderr_tail": "",
                "tier": tier,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_goal_readiness_blocks_missing_final_results(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "final_result:warm_rfam_current_exact_results.json" in failed
    assert "final_result:mmseqs_final_results.json" in failed


def test_goal_readiness_passes_when_all_evidence_exists(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)
    out = tmp_path / "goal_readiness.md"
    audit.write_markdown(result, out)

    assert result["summary"]["ready_for_goal_completion"] is True
    assert "ReactFlow Goal Readiness Audit" in out.read_text(encoding="utf-8")


def test_goal_readiness_blocks_queue_progress_failures(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "queue_progress_audit.json",
        {"summary": {"progress_healthy": False, "counts": {"pass": 2, "warn": 0, "fail": 1}}},
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "queue_progress" in failed


def test_goal_readiness_blocks_cross_family_metric_failures(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "cross_family_metric_audit.json",
        {"summary": {"cross_family_healthy": False, "counts": {"pass": 0, "warn": 0, "fail": 1}}},
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "cross_family_metrics" in failed


def test_goal_readiness_blocks_cross_family_claim_not_ready(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "cross_family_metric_audit.json",
        {
            "summary": {
                "best_novel_mean_f1": 0.0624,
                "counts": {"pass": 0, "warn": 1, "fail": 0},
                "cross_family_claim_ready": False,
                "cross_family_healthy": True,
            }
        },
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert any(row["item"] == "cross_family_metrics" for row in failed)
    assert "cross_family_claim_ready" in next(row["detail"] for row in failed if row["item"] == "cross_family_metrics")


def test_goal_readiness_blocks_system_resource_failures(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "system_resource_audit.json",
        {"summary": {"resource_healthy": False, "counts": {"pass": 2, "warn": 0, "fail": 1}}},
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "system_resources" in failed


def test_goal_readiness_blocks_final_queue_failures(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "final_queue_audit.json",
        {"summary": {"final_queue_healthy": False, "final_results_ready": False, "counts": {"pass": 2, "warn": 1, "fail": 1}}},
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "final_queue" in failed


def test_goal_readiness_blocks_profile_bottleneck_failures(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    _write_json(
        full_run_root / "profile_bottleneck_audit.json",
        {"summary": {"bottleneck_healthy": False, "counts": {"pass": 1, "warn": 0, "fail": 1}}},
    )
    for name in audit.FINAL_RESULT_FILES:
        _write_final_result_rows(full_run_root / name, audit)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row["item"] for row in result["rows"] if row["status"] == "fail"]
    assert "profile_bottlenecks" in failed


def test_goal_readiness_blocks_present_but_non_metric_final_results(tmp_path):
    audit = _load_module()
    _write_docs(tmp_path)
    full_run_root = tmp_path / "artifacts/full_runs/run"
    _write_base_audits(full_run_root)
    bad_rows = [
        {
            "count": None,
            "mean_f1": None,
            "micro_f1": None,
            "mean_mcc": None,
            "micro_mcc": None,
            "run_id": "RF-running",
            "status": "running_or_pending_json",
            "tier": "",
        }
    ]
    for name in audit.FINAL_RESULT_FILES:
        _write_json(full_run_root / name, bad_rows)

    result = audit.run_audit(tmp_path, full_run_root)

    assert result["summary"]["ready_for_goal_completion"] is False
    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert any(row["item"] == "final_result:warm_rfam_current_exact_results.json" for row in failed)
    assert "non_ok_rows=1" in next(
        row["detail"] for row in failed if row["item"] == "final_result:warm_rfam_current_exact_results.json"
    )
