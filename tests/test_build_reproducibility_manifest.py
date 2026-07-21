import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_reproducibility_manifest.py"
    spec = importlib.util.spec_from_file_location("build_reproducibility_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_reproducibility_manifest_hashes_project_and_audits(tmp_path):
    mod = _load_module()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/reactflow").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "docs/data_governance.md").write_text("data", encoding="utf-8")
    (tmp_path / "src/reactflow/model.py").write_text("x = 1\n", encoding="utf-8")
    full_run = tmp_path / "artifacts/full_runs/run"
    full_run.mkdir(parents=True)
    (full_run / "algorithm_doc_audit.json").write_text(
        json.dumps({"summary": {"strict_ready": True}}),
        encoding="utf-8",
    )
    (full_run / "final_queue_audit.json").write_text(
        json.dumps({"summary": {"final_queue_healthy": True}}),
        encoding="utf-8",
    )
    (full_run / "queue_progress_audit.json").write_text(
        json.dumps({"summary": {"progress_healthy": True}}),
        encoding="utf-8",
    )
    (full_run / "current_queue_status.md").write_text("queue", encoding="utf-8")
    (full_run / "current_queue_status.svg").write_text("<svg></svg>", encoding="utf-8")
    (full_run / "run_queue.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (full_run / "logs").mkdir()
    (full_run / "logs/watcher.pid").write_text("123\n", encoding="utf-8")
    (full_run / "logs/current_queue_status_history.jsonl").write_text("{}\n", encoding="utf-8")
    (full_run / "runs/run_a").mkdir(parents=True)
    (full_run / "runs/run_a/stderr.log").write_text("", encoding="utf-8")
    (full_run / "runs/run_a/stdout.json").write_text("{}", encoding="utf-8")

    manifest = mod.build_manifest(tmp_path, full_run, max_hash_bytes=1024)

    paths = {item["path"]: item for item in manifest["files"]}
    assert "README.md" in paths
    assert "artifacts/full_runs/run/algorithm_doc_audit.json" in paths
    assert "artifacts/full_runs/run/final_queue_audit.json" in paths
    assert "artifacts/full_runs/run/queue_progress_audit.json" in paths
    assert "artifacts/full_runs/run/current_queue_status.svg" in paths
    assert "artifacts/full_runs/run/run_queue.sh" in paths
    assert "artifacts/full_runs/run/logs/watcher.pid" in paths
    assert "artifacts/full_runs/run/logs/current_queue_status_history.jsonl" in paths
    assert "artifacts/full_runs/run/runs/run_a/stderr.log" in paths
    assert "artifacts/full_runs/run/runs/run_a/stdout.json" in paths
    assert paths["README.md"]["sha256"]
    assert manifest["audit_summaries"]["algorithm_doc_audit.json"] == {"strict_ready": True}
    assert manifest["audit_summaries"]["final_queue_audit.json"] == {"final_queue_healthy": True}
    assert manifest["audit_summaries"]["queue_progress_audit.json"] == {"progress_healthy": True}
    assert manifest["environment"]["python_executable"]
    assert "pytest" in manifest["environment"]["packages"]
    assert "mmseqs" in manifest["environment"]["tool_paths"]


def test_reproducibility_manifest_skips_large_hashes_and_writes_markdown(tmp_path):
    mod = _load_module()
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    full_run = tmp_path / "artifacts/full_runs/run"
    (full_run / "logs").mkdir(parents=True)
    large = full_run / "logs/big.log"
    large.write_bytes(b"x" * 20)

    manifest = mod.build_manifest(tmp_path, full_run, max_hash_bytes=8)
    out = tmp_path / "manifest.md"
    mod.write_markdown(manifest, out)

    large_record = next(item for item in manifest["files"] if item["path"].endswith("big.log"))
    assert large_record["sha256"] is None
    assert large_record["sha256_skipped_reason"] == "size>8"
    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Reproducibility Manifest" in text
    assert "Environment Packages" in text
    assert "Tool Paths" in text


def test_package_versions_records_missing_packages():
    mod = _load_module()

    versions = mod.package_versions(["definitely-missing-reactflow-package"])

    assert versions["definitely-missing-reactflow-package"] == {"available": False, "version": None}


def test_tool_paths_uses_environment_override(tmp_path, monkeypatch):
    mod = _load_module()
    tool = tmp_path / "custom_tool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_TOOL_BIN", str(tool))

    paths = mod.tool_paths(["custom_tool"])

    assert paths["custom_tool"] == str(tool)
