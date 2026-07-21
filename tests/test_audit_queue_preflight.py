import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_queue_preflight.py"
    spec = importlib.util.spec_from_file_location("audit_queue_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_script(path: Path, markers) -> None:
    _write(path, "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(f"# {marker}" for marker in markers) + "\n")


def _write_fixture(root: Path):
    full = root / "artifacts/full_runs/run"
    scripts = root / "scripts"
    common_tiers = ("archiveII", "PDB", "viral", "lncRNA", "human_mRNA")
    for split_name in ("rfam_current_exact_seed0", "rfam_current_mmseqs_seed0"):
        for name in ("train.jsonl", "test.jsonl", "novel.jsonl", "split_manifest.json"):
            _write(full / "splits" / split_name / name)
    for tier in common_tiers:
        _write(full / "cache" / f"{tier}.jsonl")
    _write(full / "frozen/ribonanzanet2_sharded_full/sharded_manifest.json", "{}\n")
    _write_script(
        full / "run_warm_after_export_rfam_current_exact.sh",
        (
            "RF-A1-warm",
            "RF-A2-adapter4",
            "RF-A2-adapter16",
            "warm_rfam_current_exact_results.json",
            "--backend torch",
            "--adapter-dim",
            "summarize_ablation_results.py",
            "for bs in 16 8 4 2 1",
            "out of memory|cuda out|oom|killed|MemoryError",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_contact_after_warm_rfam_current_exact.sh",
        (
            "warm_rfam_current_exact_results.json",
            "RF-A3-contact",
            "--lambda-contact",
            "contact_rfam_current_exact_results.json",
            "summarize_ablation_results.py",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_warm_tail_recovery_after_watcher_exit.sh",
        (
            "warm_tail_recovery_after_watcher_exit.pid",
            "warm_after_export_rfam_current_exact.pid",
            "warm_rfam_current_exact_results.json",
            "RF-A1-warm",
            "RF-A2-adapter4",
            "RF-A2-adapter16",
            "label_done",
            "--frozen-cache-shards 4",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
            "summarize_ablation_results.py",
        ),
    )
    _write_script(
        scripts / "run_mmseqs_final_after_exact_queue.sh",
        (
            "contact_rfam_current_exact_results.json",
            "RF-M0-base",
            "RF-M1-warm",
            "--frozen-cache-shards 4",
            "mmseqs_final_results.json",
            "rfam_current_mmseqs_seed0",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_cross_family_after_mmseqs_final.sh",
        (
            "mmseqs_final_results.json",
            "RF-CF3-family-balanced",
            "--family-balanced-batches",
            "cross_family_balanced_results.json",
            "audit_cross_family_metrics.py",
            "cross_family_claim_ready",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_contact_sweep_after_cross_family_balanced.sh",
        (
            "cross_family_balanced_results.json",
            "RF-CF1-contact-strong",
            "CONTACT_SWEEP_LAMBDAS",
            "--lambda-contact",
            "--frozen-cache-shards 4",
            "cross_family_contact_sweep_results.json",
            "cross_family_contact_sweep_metric_audit.json",
            "audit_cross_family_metrics.py",
            "cross_family_claim_ready",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_long_range_after_contact_sweep.sh",
        (
            "cross_family_contact_sweep_results.json",
            "RF-CF2-long-range",
            "LONG_RANGE_WEIGHTS",
            "--contact-long-range-min-distance",
            "--contact-long-range-weight",
            "cross_family_long_range_results.json",
            "cross_family_long_range_metric_audit.json",
            "audit_cross_family_metrics.py",
            "cross_family_claim_ready",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_capacity_after_long_range.sh",
        (
            "cross_family_long_range_results.json",
            "RF-CF5-capacity",
            "CAPACITY_GRID",
            "--hidden-size",
            "--adapter-dim",
            "--family-balanced-batches",
            "--contact-long-range-weight",
            "cross_family_capacity_results.json",
            "cross_family_capacity_metric_audit.json",
            "audit_cross_family_metrics.py",
            "cross_family_claim_ready",
            "for bs in 16 8 4 2 1",
            "instability_pattern",
            "FloatingPointError",
            "non-finite",
            "retrying smaller",
            "exhausted batch retries",
        ),
    )
    _write_script(
        scripts / "run_goal_readiness_after_final_results.sh",
        (
            "warm_rfam_current_exact_results.json",
            "contact_rfam_current_exact_results.json",
            "mmseqs_final_results.json",
            "--fail-if-not-ready",
        ),
    )
    return full


def test_queue_preflight_passes_complete_fixture(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)

    result = mod.audit_queue(tmp_path, full)

    assert result["summary"]["preflight_healthy"] is True
    assert result["summary"]["counts"]["fail"] == 0


def test_queue_preflight_fails_missing_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_mmseqs_final_after_exact_queue.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("--frozen-cache-shards 4", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:mmseqs:marker:--frozen-cache-shards 4" for row in failed)


def test_queue_preflight_fails_missing_recovery_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_warm_tail_recovery_after_watcher_exit.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("label_done", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:warm_recovery:marker:label_done" for row in failed)


def test_queue_preflight_fails_missing_cross_family_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_cross_family_after_mmseqs_final.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("--family-balanced-batches", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:cross_family:marker:--family-balanced-batches" for row in failed)


def test_queue_preflight_fails_missing_contact_sweep_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_contact_sweep_after_cross_family_balanced.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("--lambda-contact", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:contact_sweep:marker:--lambda-contact" for row in failed)


def test_queue_preflight_fails_missing_long_range_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_long_range_after_contact_sweep.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("--contact-long-range-weight", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:long_range:marker:--contact-long-range-weight" for row in failed)


def test_queue_preflight_fails_missing_capacity_marker(tmp_path):
    mod = _load_module()
    full = _write_fixture(tmp_path)
    script = tmp_path / "scripts/run_capacity_after_long_range.sh"
    script.write_text(script.read_text(encoding="utf-8").replace("--hidden-size", ""), encoding="utf-8")

    result = mod.audit_queue(tmp_path, full)

    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert result["summary"]["preflight_healthy"] is False
    assert any(row["item"] == "stage:capacity:marker:--hidden-size" for row in failed)


def test_queue_preflight_markdown(tmp_path):
    mod = _load_module()
    result = {"rows": [mod.row("pass", "x", tmp_path / "x", "ok")], "summary": {"counts": {"pass": 1, "warn": 0, "fail": 0}, "preflight_healthy": True}}
    out = tmp_path / "preflight.md"

    mod.write_markdown(result, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Queue Preflight Audit" in text
    assert "| pass | x |" in text
