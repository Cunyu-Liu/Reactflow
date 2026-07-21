import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_coverage_gate.py"
    spec = importlib.util.spec_from_file_location("audit_coverage_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coverage_gate_passes_and_writes_markdown(tmp_path):
    audit = _load_module()
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"totals": {"percent_covered": 90.53, "num_statements": 10}}),
        encoding="utf-8",
    )

    result = audit.run_audit(coverage, threshold=90.0)
    out = tmp_path / "coverage.md"
    audit.write_markdown(result, out)

    assert result["passed"] is True
    assert result["percent_covered"] == 90.53
    assert "ReactFlow Coverage Gate Audit" in out.read_text(encoding="utf-8")


def test_coverage_gate_fails_below_threshold_and_reads_display_string(tmp_path):
    audit = _load_module()
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"totals": {"percent_covered_display": "89.99"}}),
        encoding="utf-8",
    )

    result = audit.run_audit(coverage, threshold=90.0)

    assert result["passed"] is False
    assert result["percent_covered"] == 89.99
