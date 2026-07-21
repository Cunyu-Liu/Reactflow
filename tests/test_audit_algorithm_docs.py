import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_algorithm_docs.py"
    spec = importlib.util.spec_from_file_location("audit_algorithm_docs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_algorithm_doc_audit_reports_doc_gaps_and_placeholders(tmp_path):
    audit = _load_module()
    source = tmp_path / "mini_alg.py"
    source.write_text(
        '''
def good_loss(x):
    """Compute a toy loss.

    Formula: L = x^2.
    Complexity: O(1).
    """
    return x * x

def missing_complexity(x):
    """Compute a value."""
    return x + 1

def unfinished():
    """Placeholder function.

    Complexity: O(1).
    """
    raise NotImplementedError()

class Empty:
    """Container.

    Complexity: O(1).
    """
    pass
''',
        encoding="utf-8",
    )

    result = audit.run_audit([source])
    summary = result["summary"]

    assert summary["public_nodes"] == 4
    assert summary["missing_complexity"] == 1
    assert summary["placeholder_bodies"] == 2
    assert not summary["strict_ready"]
    assert any(row["qualified_name"] == "good_loss" and row["status"] == "pass" for row in result["rows"])
    assert any(row["qualified_name"] == "missing_complexity" for row in result["rows"])


def test_algorithm_doc_audit_writes_markdown(tmp_path):
    audit = _load_module()
    source = tmp_path / "mini_ok.py"
    source.write_text(
        '''
def project_score(x):
    """Project a score.

    Formula: s' = x.
    Complexity: O(1).
    """
    return x
''',
        encoding="utf-8",
    )
    result = audit.run_audit([source])
    out = tmp_path / "audit.md"

    audit.write_markdown(result, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Algorithm Documentation Audit" in text
    assert "No placeholder" in text
