import importlib.util
import os
from pathlib import Path
import subprocess


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_system_resources.py"
    spec = importlib.util.spec_from_file_location("audit_system_resources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_gpu_lines_computes_memory_fraction():
    mod = _load_module()

    rows = mod.parse_gpu_lines("0, NVIDIA RTX, 1024, 4096, 75\n1, NVIDIA RTX, 0, 4096, 0\n")

    assert len(rows) == 2
    assert rows[0]["memory_fraction"] == 0.25
    assert rows[0]["utilization_gpu_percent"] == 75.0
    assert rows[1]["memory_used_mb"] == 0.0


def test_collect_processes_reads_live_pidfile(tmp_path):
    mod = _load_module()
    pidfile = tmp_path / "self.pid"
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    processes, rows = mod.collect_processes([pidfile])

    assert rows[0]["status"] == "pass"
    assert rows[0]["item"] == "pidfile:self.pid"
    assert processes
    assert processes[0]["pid"] == os.getpid()


def test_collect_processes_warns_missing_pidfile(tmp_path):
    mod = _load_module()

    processes, rows = mod.collect_processes([tmp_path / "missing.pid"])

    assert processes == []
    assert rows[0]["status"] == "warn"


def test_descendant_pids_walks_process_tree(monkeypatch):
    mod = _load_module()

    def fake_run(args):
        parent = args[-1]
        mapping = {
            "10": "11\n12\n",
            "11": "13\n",
            "12": "",
            "13": "",
        }
        stdout = mapping.get(parent, "")
        return subprocess.CompletedProcess(args, 0 if stdout else 1, stdout, "")

    monkeypatch.setattr(mod, "_run_command", fake_run)

    assert mod.descendant_pids(10) == [11, 12, 13]


def test_system_resource_markdown(tmp_path):
    mod = _load_module()
    result = {
        "gpus": [
            {
                "index": "0",
                "memory_fraction": 0.5,
                "memory_total_mb": 4096.0,
                "memory_used_mb": 2048.0,
                "name": "GPU",
                "utilization_gpu_percent": 80.0,
            }
        ],
        "processes": [
            {
                "command": "python",
                "elapsed": "00:01",
                "pcpu": 10.0,
                "pid": 123,
                "pmem": 1.0,
                "rss_mib": 100.0,
                "role": "pidfile",
                "root_pid": 123,
            }
        ],
        "rows": [mod.row("pass", "gpu:list", None, "count=1")],
        "summary": {"counts": {"pass": 1, "warn": 0, "fail": 0}, "gpu_count": 1, "process_count": 1, "resource_healthy": True},
    }
    out = tmp_path / "resources.md"

    mod.write_markdown(result, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow System Resource Audit" in text
    assert "| 0 | GPU | 80.0 | 2048/4096 | 0.5000 |" in text
    assert "| 123 | pidfile | 123 | 10.0 | 1.0 | 100.00 | 00:01 | python |" in text
