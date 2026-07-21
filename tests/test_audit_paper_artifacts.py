import importlib.util
import json
from pathlib import Path

from reactflow.splits import SplitAssignment, SplitManifest, manifest_to_json


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_paper_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_cache_files(root):
    cache = root / "cache"
    cache.mkdir(parents=True)
    for name in ("efold_train.jsonl", "archiveII.jsonl", "PDB.jsonl", "viral.jsonl", "lncRNA.jsonl", "human_mRNA.jsonl"):
        (cache / name).write_text('{"sequence":"ACGU"}\n', encoding="utf-8")


def _write_metadata(root, *, method="mmseqs", error=None):
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "rfam_current_mmseqs_metadata.manifest.json").write_text(
        json.dumps(
            {
                "cluster_count": 2,
                "cluster_method": method,
                "input_records": 2,
                "metadata_records": 2,
                "mmseqs_error": error,
                "split_group_count": 2,
            }
        ),
        encoding="utf-8",
    )


def _write_split(root):
    split_dir = root / "splits" / "rfam_current_mmseqs_seed0"
    split_dir.mkdir(parents=True)
    manifest = SplitManifest(
        assignments=(
            SplitAssignment("a", "CL1", "c1", 10, "len_le_64", "train"),
            SplitAssignment("b", "CL2", "c2", 11, "len_le_64", "novel"),
            SplitAssignment("c", "CL3", "c3", 12, "len_le_64", "val"),
            SplitAssignment("d", "CL4", "c4", 13, "len_le_64", "test"),
        ),
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        novel_clan_fraction=0.15,
        length_bucket_boundaries=(64,),
        seed=0,
    )
    manifest_to_json(manifest, split_dir / "split_manifest.json")


def test_run_audit_passes_complete_artifacts(tmp_path):
    mod = _load_module()
    _write_cache_files(tmp_path)
    _write_metadata(tmp_path)
    _write_split(tmp_path)
    run = tmp_path / "runs" / "RF-ok"
    run.mkdir(parents=True)
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "stdout.json").write_text('{"tiers":{"PDB":{"mean_f1":0.1}}}', encoding="utf-8")
    (run / "profile.summary.json").write_text('{"phases":{}}', encoding="utf-8")
    (run / "training_checkpoint.json").write_text("{}", encoding="utf-8")

    result = mod.run_audit(tmp_path, require_final_metrics=True, run_globs=["RF-*"])

    assert result["summary"]["ok_for_paper_table"] is True
    assert result["summary"]["counts"]["fail"] == 0


def test_run_audit_fails_non_mmseqs_and_missing_metrics(tmp_path):
    mod = _load_module()
    _write_cache_files(tmp_path)
    _write_metadata(tmp_path, method="exact")
    _write_split(tmp_path)
    run = tmp_path / "runs" / "RF-running"
    run.mkdir(parents=True)
    (run / "profile.jsonl").write_text('{"phase":"model_forward","seconds":1}\n', encoding="utf-8")

    result = mod.run_audit(tmp_path, require_final_metrics=True, run_globs=["RF-*"])
    failed_items = {row["item"] for row in result["rows"] if row["status"] == "fail"}

    assert result["summary"]["ok_for_paper_table"] is False
    assert "metadata_cluster_method" in failed_items
    assert "run_metrics:RF-running" in failed_items


def test_write_markdown(tmp_path):
    mod = _load_module()
    rows = [mod.check("pass", "cache:efold_train.jsonl", tmp_path / "x", "ok")]
    summary = mod.summarize(rows)
    out = tmp_path / "audit.md"

    mod.write_markdown(rows, summary, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Paper Artifact Audit" in text
    assert "| pass | cache:efold_train.jsonl |" in text
