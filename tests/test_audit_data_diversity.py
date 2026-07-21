import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_data_diversity.py"
    spec = importlib.util.spec_from_file_location("audit_data_diversity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_dataset_audit_counts_length_source_and_long_range_pairs(tmp_path):
    module = _load_module()
    data = _write_jsonl(
        tmp_path / "tiny.jsonl",
        [
            {
                "clan": "RF1",
                "cluster": "c1",
                "family": "fam1",
                "length_bucket": "len_le_64",
                "pairs": [[0, 29], [1, 28], [5, 8], [9, 9]],
                "reactivity": [0.1] * 30,
                "reactivity_source": "real_profile",
                "sequence": "A" * 30,
                "source_id": "RF1.fa.csv_1",
            },
            {
                "clan": "RF2",
                "cluster": "c2",
                "family": "fam2",
                "length_bucket": "len_65_128",
                "pairs": [[0, 10], [1, 9]],
                "reactivity": [None] * 80,
                "reactivity_source": "structure_forward_proxy",
                "sequence": "C" * 80,
                "source_id": "RF2.fa.csv_1",
                "window": {"index": 0, "parent_length": 200, "start": 0, "end": 80},
            },
        ],
    )

    result = module.run_audit([("tiny", data)], long_range_min_distance=24)
    dataset = result["datasets"][0]

    assert result["summary"]["record_count"] == 2
    assert dataset["length"]["buckets"] == {"len_le_64": 1, "len_le_128": 1}
    assert dataset["source_mix"]["unique_groups"] == 2
    assert dataset["family_clan"]["unique_clans"] == 2
    assert dataset["structure_complexity"]["long_range_pair_count"] == 2
    assert dataset["structure_complexity"]["long_range_pair_fraction"] == pytest.approx(0.4)
    assert dataset["structure_complexity"]["max_stem_length"] == 2
    assert dataset["windowing"]["windowed_record_count"] == 1
    assert dataset["reactivity"]["with_reactivity_count"] == 1


def test_manifest_projects_curriculum_fields(tmp_path):
    module = _load_module()
    data = _write_jsonl(
        tmp_path / "single.jsonl",
        [
            {
                "pairs": [[0, 3]],
                "sequence": "GGCC",
                "source_id": "sourceA_1",
            }
        ],
    )

    audit = module.run_audit([("single", data)], long_range_min_distance=24)
    manifest = module.build_source_family_length_manifest(audit)

    assert manifest["summary"]["dataset_labels"] == ["single"]
    row = manifest["rows"][0]
    assert row["label"] == "single"
    assert "length" in row
    assert "source_mix" in row
    assert "family_clan" in row
    assert row["warnings"] == [
        "single_length_bucket",
        "mostly_missing_clan_metadata",
        "mostly_missing_family_metadata",
        "low_long_range_pair_fraction",
    ]


def test_collect_inputs_includes_existing_defaults(tmp_path):
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_jsonl(cache / "PDB.jsonl", [{"sequence": "GGCC", "pairs": [[0, 3]], "source_id": "pdb1"}])

    inputs = module.collect_inputs(full_run_root=tmp_path, input_jsonl=(), include_defaults=True)

    assert inputs == [("PDB", cache / "PDB.jsonl")]
