import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_sota_alignment_table.py"
    spec = importlib.util.spec_from_file_location("build_sota_alignment_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mmseqs_row_uses_same_split_protocol_and_runtime(tmp_path):
    module = _load_module()
    artifact = tmp_path / "mmseqs_final_results.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "run_id": "RF-M1-warm_mmseqs_torch_full_data_e1_bs16",
                    "tier": "novel_clan",
                    "status": "ok",
                    "mean_f1": 0.0447,
                    "mean_mcc": 0.0444,
                    "samples_per_second": 20.0,
                    "artifact": "runs/RF-M1",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = module.build_alignment_rows([artifact], include_default_cited=False)

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "RF-M1-warm_mmseqs_torch_full_data_e1_bs16"
    assert row["protocol"] == "same_split_local"
    assert row["split"] == "MMseqs:mmseqs_component_holdout"
    assert row["seed_count"] == "single_seed"
    assert row["runtime_s_per_sample"] == pytest.approx(0.05)


def test_public_local_row_stays_separate_from_cited_row(tmp_path):
    module = _load_module()
    artifact = tmp_path / "mmseqs_final_results.json"
    artifact.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "run_id": "RF-M1-warm_mmseqs_torch_full_data_e1_bs16",
                        "tier": "viral",
                        "status": "ok",
                        "mean_f1": 0.0156,
                        "mean_mcc": 0.0132,
                        "artifact": "runs/RF-M1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = module.build_alignment_rows([artifact], include_default_cited=True)
    viral_rows = [row for row in rows if "viral" in row["split"]]

    assert len(viral_rows) == 2
    assert {row["protocol"] for row in viral_rows} == {"cited_only", "local_closest_protocol"}
    local = next(row for row in viral_rows if row["protocol"] == "local_closest_protocol")
    cited = next(row for row in viral_rows if row["protocol"] == "cited_only")
    assert local["mean_f1"] == pytest.approx(0.0156)
    assert cited["mean_f1"] == pytest.approx(0.730)


def test_invalid_protocol_is_rejected(tmp_path):
    module = _load_module()
    artifact = tmp_path / "bad.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "model": "bad-baseline",
                    "protocol": "mixed_protocol",
                    "split": "MMseqs:novel_clan",
                    "mean_f1": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid protocol"):
        module.build_alignment_rows([artifact], include_default_cited=False)


def test_explicit_legacy_split_is_rewritten_for_new_reports(tmp_path):
    module = _load_module()
    artifact = tmp_path / "legacy.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "model": "old",
                    "protocol": "same_split_local",
                    "split": "MMseqs:novel_clan",
                    "mean_f1": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = module.build_alignment_rows([artifact], include_default_cited=False)
    assert rows[0]["split"] == "MMseqs:mmseqs_component_holdout"


def test_markdown_uses_fixed_contract_header(tmp_path):
    module = _load_module()
    out = tmp_path / "sota.md"
    module.write_markdown(
        [
            {
                "model": "RF-M0",
                "protocol": "same_split_local",
                "split": "MMseqs:novel_clan",
                "seed_count": "single_seed",
                "mean_f1": 0.0267,
                "mean_mcc": 0.0248,
                "long_f1": None,
                "long_recall": None,
                "reactivity_corr": None,
                "calibration_ece": None,
                "runtime_s_per_sample": None,
                "artifact": "mmseqs_final_results.json",
            }
        ],
        out,
    )

    markdown = out.read_text(encoding="utf-8")
    assert "| model | protocol | split | seed_count | mean_f1 | mean_mcc |" in markdown
    assert "| RF-M0 | same_split_local | MMseqs:novel_clan | single_seed | 0.0267 | 0.0248 |" in markdown
