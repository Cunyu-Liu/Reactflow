import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_external_baseline_predictions.py"
    spec = importlib.util.spec_from_file_location("evaluate_external_baseline_predictions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_complete_predictions_emit_same_split_sota_row(tmp_path):
    module = _load_module()
    long_sequence = "G" + "A" * 28 + "C"
    gold = _write_jsonl(
        tmp_path / "novel.jsonl",
        [
            {"source_id": "a", "sequence": long_sequence, "pairs": [[0, 29]]},
            {"source_id": "b", "sequence": long_sequence, "pairs": [[0, 29]]},
        ],
    )
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"source_id": "a", "sequence": long_sequence, "predicted_pairs": [[0, 29]]},
            {"source_id": "b", "sequence": long_sequence, "predicted_pairs": []},
        ],
    )
    output = tmp_path / "baseline_efold_results.json"

    payload = module.evaluate_baselines(
        gold_paths={"novel_clan": gold},
        prediction_paths={"novel_clan": predictions},
        model="eFold/RNAndria local rerun",
        protocol="same_split_local",
        seed_count="single_seed",
        output_path=output,
        one_based_predictions=False,
    )

    tier = payload["tiers"]["novel_clan"]
    assert tier["status"] == "ok"
    assert tier["gold_count"] == 2
    assert tier["matched_count"] == 2
    assert tier["mean_f1"] == pytest.approx(0.5)
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["model"] == "eFold/RNAndria local rerun"
    assert row["protocol"] == "same_split_local"
    assert row["split"] == "MMseqs:novel_clan"
    assert row["mean_f1"] == pytest.approx(0.5)
    assert row["long_recall"] == pytest.approx(0.5)


def test_missing_predictions_do_not_emit_sota_rows(tmp_path):
    module = _load_module()
    gold = _write_jsonl(
        tmp_path / "novel.jsonl",
        [{"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]}],
    )

    payload = module.evaluate_baselines(
        gold_paths={"novel_clan": gold},
        prediction_paths={},
        model="RNADiffFold local rerun",
        protocol="same_split_local",
        seed_count="single_seed",
        output_path=tmp_path / "baseline_rnadifffold_results.json",
        one_based_predictions=False,
    )

    assert payload["tiers"]["novel_clan"]["status"] == "missing_predictions"
    assert payload["rows"] == []


def test_partial_predictions_are_withheld_by_default(tmp_path):
    module = _load_module()
    gold = _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]},
            {"source_id": "b", "sequence": "GCGC", "pairs": [[0, 3]]},
        ],
    )
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [{"source_id": "a", "sequence": "GGCC", "predicted_pairs": [[0, 3]]}],
    )

    payload = module.evaluate_baselines(
        gold_paths={"in_clan": gold},
        prediction_paths={"in_clan": predictions},
        model="eFold/RNAndria local rerun",
        protocol="same_split_local",
        seed_count="single_seed",
        output_path=tmp_path / "baseline_efold_results.json",
        one_based_predictions=False,
    )

    assert payload["tiers"]["in_clan"]["status"] == "partial"
    assert payload["tiers"]["in_clan"]["matched_count"] == 1
    assert payload["rows"] == []


def test_cli_writes_protocol_safe_artifact(tmp_path, capsys):
    module = _load_module()
    gold = _write_jsonl(
        tmp_path / "public.jsonl",
        [{"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]}],
    )
    predictions = _write_jsonl(
        tmp_path / "public_predictions.jsonl",
        [{"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]}],
    )
    output = tmp_path / "baseline_efold_results.json"

    assert (
        module.main(
            [
                "--model",
                "eFold/RNAndria local rerun",
                "--gold-json",
                f"archiveII={gold}",
                "--prediction-json",
                f"archiveII={predictions}",
                "--protocol",
                "local_closest_protocol",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert printed["rows"] == 1
    assert payload["rows"][0]["split"] == "eFold-RNAndria:archiveII"
    assert payload["rows"][0]["protocol"] == "local_closest_protocol"
