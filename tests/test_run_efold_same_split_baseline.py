import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_efold_same_split_baseline.py"
    spec = importlib.util.spec_from_file_location("run_efold_same_split_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_parse_dot_bracket_supports_multiple_bracket_types():
    module = _load_module()

    assert module.parse_dot_bracket("([..])") == [[0, 5], [1, 4]]


def test_parse_dot_bracket_rejects_unbalanced_structure():
    module = _load_module()

    with pytest.raises(ValueError, match="unbalanced"):
        module.parse_dot_bracket("((..)")


def test_extract_dot_bracket_accepts_efold_sequence_mapping():
    module = _load_module()

    assert module._extract_dot_bracket({"AAACA": "((.))"}, length=5) == "((.))"


def test_extract_base_pairs_accepts_efold_sequence_mapping():
    module = _load_module()

    assert module._extract_base_pairs({"AAACA": [(1, 5), (2, 4)]}, length=5, one_based=True) == [[0, 4], [1, 3]]


def test_extract_base_pairs_ignores_efold_diagonal_artifacts():
    module = _load_module()

    assert module._extract_base_pairs(
        {"AAACA": [(1, 5), (3, 3), (2, 4)]},
        length=5,
        one_based=True,
    ) == [[0, 4], [1, 3]]


def test_prediction_record_accepts_normalized_base_pairs():
    module = _load_module()

    assert module._prediction_record({"predicted_pairs": [[0, 4], [1, 3]]}, length=5) == {
        "predicted_pairs": [[0, 4], [1, 3]]
    }


def test_prediction_function_passes_module_device(monkeypatch):
    module = _load_module()
    calls = []

    def fake_predict(sequence, *, device):
        calls.append((sequence, device))
        return {"predicted_pairs": [[0, 3]]}

    monkeypatch.setattr(module, "_predict_with_module", fake_predict)

    predict = module._prediction_function(backend="module", efold_bin=None, device="cuda")

    assert predict("GGCC") == {"predicted_pairs": [[0, 3]]}
    assert calls == [("GGCC", "cuda")]


def test_export_predictions_resume_existing_appends_missing_rows(tmp_path):
    module = _load_module()
    gold = _write_jsonl(
        tmp_path / "novel.jsonl",
        [
            {"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]},
            {"source_id": "b", "sequence": "GCGC", "pairs": [[0, 3]]},
        ],
    )
    output = tmp_path / "predictions" / "novel_clan.efold.predictions.jsonl"
    output.parent.mkdir(parents=True)
    _write_jsonl(
        output,
        [
            {
                "source_id": "a",
                "sequence": "GGCC",
                "predicted_pairs": [[0, 3]],
                "prediction_backend": "efold",
            }
        ],
    )

    summary = module.export_predictions(
        tier="novel_clan",
        gold_path=gold,
        output_path=output,
        predict_structure=lambda _sequence: {"predicted_pairs": [[0, 3]]},
        limit=None,
        progress_every=0,
        resume_existing=True,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["resumed_from"] == 1
    assert summary["prediction_count"] == 2
    assert [row["source_id"] for row in rows] == ["a", "b"]


def test_cli_wrapper_exports_predictions_and_scores_complete_tier(tmp_path):
    module = _load_module()
    fake_efold = tmp_path / "efold"
    fake_efold.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "seq = sys.argv[1]\n"
        "print(seq)\n"
        "print('(' + '.' * (len(seq) - 2) + ')' if len(seq) >= 2 else '.' * len(seq))\n",
        encoding="utf-8",
    )
    fake_efold.chmod(0o755)
    gold = _write_jsonl(
        tmp_path / "novel.jsonl",
        [
            {"source_id": "a", "sequence": "GGCC", "pairs": [[0, 3]]},
            {"source_id": "b", "sequence": "GCGC", "pairs": [[0, 3]]},
        ],
    )
    results = tmp_path / "baseline_efold_results.json"

    assert (
        module.main(
            [
                "--backend",
                "cli",
                "--efold-bin",
                str(fake_efold),
                "--gold-json",
                f"novel_clan={gold}",
                "--output-dir",
                str(tmp_path / "predictions"),
                "--results-json",
                str(results),
            ]
        )
        == 0
    )

    prediction_rows = [
        json.loads(line)
        for line in (tmp_path / "predictions" / "novel_clan.efold.predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payload = json.loads(results.read_text(encoding="utf-8"))

    assert prediction_rows[0]["predicted_pairs"] == [[0, 3]]
    assert payload["tiers"]["novel_clan"]["status"] == "ok"
    assert payload["tiers"]["novel_clan"]["matched_count"] == 2
    assert payload["rows"][0]["protocol"] == "same_split_local"
    assert payload["rows"][0]["mean_f1"] == pytest.approx(1.0)
