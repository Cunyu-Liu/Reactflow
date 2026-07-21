from dataclasses import replace
import json

from reactflow.checkpoint import write_training_checkpoint
from reactflow.cli import main
from reactflow.synthetic import make_dataset
from reactflow.train import TrainConfig, sample_to_cache_obj, train_pilot


def _fixture(tmp_path):
    samples = make_dataset(count=2, stem=3, loop=4, probe="2A3", seed=9)
    config = TrainConfig(epochs=1, seed=2)
    result = train_pilot(samples=samples, config=config)
    checkpoint = tmp_path / "checkpoint.json"
    write_training_checkpoint(checkpoint, config=config, result=result, metadata={"fixture": True})
    cache = tmp_path / "validation.jsonl"
    rows = []
    for index, sample in enumerate(samples):
        enriched = replace(
            sample,
            source_id=f"sample-{index}",
            reactivity_source="real_profile",
        )
        rows.append(json.dumps(sample_to_cache_obj(enriched), sort_keys=True))
    cache.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checkpoint, cache


def test_calibrate_then_evaluate_checkpoint_uses_locked_manifest_and_honest_tier(tmp_path, capsys):
    checkpoint, cache = _fixture(tmp_path)
    manifest = tmp_path / "decoder_manifest.json"
    assert main(
        [
            "calibrate-inference",
            "--checkpoint",
            str(checkpoint),
            "--validation-json",
            str(cache),
            "--output",
            str(manifest),
            "--coarse-count",
            "1",
            "--validation-count",
            "2",
            "--steps-grid",
            "2",
            "--samples-grid",
            "2",
            "--temperature-grid",
            "1",
            "--threshold-grid",
            "0",
            "--matching-policy",
            "nested_dp",
        ]
    ) == 0
    capsys.readouterr()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["fitted_split"] == "validation"
    assert payload["test_override_allowed"] is False
    assert payload["selected_inference"] == {
        "num_samples": 2,
        "num_steps": 2,
        "selection_metrics": payload["selected_inference"]["selection_metrics"],
    }

    output = tmp_path / "evaluation"
    assert main(
        [
            "evaluate-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--decoder-manifest",
            str(manifest),
            "--eval-json",
            f"novel_clan={cache}",
            "--output-dir",
            str(output),
            "--mode",
            "calibrated_marginal",
            "--limit-per-tier",
            "2",
        ]
    ) == 0
    summary = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "mmseqs_component_holdout:calibrated_marginal" in summary["results"]
    assert len(summary["sample_ids"]["mmseqs_component_holdout"]) == 2
    result = summary["results"]["mmseqs_component_holdout:calibrated_marginal"]
    assert set(result["distance_bins"]) == {"short", "medium", "long"}
    assert summary["length_stratified"]["mmseqs_component_holdout:calibrated_marginal"]
    probing = json.loads((output / "probing_metrics.json").read_text(encoding="utf-8"))
    assert probing["main"]["profile_count"] == 2


def test_evaluate_efold_default_refuses_uncalibrated_endpoint_path(tmp_path, capsys):
    _checkpoint, cache = _fixture(tmp_path)
    code = main(
        [
            "evaluate-efold",
            "--train-json",
            str(cache),
            "--eval-json",
            f"PDB={cache}",
            "--train-limit",
            "1",
            "--eval-limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "requires --validation-json" in captured.err
