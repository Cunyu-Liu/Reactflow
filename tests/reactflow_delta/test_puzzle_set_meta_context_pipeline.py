from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_smoke_controller_is_two_fold_prediction_only() -> None:
    script = (
        ROOT / "scripts/reactflow_delta/run_puzzle_set_meta_context_smoke_controller.sh"
    ).read_text(encoding="utf-8")
    assert "for fold in 0 1" in script
    assert "--phase P1M2" in script
    assert "--pretraining-epochs 3" in script
    assert "--point-epochs 3" in script
    assert "--calibration-epochs 3" in script
    assert "--parameter-count 6171697" in script
    assert "--trainable-parameter-count 1404417" in script
    assert "--folds 0,1" in script
    assert "archive_incomplete_fold" in script
    assert "interrupted_attempts" in script
    assert "puzzle_set_candidate_wt_decoder_fold${fold}_seed0.pt" in script
    assert "puzzle_set_null_wt_decoder_fold${fold}_seed0.pt" in script
    assert "score_puzzle_set_meta_context" not in script


def test_screen_controller_is_missing_fold_only_and_complete_before_merge() -> None:
    script = (
        ROOT
        / "scripts/reactflow_delta/run_puzzle_set_meta_context_screen_controller.sh"
    ).read_text(encoding="utf-8")
    assert "puzzle_set_fold_result_fold${fold}_seed0.json" in script
    assert "archive_incomplete_fold" in script
    assert "interrupted_attempts" in script
    assert "puzzle_set_candidate_wt_decoder_fold${fold}_seed0.pt" in script
    assert "puzzle_set_null_wt_decoder_fold${fold}_seed0.pt" in script
    assert 'wait "${pid}"' in script
    assert script.index('wait "${pid}"') < script.index(
        "merge_puzzle_set_meta_context_probe"
    )
    assert "--phase P1M3" in script
    assert "--folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" in script
    assert "--seeds 0" in script
    assert "--pretraining-epochs 200" in script
    assert "--point-epochs 40" in script
    assert "--calibration-epochs 40" in script
    assert "--parameter-count 6171697" in script
    assert "--trainable-parameter-count 1404417" in script


def test_score_once_script_runs_one_scorer_then_one_qualifier() -> None:
    script = (
        ROOT / "scripts/reactflow_delta/run_puzzle_set_meta_context_score_once.sh"
    ).read_text(encoding="utf-8")
    assert script.count("scripts.reactflow_delta.score_puzzle_set_meta_context") == 1
    assert script.count("scripts.reactflow_delta.qualify_puzzle_set_meta_context") == 1
    assert "score-once output already exists; refusing to rerun" in script
    assert script.index("score_puzzle_set_meta_context") < script.index(
        "qualify_puzzle_set_meta_context"
    )
