from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/reactflow_delta/run_post_v14_branch5_route_probe_controller.sh"
)


def test_branch5_controller_is_valid_missing_fold_only_unscored_shell() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "fold_is_complete" in text
    assert "archive_incomplete_fold" in text
    assert "interrupted_attempts" in text
    assert 'mv "${present[@]}" "${interrupted}/"' in text
    assert "puzzle_set_branch5_probe_fold%s_seed0.json" in text
    assert "puzzle_set_branch5_probe_predictions_fold%s_seed0.npz" in text
    assert "puzzle_set_branch5_probe_ridge_fold%s_seed0.json" in text
    assert "scripts.reactflow_delta.run_post_v14_branch5_route_probe" in text
    assert "scripts.reactflow_delta.merge_post_v14_branch5_route_probe" in text
    assert "puzzle_set_branch5_probe_complete_unscored_merge.json" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "fold=worker; fold<20; fold+=worker_count" in text
    assert "score_post_v14_branch5_route_probe" not in text
    assert "qualify_post_v14_branch5_route_probe" not in text
    assert "held_score_read_allowed" not in text
    assert "rm " not in text
    assert "kill " not in text
    assert "nvidia-smi" not in text


def test_branch5_controller_freezes_all_scientific_input_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    expected = (
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "openknot_m2/OK7a_M2_data.v4.5.2.csv",
        "/mnt/cunyuliu/reactflow_delta_target_identity_correction/"
        "tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json",
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/"
        "ensemble_delta_cache.h5",
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/"
        "constrained_cache.h5",
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/"
        "source_binding/post_v14_branch5_safe_source_manifest.json",
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/" "b5rp1_seed0",
    )
    for path in expected:
        assert path in text
    assert "REACTFLOW_" not in text
