from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(
    "scripts/reactflow_delta/run_independent_rnet_distill_screen_controller.sh"
)
FORMAL_SCRIPT = Path(
    "scripts/reactflow_delta/run_independent_rnet_distill_formal_controller.sh"
)


def test_controller_shell_syntax_and_phase_rejection() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    completed = subprocess.run(
        ["bash", str(SCRIPT), "RND9"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "unsupported independent RNet downstream phase" in completed.stderr


def test_controller_freezes_schedules_and_cuda_mapping() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "folds=(0 1)" in text
    assert "folds=({0..19})" in text
    assert "point_epochs=3" in text and "calibration_epochs=3" in text
    assert "point_epochs=40" in text and "calibration_epochs=40" in text
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in text
    assert "--device cuda:0" in text
    assert "nvidia-smi" not in text
    assert "free_vram" not in text.lower()
    assert "run_model_rescue_v14" not in text


def test_controller_is_missing_only_and_never_merges_after_worker_failure() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    task_block = text.index('if ((${#tasks[@]} > 0)); then')
    no_gpu_check = text.index('if ((${#gpus[@]} == 0)); then')
    merge_call = text.index("scripts.reactflow_delta.merge_independent_rnet_distill")
    failure_exit = text.index("if ((failed != 0)); then")
    assert task_block < no_gpu_check < failure_exit < merge_call
    assert "if ! task_is_complete" in text
    assert "if task_is_complete" in text
    assert "rnet_distill_complete_unscored_merge.json" in text


def test_formal_controller_freezes_exact_hundred_task_cuda_queue() -> None:
    subprocess.run(["bash", "-n", str(FORMAL_SCRIPT)], check=True)
    text = FORMAL_SCRIPT.read_text(encoding="utf-8")
    assert "phase=RND6P" in text
    assert (
        "experiment_id="
        "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY"
    ) in text
    assert "for seed in {0..4}; do" in text
    assert "for fold in {0..19}; do" in text
    assert "--point-epochs 40" in text
    assert "--calibration-epochs 40" in text
    assert '--seed "${seed}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in text
    assert "--device cuda:0" in text
    assert "nvidia-smi" not in text
    assert "free_vram" not in text.lower()
    assert "run_model_rescue_v14" not in text


def test_formal_controller_is_missing_only_and_fail_closed_at_terminal_states() -> None:
    text = FORMAL_SCRIPT.read_text(encoding="utf-8")
    score_guard = text.index('if [[ -e "${formal_score}"')
    task_block = text.index('if ((${#tasks[@]} > 0)); then')
    mixed_state_guard = text.index(
        'if ((${#tasks[@]} > 0)) && [[ -e "${merged}" || -e "${assembly}" ]]'
    )
    no_gpu_check = text.index('if ((${#gpus[@]} == 0)); then')
    worker_failure_exit = text.index("if ((failed != 0)); then")
    result_marker_sweep = text.rindex("for seed in {0..4}; do")
    merge_call = text.index("scripts.reactflow_delta.merge_independent_rnet_distill")
    assembly_call = text.index(
        "scripts.reactflow_delta.assemble_independent_rnet_distill_formal"
    )
    assert score_guard < mixed_state_guard < task_block < no_gpu_check
    assert no_gpu_check < worker_failure_exit < result_marker_sweep < merge_call
    assert merge_call < assembly_call
    assert 'if [[ ! -f "${merged}" ]]; then' in text
    assert 'if [[ ! -f "${assembly}" ]]; then' in text
    assert "missing prediction tasks alongside an existing merge or assembly" in text
    assert "assembly exists without its canonical merge" in text


def test_formal_controller_uses_exact_canonical_terminal_paths() -> None:
    text = FORMAL_SCRIPT.read_text(encoding="utf-8")
    assert "out=${artifact_root}/rnd6_formal_seeds0_4" in text
    assert "merged=${out}/rnet_distill_complete_unscored_merge.json" in text
    assert (
        "assembly=${assembled_dir}/"
        "rnet_distill_five_seed_prediction_only_assembly.json"
    ) in text
    assert "formal_score=${out}/rnet_distill_complete_formal_score.json" in text
    assert "formal_qualification=${out}/rnet_distill_formal_qualification.json" in text
    assert '--phase "${phase}"' in text
    assert '--out-dir "${assembled_dir}"' in text
    assert '--out-json "${assembly}"' in text
    assert 'mkdir -p "${assembled_dir}"' not in text
