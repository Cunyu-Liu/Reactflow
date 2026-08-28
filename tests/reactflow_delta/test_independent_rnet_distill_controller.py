from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(
    "scripts/reactflow_delta/run_independent_rnet_distill_screen_controller.sh"
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
