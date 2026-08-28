from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(
    "scripts/reactflow_delta/run_independent_rnet_distill_screen_controller.sh"
)
FORMAL_SCRIPT = Path(
    "scripts/reactflow_delta/run_independent_rnet_distill_formal_controller.sh"
)


def _sandboxed_controller(
    tmp_path: Path,
    source: Path,
    *,
    preflight_status: int,
) -> tuple[Path, Path, Path, dict[str, str]]:
    repo = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    bin_dir = tmp_path / "bin"
    repo.mkdir()
    bin_dir.mkdir()
    calls = tmp_path / "python_calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {calls}
if [[ "$*" == *assert_run_authority* ]]; then
  exit 0
fi
if [[ "$*" == *require_cuda_device* ]]; then
  printf 'synthetic CUDA probe stderr\\n' >&2
  exit {preflight_status}
fi
printf 'unexpected fake-python command: %s\\n' "$*" >&2
exit 91
""".format(
            calls=shlex.quote(str(calls)),
            preflight_status=preflight_status,
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_date = bin_dir / "date"
    fake_date.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '2026-08-28T12:34:56+08:00'\n",
        encoding="utf-8",
    )
    fake_date.chmod(0o755)

    text = source.read_text(encoding="utf-8")
    text = re.sub(r"^repo=.*$", f"repo={shlex.quote(str(repo))}", text, count=1, flags=re.M)
    text = re.sub(
        r"^python_bin=.*$",
        f"python_bin={shlex.quote(str(fake_python))}",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"^artifact_root=.*$",
        f"artifact_root={shlex.quote(str(artifact_root))}",
        text,
        count=1,
        flags=re.M,
    )
    controller = tmp_path / source.name
    controller.write_text(text, encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return controller, artifact_root, calls, env


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


@pytest.mark.parametrize(
    ("source", "arguments", "out_name", "phase", "gpu"),
    [
        (SCRIPT, ["RND2", "3"], "rnd2_smoke_seed0", "RND2", "3"),
        (FORMAL_SCRIPT, ["4"], "rnd6_formal_seeds0_4", "RND6P", "4"),
    ],
)
def test_cuda_preflight_failure_is_persisted_before_any_worker_or_merge(
    tmp_path: Path,
    source: Path,
    arguments: list[str],
    out_name: str,
    phase: str,
    gpu: str,
) -> None:
    controller, artifact_root, calls, env = _sandboxed_controller(
        tmp_path,
        source,
        preflight_status=17,
    )

    completed = subprocess.run(
        ["bash", str(controller), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1
    assert f"{phase} CUDA preflight failed" in completed.stderr
    evidence_path = artifact_root / out_name / "logs" / f"cuda_preflight_gpu{gpu}.log"
    evidence = evidence_path.read_text(encoding="utf-8")
    assert "2026-08-28T12:34:56+08:00" in evidence
    assert f"phase={phase}" in evidence
    assert f"physical_gpu={gpu}" in evidence
    assert "logical_device=cuda:0" in evidence
    assert "command=CUDA_VISIBLE_DEVICES=" in evidence
    assert "require_cuda_device" in evidence
    assert "stderr=following" in evidence
    assert "synthetic CUDA probe stderr" in evidence
    assert "event=cuda_preflight_failed" in evidence
    assert "status=17" in evidence

    python_calls = calls.read_text(encoding="utf-8")
    assert "assert_run_authority" in python_calls
    assert "require_cuda_device" in python_calls
    assert "run_independent_rnet_distill_downstream" not in python_calls
    assert "merge_independent_rnet_distill" not in python_calls
    assert "assemble_independent_rnet_distill_formal" not in python_calls


def test_successful_cuda_preflight_remains_nonblocking(tmp_path: Path) -> None:
    controller, artifact_root, calls, env = _sandboxed_controller(
        tmp_path,
        SCRIPT,
        preflight_status=0,
    )
    controller_text = controller.read_text(encoding="utf-8")
    function_text = controller_text[
        controller_text.index("preflight_gpu() {") : controller_text.index(
            "\nresult_path() {"
        )
    ]
    harness = tmp_path / "successful_preflight.sh"
    harness.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "phase=RND2",
                f"out={shlex.quote(str(artifact_root / 'rnd2_smoke_seed0'))}",
                f"python_bin={shlex.quote(str(tmp_path / 'bin' / 'python'))}",
                'mkdir -p "${out}/logs"',
                function_text,
                'preflight_gpu "3"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = (
        artifact_root / "rnd2_smoke_seed0" / "logs" / "cuda_preflight_gpu3.log"
    ).read_text(encoding="utf-8")
    assert "synthetic CUDA probe stderr" in evidence
    assert "event=cuda_preflight_pass" in evidence
    assert "status=0" in evidence
    python_calls = calls.read_text(encoding="utf-8")
    assert len(python_calls.splitlines()) == 1
    assert "require_cuda_device('cuda:0')" in python_calls


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
    authority_check = text.index("assert_run_authority")
    task_discovery = text.index("tasks=()")
    stale_merge_guard = text.index(
        'if ((${#tasks[@]} > 0)) && [[ -e "${merged}" ]]'
    )
    task_block = text.index('if ((${#tasks[@]} > 0)); then')
    no_gpu_check = text.index('if ((${#gpus[@]} == 0)); then')
    merge_call = text.index("scripts.reactflow_delta.merge_independent_rnet_distill")
    failure_exit = text.index("if ((failed != 0)); then")
    assert authority_check < task_discovery < stale_merge_guard < task_block
    assert task_block < no_gpu_check < failure_exit < merge_call
    assert "if ! task_is_complete" in text
    assert "if task_is_complete" in text
    assert "rnet_distill_complete_unscored_merge.json" in text
    assert "missing prediction tasks alongside an existing merge" in text
    assert "--validate-existing" in text


def test_screen_controller_binds_requested_phase_before_any_recovery_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "assert_run_authority" in text
    assert "'${phase}'" in text
    assert "-m scripts.reactflow_delta.validate_independent_rnet_distill_contract" not in text
    authority_check = text.index("assert_run_authority")
    task_discovery = text.index("tasks=()")
    existing_merge_validation = text.rindex("--validate-existing")
    assert authority_check < task_discovery < existing_merge_validation


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
    assert "assert_run_authority" in text
    assert "'${phase}'" in text


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
    assert '--repo-root "${repo}"' in text
    assert 'mkdir -p "${assembled_dir}"' not in text
