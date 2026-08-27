from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ControllerCase:
    name: str
    script: Path
    production_out_assignment: str
    gpus: tuple[str, ...]
    expected_modules_without_training: tuple[str, ...]


CASES = (
    ControllerCase(
        name="smoke",
        script=ROOT
        / "scripts/reactflow_delta/run_puzzle_set_meta_context_smoke_controller.sh",
        production_out_assignment=(
            "out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/"
            "p1m2_real_smoke"
        ),
        gpus=("7",),
        expected_modules_without_training=(
            "scripts.reactflow_delta.merge_puzzle_set_meta_context_probe",
            "scripts.reactflow_delta.qualify_puzzle_set_meta_context_smoke",
        ),
    ),
    ControllerCase(
        name="screen",
        script=ROOT
        / "scripts/reactflow_delta/run_puzzle_set_meta_context_screen_controller.sh",
        production_out_assignment=(
            "out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/"
            "p1m3_screen_seed0"
        ),
        gpus=("3", "7"),
        expected_modules_without_training=(
            "scripts.reactflow_delta.merge_puzzle_set_meta_context_probe",
        ),
    ),
    ControllerCase(
        name="formal",
        script=ROOT
        / "scripts/reactflow_delta/run_puzzle_set_meta_context_formal_controller.sh",
        production_out_assignment=(
            "out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/"
            "p1m4_formal_seeds0_4"
        ),
        gpus=("3", "7"),
        expected_modules_without_training=(
            "scripts.reactflow_delta.merge_puzzle_set_meta_context_probe",
            "scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal",
        ),
    ),
    ControllerCase(
        name="branch5",
        script=ROOT
        / "scripts/reactflow_delta/run_post_v14_branch5_route_probe_controller.sh",
        production_out_assignment=(
            "out=/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/"
            "b5rp1_seed0"
        ),
        gpus=("3", "7"),
        expected_modules_without_training=(
            "scripts.reactflow_delta.merge_post_v14_branch5_route_probe",
        ),
    ),
)


def _write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys


            args = sys.argv[1:]


            def record(event, **fields):
                with Path(os.environ["FAKE_EVENTS"]).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps({"event": event, **fields}) + "\\n")


            cuda_visible_devices_value = os.environ.get("CUDA_VISIBLE_DEVICES")
            if args and args[0] == "-c":
                record(
                    "preflight",
                    cuda_visible_devices_value=cuda_visible_devices_value,
                )
                if cuda_visible_devices_value == os.environ.get("FAKE_FAIL_GPU"):
                    print(
                        "CUDA_REQUIRED: fake "
                        f"cuda_visible_devices_value={cuda_visible_devices_value} "
                        "logical_device=cuda:0 unavailable",
                        file=sys.stderr,
                    )
                    raise SystemExit(73)
                raise SystemExit(0)


            def option(name):
                return args[args.index(name) + 1]


            module = option("-m")
            record(
                "module",
                module=module,
                cuda_visible_devices_value=cuda_visible_devices_value,
            )
            if module == "scripts.reactflow_delta.run_puzzle_set_meta_context_probe":
                out_dir = Path(option("--out-dir"))
                seed = option("--seed")
                for fold in option("--folds").split(","):
                    path = out_dir / f"puzzle_set_fold_result_fold{fold}_seed{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}\\n", encoding="utf-8")
            elif module == "scripts.reactflow_delta.run_post_v14_branch5_route_probe":
                out_dir = Path(option("--out-dir"))
                for fold in option("--folds").split(","):
                    paths = (
                        out_dir / f"puzzle_set_branch5_probe_fold{fold}_seed0.json",
                        out_dir
                        / f"puzzle_set_branch5_probe_predictions_fold{fold}_seed0.npz",
                        out_dir / f"puzzle_set_branch5_probe_ridge_fold{fold}_seed0.json",
                    )
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("fake\\n", encoding="utf-8")
            elif module in {
                "scripts.reactflow_delta.merge_puzzle_set_meta_context_probe",
                "scripts.reactflow_delta.qualify_puzzle_set_meta_context_smoke",
                "scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal",
                "scripts.reactflow_delta.merge_post_v14_branch5_route_probe",
            }:
                out_json = Path(option("--out-json"))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text("{}\\n", encoding="utf-8")
            else:
                print(f"unexpected fake module: {module}", file=sys.stderr)
                raise SystemExit(92)
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def _write_controller_copy(
    tmp_path: Path,
    case: ControllerCase,
    out_dir: Path,
    fake_python: Path,
) -> Path:
    source = case.script.read_text(encoding="utf-8")
    replacements = (
        (
            'repo=$(cd "$(dirname "$0")/../.." && pwd)',
            f"repo={shlex.quote(str(ROOT))}",
        ),
        (
            "python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python",
            f"python_bin={shlex.quote(str(fake_python))}",
        ),
        (
            case.production_out_assignment,
            f"out={shlex.quote(str(out_dir))}",
        ),
    )
    for production_assignment, test_assignment in replacements:
        assert source.count(production_assignment) == 1
        source = source.replace(production_assignment, test_assignment, 1)
    script_copy = tmp_path / f"{case.name}_controller.sh"
    script_copy.write_text(source, encoding="utf-8")
    script_copy.chmod(0o755)
    return script_copy


def _controller_fixture(
    tmp_path: Path,
    case: ControllerCase,
    out_dir: Path,
    *,
    fail_gpu: str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    event_path = tmp_path / f"{case.name}_events.jsonl"
    env = os.environ.copy()
    env["FAKE_EVENTS"] = str(event_path)
    script_copy = _write_controller_copy(
        tmp_path,
        case,
        out_dir,
        _write_fake_python(tmp_path),
    )
    if fail_gpu is not None:
        env["FAKE_FAIL_GPU"] = fail_gpu
    return env, event_path, script_copy


def _populate_complete(case: ControllerCase, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if case.name == "smoke":
        seeds = (0,)
        folds = range(2)
    elif case.name == "screen":
        seeds = (0,)
        folds = range(20)
    elif case.name == "formal":
        seeds = range(5)
        folds = range(20)
    else:
        for fold in range(20):
            for name in (
                f"puzzle_set_branch5_probe_fold{fold}_seed0.json",
                f"puzzle_set_branch5_probe_predictions_fold{fold}_seed0.npz",
                f"puzzle_set_branch5_probe_ridge_fold{fold}_seed0.json",
            ):
                (out_dir / name).write_text("complete\n", encoding="utf-8")
        return
    for seed in seeds:
        for fold in folds:
            (out_dir / f"puzzle_set_fold_result_fold{fold}_seed{seed}.json").write_text(
                "{}\n", encoding="utf-8"
            )


def _make_first_task_incomplete(case: ControllerCase, out_dir: Path) -> None:
    if case.name == "branch5":
        (out_dir / "puzzle_set_branch5_probe_ridge_fold0_seed0.json").unlink()
        return
    (out_dir / "puzzle_set_fold_result_fold0_seed0.json").unlink()
    (out_dir / "puzzle_set_predictions_fold0_seed0.npz").write_text(
        "partial\n", encoding="utf-8"
    )


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_snapshot(out_dir: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(out_dir)): None if path.is_dir() else path.read_bytes()
        for path in sorted(out_dir.rglob("*"))
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_failed_gpu_preflight_keeps_artifacts_byte_for_byte_unchanged(
    tmp_path: Path, case: ControllerCase
) -> None:
    subprocess.run(["bash", "-n", str(case.script)], check=True)
    production_source = case.script.read_text(encoding="utf-8")
    assert "physical_gpu=" not in production_source
    assert (
        "cuda_visible_devices_value=%s logical_device=cuda:0"
        in production_source
    )
    assert case.production_out_assignment in production_source
    for test_hook in (
        "_CONTROLLER_REPO",
        "_CONTROLLER_PYTHON_BIN",
        "_CONTROLLER_OUT",
        "FAKE_",
        "relocation",
        "isolated controller tests",
    ):
        assert test_hook not in production_source
    out_dir = tmp_path / case.name
    _populate_complete(case, out_dir)
    _make_first_task_incomplete(case, out_dir)
    before = _artifact_snapshot(out_dir)
    fail_gpu = case.gpus[-1]
    env, event_path, script_copy = _controller_fixture(
        tmp_path, case, out_dir, fail_gpu=fail_gpu
    )

    completed = subprocess.run(
        ["bash", str(script_copy), *case.gpus],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "CUDA_REQUIRED" in completed.stderr
    assert f"cuda_visible_devices_value={fail_gpu}" in completed.stderr
    assert "logical_device=cuda:0" in completed.stderr
    assert _artifact_snapshot(out_dir) == before
    events = _read_events(event_path)
    assert [
        event["cuda_visible_devices_value"] for event in events
    ] == list(case.gpus)
    assert all(event["event"] == "preflight" for event in events)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_complete_tasks_skip_gpu_preflight_and_continue_existing_postprocessing(
    tmp_path: Path, case: ControllerCase
) -> None:
    out_dir = tmp_path / case.name
    _populate_complete(case, out_dir)
    env, event_path, script_copy = _controller_fixture(
        tmp_path, case, out_dir, fail_gpu=case.gpus[0]
    )

    completed = subprocess.run(
        ["bash", str(script_copy), *case.gpus],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = _read_events(event_path)
    assert not [event for event in events if event["event"] == "preflight"]
    assert tuple(event["module"] for event in events) == (
        case.expected_modules_without_training
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_all_gpu_preflights_finish_before_fake_training_and_partial_archive(
    tmp_path: Path, case: ControllerCase
) -> None:
    out_dir = tmp_path / case.name
    _populate_complete(case, out_dir)
    _make_first_task_incomplete(case, out_dir)
    env, event_path, script_copy = _controller_fixture(tmp_path, case, out_dir)

    completed = subprocess.run(
        ["bash", str(script_copy), *case.gpus],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = _read_events(event_path)
    preflight_indices = [
        index for index, event in enumerate(events) if event["event"] == "preflight"
    ]
    module_indices = [
        index for index, event in enumerate(events) if event["event"] == "module"
    ]
    assert [
        events[index]["cuda_visible_devices_value"] for index in preflight_indices
    ] == list(case.gpus)
    assert max(preflight_indices) < min(module_indices)
    assert list((out_dir / "interrupted_attempts").glob("*/*"))
