from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCREEN = (
    ROOT / "scripts/reactflow_delta/run_model_rescue_v14_screen_controller.sh"
)
FORMAL = (
    ROOT / "scripts/reactflow_delta/run_model_rescue_v14_formal_controller.sh"
)
PRODUCTION_REPO_ASSIGNMENT = (
    "repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827"
)
PRODUCTION_PYTHON_ASSIGNMENT = (
    "python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python"
)
PRODUCTION_OUT_ASSIGNMENTS = {
    SCREEN: "out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0",
    FORMAL: (
        "out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
        "v14m4_formal_seeds0_4"
    ),
}


def _write_fake_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "fake_python"
    runner.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys
            import time


            args = sys.argv[1:]


            def option(name):
                return args[args.index(name) + 1]


            def record(event, **fields):
                payload = {"event": event, **fields, "time": time.monotonic()}
                line = (json.dumps(payload, sort_keys=True) + "\\n").encode()
                fd = os.open(
                    os.environ["FAKE_EVENTS"],
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o644,
                )
                try:
                    os.write(fd, line)
                finally:
                    os.close(fd)


            if args[0] == "-c":
                gpu = os.environ["CUDA_VISIBLE_DEVICES"]
                record("preflight", gpu=gpu, logical_device="cuda:0")
                if gpu == os.environ.get("FAKE_FAIL_PREFLIGHT_GPU"):
                    print(
                        f"FAKE CUDA preflight failure with CUDA_VISIBLE_DEVICES={gpu}",
                        file=sys.stderr,
                    )
                    raise SystemExit(8)
                raise SystemExit(0)

            module = option("-m")
            if module == "scripts.reactflow_delta.run_model_rescue_v14":
                folds = option("--folds")
                seed = option("--seed")
                if "," in folds:
                    record("invalid_batch", seed=seed, folds=folds)
                    raise SystemExit(91)
                fold = folds
                task = f"{seed}:{fold}"
                gpu = os.environ["CUDA_VISIBLE_DEVICES"]
                record("start", module="run", seed=int(seed), fold=int(fold), gpu=gpu)
                if task == os.environ.get("FAKE_SLOW_TASK"):
                    delay = float(os.environ.get("FAKE_SLOW_SECONDS", "0.30"))
                else:
                    delay = float(os.environ.get("FAKE_DEFAULT_SECONDS", "0.01"))
                time.sleep(delay)
                if task == os.environ.get("FAKE_FAIL_TASK"):
                    record(
                        "end",
                        module="run",
                        seed=int(seed),
                        fold=int(fold),
                        gpu=gpu,
                        status="failed",
                    )
                    raise SystemExit(9)
                out_dir = Path(option("--out-dir"))
                out_dir.mkdir(parents=True, exist_ok=True)
                result = out_dir / f"v14_fold_result_fold{fold}_seed{seed}.json"
                result.write_text("{}\\n", encoding="utf-8")
                record(
                    "end",
                    module="run",
                    seed=int(seed),
                    fold=int(fold),
                    gpu=gpu,
                    status="ok",
                )
            elif module == "scripts.reactflow_delta.merge_model_rescue_v14":
                out_json = Path(option("--out-json"))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text("{}\\n", encoding="utf-8")
                record("merge", phase=option("--phase"))
            elif module == "scripts.reactflow_delta.assemble_model_rescue_v14_formal":
                out_json = Path(option("--out-json"))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text("{}\\n", encoding="utf-8")
                record("assemble")
            else:
                record("unexpected_module", module=module)
                raise SystemExit(92)
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def _write_test_controller(
    tmp_path: Path,
    script: Path,
    python_bin: Path,
    out_dir: Path,
) -> Path:
    text = script.read_text(encoding="utf-8")
    replacements = {
        PRODUCTION_REPO_ASSIGNMENT: f"repo={shlex.quote(str(tmp_path))}",
        PRODUCTION_PYTHON_ASSIGNMENT: f"python_bin={shlex.quote(str(python_bin))}",
        PRODUCTION_OUT_ASSIGNMENTS[script]: f"out={shlex.quote(str(out_dir))}",
    }
    for production, isolated in replacements.items():
        assert text.count(production) == 1
        text = text.replace(production, isolated)
    test_script = tmp_path / f"test_{script.name}"
    test_script.write_text(text, encoding="utf-8")
    test_script.chmod(0o755)
    return test_script


def _controller_env(
    tmp_path: Path,
    script: Path,
    out_dir: Path,
    *,
    slow_task: str | None = None,
    fail_task: str | None = None,
    fail_preflight_gpu: str | None = None,
) -> tuple[Path, dict[str, str], Path]:
    events = tmp_path / "events.jsonl"
    fake_runner = _write_fake_runner(tmp_path)
    test_script = _write_test_controller(tmp_path, script, fake_runner, out_dir)
    env = os.environ.copy()
    env.update(
        {
            "FAKE_EVENTS": str(events),
            "FAKE_DEFAULT_SECONDS": "0.01",
        }
    )
    if slow_task is not None:
        env["FAKE_SLOW_TASK"] = slow_task
        env["FAKE_SLOW_SECONDS"] = "0.35"
    if fail_task is not None:
        env["FAKE_FAIL_TASK"] = fail_task
    if fail_preflight_gpu is not None:
        env["FAKE_FAIL_PREFLIGHT_GPU"] = fail_preflight_gpu
    return test_script, env, events


def _read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_events(events: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if event["event"] == kind and event.get("module") == "run"
    ]


@pytest.mark.parametrize("script", [SCREEN, FORMAL])
def test_v14_controller_shell_is_valid_and_uses_one_task_dynamic_queue(
    script: Path,
) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "wait -n -p finished_pid" in text
    assert "scripts.reactflow_delta.gpu_runtime import require_cuda_device" in text
    assert "require_cuda_device('cuda:0')" in text
    assert '--folds "${fold}"' in text
    assert 'tasks+=("${seed}:${fold}")' in text or 'tasks+=("0:${fold}")' in text
    assert "V14_CONTROLLER_" not in text
    assert "cuda_visible_devices_value=" in text
    assert "physical_gpu=" not in text
    assert "nvidia-smi" not in text


def test_screen_queue_skips_complete_fold_and_reuses_gpu_before_straggler_finishes(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "screen"
    out_dir.mkdir()
    (out_dir / "v14_fold_result_fold3_seed0.json").write_text(
        "{}\n", encoding="utf-8"
    )
    test_script, env, event_path = _controller_env(
        tmp_path, SCREEN, out_dir, slow_task="0:0"
    )

    completed = subprocess.run(
        ["bash", str(test_script), "4", "7"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = _read_events(event_path)
    starts = _run_events(events, "start")
    tasks = [(int(event["seed"]), int(event["fold"])) for event in starts]
    expected = [(0, fold) for fold in range(20) if fold != 3]
    assert Counter(tasks) == Counter(expected)
    assert len(tasks) == len(set(tasks))

    preflight_indices = [
        index
        for index, event in enumerate(events)
        if event["event"] == "preflight"
    ]
    first_task_start = min(
        index
        for index, event in enumerate(events)
        if event["event"] == "start" and event.get("module") == "run"
    )
    assert {events[index]["gpu"] for index in preflight_indices} == {"4", "7"}
    assert max(preflight_indices) < first_task_start

    start_fold2 = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "start" and event.get("fold") == 2
    )
    end_straggler = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "end" and event.get("fold") == 0
    )
    fold2_event = events[start_fold2]
    assert fold2_event["gpu"] == "7"
    assert start_fold2 < end_straggler

    merge_index = next(
        index for index, event in enumerate(events) if event["event"] == "merge"
    )
    last_task_end = max(
        index
        for index, event in enumerate(events)
        if event["event"] == "end" and event.get("module") == "run"
    )
    assert merge_index > last_task_end
    assert (out_dir / "v14m3_complete_unscored_merge.json").is_file()


def test_formal_queue_runs_each_missing_seed_fold_once_then_merges_and_assembles(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "formal"
    out_dir.mkdir()
    missing = {(0, 0), (0, 1), (1, 0), (4, 19)}
    for seed in range(5):
        for fold in range(20):
            if (seed, fold) not in missing:
                (out_dir / f"v14_fold_result_fold{fold}_seed{seed}.json").write_text(
                    "{}\n", encoding="utf-8"
                )
    test_script, env, event_path = _controller_env(
        tmp_path, FORMAL, out_dir, slow_task="0:0"
    )

    completed = subprocess.run(
        ["bash", str(test_script), "2", "6"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = _read_events(event_path)
    starts = _run_events(events, "start")
    tasks = [(int(event["seed"]), int(event["fold"])) for event in starts]
    assert Counter(tasks) == Counter(missing)
    assert len(tasks) == len(set(tasks))

    preflight_indices = [
        index
        for index, event in enumerate(events)
        if event["event"] == "preflight"
    ]
    first_task_start = min(
        index
        for index, event in enumerate(events)
        if event["event"] == "start" and event.get("module") == "run"
    )
    assert {events[index]["gpu"] for index in preflight_indices} == {"2", "6"}
    assert max(preflight_indices) < first_task_start

    task_end_indices = [
        index
        for index, event in enumerate(events)
        if event["event"] == "end" and event.get("module") == "run"
    ]
    merge_index = next(
        index for index, event in enumerate(events) if event["event"] == "merge"
    )
    assemble_index = next(
        index for index, event in enumerate(events) if event["event"] == "assemble"
    )
    assert max(task_end_indices) < merge_index < assemble_index
    assert (out_dir / "v14m4_complete_unscored_merge.json").is_file()
    assert (out_dir / "v14m4_five_seed_prediction_only_assembly.json").is_file()


@pytest.mark.parametrize("script", [SCREEN, FORMAL])
def test_v14_controller_failure_stops_dispatch_waits_for_running_task_and_skips_merge(
    tmp_path: Path,
    script: Path,
) -> None:
    out_dir = tmp_path / script.stem
    test_script, env, event_path = _controller_env(
        tmp_path,
        script,
        out_dir,
        slow_task="0:1",
        fail_task="0:0",
    )

    completed = subprocess.run(
        ["bash", str(test_script), "1", "5"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    events = _read_events(event_path)
    starts = _run_events(events, "start")
    started_tasks = {(int(event["seed"]), int(event["fold"])) for event in starts}
    assert started_tasks == {(0, 0), (0, 1)}
    assert any(
        event["event"] == "end"
        and event.get("fold") == 1
        and event.get("status") == "ok"
        for event in events
    )
    assert not any(event["event"] in {"merge", "assemble"} for event in events)
    assert not list(out_dir.glob("*complete_unscored_merge.json"))
    assert "failed" in completed.stderr


@pytest.mark.parametrize("script", [SCREEN, FORMAL])
def test_v14_controller_preflight_failure_makes_no_experiment_directory_or_runner(
    tmp_path: Path,
    script: Path,
) -> None:
    out_dir = tmp_path / "must_not_exist"
    test_script, env, event_path = _controller_env(
        tmp_path,
        script,
        out_dir,
        fail_preflight_gpu="5",
    )

    completed = subprocess.run(
        ["bash", str(test_script), "1", "5", "7"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not out_dir.exists()
    events = _read_events(event_path)
    assert [event["gpu"] for event in events] == ["1", "5", "7"]
    assert all(event["event"] == "preflight" for event in events)
    assert "cuda_visible_devices_value=5" in completed.stderr
    assert "logical_device=cuda:0" in completed.stderr


@pytest.mark.parametrize(
    ("script", "seed_count", "expected_terminal_events"),
    [
        (SCREEN, 1, ["merge"]),
        (FORMAL, 5, ["merge", "assemble"]),
    ],
)
def test_v14_controller_with_no_missing_tasks_needs_no_gpu_or_preflight(
    tmp_path: Path,
    script: Path,
    seed_count: int,
    expected_terminal_events: list[str],
) -> None:
    out_dir = tmp_path / "complete"
    out_dir.mkdir()
    for seed in range(seed_count):
        for fold in range(20):
            (out_dir / f"v14_fold_result_fold{fold}_seed{seed}.json").write_text(
                "{}\n", encoding="utf-8"
            )
    test_script, env, event_path = _controller_env(
        tmp_path,
        script,
        out_dir,
        fail_preflight_gpu="unused",
    )

    completed = subprocess.run(
        ["bash", str(test_script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = _read_events(event_path)
    assert [event["event"] for event in events] == expected_terminal_events


def _populate_v14_results(out_dir: Path, seed_count: int) -> None:
    out_dir.mkdir()
    for seed in range(seed_count):
        for fold in range(20):
            (out_dir / f"v14_fold_result_fold{fold}_seed{seed}.json").write_text(
                "{}\n", encoding="utf-8"
            )


def test_screen_controller_preserves_existing_canonical_merge(tmp_path: Path) -> None:
    out_dir = tmp_path / "screen_complete"
    _populate_v14_results(out_dir, 1)
    merged = out_dir / "v14m3_complete_unscored_merge.json"
    merged.write_bytes(b"canonical-v14m3-merge\n")
    test_script, env, event_path = _controller_env(tmp_path, SCREEN, out_dir)

    completed = subprocess.run(
        ["bash", str(test_script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert merged.read_bytes() == b"canonical-v14m3-merge\n"
    assert not event_path.exists()


def test_formal_controller_resumes_assembly_without_remerging(tmp_path: Path) -> None:
    out_dir = tmp_path / "formal_complete"
    _populate_v14_results(out_dir, 5)
    merged = out_dir / "v14m4_complete_unscored_merge.json"
    merged.write_bytes(b"canonical-v14m4-merge\n")
    test_script, env, event_path = _controller_env(tmp_path, FORMAL, out_dir)

    completed = subprocess.run(
        ["bash", str(test_script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert merged.read_bytes() == b"canonical-v14m4-merge\n"
    assert [event["event"] for event in _read_events(event_path)] == ["assemble"]
