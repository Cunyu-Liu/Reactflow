#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "$script_dir/.." && pwd)"
cd "${REACTFLOW_ROOT:-$default_root}"
case ":${PYTHONPATH:-}:" in
  *":src:"*) ;;
  *) export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

latest="${LATEST_RUN:-artifacts/full_runs/full_ablation_20260709_003012}"
python_bin="${PYTHON_BIN:-python3}"
exact_total_samples="${TOTAL_SAMPLES:-206443}"
mmseqs_total_samples="${MMSEQS_TOTAL_SAMPLES:-228282}"
current_run="${CURRENT_RUN:-RF-A1-warm_rfam_current_exact_torch_full_data_e1_bs16}"
run_dir="$latest/runs/$current_run"
queue_globs=()
IFS=',' read -r -a queue_globs <<< "${QUEUE_GLOBS:-*_rfam_current_exact_torch_full_data_e1_bs*,RF-M*_mmseqs_torch_full_data_e1_bs*,RF-CF*_mmseqs_torch_full_data_e1_bs*}"
glob_args=()
for pattern in "${queue_globs[@]}"; do
  glob_args+=(--glob "$pattern")
done

mkdir -p "$latest/logs"

monitor_total_samples() {
  local candidate="$1"
  case "$(basename "$candidate")" in
    *_mmseqs_*) echo "$mmseqs_total_samples" ;;
    *) echo "$exact_total_samples" ;;
  esac
}

monitor_run_dir() {
  local candidate="$1"
  if [ -f "$candidate/profile.jsonl" ] && [ ! -f "$candidate/profile.summary.json" ]; then
    local run_name total
    run_name="$(basename "$candidate")"
    total="$(monitor_total_samples "$candidate")"
    PYTHONPATH=src "$python_bin" scripts/monitor_reactflow_run.py \
      --run-dir "$candidate" \
      --total-samples "$total" \
      --output-json "$candidate/monitor_snapshot.json" \
      --output-md "$candidate/monitor_snapshot.md" \
      > "$latest/logs/refresh_full_run_status.monitor.${run_name}.json"
    printf '{"run_id":"%s","total_samples":%s,"monitor_json":"%s"}\n' \
      "$run_name" "$total" "$candidate/monitor_snapshot.json" \
      >> "$latest/logs/refresh_full_run_status.monitor_runs.jsonl"
  fi
}

monitor_run_dirs=()
for pattern in "${queue_globs[@]}"; do
  for candidate in "$latest"/runs/$pattern; do
    if [ ! -d "$candidate" ]; then
      continue
    fi
    already_seen=0
    for seen in "${monitor_run_dirs[@]}"; do
      if [ "$seen" = "$candidate" ]; then
        already_seen=1
        break
      fi
    done
    if [ "$already_seen" -eq 0 ]; then
      monitor_run_dirs+=("$candidate")
    fi
  done
done
if [ "${#monitor_run_dirs[@]}" -eq 0 ] && [ -d "$run_dir" ]; then
  monitor_run_dirs+=("$run_dir")
fi

: > "$latest/logs/refresh_full_run_status.monitor_runs.jsonl"
for candidate in "${monitor_run_dirs[@]}"; do
  monitor_run_dir "$candidate"
done

PYTHONPATH=src "$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  "${glob_args[@]}" \
  --output-json "$latest/current_queue_status.json" \
  --output-md "$latest/current_queue_status.md" \
  --output-svg "$latest/current_queue_status.svg" \
  --title "ReactFlow current queue status" \
  > "$latest/logs/refresh_full_run_status.summary.json"

PYTHONPATH=src "$python_bin" scripts/audit_queue_progress.py \
  --queue-json "$latest/current_queue_status.json" \
  --history-jsonl "$latest/logs/current_queue_status_history.jsonl" \
  --append-current \
  --output-json "$latest/queue_progress_audit.json" \
  --output-md "$latest/queue_progress_audit.md" \
  > "$latest/logs/refresh_full_run_status.queue_progress.json"

PYTHONPATH=src "$python_bin" scripts/audit_active_eval_progress.py \
  --full-run-root "$latest" \
  --queue-json "$latest/current_queue_status.json" \
  --output-json "$latest/active_eval_progress_audit.json" \
  --output-md "$latest/active_eval_progress_audit.md" \
  > "$latest/logs/refresh_full_run_status.active_eval_progress.json"

PYTHONPATH=src "$python_bin" scripts/audit_cross_family_metrics.py \
  --results-json "$latest/current_queue_status.json" \
  --min-novel-mean-f1 "${CROSS_FAMILY_MIN_NOVEL_MEAN_F1:-0.15}" \
  --max-generalization-gap "${CROSS_FAMILY_MAX_GENERALIZATION_GAP:-0.10}" \
  --output-json "$latest/cross_family_metric_audit.json" \
  --output-md "$latest/cross_family_metric_audit.md" \
  > "$latest/logs/refresh_full_run_status.cross_family.json"

PYTHONPATH=src "$python_bin" scripts/audit_paper_artifacts.py \
  --full-run-root "$latest" \
  --run-glob "$current_run" \
  --output-json "$latest/paper_artifact_audit.json" \
  --output-md "$latest/paper_artifact_audit.md" \
  > "$latest/logs/refresh_full_run_status.audit.json"

PYTHONPATH=src "$python_bin" scripts/audit_queue_preflight.py \
  --project-root . \
  --full-run-root "$latest" \
  --output-json "$latest/queue_preflight_audit.json" \
  --output-md "$latest/queue_preflight_audit.md" \
  > "$latest/logs/refresh_full_run_status.queue_preflight.json"

PYTHONPATH=src "$python_bin" scripts/audit_algorithm_docs.py src/reactflow \
  --output-json "$latest/algorithm_doc_audit.json" \
  --output-md "$latest/algorithm_doc_audit.md" \
  --fail-on-placeholder \
  > "$latest/logs/refresh_full_run_status.algorithm_docs.json"

pid_args=()
add_pidfile_if_present() {
  local pidfile="$1"
  if [ -f "$pidfile" ]; then
    pid_args+=(--pidfile "$pidfile")
  fi
}

# Only unfinished queue stages are liveness-checked here.  Once a stage has
# emitted a non-empty final result, the JSON contract in audit_final_queue.py is
# stronger evidence than an old one-shot watcher PID that may have exited cleanly.
if [ ! -s "$latest/warm_rfam_current_exact_results.json" ]; then
  add_pidfile_if_present "$latest/logs/warm_after_export_rfam_current_exact.pid"
  add_pidfile_if_present "$latest/logs/warm_tail_recovery_after_watcher_exit.pid"
fi
if [ ! -s "$latest/contact_rfam_current_exact_results.json" ]; then
  add_pidfile_if_present "$latest/logs/contact_after_warm_rfam_current_exact.pid"
fi
if [ ! -s "$latest/mmseqs_final_results.json" ]; then
  add_pidfile_if_present "$latest/logs/mmseqs_final_after_exact_queue.pid"
fi
if [ ! -s "$latest/cross_family_balanced_results.json" ]; then
  add_pidfile_if_present "$latest/logs/cross_family_after_mmseqs_final.pid"
fi
if [ ! -s "$latest/cross_family_contact_sweep_results.json" ]; then
  add_pidfile_if_present "$latest/logs/contact_sweep_after_cross_family_balanced.pid"
fi
if [ ! -s "$latest/cross_family_long_range_results.json" ]; then
  add_pidfile_if_present "$latest/logs/long_range_after_contact_sweep.pid"
fi
if [ ! -s "$latest/cross_family_capacity_results.json" ]; then
  add_pidfile_if_present "$latest/logs/capacity_after_long_range.pid"
fi
if [ ! -s "$latest/warm_rfam_current_exact_results.json" ] \
  || [ ! -s "$latest/contact_rfam_current_exact_results.json" ] \
  || [ ! -s "$latest/mmseqs_final_results.json" ]; then
  add_pidfile_if_present "$latest/logs/goal_readiness_after_final_results.pid"
fi
for manual_pidfile in "$latest"/logs/manual_parallel_*.pid; do
  [ -f "$manual_pidfile" ] || continue
  manual_name="$(basename "$manual_pidfile")"
  manual_run="${manual_name#manual_parallel_}"
  manual_run="${manual_run%.pid}"
  manual_dir="$latest/runs/$manual_run"
  if [ -s "$manual_dir/stdout.json" ] && [ -s "$manual_dir/training_checkpoint.json" ]; then
    continue
  fi
  add_pidfile_if_present "$manual_pidfile"
done

runtime_run_args=()
for candidate in "${monitor_run_dirs[@]}"; do
  runtime_run_args+=(--run-dir "$candidate")
done
if [ "${#runtime_run_args[@]}" -eq 0 ]; then
  runtime_run_args+=(--run-dir "$run_dir")
fi

PYTHONPATH=src "$python_bin" scripts/audit_runtime_health.py \
  "${runtime_run_args[@]}" \
  "${pid_args[@]}" \
  --output-json "$latest/runtime_health_audit.json" \
  --output-md "$latest/runtime_health_audit.md" \
  > "$latest/logs/refresh_full_run_status.runtime.json"

PYTHONPATH=src "$python_bin" scripts/audit_system_resources.py \
  "${pid_args[@]}" \
  --output-json "$latest/system_resource_audit.json" \
  --output-md "$latest/system_resource_audit.md" \
  > "$latest/logs/refresh_full_run_status.resources.json"

PYTHONPATH=src "$python_bin" scripts/audit_profile_bottlenecks.py \
  "${runtime_run_args[@]}" \
  --output-json "$latest/profile_bottleneck_audit.json" \
  --output-md "$latest/profile_bottleneck_audit.md" \
  > "$latest/logs/refresh_full_run_status.profile_bottlenecks.json"

PYTHONPATH=src "$python_bin" scripts/audit_final_queue.py \
  --full-run-root "$latest" \
  --output-json "$latest/final_queue_audit.json" \
  --output-md "$latest/final_queue_audit.md" \
  > "$latest/logs/refresh_full_run_status.final_queue.json"

PYTHONPATH=src "$python_bin" scripts/update_ablation_ledger_status.py \
  --ledger docs/ablation_experiment_filled.md \
  --full-run-root "$latest" \
  > "$latest/logs/refresh_full_run_status.ledger.json"

PYTHONPATH=src "$python_bin" scripts/build_reproducibility_manifest.py \
  --project-root . \
  --full-run-root "$latest" \
  --output-json "$latest/reproducibility_manifest.json" \
  --output-md "$latest/reproducibility_manifest.md" \
  > "$latest/logs/refresh_full_run_status.reproducibility.json"

PYTHONPATH=src "$python_bin" scripts/audit_goal_readiness.py \
  --project-root . \
  --full-run-root "$latest" \
  --output-json "$latest/goal_readiness_audit.json" \
  --output-md "$latest/goal_readiness_audit.md" \
  > "$latest/logs/refresh_full_run_status.goal_readiness.json"

echo "[$(date -Iseconds)] refreshed status for $current_run"
