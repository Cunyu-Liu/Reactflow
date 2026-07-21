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
wait_seconds="${WAIT_SECONDS:-300}"
logdir="$latest/logs"
log="$logdir/goal_readiness_after_final_results.log"

mkdir -p "$logdir"

required_results=(
  "$latest/warm_rfam_current_exact_results.json"
  "$latest/contact_rfam_current_exact_results.json"
  "$latest/mmseqs_final_results.json"
)

echo "[$(date -Iseconds)] waiting for final result files before goal readiness" | tee -a "$log"

while true; do
  missing=()
  for result in "${required_results[@]}"; do
    if [ ! -s "$result" ]; then
      missing+=("$result")
    fi
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    break
  fi
  echo "[$(date -Iseconds)] still waiting for ${#missing[@]} result files: ${missing[*]}" | tee -a "$log"
  sleep "$wait_seconds"
done

echo "[$(date -Iseconds)] final result files detected; refreshing full status" | tee -a "$log"
bash scripts/refresh_full_run_status.sh | tee -a "$log"

echo "[$(date -Iseconds)] running strict goal-readiness check" | tee -a "$log"
PYTHONPATH=src "$python_bin" scripts/audit_goal_readiness.py \
  --project-root . \
  --full-run-root "$latest" \
  --output-json "$latest/goal_readiness_audit.json" \
  --output-md "$latest/goal_readiness_audit.md" \
  --fail-if-not-ready \
  | tee -a "$log"

echo "[$(date -Iseconds)] final goal readiness complete" | tee -a "$log"
