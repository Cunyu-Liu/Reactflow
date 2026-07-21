#!/usr/bin/env bash
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "$script_dir/.." && pwd)"
cd "${REACTFLOW_ROOT:-$default_root}"
case ":${PYTHONPATH:-}:" in
  *":src:"*) ;;
  *) export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

latest="${LATEST_RUN:-artifacts/full_runs/full_ablation_20260709_003012}"
split="${SPLIT_DIR:-$latest/splits/rfam_current_exact_seed0}"
frozen="${FROZEN_DIR:-$latest/frozen/ribonanzanet2_sharded_full}"
logdir="$latest/logs"
log="$logdir/warm_tail_recovery_after_watcher_exit.log"
pidfile="$logdir/warm_tail_recovery_after_watcher_exit.pid"
warm_pidfile="$logdir/warm_after_export_rfam_current_exact.pid"
warm_results="$latest/warm_rfam_current_exact_results.json"
python_bin="${TORCH_PYTHON:-${PYTHON_BIN:-python3}}"
torch_device="${TORCH_DEVICE:-cuda}"
instability_pattern="out of memory|cuda out|oom|killed|MemoryError|FloatingPointError|non-finite|nan|inf|diverg|converg"

mkdir -p "$logdir"
echo $$ > "$pidfile"

cleanup_pidfile() {
  rm -f "$pidfile"
}
trap cleanup_pidfile EXIT

echo "[$(date -Iseconds)] warm tail recovery watcher started" | tee -a "$log"

warm_watcher_alive() {
  if [ -f "$warm_pidfile" ]; then
    pid="$(cat "$warm_pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -u "$(id -un)" -f "run_warm_after_export_rfam_current_exact.sh" >/dev/null 2>&1
}

label_done() {
  "$python_bin" - "$latest" "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
for run_dir in sorted((root / "runs").glob(f"{label}_rfam_current_exact_torch_full_data_e1_bs*")):
    stderr = run_dir / "stderr.log"
    if stderr.exists() and stderr.stat().st_size > 0:
        continue
    for name in ("eval_summary.json", "eval_summary.recovered.json", "stdout.json", "stdout.recovered.json"):
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        start = text.find("{")
        if start < 0:
            continue
        try:
            obj = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("tiers" in obj or "metrics" in obj or "summary" in obj):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

run_one() {
  label="$1"
  dim="$2"
  if label_done "$label"; then
    echo "[$(date -Iseconds)] $label already has parseable metrics; skipping" | tee -a "$log"
    return 0
  fi
  for bs in 16 8 4 2 1; do
    out="$latest/runs/${label}_rfam_current_exact_torch_full_data_e1_bs${bs}"
    mkdir -p "$out"
    echo "[$(date -Iseconds)] recovery running $label adapter_dim=$dim batch=$bs" | tee -a "$log"
    "$python_bin" -m reactflow.cli evaluate-efold \
      --train-json "$split/train.jsonl" \
      --train-limit 1000000000 --eval-limit 1000000000 \
      --epochs 1 --learning-rate 0.2 --hidden-size 8 --lambda-react 0 \
      --bucket-boundaries 64,128,256,384 \
      --profile-path "$out/profile.jsonl" --output-dir "$out" \
      --backend torch --torch-device "$torch_device" --batch-size "$bs" \
      --adapter-dim "$dim" --adapter-lr 0.05 --frozen-dir "$frozen" --frozen-cache-shards 4 \
      --eval-json "in_clan=$split/test.jsonl" \
      --eval-json "novel_clan=$split/novel.jsonl" \
      --eval-json "archiveII=$latest/cache/archiveII.jsonl" \
      --eval-json "PDB=$latest/cache/PDB.jsonl" \
      --eval-json "viral=$latest/cache/viral.jsonl" \
      --eval-json "lncRNA=$latest/cache/lncRNA.jsonl" \
      --eval-json "human_mRNA=$latest/cache/human_mRNA.jsonl" \
      > "$out/stdout.json" 2> "$out/stderr.log"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -Iseconds)] recovery completed $label batch=$bs" | tee -a "$log"
      return 0
    fi
    if grep -Eiq "$instability_pattern" "$out/stderr.log"; then
      echo "[$(date -Iseconds)] recovery OOM/instability for $label batch=$bs, retrying smaller" | tee -a "$log"
      continue
    fi
    echo "[$(date -Iseconds)] recovery failed $label batch=$bs rc=$rc" | tee -a "$log"
    tail -80 "$out/stderr.log" | tee -a "$log"
    return "$rc"
  done
  echo "[$(date -Iseconds)] recovery exhausted batch retries for $label" | tee -a "$log"
  return 1
}

while true; do
  if [ -s "$warm_results" ]; then
    echo "[$(date -Iseconds)] warm results already present; recovery watcher exiting" | tee -a "$log"
    exit 0
  fi
  if warm_watcher_alive; then
    sleep 300
    continue
  fi
  break
done

echo "[$(date -Iseconds)] main warm watcher is absent and warm results are missing; starting recovery" | tee -a "$log"

run_one RF-A1-warm 8 || exit $?
run_one RF-A2-adapter4 4 || exit $?
run_one RF-A2-adapter16 16 || exit $?

"$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  --glob '*_rfam_current_exact_torch_full_data_e1_bs*' \
  --output-json "$latest/warm_rfam_current_exact_results.json" \
  --output-md "$latest/warm_rfam_current_exact_results.md" \
  --output-svg "$latest/warm_rfam_current_exact_mean_f1.svg" \
  --title 'ReactFlow warm-start ablations on rfam_current_exact split' \
  | tee -a "$log"

echo "[$(date -Iseconds)] warm tail recovery watcher finished" | tee -a "$log"
