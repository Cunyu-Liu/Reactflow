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
split="${MMSEQS_SPLIT_DIR:-$latest/splits/rfam_current_mmseqs_seed0}"
frozen="${FROZEN_DIR:-$latest/frozen/ribonanzanet2_sharded_full}"
logdir="$latest/logs"
log="$logdir/cross_family_after_mmseqs_final.log"
pidfile="$logdir/cross_family_after_mmseqs_final.pid"
mmseqs_pidfile="$logdir/mmseqs_final_after_exact_queue.pid"
mmseqs_results="$latest/mmseqs_final_results.json"
cf_results="$latest/cross_family_balanced_results.json"
cf_audit="$latest/cross_family_balanced_metric_audit.json"
if [ -n "${TORCH_PYTHON:-}" ]; then
  python_bin="$TORCH_PYTHON"
elif [ -n "${PYTHON_BIN:-}" ]; then
  python_bin="$PYTHON_BIN"
elif [ -x /home/cunyuliu/miniconda3/envs/editflow/bin/python ]; then
  python_bin="/home/cunyuliu/miniconda3/envs/editflow/bin/python"
else
  python_bin="python3"
fi
torch_device="${TORCH_DEVICE:-cuda}"
min_novel_mean_f1="${CROSS_FAMILY_MIN_NOVEL_MEAN_F1:-0.15}"
max_generalization_gap="${CROSS_FAMILY_MAX_GENERALIZATION_GAP:-0.10}"
instability_pattern="out of memory|cuda out|oom|killed|MemoryError|FloatingPointError|non-finite|nan|inf|diverg|converg"

mkdir -p "$logdir"
echo $$ > "$pidfile"

cleanup_pidfile() {
  rm -f "$pidfile"
}
trap cleanup_pidfile EXIT

echo "[$(date -Iseconds)] waiting for MMseqs final results before RF-CF3-family-balanced" | tee -a "$log"

mmseqs_watcher_alive() {
  if [ -f "$mmseqs_pidfile" ]; then
    pid="$(cat "$mmseqs_pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -u "$(id -un)" -f "run_mmseqs_final_after_exact_queue.sh" >/dev/null 2>&1
}

claim_ready() {
  local result_json="$1"
  local output_json="$2"
  local output_md="$3"
  "$python_bin" scripts/audit_cross_family_metrics.py \
    --results-json "$result_json" \
    --min-novel-mean-f1 "$min_novel_mean_f1" \
    --max-generalization-gap "$max_generalization_gap" \
    --output-json "$output_json" \
    --output-md "$output_md" \
    > "$logdir/cross_family_after_mmseqs_final.audit.json"
  "$python_bin" - "$output_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload["summary"].get("cross_family_claim_ready") else 1)
PY
}

while [ ! -s "$mmseqs_results" ]; do
  if ! mmseqs_watcher_alive; then
    echo "[$(date -Iseconds)] MMseqs watcher is not running and $mmseqs_results is absent" | tee -a "$log"
    exit 2
  fi
  sleep 300
done

echo "[$(date -Iseconds)] MMseqs final results found; auditing cross-family claim gate" | tee -a "$log"
if claim_ready "$mmseqs_results" "$latest/mmseqs_final_cross_family_metric_audit.json" "$latest/mmseqs_final_cross_family_metric_audit.md"; then
  echo "[$(date -Iseconds)] MMseqs final already satisfies cross-family claim gate; RF-CF3 skipped" | tee -a "$log"
  exit 0
fi

echo "[$(date -Iseconds)] cross-family claim gate unmet; starting RF-CF3-family-balanced" | tee -a "$log"

run_family_balanced() {
  for bs in 16 8 4 2 1; do
    out="$latest/runs/RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs${bs}"
    mkdir -p "$out"
    echo "[$(date -Iseconds)] running RF-CF3-family-balanced batch=$bs" | tee -a "$log"
    "$python_bin" -m reactflow.cli evaluate-efold \
      --train-json "$split/train.jsonl" \
      --train-limit 1000000000 --eval-limit 1000000000 \
      --epochs 1 --learning-rate 0.2 --hidden-size 8 --lambda-react 0 \
      --bucket-boundaries 64,128,256,384 \
      --family-balanced-batches \
      --profile-path "$out/profile.jsonl" --output-dir "$out" \
      --backend torch --torch-device "$torch_device" --batch-size "$bs" \
      --adapter-dim 8 --adapter-lr 0.05 --frozen-dir "$frozen" --frozen-cache-shards 4 \
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
      echo "[$(date -Iseconds)] completed RF-CF3-family-balanced batch=$bs" | tee -a "$log"
      return 0
    fi
    if grep -Eiq "$instability_pattern" "$out/stderr.log"; then
      echo "[$(date -Iseconds)] OOM/instability for RF-CF3-family-balanced batch=$bs, retrying smaller" | tee -a "$log"
      continue
    fi
    echo "[$(date -Iseconds)] failed RF-CF3-family-balanced batch=$bs rc=$rc" | tee -a "$log"
    tail -80 "$out/stderr.log" | tee -a "$log"
    return "$rc"
  done
  echo "[$(date -Iseconds)] exhausted batch retries for RF-CF3-family-balanced" | tee -a "$log"
  return 1
}

run_family_balanced || exit $?

"$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  --glob 'RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs*' \
  --output-json "$cf_results" \
  --output-md "$latest/cross_family_balanced_results.md" \
  --output-svg "$latest/cross_family_balanced_mean_f1.svg" \
  --title 'ReactFlow RF-CF3 family-balanced MMseqs ablation' \
  | tee -a "$log"

claim_ready "$cf_results" "$cf_audit" "$latest/cross_family_balanced_metric_audit.md" || true

echo "[$(date -Iseconds)] RF-CF3-family-balanced watcher finished" | tee -a "$log"
