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
log="$logdir/contact_sweep_after_cross_family_balanced.log"
pidfile="$logdir/contact_sweep_after_cross_family_balanced.pid"
cf_results="$latest/cross_family_balanced_results.json"
cf_pidfile="$logdir/cross_family_after_mmseqs_final.pid"
contact_results="$latest/cross_family_contact_sweep_results.json"
contact_audit="$latest/cross_family_contact_sweep_metric_audit.json"
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
contact_lambdas="${CONTACT_SWEEP_LAMBDAS:-0.1 0.2 0.4 0.8}"
negative_weight="${CONTACT_NEGATIVE_WEIGHT:-0.25}"
min_novel_mean_f1="${CROSS_FAMILY_MIN_NOVEL_MEAN_F1:-0.15}"
max_generalization_gap="${CROSS_FAMILY_MAX_GENERALIZATION_GAP:-0.10}"
instability_pattern="out of memory|cuda out|oom|killed|MemoryError|FloatingPointError|non-finite|nan|inf|diverg|converg"

mkdir -p "$logdir"
echo $$ > "$pidfile"

cleanup_pidfile() {
  rm -f "$pidfile"
}
trap cleanup_pidfile EXIT

echo "[$(date -Iseconds)] waiting for RF-CF3 balanced results before RF-CF1-contact-strong sweep" | tee -a "$log"

balanced_watcher_alive() {
  if [ -f "$cf_pidfile" ]; then
    pid="$(cat "$cf_pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -u "$(id -un)" -f "RF-CF3-family-balanced|run_cross_family_after_mmseqs_final.sh" >/dev/null 2>&1
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
    > "$logdir/contact_sweep_after_cross_family_balanced.audit.json"
  "$python_bin" - "$output_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload["summary"].get("cross_family_claim_ready") else 1)
PY
}

while [ ! -s "$cf_results" ]; do
  if ! balanced_watcher_alive; then
    echo "[$(date -Iseconds)] RF-CF3 watcher is not running and $cf_results is absent" | tee -a "$log"
    exit 2
  fi
  sleep 300
done

echo "[$(date -Iseconds)] RF-CF3 balanced results found; auditing claim gate" | tee -a "$log"
if claim_ready "$cf_results" "$latest/cross_family_balanced_metric_audit.json" "$latest/cross_family_balanced_metric_audit.md"; then
  echo "[$(date -Iseconds)] RF-CF3 already satisfies cross-family claim gate; RF-CF1 skipped" | tee -a "$log"
  exit 0
fi

echo "[$(date -Iseconds)] RF-CF3 still below claim gate; starting RF-CF1-contact-strong sweep lambdas=$contact_lambdas" | tee -a "$log"

lambda_label() {
  echo "$1" | tr '.' 'p' | tr -cd '0-9p'
}

run_one_lambda() {
  local lam="$1"
  local label
  label="$(lambda_label "$lam")"
  for bs in 16 8 4 2 1; do
    out="$latest/runs/RF-CF1-contact-lam${label}_mmseqs_torch_full_data_e1_bs${bs}"
    mkdir -p "$out"
    echo "[$(date -Iseconds)] running RF-CF1-contact-strong lambda=$lam neg_weight=$negative_weight batch=$bs" | tee -a "$log"
    "$python_bin" -m reactflow.cli evaluate-efold \
      --train-json "$split/train.jsonl" \
      --train-limit 1000000000 --eval-limit 1000000000 \
      --epochs 1 --learning-rate 0.2 --hidden-size 8 --lambda-react 0 \
      --lambda-contact "$lam" --contact-negative-weight "$negative_weight" \
      --bucket-boundaries 64,128,256,384 \
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
      echo "[$(date -Iseconds)] completed RF-CF1-contact-strong lambda=$lam batch=$bs" | tee -a "$log"
      return 0
    fi
    if grep -Eiq "$instability_pattern" "$out/stderr.log"; then
      echo "[$(date -Iseconds)] OOM/instability for RF-CF1-contact-strong lambda=$lam batch=$bs, retrying smaller" | tee -a "$log"
      continue
    fi
    echo "[$(date -Iseconds)] failed RF-CF1-contact-strong lambda=$lam batch=$bs rc=$rc" | tee -a "$log"
    tail -80 "$out/stderr.log" | tee -a "$log"
    return "$rc"
  done
  echo "[$(date -Iseconds)] exhausted batch retries for RF-CF1-contact-strong lambda=$lam" | tee -a "$log"
  return 1
}

for lam in $contact_lambdas; do
  run_one_lambda "$lam" || exit $?
done

"$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  --glob 'RF-CF1-contact-lam*_mmseqs_torch_full_data_e1_bs*' \
  --output-json "$contact_results" \
  --output-md "$latest/cross_family_contact_sweep_results.md" \
  --output-svg "$latest/cross_family_contact_sweep_mean_f1.svg" \
  --title 'ReactFlow RF-CF1 contact-strong MMseqs sweep' \
  | tee -a "$log"

claim_ready "$contact_results" "$contact_audit" "$latest/cross_family_contact_sweep_metric_audit.md" || true

echo "[$(date -Iseconds)] RF-CF1-contact-strong sweep watcher finished" | tee -a "$log"
