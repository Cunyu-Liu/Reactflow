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
split="${MMSEQS_SPLIT_DIR:-$latest/splits/rfam_current_mmseqs_seed0}"
frozen="${FROZEN_DIR:-$latest/frozen/ribonanzanet2_sharded_full}"
logdir="$latest/logs"
log="$logdir/mmseqs_final_after_exact_queue.log"
python_bin="${TORCH_PYTHON:-${PYTHON_BIN:-python3}}"
torch_device="${TORCH_DEVICE:-cuda}"
wait_for_contact="${WAIT_FOR_CONTACT_RESULTS:-1}"
instability_pattern="out of memory|cuda out|oom|killed|MemoryError|FloatingPointError|non-finite|nan|inf|diverg|converg"

mkdir -p "$logdir"
echo "[$(date -Iseconds)] waiting before MMseqs final split runs" | tee -a "$log"

if [ "$wait_for_contact" != "0" ]; then
  contact_results="$latest/contact_rfam_current_exact_results.json"
  while true; do
    if [ -s "$contact_results" ]; then
      break
    fi
    if ! pgrep -u "$(id -un)" -f "run_contact_after_warm_rfam_current_exact.sh" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] contact watcher is not running and $contact_results is absent" | tee -a "$log"
      exit 2
    fi
    sleep 300
  done
else
  warm_results="$latest/warm_rfam_current_exact_results.json"
  while [ ! -s "$warm_results" ]; do
    sleep 300
  done
fi

if [ ! -s "$split/split_manifest.json" ]; then
  echo "[$(date -Iseconds)] missing MMseqs split manifest: $split/split_manifest.json" | tee -a "$log"
  exit 2
fi

run_eval() {
  label="$1"
  adapter_dim="$2"
  extra_args=()
  if [ "$adapter_dim" -gt 0 ]; then
    extra_args+=(--adapter-dim "$adapter_dim" --adapter-lr 0.05 --frozen-dir "$frozen" --frozen-cache-shards 4)
  fi

  for bs in 16 8 4 2 1; do
    out="$latest/runs/${label}_mmseqs_torch_full_data_e1_bs${bs}"
    mkdir -p "$out"
    echo "[$(date -Iseconds)] running $label on MMseqs split adapter_dim=$adapter_dim batch=$bs" | tee -a "$log"
    "$python_bin" -m reactflow.cli evaluate-efold \
      --train-json "$split/train.jsonl" \
      --train-limit 1000000000 --eval-limit 1000000000 \
      --epochs 1 --learning-rate 0.2 --hidden-size 8 --lambda-react 0 \
      --bucket-boundaries 64,128,256,384 \
      --profile-path "$out/profile.jsonl" --output-dir "$out" \
      --backend torch --torch-device "$torch_device" --batch-size "$bs" \
      "${extra_args[@]}" \
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
      echo "[$(date -Iseconds)] completed $label batch=$bs" | tee -a "$log"
      return 0
    fi
    if grep -Eiq "$instability_pattern" "$out/stderr.log"; then
      echo "[$(date -Iseconds)] OOM/instability for $label batch=$bs, retrying smaller" | tee -a "$log"
      continue
    fi
    echo "[$(date -Iseconds)] failed $label batch=$bs rc=$rc" | tee -a "$log"
    tail -80 "$out/stderr.log" | tee -a "$log"
    return "$rc"
  done
  echo "[$(date -Iseconds)] exhausted batch retries for $label" | tee -a "$log"
  return 1
}

run_eval RF-M0-base 0
run_eval RF-M1-warm 8

"$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  --glob 'RF-M*_mmseqs_torch_full_data_e1_bs*' \
  --output-json "$latest/mmseqs_final_results.json" \
  --output-md "$latest/mmseqs_final_results.md" \
  --output-svg "$latest/mmseqs_final_mean_f1.svg" \
  --title 'ReactFlow final MMseqs split ablations' \
  | tee -a "$log"

echo "[$(date -Iseconds)] MMseqs final split queue finished" | tee -a "$log"
