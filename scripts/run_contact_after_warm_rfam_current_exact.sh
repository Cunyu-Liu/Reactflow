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
logdir="$latest/logs"
log="$logdir/contact_after_warm_rfam_current_exact.log"
warm_results="$latest/warm_rfam_current_exact_results.json"
python_bin="${TORCH_PYTHON:-${PYTHON_BIN:-python3}}"
torch_device="${TORCH_DEVICE:-cuda}"
lambda_contact="${CONTACT_LAMBDA:-0.2}"
negative_weight="${CONTACT_NEGATIVE_WEIGHT:-0.25}"
instability_pattern="out of memory|cuda out|oom|killed|MemoryError|FloatingPointError|non-finite|nan|inf|diverg|converg"

mkdir -p "$logdir"
echo "[$(date -Iseconds)] waiting for warm-start results before RF-A3-contact" | tee -a "$log"

while true; do
  if [ -f "$warm_results" ]; then
    break
  fi
  if ! pgrep -u "$(id -un)" -f "run_warm_after_export_rfam_current_exact.sh" >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] warm watcher is not running and $warm_results is absent" | tee -a "$log"
    exit 2
  fi
  sleep 300
done

echo "[$(date -Iseconds)] warm results found; starting RF-A3-contact" | tee -a "$log"

run_contact() {
  for bs in 16 8 4 2 1; do
    out="$latest/runs/RF-A3-contact_rfam_current_exact_torch_full_data_e1_bs${bs}"
    mkdir -p "$out"
    echo "[$(date -Iseconds)] running RF-A3-contact lambda=$lambda_contact neg_weight=$negative_weight batch=$bs" | tee -a "$log"
    "$python_bin" -m reactflow.cli evaluate-efold \
      --train-json "$split/train.jsonl" \
      --train-limit 1000000000 --eval-limit 1000000000 \
      --epochs 1 --learning-rate 0.2 --hidden-size 8 --lambda-react 0 \
      --lambda-contact "$lambda_contact" --contact-negative-weight "$negative_weight" \
      --bucket-boundaries 64,128,256,384 \
      --profile-path "$out/profile.jsonl" --output-dir "$out" \
      --backend torch --torch-device "$torch_device" --batch-size "$bs" \
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
      echo "[$(date -Iseconds)] completed RF-A3-contact batch=$bs" | tee -a "$log"
      return 0
    fi
    if grep -Eiq "$instability_pattern" "$out/stderr.log"; then
      echo "[$(date -Iseconds)] OOM/instability for RF-A3-contact batch=$bs, retrying smaller" | tee -a "$log"
      continue
    fi
    echo "[$(date -Iseconds)] failed RF-A3-contact batch=$bs rc=$rc" | tee -a "$log"
    tail -80 "$out/stderr.log" | tee -a "$log"
    return "$rc"
  done
  echo "[$(date -Iseconds)] exhausted batch retries for RF-A3-contact" | tee -a "$log"
  return 1
}

run_contact || exit $?

"$python_bin" scripts/summarize_ablation_results.py \
  --run-root "$latest/runs" \
  --glob 'RF-A3-contact_rfam_current_exact_torch_full_data_e1_bs*' \
  --output-json "$latest/contact_rfam_current_exact_results.json" \
  --output-md "$latest/contact_rfam_current_exact_results.md" \
  --output-svg "$latest/contact_rfam_current_exact_mean_f1.svg" \
  --title 'ReactFlow contact auxiliary ablation on rfam_current_exact split' \
  | tee -a "$log"

echo "[$(date -Iseconds)] RF-A3-contact watcher finished" | tee -a "$log"
