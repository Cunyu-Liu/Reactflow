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
cache="${CACHE_JSONL:-$latest/cache/efold_train.jsonl}"
metadata="${METADATA_TSV:-$latest/metadata/rfam_current_mmseqs_metadata.tsv}"
manifest="${METADATA_MANIFEST:-$latest/metadata/rfam_current_mmseqs_metadata.manifest.json}"
out_dir="${SPLIT_OUT_DIR:-$latest/splits/rfam_current_mmseqs_seed0}"
logdir="$latest/logs"
log="$logdir/split_after_mmseqs_metadata.log"
metadata_pidfile="$logdir/build_rfam_metadata_mmseqs.pid"
python_bin="${PYTHON_BIN:-python3}"

mkdir -p "$logdir"
echo "[$(date -Iseconds)] waiting for MMseqs metadata manifest: $manifest" | tee -a "$log"

while true; do
  if [ -s "$manifest" ] && [ -s "$metadata" ]; then
    break
  fi
  if [ -f "$metadata_pidfile" ] && ! kill -0 "$(cat "$metadata_pidfile")" 2>/dev/null; then
    echo "[$(date -Iseconds)] metadata process ended before manifest was ready" | tee -a "$log"
    exit 2
  fi
  sleep 120
done

echo "[$(date -Iseconds)] metadata ready; building MMseqs split at $out_dir" | tee -a "$log"
mkdir -p "$out_dir"
"$python_bin" -m reactflow.cli split-efold-cache "$cache" \
  --output-dir "$out_dir" \
  --metadata-tsv "$metadata" \
  --bucket-boundaries 64,128,256,384 \
  --novel-clan-fraction 0.15 --seed 0 \
  > "$out_dir/stdout.json" 2> "$out_dir/stderr.log"

echo "[$(date -Iseconds)] MMseqs split finished" | tee -a "$log"
