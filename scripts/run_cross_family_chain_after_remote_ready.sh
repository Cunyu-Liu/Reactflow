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
require_full_preflight="${REQUIRE_FULL_PREFLIGHT:-0}"
logdir="$latest/logs"
log="$logdir/cross_family_chain_after_remote_ready.log"
pidfile="$logdir/cross_family_chain_after_remote_ready.pid"

mkdir -p "$logdir"
echo $$ > "$pidfile"

cleanup_pidfile() {
  rm -f "$pidfile"
}
trap cleanup_pidfile EXIT

frozen_complete() {
  "$python_bin" - "$latest/frozen/ribonanzanet2_sharded_full/sharded_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
if not manifest.exists():
    raise SystemExit(1)
root = manifest.parent
payload = json.loads(manifest.read_text(encoding="utf-8"))
missing = []
for shard in payload.get("shards", []):
    shard_dir = root / str(shard.get("path", ""))
    for name in ("features.npz", "index.jsonl", "provenance.json"):
        path = shard_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(str(path))
            break
if missing:
    print(f"missing_or_empty_shards={len(missing)} first={missing[0]}", file=sys.stderr)
    raise SystemExit(1)
print(f"complete_shards={len(payload.get('shards', []))}")
PY
}

preflight_healthy() {
  "$python_bin" scripts/audit_queue_preflight.py \
    --project-root "$(pwd)" \
    --full-run-root "$latest" \
    --output-json "$latest/queue_preflight_audit.json" \
    --output-md "$latest/queue_preflight_audit.md" \
    > "$logdir/cross_family_chain_after_remote_ready.preflight.json"
  "$python_bin" - "$latest/queue_preflight_audit.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("summary", {}).get("preflight_healthy") else 1)
PY
}

cross_family_inputs_ready() {
  "$python_bin" - "$latest" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "mmseqs_final_results.json",
    root / "splits/rfam_current_mmseqs_seed0/train.jsonl",
    root / "splits/rfam_current_mmseqs_seed0/test.jsonl",
    root / "splits/rfam_current_mmseqs_seed0/novel.jsonl",
    root / "splits/rfam_current_mmseqs_seed0/split_manifest.json",
    root / "cache/archiveII.jsonl",
    root / "cache/PDB.jsonl",
    root / "cache/viral.jsonl",
    root / "cache/lncRNA.jsonl",
    root / "cache/human_mRNA.jsonl",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size <= 0]
if missing:
    print(f"missing_cross_family_inputs={len(missing)} first={missing[0]}", file=sys.stderr)
    raise SystemExit(1)
print("cross_family_inputs_ready=true")
PY
}

start_once() {
  local name="$1"
  local script="$2"
  local pid="$logdir/${name}.pid"
  if [ -s "$pid" ] && kill -0 "$(cat "$pid")" 2>/dev/null; then
    echo "[$(date -Iseconds)] $name already running pid=$(cat "$pid")" | tee -a "$log"
    return 0
  fi
  nohup bash "$script" > "$logdir/${name}.launch.log" 2>&1 &
  echo $! > "$pid"
  echo "[$(date -Iseconds)] started $name pid=$(cat "$pid")" | tee -a "$log"
}

echo "[$(date -Iseconds)] waiting for RF-CF inputs and frozen shards before RF-CF chain" | tee -a "$log"
while true; do
  if frozen_complete >> "$log" 2>&1 && cross_family_inputs_ready >> "$log" 2>&1; then
    if [ "$require_full_preflight" = "1" ]; then
      preflight_healthy && break
    else
      preflight_healthy >> "$log" 2>&1 || echo "[$(date -Iseconds)] full queue preflight still not healthy; proceeding with RF-CF targeted readiness" | tee -a "$log"
      break
    fi
  fi
  echo "[$(date -Iseconds)] not ready yet; sleeping ${wait_seconds}s" | tee -a "$log"
  sleep "$wait_seconds"
done

start_once cross_family_after_mmseqs_final scripts/run_cross_family_after_mmseqs_final.sh
start_once contact_sweep_after_cross_family_balanced scripts/run_contact_sweep_after_cross_family_balanced.sh
start_once long_range_after_contact_sweep scripts/run_long_range_after_contact_sweep.sh
start_once capacity_after_long_range scripts/run_capacity_after_long_range.sh

echo "[$(date -Iseconds)] RF-CF chain launch complete" | tee -a "$log"
