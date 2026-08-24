#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/cunyuliu/reactflow_delta_model_rescue_v7
REPO=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v7_20260824
PYTHON="$BASE/runtime_v7_clean/bin/python"
OFFICIAL="$BASE/official/RiNALMo"
WEIGHTS="$BASE/weights/rinalmo_giga_pretrained.pt"
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
SETUP="$BASE/setup"
SMOKE="$BASE/v7m1_real_smoke"
FULL="$BASE/v7m1_rinalmo_dependency_cache"
EXPECTED_WEIGHT_BYTES=2603787622
MIN_FREE_MIB=8192
POLL_SECONDS=900

mkdir -p "$SETUP"
exec >> "$SETUP/v7m1_cache_controller.log" 2>&1

select_gpu() {
  local selection
  selection="$({
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
  } | awk -F, -v minimum="$MIN_FREE_MIB" '
    {
      gsub(/ /, "", $1)
      gsub(/ /, "", $2)
      if ($1 >= 0 && $1 <= 7 && $2 >= minimum) print $1, $2
    }
  ' | sort -k2,2nr | head -n 1)"
  if [[ -n "$selection" ]]; then
    printf '%s\n' "${selection%% *}"
  fi
}

wait_for_setup() {
  while [[ ! -f "$SETUP/runtime_setup_complete" || ! -f "$SETUP/weight_download_complete" ]]; do
    printf '%s runtime=%s weight=%s\n' \
      "$(date --iso-8601=seconds)" \
      "$(test -f "$SETUP/runtime_setup_complete" && echo ready || echo waiting)" \
      "$(test -f "$SETUP/weight_download_complete" && echo ready || echo waiting)"
    sleep "$POLL_SECONDS"
  done
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu="$(select_gpu)"
    if [[ -n "$gpu" ]]; then
      printf '%s selected_physical_gpu=%s\n' "$(date --iso-8601=seconds)" "$gpu" >&2
      printf '%s\n' "$gpu"
      return 0
    fi
    printf '%s no_gpu_with_%s_mib_free\n' \
      "$(date --iso-8601=seconds)" "$MIN_FREE_MIB" >&2
    sleep "$POLL_SECONDS"
  done
}

wait_for_setup

if [[ "$(stat -c %s "$WEIGHTS")" -ne "$EXPECTED_WEIGHT_BYTES" ]]; then
  echo "official weight length does not match the frozen Zenodo artifact"
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "frozen v7 runtime is incomplete"
  exit 1
fi
if [[ "$(git -C "$OFFICIAL" rev-parse HEAD)" != "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955" ]]; then
  echo "official RiNALMo code commit drifted"
  exit 1
fi

"$PYTHON" -c 'import flash_attn, rinalmo, torch; print({"torch": torch.__version__, "cuda": torch.version.cuda})'

if [[ -e "$SMOKE/dependency_cache.h5" || -e "$FULL/dependency_cache.h5" ]]; then
  echo "v7m1 controller refuses to overwrite an existing cache artifact"
  exit 1
fi

mkdir -p "$SMOKE"
gpu="$(wait_for_gpu)"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
  scripts/reactflow_delta/build_model_rescue_v7_dependency_cache.py \
  --repo-root "$REPO" \
  --m2-csv "$M2" \
  --model-code-root "$OFFICIAL" \
  --weights "$WEIGHTS" \
  --device cuda:0 \
  --attention-backend flash \
  --batch-size 4 \
  --max-constructs 2 \
  --out-h5 "$SMOKE/dependency_cache.h5" \
  --out-manifest "$SMOKE/manifest.json"

smoke_mutants="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_registered_mutants"])' "$SMOKE/manifest.json")"
"$PYTHON" scripts/reactflow_delta/qualify_model_rescue_v7_dependency_cache.py \
  --repo-root "$REPO" \
  --cache "$SMOKE/dependency_cache.h5" \
  --manifest "$SMOKE/manifest.json" \
  --m2-csv "$M2" \
  --out-json "$SMOKE/qualification.json" \
  --expected-constructs 2 \
  --expected-mutants "$smoke_mutants" \
  --expected-length 177

smoke_status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SMOKE/qualification.json")"
if [[ "$smoke_status" != "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS" ]]; then
  echo "v7m1 real-data smoke did not qualify"
  exit 1
fi
touch "$SMOKE/engineering_smoke_complete"

mkdir -p "$FULL"
gpu="$(wait_for_gpu)"
CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
  scripts/reactflow_delta/build_model_rescue_v7_dependency_cache.py \
  --repo-root "$REPO" \
  --m2-csv "$M2" \
  --model-code-root "$OFFICIAL" \
  --weights "$WEIGHTS" \
  --device cuda:0 \
  --attention-backend flash \
  --batch-size 4 \
  --out-h5 "$FULL/dependency_cache.h5" \
  --out-manifest "$FULL/manifest.json"

"$PYTHON" scripts/reactflow_delta/qualify_model_rescue_v7_dependency_cache.py \
  --repo-root "$REPO" \
  --cache "$FULL/dependency_cache.h5" \
  --manifest "$FULL/manifest.json" \
  --m2-csv "$M2" \
  --out-json "$FULL/qualification.json" \
  --expected-constructs 160 \
  --expected-mutants 13976 \
  --expected-length 177

full_status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$FULL/qualification.json")"
if [[ "$full_status" != "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS" ]]; then
  echo "v7m1 full outcome-blind cache did not qualify"
  exit 1
fi
touch "$FULL/v7m1_cache_complete"
printf '%s v7m1_cache_complete\n' "$(date --iso-8601=seconds)"
