#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v13_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0
v14_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
source_manifest=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/source_binding/puzzle_set_source_manifest.json
out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/p1m2_real_smoke

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s CUDA_VISIBLE_DEVICES_VALUE\n' "$0" >&2
  exit 2
fi
gpu=$1
cd "${repo}"

require_gpu() {
  local cuda_visible_devices_value=$1
  local status
  if CUDA_VISIBLE_DEVICES="${cuda_visible_devices_value}" "${python_bin}" -c \
    "from scripts.reactflow_delta.gpu_runtime import require_cuda_device; require_cuda_device('cuda:0')"; then
    return 0
  else
    status=$?
  fi
  printf 'CUDA_REQUIRED: cuda_visible_devices_value=%s logical_device=cuda:0 preflight_status=%s\n' \
    "${cuda_visible_devices_value}" "${status}" >&2
  return "${status}"
}

archive_incomplete_fold() {
  local fold=$1
  local result="${out}/puzzle_set_fold_result_fold${fold}_seed0.json"
  if [[ -f "${result}" ]]; then
    return 0
  fi
  local partial=(
    "${out}/puzzle_set_predictions_fold${fold}_seed0.npz"
    "${out}/puzzle_set_candidate_point_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_point_fold${fold}_seed0.pt"
    "${out}/puzzle_set_candidate_wt_decoder_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_wt_decoder_fold${fold}_seed0.pt"
    "${out}/puzzle_set_candidate_residual_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_residual_fold${fold}_seed0.pt"
  )
  local present=()
  local path
  for path in "${partial[@]}"; do
    if [[ -e "${path}" ]]; then
      present+=("${path}")
    fi
  done
  if [[ "${#present[@]}" -gt 0 ]]; then
    local interrupted="${out}/interrupted_attempts/fold${fold}_$(date +%Y%m%dT%H%M%S)"
    mkdir -p "${interrupted}"
    mv "${present[@]}" "${interrupted}/"
  fi
}

missing=()
for fold in 0 1; do
  if [[ ! -f "${out}/puzzle_set_fold_result_fold${fold}_seed0.json" ]]; then
    missing+=("${fold}")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  require_gpu "${gpu}"
fi

mkdir -p "${out}/logs" "${out}/interrupted_attempts"

if [[ "${#missing[@]}" -gt 0 ]]; then
  for fold in "${missing[@]}"; do
    archive_incomplete_fold "${fold}"
  done
  csv=$(IFS=,; printf '%s' "${missing[*]}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_puzzle_set_meta_context_probe \
      --repo-root "${repo}" \
      --phase P1M2 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --v13-dir "${v13_dir}" \
      --v14-dir "${v14_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --source-manifest "${source_manifest}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${csv}" \
      --pretraining-epochs 3 \
      --point-epochs 3 \
      --calibration-epochs 3 \
      --seed 0 \
      >> "${out}/logs/smoke.log" 2>&1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_puzzle_set_meta_context_probe \
  --repo-root "${repo}" \
  --input-dir "${out}" \
  --phase P1M2 \
  --folds 0,1 \
  --seeds 0 \
  --pretraining-epochs 3 \
  --point-epochs 3 \
  --calibration-epochs 3 \
  --parameter-count 6171697 \
  --trainable-parameter-count 1404417 \
  --out-json "${out}/p1m2_complete_unscored_merge.json"

"${python_bin}" -m scripts.reactflow_delta.qualify_puzzle_set_meta_context_smoke \
  --repo-root "${repo}" \
  --merged-json "${out}/p1m2_complete_unscored_merge.json" \
  --out-json "${out}/p1m2_engineering_smoke_qualification.json"
