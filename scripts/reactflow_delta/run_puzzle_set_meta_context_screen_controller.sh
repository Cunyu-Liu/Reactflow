#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v10_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
v13_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0
v14_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/p1m3_screen_seed0

if [[ "$#" -lt 1 || "$#" -gt 8 ]]; then
  printf 'usage: %s PHYSICAL_GPU [PHYSICAL_GPU ...]\n' "$0" >&2
  exit 2
fi
gpus=("$@")
mkdir -p "${out}/logs" "${out}/interrupted_attempts"
cd "${repo}"

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

run_worker() {
  local worker=$1
  local gpu=$2
  shift 2
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if [[ ! -f "${out}/puzzle_set_fold_result_fold${fold}_seed0.json" ]]; then
      archive_incomplete_fold "${fold}"
      missing+=("${fold}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv=$(IFS=,; printf '%s' "${missing[*]}")
  printf '%s worker=%s physical_gpu=%s folds=%s start\n' \
    "$(date --iso-8601=seconds)" "${worker}" "${gpu}" "${csv}" \
    >> "${out}/logs/worker${worker}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_puzzle_set_meta_context_probe \
      --repo-root "${repo}" \
      --phase P1M3 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --v10-dir "${v10_dir}" \
      --v13-dir "${v13_dir}" \
      --v14-dir "${v14_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${csv}" \
      --pretraining-epochs 200 \
      --point-epochs 40 \
      --calibration-epochs 40 \
      --seed 0 \
      >> "${out}/logs/worker${worker}.log" 2>&1
}

pids=()
worker_count=${#gpus[@]}
for ((worker=0; worker<worker_count; worker++)); do
  assigned=()
  for ((fold=worker; fold<20; fold+=worker_count)); do
    assigned+=("${fold}")
  done
  run_worker "${worker}" "${gpus[$worker]}" "${assigned[@]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more P1M3 prediction workers failed\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_puzzle_set_meta_context_probe \
  --input-dir "${out}" \
  --phase P1M3 \
  --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --seeds 0 \
  --pretraining-epochs 200 \
  --point-epochs 40 \
  --calibration-epochs 40 \
  --parameter-count 6171697 \
  --trainable-parameter-count 1404417 \
  --out-json "${out}/p1m3_complete_unscored_merge.json"
