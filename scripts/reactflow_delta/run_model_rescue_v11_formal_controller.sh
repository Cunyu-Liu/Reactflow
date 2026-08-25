#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v11_20260825
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v10_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v11/v11m4_formal_seeds0_4

if [[ "$#" -lt 1 || "$#" -gt 8 ]]; then
  printf 'usage: %s PHYSICAL_GPU [PHYSICAL_GPU ...]\n' "$0" >&2
  exit 2
fi

gpus=("$@")
mkdir -p "${out}/logs" "${out}/assembled"
cd "${repo}"

run_worker() {
  local worker=$1
  local gpu=$2
  local worker_count=$3
  local seed fold flat_index csv
  for seed in 0 1 2 3 4; do
    local missing=()
    for fold in {0..19}; do
      flat_index=$((seed * 20 + fold))
      if ((flat_index % worker_count != worker)); then
        continue
      fi
      if [[ ! -f "${out}/v11_fold_result_fold${fold}_seed${seed}.json" ]]; then
        missing+=("${fold}")
      fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
      continue
    fi
    csv=$(IFS=,; printf '%s' "${missing[*]}")
    printf '%s worker=%s physical_gpu=%s seed=%s folds=%s start\n' \
      "$(date --iso-8601=seconds)" "${worker}" "${gpu}" "${seed}" "${csv}" \
      >> "${out}/logs/worker${worker}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
      scripts.reactflow_delta.run_model_rescue_v11 \
        --repo-root "${repo}" \
        --phase V11M4 \
        --m2-csv "${m2}" \
        --v8-dir "${v8_dir}" \
        --v10-dir "${v10_dir}" \
        --tic2a-merged-json "${tic2a}" \
        --unconstrained-cache "${unconstrained}" \
        --constrained-cache "${constrained}" \
        --out-dir "${out}" \
        --device cuda:0 \
        --folds "${csv}" \
        --point-epochs 40 \
        --calibration-epochs 40 \
        --seed "${seed}" \
        >> "${out}/logs/worker${worker}.log" 2>&1
  done
}

worker_pids=()
worker_count=${#gpus[@]}
for ((worker=0; worker<worker_count; worker++)); do
  run_worker "${worker}" "${gpus[$worker]}" "${worker_count}" &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more V11M4 prediction workers failed\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_model_rescue_v11 \
  --input-dir "${out}" \
  --phase V11M4 \
  --out-json "${out}/v11m4_complete_unscored_merge.json"
"${python_bin}" -m scripts.reactflow_delta.assemble_model_rescue_v11_formal \
  --merged-json "${out}/v11m4_complete_unscored_merge.json" \
  --out-dir "${out}/assembled" \
  --out-json "${out}/v11m4_five_seed_prediction_only_assembly.json"
touch "${out}/v11m4_complete_prediction_only_assembly_pass"
