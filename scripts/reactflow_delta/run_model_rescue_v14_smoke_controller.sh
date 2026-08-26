#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v10_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m2_real_smoke

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  printf 'usage: %s PHYSICAL_GPU [PHYSICAL_GPU]\n' "$0" >&2
  exit 2
fi

gpus=("$@")
mkdir -p "${out}/logs"
cd "${repo}"

run_fold() {
  local fold=$1
  local gpu=$2
  if [[ -f "${out}/v14_fold_result_fold${fold}_seed0.json" ]]; then
    return 0
  fi
  printf '%s fold=%s physical_gpu=%s start\n' \
    "$(date --iso-8601=seconds)" "${fold}" "${gpu}" \
    >> "${out}/logs/fold${fold}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v14 \
      --repo-root "${repo}" \
      --phase V14M2 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --v10-dir "${v10_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${fold}" \
      --pretraining-epochs 3 \
      --point-epochs 3 \
      --calibration-epochs 3 \
      --seed 0 \
      >> "${out}/logs/fold${fold}.log" 2>&1
}

pids=()
for fold in 0 1; do
  gpu=${gpus[$((fold % ${#gpus[@]}))]}
  run_fold "${fold}" "${gpu}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more V14M2 smoke workers failed\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_model_rescue_v14 \
  --input-dir "${out}" \
  --phase V14M2 \
  --out-json "${out}/v14m2_complete_unscored_merge.json"
"${python_bin}" -m scripts.reactflow_delta.qualify_model_rescue_v14_smoke \
  --merged-json "${out}/v14m2_complete_unscored_merge.json" \
  --out-json "${out}/v14m2_smoke_qualification.json"
