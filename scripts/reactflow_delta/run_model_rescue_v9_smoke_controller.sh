#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v9_20260824
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v9/v9m1_real_smoke

mkdir -p "${out}/logs"
cd "${repo}"

run_fold() {
  local fold=$1
  local gpu=$2
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v9 \
      --repo-root "${repo}" \
      --phase V9M1 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${fold}" \
      --epochs 3 \
      --seed 0 \
      >> "${out}/logs/fold${fold}.log" 2>&1
}

run_fold 0 6 &
pid0=$!
run_fold 1 7 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

"${python_bin}" -m scripts.reactflow_delta.qualify_model_rescue_v9_smoke \
  --input-dir "${out}" \
  --out-json "${out}/v9m1_smoke_qualification.json"
