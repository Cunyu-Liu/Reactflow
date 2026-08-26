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
out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/p1m2_real_smoke

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s PHYSICAL_GPU\n' "$0" >&2
  exit 2
fi
gpu=$1
mkdir -p "${out}/logs"
cd "${repo}"

missing=()
for fold in 0 1; do
  if [[ ! -f "${out}/puzzle_set_fold_result_fold${fold}_seed0.json" ]]; then
    missing+=("${fold}")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  csv=$(IFS=,; printf '%s' "${missing[*]}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_puzzle_set_meta_context_probe \
      --repo-root "${repo}" \
      --phase P1M2 \
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
      --point-epochs 3 \
      --calibration-epochs 3 \
      --seed 0 \
      >> "${out}/logs/smoke.log" 2>&1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_puzzle_set_meta_context_probe \
  --input-dir "${out}" \
  --phase P1M2 \
  --folds 0,1 \
  --seeds 0 \
  --point-epochs 3 \
  --calibration-epochs 3 \
  --parameter-count 6170417 \
  --trainable-parameter-count 1403137 \
  --out-json "${out}/p1m2_complete_unscored_merge.json"
