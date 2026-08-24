#!/usr/bin/env bash
set -euo pipefail

repo_root=${REACTFLOW_V8_REPO_ROOT:-/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v8_20260824}
m2_csv=${REACTFLOW_V8_M2_CSV:-/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv}
v8_dir=${REACTFLOW_V8_M1_DIR:-/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0}
tic2a_merged=${REACTFLOW_TIC2A_MERGED:-/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json}

cd "${repo_root}"

python -m scripts.reactflow_delta.merge_model_rescue_v8_mean_screen \
  --input-dir "${v8_dir}" \
  --qualification-json "${v8_dir}/v8m1_corrected_expert_qualification.json" \
  --out-json "${v8_dir}/v8m2_complete_unscored_merge.json"

python -m scripts.reactflow_delta.score_model_rescue_v8_mean_screen \
  --repo-root "${repo_root}" \
  --v8-merged-json "${v8_dir}/v8m2_complete_unscored_merge.json" \
  --tic2a-merged-json "${tic2a_merged}" \
  --m2-csv "${m2_csv}" \
  --out-json "${v8_dir}/v8m2_complete_mean_screen_scores.json"

python -m scripts.reactflow_delta.qualify_model_rescue_v8_mean_screen \
  --score-json "${v8_dir}/v8m2_complete_mean_screen_scores.json" \
  --out-json "${v8_dir}/v8m2_mean_screen_qualification.json"
