#!/usr/bin/env bash
set -euo pipefail

repo_root=${REACTFLOW_POST_V13_REPO_ROOT:-/home/cunyuliu/reactflow_delta_worktrees/post_v13_diagnostics_20260827}
out_dir=${REACTFLOW_POST_V13_OUT_DIR:-/mnt/cunyuliu/reactflow_delta_post_v13_diagnostics/pv13d2_prediction_only}
python_bin=${REACTFLOW_PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}
m2_csv=${REACTFLOW_M2_CSV:-/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv}
unconstrained_cache=${REACTFLOW_V5_CACHE:-/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5}
constrained_cache=${REACTFLOW_V6_CACHE:-/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5}
corrected_baseline_dir=${REACTFLOW_TIC2A_DIR:-/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines}

mkdir -p "$out_dir"
missing=()
for fold in $(seq 0 19); do
  result="$out_dir/post_v13_diag_fold_result_fold${fold}.json"
  if [[ ! -f "$result" ]]; then
    missing+=("$fold")
  fi
done

if (( ${#missing[@]} > 0 )); then
  fold_csv=$(IFS=,; printf '%s' "${missing[*]}")
  cd "$repo_root"
  "$python_bin" scripts/reactflow_delta/run_post_v13_route_diagnostics.py \
    --repo-root "$repo_root" \
    --m2-csv "$m2_csv" \
    --unconstrained-cache "$unconstrained_cache" \
    --constrained-cache "$constrained_cache" \
    --corrected-baseline-dir "$corrected_baseline_dir" \
    --out-dir "$out_dir" \
    --folds "$fold_csv"
fi

cd "$repo_root"
"$python_bin" scripts/reactflow_delta/merge_post_v13_route_diagnostics.py \
  --input-dir "$out_dir" \
  --out-json "$out_dir/pv13d2_complete_unscored_merge.json"
