#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/p1m3_screen_seed0
merged=${out}/p1m3_complete_unscored_merge.json
score=${out}/p1m3_complete_score.json
qualification=${out}/p1m3_qualification.json
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
v13_score=/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0/v13m3_complete_score.json

cd "${repo}"
for required in "${merged}" "${m2}" "${tic2a}" "${v13_score}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen puzzle-set score input is missing: %s\n' "${required}" >&2
    exit 1
  fi
done
if [[ -e "${score}" || -e "${qualification}" ]]; then
  printf 'puzzle-set score-once output already exists; refusing to rerun\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.score_puzzle_set_meta_context \
  --repo-root "${repo}" \
  --merged-json "${merged}" \
  --tic2a-merged-json "${tic2a}" \
  --v13-score-json "${v13_score}" \
  --m2-csv "${m2}" \
  --out-json "${score}"

"${python_bin}" -m scripts.reactflow_delta.qualify_puzzle_set_meta_context \
  --repo-root "${repo}" \
  --score-json "${score}" \
  --out-json "${qualification}"
