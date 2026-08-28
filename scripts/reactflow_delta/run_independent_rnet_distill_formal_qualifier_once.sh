#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
formal_dir=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd6_formal_seeds0_4
screen_qualification=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0/rnet_distill_qualification.json
score=${formal_dir}/rnet_distill_complete_formal_score.json
qualification=${formal_dir}/rnet_distill_formal_qualification.json

cd "${repo}"
for required in "${screen_qualification}" "${score}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen RND6Q qualifier input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ -e "${qualification}" ]]; then
  printf 'RND6Q canonical qualification already exists; refusing rerun\n' >&2
  exit 2
fi

"${python_bin}" -m scripts.reactflow_delta.qualify_independent_rnet_distill_formal \
  --repo-root "${repo}" \
  --screen-qualification-json "${screen_qualification}" \
  --score-json "${score}" \
  --out-json "${qualification}"
