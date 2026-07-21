#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

remote="${REMOTE:-cunyuliu@36.137.135.49}"
remote_port="${REMOTE_PORT:-22}"
remote_root="${REMOTE_ROOT:-/home/cunyuliu/reactflow}"
latest="${LATEST_RUN:-artifacts/full_runs/full_ablation_20260709_003012}"
split_name="${SPLIT_NAME:-rfam_current_exact_seed0}"
split_rel="$latest/splits/$split_name"

cd "$project_root"

if [ ! -s "$split_rel/split_manifest.json" ]; then
  echo "missing split manifest: $split_rel/split_manifest.json" >&2
  exit 1
fi

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=30 -p "$remote_port")
ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_root/$latest/splits'"

COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C "$latest/splits" "$split_name" \
  | ssh "${ssh_opts[@]}" "$remote" "tar --no-xattrs -xf - -C '$remote_root/$latest/splits'"
