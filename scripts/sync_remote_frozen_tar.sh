#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

remote="${REMOTE:-cunyuliu@36.137.135.49}"
remote_port="${REMOTE_PORT:-22}"
remote_root="${REMOTE_ROOT:-/home/cunyuliu/reactflow}"
latest="${LATEST_RUN:-artifacts/full_runs/full_ablation_20260709_003012}"
frozen_rel="$latest/frozen/ribonanzanet2_sharded_full"

cd "$project_root"

if [ ! -s "$frozen_rel/sharded_manifest.json" ]; then
  echo "missing frozen manifest: $frozen_rel/sharded_manifest.json" >&2
  exit 1
fi

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=30 -p "$remote_port")
ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_root/$latest/frozen'"

# NPZ shards are already compressed; stream an uncompressed tar and suppress
# macOS AppleDouble / extended attribute records for cleaner Linux extraction.
COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C "$latest/frozen" ribonanzanet2_sharded_full \
  | ssh "${ssh_opts[@]}" "$remote" "tar --no-xattrs -xf - -C '$remote_root/$latest/frozen'"
