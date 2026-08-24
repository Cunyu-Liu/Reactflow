#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/cunyuliu/reactflow_delta_model_rescue_v7
RUNTIME="$BASE/runtime"
PYTHON="$RUNTIME/bin/python"
OFFICIAL="$BASE/official/RiNALMo"
SETUP="$BASE/setup"
CONDA=/home/cunyuliu/miniconda3/bin/conda

export CUDA_HOME="$RUNTIME"
export PATH="$RUNTIME/bin:$PATH"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.0
export CONDA_PKGS_DIRS="$BASE/conda_pkgs"

if [[ ! -x "$PYTHON" || ! -x "$RUNTIME/bin/nvcc" ]]; then
  echo "v7 recovery requires the completed Conda runtime transaction"
  exit 1
fi
if [[ ! -x "$CONDA" ]]; then
  echo "v7 recovery requires the server Conda executable"
  exit 1
fi
if [[ "$(git -C "$OFFICIAL" rev-parse HEAD)" != "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955" ]]; then
  echo "official RiNALMo code commit drifted"
  exit 1
fi

"$CONDA" install -y --force-reinstall -p "$RUNTIME" -c conda-forge \
  'pip=23.3' \
  'setuptools=68.2.2' \
  'wheel=0.41.2'
"$PYTHON" -m pip install packaging==23.2 ninja==1.11.1.1
"$PYTHON" -m pip install --no-build-isolation flash-attn==2.3.2
"$PYTHON" -m pip install --no-build-isolation --no-deps -e "$OFFICIAL"

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch

import flash_attn
import rinalmo

assert metadata.version("pip").startswith("23.3")
assert metadata.version("setuptools") == "68.2.2"
assert metadata.version("wheel") == "0.41.2"
assert metadata.version("flash-attn") == "2.3.2"
assert torch.__version__.split("+")[0] == "2.1.0"
assert torch.version.cuda == "11.8"
assert rinalmo is not None and flash_attn is not None
print({"torch": torch.__version__, "cuda": torch.version.cuda, "flash_attn": metadata.version("flash-attn")})
PY

mkdir -p "$SETUP"
touch "$SETUP/runtime_setup_complete"
echo "v7_runtime_setup_complete"
