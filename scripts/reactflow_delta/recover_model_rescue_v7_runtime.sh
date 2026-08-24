#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/cunyuliu/reactflow_delta_model_rescue_v7
RUNTIME="$BASE/runtime_v7_clean"
PYTHON="$RUNTIME/bin/python"
OFFICIAL="$BASE/official/RiNALMo"
SETUP="$BASE/setup"
CONDA=/home/cunyuliu/miniconda3/bin/conda
FLASH_WHEEL="$BASE/wheels/flash_attn-2.3.2-cp311-cp311-linux_x86_64.whl"

export CUDA_HOME="$RUNTIME"
export PATH="$RUNTIME/bin:$PATH"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export CONDA_PKGS_DIRS="$BASE/conda_pkgs"
export PIP_CACHE_DIR="$BASE/pip_cache"
export TMPDIR="$BASE/tmp"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -x "$CONDA" ]]; then
  echo "v7 recovery requires the server Conda executable"
  exit 1
fi
if [[ "$(git -C "$OFFICIAL" rev-parse HEAD)" != "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955" ]]; then
  echo "official RiNALMo code commit drifted"
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  "$CONDA" create -y -p "$RUNTIME" --override-channels \
    -c pytorch -c nvidia -c conda-forge \
    'python=3.11' \
    'pytorch=2.1.0=py3.11_cuda11.8_cudnn8.7.0_0' \
    'pytorch-cuda=11.8' \
    'cuda-nvcc=11.8' \
    'cuda-cudart-dev=11.8' \
    'cuda-libraries-dev=11.8' \
    'numpy=1.24.4' \
    'pandas=2.0.3' \
    'h5py=3.9.0' \
    'pyyaml=6.0.1' \
    'pip=23.3' \
    'setuptools=68.2.2' \
    'wheel=0.41.2'
fi
if [[ ! -x "$PYTHON" || ! -x "$RUNTIME/bin/nvcc" ]]; then
  echo "v7 clean Conda runtime transaction is incomplete"
  exit 1
fi
if [[ ! -f "$RUNTIME/include/cuda_runtime.h" ]]; then
  "$CONDA" install -y -p "$RUNTIME" --override-channels \
    -c nvidia -c conda-forge \
    'cuda-cudart-dev=11.8'
fi
if [[ ! -f "$RUNTIME/include/cusparse.h" ]]; then
  "$CONDA" install -y -p "$RUNTIME" --override-channels \
    -c nvidia -c conda-forge \
    'cuda-libraries-dev=11.8'
fi
"$PYTHON" -m pip install \
  packaging==23.2 \
  ninja==1.11.1.1 \
  einops==0.6.1 \
  ml-collections==0.1.1 \
  gdown==5.1.0
if [[ ! -f "$FLASH_WHEEL" ]]; then
  "$PYTHON" -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$BASE/wheels" \
    flash-attn==2.3.2
fi
if [[ ! -f "$FLASH_WHEEL" ]]; then
  echo "v7 forced local FlashAttention build did not produce the exact wheel"
  exit 1
fi
"$PYTHON" -m pip install --no-deps "$FLASH_WHEEL"
"$PYTHON" -m pip install --no-build-isolation --no-deps -e "$OFFICIAL"

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import h5py
import numpy
import pandas
import torch
import yaml

import einops
import flash_attn
import gdown
import ml_collections
import rinalmo

assert metadata.version("pip").startswith("23.3")
assert metadata.version("setuptools") == "68.2.2"
assert metadata.version("wheel") == "0.41.2"
assert metadata.version("packaging") == "23.2"
assert metadata.version("ninja") == "1.11.1.1"
assert metadata.version("einops") == "0.6.1"
assert metadata.version("ml-collections") == "0.1.1"
assert metadata.version("gdown") == "5.1.0"
assert metadata.version("flash-attn") == "2.3.2"
assert torch.__version__.split("+")[0] == "2.1.0"
assert torch.version.cuda == "11.8"
assert numpy.__version__ == "1.24.4"
assert pandas.__version__ == "2.0.3"
assert h5py.__version__ == "3.9.0"
assert yaml.__version__ == "6.0.1"
assert all(
    module is not None
    for module in (einops, flash_attn, gdown, ml_collections, rinalmo)
)
print({"torch": torch.__version__, "cuda": torch.version.cuda, "flash_attn": metadata.version("flash-attn")})
PY

mkdir -p "$SETUP"
touch "$SETUP/runtime_setup_complete"
echo "v7_runtime_setup_complete"
