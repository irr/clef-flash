#!/usr/bin/env bash
# Serve Cloudflare/clef-flash locally (System One / Jev-compatible).
# Official weights: https://huggingface.co/Cloudflare/clef-flash
#
# gfx1151 (Ryzen AI Max / Radeon 8060S, including GMKtec EVO-X2) cannot use the
# pytorch.org ROCm 6.4 wheel: it segfaults or raises hipErrorNoBinaryForGpu.
# This script installs TheRock gfx1151 wheels and leaves HSA_OVERRIDE_GFX_VERSION
# unset. Verified: torch 2.10.0+rocm7.13.0a20260513, device name "Radeon 8060S Graphics".
#
#   ./run.sh
#   PORT=8000 ./run.sh
#   CLEF_DEVICE=cpu ./run.sh
#   ./run.sh --smoke
#   ./run.sh --debug   # log request, rendered prompt, response and latency
#   CLEF_CAUSAL_CONV1D=0 ./run.sh   # skip building causal-conv1d (default: build once)
#   CLEF_GPU=nvidia ./run.sh        # force vendor: amd | nvidia | none (default: auto-detect)
#
# NVIDIA hosts (nvidia-smi / /dev/nvidia0) get the CUDA torch wheel (TORCH_INDEX_URL
# defaults to https://download.pytorch.org/whl/cu128) and causal-conv1d built for the
# detected compute capability (TORCH_CUDA_ARCH_LIST).
#
#   curl -s http://0.0.0.0:8000/v1/systemone \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"clef-flash","state":"Checkout is failing for every customer.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Cloudflare/clef-flash}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/clef-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CLEF_DEVICE="${CLEF_DEVICE:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SMOKE=0
DEBUG="${DEBUG:-0}"
CLEF_CAUSAL_CONV1D="${CLEF_CAUSAL_CONV1D:-1}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --debug) DEBUG=1 ;;
    -h|--help)
      sed -n "2,25p" "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# GPU vendor: amd | nvidia | none (override with CLEF_GPU).
GPU_VENDOR="${CLEF_GPU:-}"
if [[ -z "$GPU_VENDOR" ]]; then
  if [[ -e /dev/kfd || -d /opt/rocm || -n "${ROCM_PATH:-}" ]]; then
    GPU_VENDOR=amd
  elif [[ -e /dev/nvidia0 ]] \
       || { command -v nvidia-smi >/dev/null 2>&1 && [[ -n "$(nvidia-smi -L 2>/dev/null)" ]]; }; then
    GPU_VENDOR=nvidia
  else
    GPU_VENDOR=none
  fi
fi
case "$GPU_VENDOR" in
  amd|nvidia|none) ;;
  *) echo "CLEF_GPU must be amd, nvidia or none (got: $GPU_VENDOR)" >&2; exit 2 ;;
esac
USE_ROCM=0
USE_CUDA=0
[[ "$GPU_VENDOR" == amd ]] && USE_ROCM=1
[[ "$GPU_VENDOR" == nvidia ]] && USE_CUDA=1

if [[ "$USE_ROCM" == 1 ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx1151/}"
  unset HSA_OVERRIDE_GFX_VERSION
  export HSA_USE_SVM="${HSA_USE_SVM:-0}"
  export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  # Without this, SDPA flash/mem-efficient kernels are disabled on gfx1151 (warns
  # "still experimental") and attention uses the slower math path.
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
  echo "ROCm: native gfx1151, HSA_OVERRIDE_GFX_VERSION unset, HSA_USE_SVM=${HSA_USE_SVM}, HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA}"
elif [[ "$USE_CUDA" == 1 ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d ' \r' | sort -u | paste -sd';' -)"
  fi
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
  echo "CUDA: NVIDIA GPU, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset}"
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ROOT/.venv"
fi

if [[ "$USE_ROCM" == 1 ]]; then
  echo "installing torch from ${TORCH_INDEX_URL}"
  uv pip install --python "$ROOT/.venv/bin/python" --reinstall \
    --index-url "$TORCH_INDEX_URL" torch
elif [[ "$USE_CUDA" == 1 ]]; then
  echo "installing torch from ${TORCH_INDEX_URL}"
  uv pip install --python "$ROOT/.venv/bin/python" \
    --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple torch torchvision
else
  uv pip install --python "$ROOT/.venv/bin/python" torch torchvision
fi

uv pip install --python "$ROOT/.venv/bin/python" \
  "transformers>=5.10.2" \
  "huggingface_hub>=0.34" \
  "safetensors>=0.4" \
  "pillow>=10" \
  "accelerate>=1.0" \
  "flash-linear-attention"

# Optional kernel for the GatedDeltaNet conv. Builds from source (~2 min, once;
# verified on gfx1151 with hipcc). If it fails the torch fallback is used, only slower.
if [[ "$CLEF_CAUSAL_CONV1D" == 1 ]] \
   && ! "$ROOT/.venv/bin/python" -c 'import causal_conv1d' >/dev/null 2>&1; then
  if [[ "$USE_CUDA" == 1 ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      uv pip install --python "$ROOT/.venv/bin/python" setuptools wheel ninja packaging
      MAX_JOBS="${MAX_JOBS:-8}" \
        uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation causal-conv1d \
        || echo "causal-conv1d unavailable, using torch fallback" >&2
    else
      echo "nvcc not found; skipping causal-conv1d build, using torch fallback" >&2
    fi
  else
    uv pip install --python "$ROOT/.venv/bin/python" setuptools wheel ninja packaging
    CAUSAL_CONV1D_FORCE_BUILD=TRUE PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}" \
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}" MAX_JOBS="${MAX_JOBS:-8}" \
      uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation causal-conv1d \
      || echo "causal-conv1d unavailable, using torch fallback" >&2
  fi
fi

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/joint_schema_model.py" ]]; then
  echo "downloading ${MODEL_ID} -> ${MODEL_DIR}"
  uv run --python "$ROOT/.venv/bin/python" hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  echo "using existing snapshot ${MODEL_DIR}"
fi

if [[ "$GPU_VENDOR" != none && "$CLEF_DEVICE" != "cpu" ]]; then
  echo "probing GPU before loading the 19 GB weights"
  uv run --python "$ROOT/.venv/bin/python" python -c \
    'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(4, device="cuda").sum().item())'
fi

export CLEF_MODEL_DIR="$MODEL_DIR"
export CLEF_DEVICE
export CLEF_HOST="$HOST"
export CLEF_PORT="$PORT"
export CLEF_SMOKE="$SMOKE"
export CLEF_DEBUG="$DEBUG"

exec uv run --python "$ROOT/.venv/bin/python" python "$ROOT/serve_clef_flash.py"
