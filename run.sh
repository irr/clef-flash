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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx1151/}"
SMOKE=0
DEBUG="${DEBUG:-0}"
CLEF_CAUSAL_CONV1D="${CLEF_CAUSAL_CONV1D:-1}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --debug) DEBUG=1 ;;
    -h|--help)
      sed -n "2,20p" "$0"
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

USE_ROCM=0
if [[ -e /dev/kfd || -d /opt/rocm || -n "${ROCM_PATH:-}" ]]; then
  USE_ROCM=1
fi

if [[ "$USE_ROCM" == 1 ]]; then
  unset HSA_OVERRIDE_GFX_VERSION
  export HSA_USE_SVM="${HSA_USE_SVM:-0}"
  export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  # Without this, SDPA flash/mem-efficient kernels are disabled on gfx1151 (warns
  # "still experimental") and attention uses the slower math path.
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
  echo "ROCm: native gfx1151, HSA_OVERRIDE_GFX_VERSION unset, HSA_USE_SVM=${HSA_USE_SVM}, HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA}"
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ROOT/.venv"
fi

if [[ "$USE_ROCM" == 1 ]]; then
  echo "installing torch from ${TORCH_INDEX_URL}"
  uv pip install --python "$ROOT/.venv/bin/python" --reinstall \
    --index-url "$TORCH_INDEX_URL" torch
else
  uv pip install --python "$ROOT/.venv/bin/python" torch
fi

uv pip install --python "$ROOT/.venv/bin/python" \
  "transformers>=5.10.2" \
  "huggingface_hub>=0.34" \
  "safetensors>=0.4" \
  "pillow>=10" \
  "accelerate>=1.0" \
  "flash-linear-attention"

# Optional HIP kernel for the GatedDeltaNet conv. Builds from source (~2 min, once;
# verified on gfx1151 with hipcc). If it fails the torch fallback is used, only slower.
if [[ "$CLEF_CAUSAL_CONV1D" == 1 ]] \
   && ! "$ROOT/.venv/bin/python" -c 'import causal_conv1d' >/dev/null 2>&1; then
  uv pip install --python "$ROOT/.venv/bin/python" setuptools wheel ninja packaging
  CAUSAL_CONV1D_FORCE_BUILD=TRUE PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}" \
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}" MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation causal-conv1d \
    || echo "causal-conv1d unavailable, using torch fallback" >&2
fi

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/joint_schema_model.py" ]]; then
  echo "downloading ${MODEL_ID} -> ${MODEL_DIR}"
  uv run --python "$ROOT/.venv/bin/python" hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  echo "using existing snapshot ${MODEL_DIR}"
fi

if [[ "$USE_ROCM" == 1 && "$CLEF_DEVICE" != "cpu" ]]; then
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
