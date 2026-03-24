#!/usr/bin/env bash
# Build the ck_baseline PyTorch extension (conv3d_my_impl).
# Run from conv3amd:  ./build_ck_baseline.sh   (do not "source" this script)
#
# PYTORCH_ROCM_ARCH avoids PyTorch querying the GPU during build (which can
# pull in CUDA libs and fail with libcaffe2_nvrtc.so on ROCm). Override if needed:
#   PYTORCH_ROCM_ARCH=gfx942 ./build_ck_baseline.sh
#
# TORCH_CUDA_ARCH_LIST: when no CUDA devices are visible (e.g. ROCm node), PyTorch's
# extension build can get an empty arch list and raise IndexError; set a fallback so
# the list is non-empty (ROCm build will use PYTORCH_ROCM_ARCH for the actual arch).

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/ck_baseline" || { echo "Missing ck_baseline dir"; exit 1; }
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# torch/extension.h includes pybind11; many conda stacks omit it from default -I paths.
if ! python -c "import pybind11" 2>/dev/null; then
  echo "Installing pybind11 (needed to compile conv3d_my.cpp / kernel)..."
  pip install -q pybind11
fi

python setup.py build_ext --inplace
status=$?
cd "$ROOT"
if [[ $status -eq 0 ]]; then
  echo "Done. Extension built in ck_baseline/"
else
  echo "Build failed (exit $status)."
fi
[[ "${BASH_SOURCE[0]:-}" = "$0" ]] && exit $status || return $status
