"""
Build the "my" Conv3d C++ extension (PyTorch) — float8_e4m3fn only.
Uses conv3d_my_kernel.cu for both ROCm (hipcc) and CUDA (nvcc).
Use with: python setup.py build_ext --inplace  (or ./build_ck_baseline.sh from parent dir).
Requires PyTorch built for ROCm (or CUDA) so that the extension compiles for GPU.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _include_dirs() -> list[str]:
    """torch/extension.h pulls in pybind11; conda/nvcc often need an explicit include path."""
    dirs: list[str] = []
    try:
        import pybind11

        dirs.append(pybind11.get_include())
    except ImportError:
        pass
    torch_root = os.path.dirname(torch.__file__)
    torch_inc = os.path.join(torch_root, "include")
    if os.path.isfile(os.path.join(torch_inc, "pybind11", "pybind11.h")):
        dirs.append(torch_inc)
    if not dirs:
        print(
            "ck_baseline setup: pybind11 headers not found. Install in this env:\n"
            "  pip install pybind11\n"
            "or: conda install pybind11 -c conda-forge",
            file=sys.stderr,
        )
    return list(dict.fromkeys(dirs))


sources = ["conv3d_my.cpp", "conv3d_my_kernel.cu"]
extra_cxx = ["-O3", "-g", "-std=c++17"]
# PyTorch ROCm adds -D__HIP_NO_HALF_* which breaks rocWMMA (static_cast<__half>(0.0f) in headers).
_hip_rocwmma_fix = (
    ["-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__"]
    if getattr(torch.version, "hip", None) is not None
    else []
)
extra_compile_args: dict[str, list[str]] = {
    "cxx": extra_cxx + _hip_rocwmma_fix,
    # -g for hipcc/nvcc so profilers (e.g. ATT / RCV) get line-level source mapping.
    "nvcc": ["-O3", "-g", "-std=c++17"] + _hip_rocwmma_fix,
}

include_dirs = _include_dirs()
# Optional: path to rocwmma headers (directory containing rocwmma/rocwmma.hpp), e.g.
#   export ROCWMMA_INCLUDE=/path/to/rocwmma/library/include
_rocwmma = os.environ.get("ROCWMMA_INCLUDE")
if _rocwmma:
    include_dirs.append(_rocwmma)

setup(
    name="conv3d_my_impl",
    ext_modules=[
        CUDAExtension(
            name="conv3d_my_impl",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
