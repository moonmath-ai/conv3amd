"""
PyTorch C++ extension: BF16 conv3d via Composable Kernel (explicit Xdl_CShuffle_V3 instances).

Build (ROCm):
  export CK_ROOT=/path/to/composablekernel   # e.g. .../rocm-libraries/projects/composablekernel
  cd ck_baseline_bf16 && python setup.py build_ext --inplace

Requires: PyTorch ROCm, hipcc, Composable Kernel headers under CK_ROOT/{include,library/include}.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _ck_root() -> str:
    env = os.environ.get("CK_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    # Default: sibling rocm-libraries layout from conv3amd
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.abspath(
        os.path.join(here, "..", "..", "rocm-libraries", "projects", "composablekernel")
    )
    return guess


def _include_dirs() -> list[str]:
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

    ck = _ck_root()
    inc_ck = os.path.join(ck, "include")
    inc_lib = os.path.join(ck, "library", "include")
    if not os.path.isdir(inc_ck):
        print(
            f"ck_baseline_bf16: CK_ROOT={ck!r} missing include/. Set CK_ROOT to Composable Kernel root.",
            file=sys.stderr,
        )
    else:
        dirs.append(inc_ck)
    if os.path.isdir(inc_lib):
        dirs.append(inc_lib)

    return list(dict.fromkeys(dirs))


sources = ["conv3d_ck_bind.cpp", "conv3d_ck_launch.cu"]
extra_cxx = ["-O3", "-g", "-std=c++17"]
_hip_fix = (
    ["-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__"]
    if getattr(torch.version, "hip", None) is not None
    else []
)
ck_defs = [
    "-DCK_USE_XDL",
    "-DCK_ENABLE_BF16",
    "-DCK_ENABLE_FP16",
    "-DCK_ENABLE_FP32",
]
extra_compile_args: dict[str, list[str]] = {
    "cxx": extra_cxx + _hip_fix + ck_defs,
    "nvcc": ["-O3", "-g", "-std=c++17"] + _hip_fix + ck_defs,
}

setup(
    name="conv3d_ck_bf16_impl",
    ext_modules=[
        CUDAExtension(
            name="conv3d_ck_bf16_impl",
            sources=sources,
            include_dirs=_include_dirs(),
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
