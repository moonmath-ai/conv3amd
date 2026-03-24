"""
Stage-0 causal 3D conv (BF16) using Composable Kernel (explicit v3 grouped-conv instances).

Build: ``CK_ROOT=/path/to/composablekernel python setup.py build_ext --inplace``

Matches ``CausalConv3d`` + ``nn.Conv3d`` numerically in BF16 (within rtol/atol): same F.pad
then conv with padding=(0,0,0).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE0_IN_CHANNELS = 128
STAGE0_OUT_CHANNELS = 128
STAGE0_KERNEL_SIZE = (3, 3, 3)
STAGE0_T = 4
STAGE0_H = 192
STAGE0_W = 320

_impl = None


def _get_impl():
    global _impl
    if _impl is None:
        try:
            import conv3d_ck_bf16_impl as m

            _impl = m
        except ImportError as e:
            raise ImportError(
                "Build ck_baseline_bf16: export CK_ROOT=<composablekernel> && "
                "cd ck_baseline_bf16 && python setup.py build_ext --inplace"
            ) from e
    return _impl


def _verify_stage0_io(x: torch.Tensor, weight: torch.Tensor, *, ctx: str) -> None:
    exp_w = (STAGE0_OUT_CHANNELS, STAGE0_IN_CHANNELS, *STAGE0_KERNEL_SIZE)
    if tuple(weight.shape) != exp_w:
        raise ValueError(f"{ctx}: expected weight {exp_w}, got {tuple(weight.shape)}")
    n, c, t, h, w = x.shape
    if (c, t, h, w) != (STAGE0_IN_CHANNELS, STAGE0_T, STAGE0_H, STAGE0_W):
        raise ValueError(
            f"{ctx}: expected spatial (C,T,H,W)=({STAGE0_IN_CHANNELS},{STAGE0_T},"
            f"{STAGE0_H},{STAGE0_W}), got {(c, t, h, w)}"
        )


class CausalConv3dBF16CK(nn.Module):
    """Same padding contract as ``conv3amd.CausalConv3d``; forward uses CK BF16 conv."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=True):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        if (in_channels, out_channels, k) != (
            STAGE0_IN_CHANNELS,
            STAGE0_OUT_CHANNELS,
            STAGE0_KERNEL_SIZE,
        ):
            raise ValueError(
                "CausalConv3dBF16CK: stage 0 only — in/out 128, kernel 3x3x3"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *k, dtype=torch.bfloat16)
        )
        # (W_l, W_r, H_l, H_r, T_l, T_r) for F.pad on (N,C,D,H,W) last dims W,H,D
        p = padding
        self._padding = (p, p, p, p, 2 * p, 0)
        self.padding = (0, 0, 0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"CausalConv3dBF16CK: need bfloat16, got {x.dtype}")
        _verify_stage0_io(x, self.weight, ctx="CausalConv3dBF16CK")
        x = F.pad(x, self._padding)
        # Conv on padded tensor: symmetric spatial pad 1 per side already in x; temporal asymmetric.
        # F.conv3d with padding=0 matches physical pad baked into x.
        return _get_impl().conv3d_bf16_ck_forward(
            x,
            self.weight,
            0,
            0,
            0,
            0,
            0,
            0,
        )


def causal_conv3d_bf16_ck(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unpadded conv3d (BF16). For stage-0 causal, pad x like ``CausalConv3d`` before calling."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("causal_conv3d_bf16_ck: bfloat16 only")
    _verify_stage0_io(x, weight, ctx="causal_conv3d_bf16_ck")
    return _get_impl().conv3d_bf16_ck_forward(x, weight, 0, 0, 0, 0, 0, 0)


def compare_with_nn(
    batch: int = 1,
    seed: int | None = 0,
    rtol: float = 1e-2,
    atol: float = 2e-2,
) -> bool:
    """Return True if CausalConv3dBF16CK matches CausalConv3d (BF16) within tolerances."""
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    class CausalConv3d(nn.Conv3d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._padding = (
                self.padding[2],
                self.padding[2],
                self.padding[1],
                self.padding[1],
                2 * self.padding[0],
                0,
            )
            self.padding = (0, 0, 0)

        def forward(self, x):
            x = F.pad(x, self._padding)
            return super().forward(x)

    c, t, h, w = STAGE0_IN_CHANNELS, STAGE0_T, STAGE0_H, STAGE0_W
    m_nn = CausalConv3d(c, c, 3, padding=1, bias=False).to(device, torch.bfloat16)
    m_ck = CausalConv3dBF16CK(c, c, 3, padding=1, bias=False).to(device, torch.bfloat16)
    with torch.no_grad():
        m_ck.weight.copy_(m_nn.weight)
    m_nn.eval()
    m_ck.eval()
    x = torch.randn(batch, c, t, h, w, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        y_nn = m_nn(x)
        y_ck = m_ck(x)
    ok = torch.allclose(y_nn, y_ck, rtol=rtol, atol=atol)
    if not ok:
        d = (y_nn.float() - y_ck.float()).abs()
        print(
            f"compare_with_nn: max_abs={d.max().item():.6f} mean_abs={d.mean().item():.6f}"
        )
    return bool(ok)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    if not torch.cuda.is_available():
        print("CUDA/ROCm required", file=sys.stderr)
        sys.exit(1)
    ok = compare_with_nn()
    print("pass" if ok else "fail")
    sys.exit(0 if ok else 1)
