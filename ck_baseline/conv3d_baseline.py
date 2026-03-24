"""
3D conv module (``float8_e4m3fn`` only) backed by ``conv3d_my_impl``.

Build: ``python setup.py build_ext --inplace`` from ``ck_baseline/``.

No bias in the extension; ``bias=`` on the module is accepted for API compatibility only.

Example::

    layer = CausalConv3dFP8(128, 128, 3, padding=1).cuda().to(torch.float8_e4m3fn)
    x = torch.randn(B, 128, 4, 192, 320, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    y = layer(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stage 0 (wan2pt1 chunk): fixed geometry enforced in forward / functionals
# ---------------------------------------------------------------------------

STAGE0_IN_CHANNELS = 128
STAGE0_OUT_CHANNELS = 128
STAGE0_KERNEL_SIZE = (3, 3, 3)
# Spatial layout for x: (B, C, T, H, W)
STAGE0_T = 4
STAGE0_H = 192
STAGE0_W = 320

# OCP float8 e4m3fn only (no bf16 / e5m2 in this baseline).
FLOAT8_E4M3FN = torch.float8_e4m3fn


def _verify_stage0_io(x: torch.Tensor, weight: torch.Tensor, *, ctx: str) -> None:
    """Require weight (128,128,3,3,3) and x (B,128,4,192,320) with B >= 1."""
    exp_w = (STAGE0_OUT_CHANNELS, STAGE0_IN_CHANNELS, *STAGE0_KERNEL_SIZE)
    if tuple(weight.shape) != exp_w:
        raise ValueError(
            f"{ctx}: expected weight shape {exp_w} (out,in,kT,kH,kW), got {tuple(weight.shape)}"
        )
    if x.dim() != 5:
        raise ValueError(f"{ctx}: expected x dim 5 (N,C,T,H,W), got {x.dim()}")
    n, c, t, h, w = x.shape
    if n < 1:
        raise ValueError(f"{ctx}: expected batch N >= 1, got N={n}")
    if (c, t, h, w) != (STAGE0_IN_CHANNELS, STAGE0_T, STAGE0_H, STAGE0_W):
        raise ValueError(
            f"{ctx}: expected x shape (N,{STAGE0_IN_CHANNELS},{STAGE0_T},{STAGE0_H},{STAGE0_W}), "
            f"got (N,{c},{t},{h},{w})"
        )


# ---------------------------------------------------------------------------
# Extension (lazy import)
# ---------------------------------------------------------------------------

_impl = None


def _get_impl():
    global _impl
    if _impl is None:
        try:
            import conv3d_my_impl as m

            _impl = m
        except ImportError as e:
            raise ImportError(
                "Build the extension: cd ck_baseline && python setup.py build_ext --inplace"
            ) from e
    return _impl


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _ks(kernel_size):
    return kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3


def _require_float8_e4m3fn(x, weight, *, ctx: str):
    if x.dtype != FLOAT8_E4M3FN:
        raise TypeError(f"{ctx}: x must be float8_e4m3fn, got {x.dtype}")
    if weight.dtype != FLOAT8_E4M3FN:
        raise TypeError(f"{ctx}: weight must be float8_e4m3fn, got {weight.dtype}")


# ---------------------------------------------------------------------------
# FP8 (float8_e4m3fn) only
# ---------------------------------------------------------------------------


class CausalConv3dFP8(nn.Module):
    """3D conv, float8_e4m3fn only. ``padding`` / ``bias`` are ignored (API compatibility only).

    Stage 0: ``in_channels`` / ``out_channels`` must be 128, ``kernel_size`` must be 3,
    and forward input ``x`` must be ``(B, 128, 4, 192, 320)`` with ``B >= 1``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=True):
        super().__init__()
        k = _ks(kernel_size)
        if (in_channels, out_channels, k) != (STAGE0_IN_CHANNELS, STAGE0_OUT_CHANNELS, STAGE0_KERNEL_SIZE):
            raise ValueError(
                "CausalConv3dFP8: stage 0 requires "
                f"in_channels={STAGE0_IN_CHANNELS}, out_channels={STAGE0_OUT_CHANNELS}, "
                f"kernel_size=3; got in={in_channels}, out={out_channels}, kernel_size={kernel_size!r}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *k, dtype=FLOAT8_E4M3FN)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            w = torch.empty(self.weight.shape, dtype=torch.float32, device=self.weight.device)
            nn.init.normal_(w, mean=0.0, std=0.02)
            self.weight.copy_(w.to(FLOAT8_E4M3FN))

    def forward(self, x):
        _require_float8_e4m3fn(x, self.weight, ctx="CausalConv3dFP8")
        _verify_stage0_io(x, self.weight, ctx="CausalConv3dFP8")
        return _get_impl().conv3d_fp8_forward(x, self.weight)


def causal_conv3d_fp8(x, weight):
    """Low-level forward (float8_e4m3fn only). Same stage-0 shapes as ``CausalConv3dFP8``."""
    _require_float8_e4m3fn(x, weight, ctx="causal_conv3d_fp8")
    _verify_stage0_io(x, weight, ctx="causal_conv3d_fp8")
    return _get_impl().conv3d_fp8_forward(x, weight)
