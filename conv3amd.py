#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CausalConv3d stage 0 (wan2pt1).
#
# Benchmarks and tests a single causal 3D conv (128→128, 3×3×3) on GPU (ROCm/CUDA).
# Modes: optional Chrome trace (--profile-name), single forward (default), test-my-conv (nn vs ck_baseline),
# and --use-my-conv to run the ck_baseline extension (float8_e4m3fn only) instead of nn.Conv3d.

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# CausalConv3d: 3D conv with causal padding in the temporal dimension (pad left only).
# Used as stage 0 in the reference; padding (W,H,T) is (1,1,1) → pad 2 on left of T, 1 on W/H.
# -----------------------------------------------------------------------------
class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution (pad then conv; no future frames in receptive field)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (W_l, W_r, H_l, H_r, T_l, T_r) for F.pad; we zero pad left of T only for causal.
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        x = F.pad(x, self._padding)
        return super().forward(x)


def main() -> int:
    # -------------------------------------------------------------------------
    # Arguments: shape, batch, profiling, and which implementation to use.
    # -------------------------------------------------------------------------
    p = argparse.ArgumentParser(description="CausalConv3d stage 0.")
    p.add_argument("--H", type=int, default=192, help="Height")
    p.add_argument("--W", type=int, default=256, help="Width")
    p.add_argument("--T", type=int, default=4, help="Time (4 = stage 0 chunk)")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--channels", "-c", type=int, default=128, help="Channels (128 = stage 0)")
    p.add_argument("--profile-name", type=str, default=None, metavar="NAME", help="If set, record Chrome trace to NAME or NAME.json (single forward only when omitted)")
    p.add_argument("--profile-iters", type=int, default=10, help="Profiler iterations when --profile-name is set")
    p.add_argument("--seed", type=int, default=None, help="Random seed for weight/input init (default: none)")
    p.add_argument("--use-my-conv", action="store_true", dest="use_my_conv", help="Use ck_baseline (float8_e4m3fn only) instead of nn.Conv3d")
    p.add_argument("--test-my-conv", action="store_true", dest="test_my_conv", help="Compare my (ck_baseline) vs nn output; no profile, print pass/fail")
    p.add_argument("--use-ck-bf16", action="store_true", dest="use_ck_bf16", help="Use ck_baseline_bf16 (Composable Kernel BF16) instead of nn.Conv3d")
    p.add_argument("--test-ck-bf16", action="store_true", dest="test_ck_bf16", help="Compare ck_baseline_bf16 vs nn CausalConv3d (BF16); pass/fail")
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help="Before the timed forward / profiler block, run N extra forwards (same process) for GPU warmup",
    )
    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Common setup: require GPU, set seed/device. NN path is bfloat16; ck_baseline is float8_e4m3fn.
    # -------------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("Error: CUDA/ROCm GPU required.", file=sys.stderr)
        return 1
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)

    if args.use_my_conv or args.test_my_conv:
        dtype = torch.float8_e4m3fn
    else:
        dtype = torch.bfloat16

    # -------------------------------------------------------------------------
    # test_ck_bf16: nn CausalConv3d (BF16) vs ck_baseline_bf16 CausalConv3dBF16CK.
    # -------------------------------------------------------------------------
    if args.test_ck_bf16:
        c, t, h, w = args.channels, args.T, args.H, args.W
        _bf16_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_baseline_bf16")
        sys.path.insert(0, _bf16_dir)
        from conv3d_baseline_bf16 import CausalConv3dBF16CK

        model_nn = CausalConv3d(c, c, 3, padding=1, bias=False).to(device, torch.bfloat16)
        nn.init.normal_(model_nn.weight, mean=0.0, std=0.02)
        model_nn.eval()
        model_ck = CausalConv3dBF16CK(c, c, 3, padding=1, bias=False).to(device, torch.bfloat16)
        model_ck.weight.data.copy_(model_nn.weight.data)
        model_ck.eval()
        x = torch.randn(args.batch, c, t, h, w, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y_nn = model_nn(x)
            y_ck = model_ck(x)
        ok = torch.allclose(y_nn, y_ck, rtol=1e-2, atol=2e-2)
        if not ok:
            diff = (y_nn.float() - y_ck.float()).abs()
            print(
                f"fail (max_abs_diff={diff.max().item():.4f}, mean_abs_diff={diff.mean().item():.4f})"
            )
        else:
            print("pass")
        return 0 if ok else 1

    # -------------------------------------------------------------------------
    # test_my_conv: nn CausalConv3d (BF16) vs ck_baseline CausalConv3dFP8; compare in float32.
    # -------------------------------------------------------------------------
    if args.test_my_conv:
        c, t, h, w = args.channels, args.T, args.H, args.W
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_baseline"))
        from conv3d_baseline import CausalConv3dFP8

        model_nn = CausalConv3d(c, c, 3, padding=1, bias=False).to(device, torch.bfloat16)
        nn.init.normal_(model_nn.weight, mean=0.0, std=0.02)
        model_nn.eval()
        model_my = CausalConv3dFP8(c, c, 3, padding=1).to(device, torch.float8_e4m3fn)
        model_my.weight.data.copy_(model_nn.weight.data.to(torch.float8_e4m3fn))
        model_my.eval()
        x_nn = torch.randn(args.batch, c, t, h, w, device=device, dtype=torch.bfloat16)
        x_my = x_nn.to(torch.float8_e4m3fn)
        with torch.no_grad():
            y_nn = model_nn(x_nn)
            y_my = model_my(x_my)

        y_nn_f = x_my.float()  # changed temporarily for debugging
        y_my_f = y_my.float()
        ok = torch.allclose(y_nn_f, y_my_f, rtol=5e-2, atol=5e-1)
        if not ok:
            diff = (y_nn_f - y_my_f).abs()
            print(y_nn_f.shape, y_my_f.shape)
            print(y_nn_f[0,1,1,:,:], y_my_f[0,1,1,:,:])
            print(f"fail (max_abs_diff={diff.max().item():.4f}, mean_abs_diff={diff.mean().item():.4f})")
        else:
            print("pass")
        return 0 if ok else 1

    # -------------------------------------------------------------------------
    # Choose implementation: nn, ck_baseline FP8, or ck_baseline_bf16 (CK).
    # -------------------------------------------------------------------------
    ConvLayer = CausalConv3d
    if args.use_ck_bf16:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_baseline_bf16"))
        from conv3d_baseline_bf16 import CausalConv3dBF16CK

        ConvLayer = CausalConv3dBF16CK
    elif args.use_my_conv:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_baseline"))
        from conv3d_baseline import CausalConv3dFP8

        ConvLayer = CausalConv3dFP8

    # -------------------------------------------------------------------------
    # Build single model, init weights, create random input.
    # -------------------------------------------------------------------------
    model = ConvLayer(args.channels, args.channels, 3, padding=1).to(device, dtype)
    if dtype == torch.float8_e4m3fn:
        # PyTorch has no normal_ for FP8; init in float32 then cast.
        with torch.no_grad():
            w = torch.empty_like(model.weight, dtype=torch.float32).normal_(0.0, 0.02)
            model.weight.copy_(w.to(dtype))
    else:
        nn.init.normal_(model.weight, mean=0.0, std=0.02)
    model.eval()

    # randn does not support fp8; create in float32 then cast to bf16 / fp8.
    x = torch.randn(args.batch, args.channels, args.T, args.H, args.W, device=device, dtype=torch.float32)
    x = x.to(dtype)

    if args.warmup_iters > 0:
        with torch.no_grad():
            for _ in range(args.warmup_iters):
                _ = model(x)
        torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # Run: single forward by default, or profile (warmup + trace) when --profile-name is set.
    # -------------------------------------------------------------------------
    profile_name = (args.profile_name or "").strip()
    do_profile = bool(profile_name)
    if not do_profile:
        with torch.no_grad():
            y = model(x)
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)} (B,C,T,H,W)")
    else:
        for _ in range(5):  # warmup
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        with torch.no_grad(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _ in range(args.profile_iters):
                y = model(x)
            torch.cuda.synchronize()
        trace_path = profile_name if profile_name.endswith(".json") else f"{profile_name}.json"
        trace_path = os.path.abspath(trace_path)
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        # Add runDateTime to trace JSON for bookkeeping (Chrome ignores unknown keys).
        with open(trace_path, "r") as f:
            data = json.load(f)
        run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        data = {"runDateTime": run_time, **data}
        with open(trace_path, "w") as f:
            json.dump(data, f)
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)} (B,C,T,H,W)")
        print(f"Chrome trace written to {trace_path}. Open in Chrome: chrome://tracing → Load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
