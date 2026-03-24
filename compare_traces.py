#!/usr/bin/env python3
"""Compare two Chrome trace JSON files (PyTorch profiler export).

Example:
  python compare_traces.py trace_bf16.json ck_baseline_bf16/trace_bf16_ck.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_trace_events(path: Path) -> list:
    with path.open() as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def collect_durations(events: list, cat: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("cat") != cat:
            continue
        if e.get("ph") != "X":
            continue
        dur = e.get("dur")
        if dur is None:
            continue
        rows.append((e.get("name") or "", float(dur)))
    return rows


def summarize(rows: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for n, d in rows:
        by[n].append(d)
    return {
        n: {"count": len(ds), "total_us": sum(ds), "mean_us": sum(ds) / len(ds)}
        for n, ds in by.items()
    }


def print_table(
    title: str,
    left_label: str,
    right_label: str,
    s_left: dict,
    s_right: dict,
    min_ms: float = 0.01,
) -> None:
    names = set(s_left) | set(s_right)

    def key(n: str) -> float:
        return max(s_left.get(n, {}).get("total_us", 0), s_right.get(n, {}).get("total_us", 0))

    total_l = sum(x["total_us"] for x in s_left.values())
    total_r = sum(x["total_us"] for x in s_right.values())

    print(f"\n=== {title} ===")
    hdr = f"{'name':<78} {left_label + '_ms':>12} {right_label + '_ms':>12} {'Δ_ms':>10}"
    print(hdr)
    print("-" * len(hdr))
    shown = 0
    for n in sorted(names, key=key, reverse=True):
        a, b = s_left.get(n), s_right.get(n)
        t_l = (a["total_us"] if a else 0) / 1000.0
        t_r = (b["total_us"] if b else 0) / 1000.0
        if t_l < min_ms and t_r < min_ms:
            continue
        short = (n[:76] + "..") if len(n) > 78 else n
        print(f"{short:<78} {t_l:>12.3f} {t_r:>12.3f} {t_r - t_l:>+10.3f}")
        shown += 1
    if shown == 0:
        print("(no rows above min_ms)")
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<78} {total_l/1000:>12.3f} {total_r/1000:>12.3f} "
        f"{(total_r - total_l)/1000:>+10.3f}"
    )

    only_l = sorted(set(s_left) - set(s_right), key=lambda n: s_left[n]["total_us"], reverse=True)
    only_r = sorted(set(s_right) - set(s_left), key=lambda n: s_right[n]["total_us"], reverse=True)
    if only_l:
        print(f"\nOnly in {left_label} ({len(only_l)} names):")
        for n in only_l[:25]:
            print(f"  {int(s_left[n]['count'])}×  {n[:120]}")
        if len(only_l) > 25:
            print(f"  ... +{len(only_l) - 25} more")
    if only_r:
        print(f"\nOnly in {right_label} ({len(only_r)} names):")
        for n in only_r[:25]:
            print(f"  {int(s_right[n]['count'])}×  {n[:120]}")
        if len(only_r) > 25:
            print(f"  ... +{len(only_r) - 25} more")


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two Chrome trace JSON files.")
    p.add_argument("trace_a", type=Path, help="First trace (e.g. trace_bf16.json)")
    p.add_argument("trace_b", type=Path, help="Second trace (e.g. trace_bf16_ck.json)")
    p.add_argument(
        "--label-a",
        default="A",
        help="Label for first trace column (default: A)",
    )
    p.add_argument(
        "--label-b",
        default="B",
        help="Label for second trace column (default: B)",
    )
    p.add_argument(
        "--min-ms",
        type=float,
        default=0.01,
        help="Hide rows where both totals are below this (default: 0.01)",
    )
    args = p.parse_args()

    if not args.trace_a.is_file() or not args.trace_b.is_file():
        print("Error: trace file not found.", file=__import__("sys").stderr)
        return 1

    da = json.loads(args.trace_a.read_text())
    db = json.loads(args.trace_b.read_text())
    print(f"A: {args.trace_a.resolve()}")
    print(f"   runDateTime: {da.get('runDateTime', 'n/a')}")
    print(f"B: {args.trace_b.resolve()}")
    print(f"   runDateTime: {db.get('runDateTime', 'n/a')}")

    ea, eb = load_trace_events(args.trace_a), load_trace_events(args.trace_b)

    print_table(
        "GPU kernels (cat=kernel)",
        args.label_a,
        args.label_b,
        summarize(collect_durations(ea, "kernel")),
        summarize(collect_durations(eb, "kernel")),
        min_ms=args.min_ms,
    )
    print_table(
        "CUDA/HIP runtime (cat=cuda_runtime)",
        args.label_a,
        args.label_b,
        summarize(collect_durations(ea, "cuda_runtime")),
        summarize(collect_durations(eb, "cuda_runtime")),
        min_ms=args.min_ms,
    )
    print_table(
        "CPU aten (cat=cpu_op)",
        args.label_a,
        args.label_b,
        summarize(collect_durations(ea, "cpu_op")),
        summarize(collect_durations(eb, "cpu_op")),
        min_ms=max(args.min_ms, 0.05),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
