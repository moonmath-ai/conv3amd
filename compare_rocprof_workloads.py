#!/usr/bin/env python3
"""
Compare two rocprof-compute workload directories (same layout as rocprof_use_my_conv).

Shows:
  - pmc_kernel_top.csv: Mean(ns) per kernel (hardware timer–style totals)
  - summary_percentages.csv: merged A vs B when present
  - ui_thread_trace/stats_*_dispatch_*.csv: paired by dispatch id — row counts, summed
    Hitcount/Latency/Stall/Idle, whether (Vaddr, Instruction) sequence matches, and
    largest per-instruction Latency deltas where keys align.

ATT instruction stats are aggregated over whichever waves hit the traced CU; row order
and counts can differ between runs even when the kernel binary is the same. Use
pmc_kernel_top / PMC for “how long did the dispatch take” more than raw ATT row parity.

Usage:
  python compare_rocprof_workloads.py \\
    conv3amd/rocprof_out/20260325_105310 \\
    conv3amd/rocprof_out/20260325_110154

  python compare_rocprof_workloads.py run_a run_b --dispatch 6
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_kernel_top(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return _read_csv_rows(path)


def load_summary_map(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    rows = _read_csv_rows(path)
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        sec = (r.get("Section") or "").strip()
        name = (r.get("Name") or "").strip()
        val = (r.get("Value") or "").strip()
        if sec or name:
            out[(sec, name)] = val
    return out


def find_stats_by_dispatch(ui_trace: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not ui_trace.is_dir():
        return out
    for p in ui_trace.glob("stats_ui_output_agent_*_dispatch_*.csv"):
        m = re.search(r"_dispatch_(\d+)\.csv$", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_stats(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path)


def stats_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("Vaddr", "").strip(), row.get("Instruction", "").strip())


def aggregate_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, int]]:
    """Merge duplicate (Vaddr, Instruction) by summing numeric columns."""
    acc: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        k = stats_key(r)
        if not k[1] and not k[0]:
            continue
        d = acc.setdefault(k, {"Hitcount": 0, "Latency": 0, "Stall": 0, "Idle": 0})
        for col in ("Hitcount", "Latency", "Stall", "Idle"):
            try:
                d[col] += int(float(r.get(col) or 0))
            except (TypeError, ValueError):
                pass
    return acc


def print_kernel_top(a: Path, b: Path, label_a: str, label_b: str) -> None:
    pa, pb = a / "pmc_kernel_top.csv", b / "pmc_kernel_top.csv"
    ra, rb = load_kernel_top(pa), load_kernel_top(pb)
    if not ra and not rb:
        print("\n=== pmc_kernel_top.csv ===\n(missing both)")
        return
    print("\n=== pmc_kernel_top.csv (Mean(ns) ≈ per-invocation GPU time) ===")
    by_b = {r.get("Kernel_Name", ""): r for r in rb}
    names = sorted(set(r.get("Kernel_Name", "") for r in ra) | set(by_b))
    print(f"{'Kernel':<72} {label_a:>14} {label_b:>14} {'Δ_ns':>10}")
    print("-" * 114)
    for name in names:
        if not name:
            continue
        xa, xb = next((x for x in ra if x.get("Kernel_Name") == name), None), by_b.get(name)
        ma = float(xa["Mean(ns)"]) if xa and xa.get("Mean(ns)") else float("nan")
        mb = float(xb["Mean(ns)"]) if xb and xb.get("Mean(ns)") else float("nan")
        short = (name[:70] + "..") if len(name) > 72 else name
        if xa is None:
            print(f"{short:<72} {'—':>14} {mb:>14.1f} {'':>10}")
        elif xb is None:
            print(f"{short:<72} {ma:>14.1f} {'—':>14} {'':>10}")
        else:
            d = mb - ma
            print(f"{short:<72} {ma:>14.1f} {mb:>14.1f} {d:>+10.1f}")


def print_summary_diff(a: Path, b: Path, label_a: str, label_b: str) -> None:
    pa, pb = a / "summary_percentages.csv", b / "summary_percentages.csv"
    ma, mb = load_summary_map(pa), load_summary_map(pb)
    if not ma and not mb:
        print("\n=== summary_percentages.csv ===\n(missing both)")
        return
    keys = sorted(set(ma) | set(mb), key=lambda k: (k[0], k[1]))
    print("\n=== summary_percentages.csv ===")
    print(f"{'Section':<28} {'Name':<38} {label_a:>12} {label_b:>12}")
    print("-" * 94)
    shown = 0
    for k in keys:
        va, vb = ma.get(k, ""), mb.get(k, "")
        if va == vb:
            continue
        sec, name = k
        sn = (name[:36] + "..") if len(name) > 38 else name
        print(f"{sec:<28} {sn:<38} {va:>12} {vb:>12}")
        shown += 1
    if shown == 0:
        print("(all listed keys identical, or only one side present)")
    else:
        print(f"--- {shown} differing rows (same key)")


def print_stats_pair(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    dispatch: int | None,
) -> None:
    ua, ub = path_a / "ui_thread_trace", path_b / "ui_thread_trace"
    da, db = find_stats_by_dispatch(ua), find_stats_by_dispatch(ub)
    ids = sorted(set(da) & set(db))
    if dispatch is not None:
        ids = [dispatch] if dispatch in da and dispatch in db else []
    if not ids:
        if dispatch is not None:
            print(
                f"\n=== stats ATT (dispatch {dispatch}) ===\n"
                f"missing ui_thread_trace or stats for dispatch on one side"
            )
        else:
            print("\n=== stats ATT ===\n(no common dispatch_* stats CSV pairs)")
        return

    for did in ids:
        fa, fb = da[did], db[did]
        ra, rb = load_stats(fa), load_stats(fb)
        print(f"\n=== stats ATT dispatch_{did} ===")
        print(f"  A: {fa.name}  ({len(ra)} rows)")
        print(f"  B: {fb.name}  ({len(rb)} rows)")

        seq_a = [stats_key(r) for r in ra]
        seq_b = [stats_key(r) for r in rb]
        print(f"  (Vaddr, Instruction) sequence identical: {seq_a == seq_b}")

        def totals(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
            h = l = s = i = 0
            for r in rows:
                for col, var in (
                    ("Hitcount", "h"),
                    ("Latency", "l"),
                    ("Stall", "s"),
                    ("Idle", "i"),
                ):
                    try:
                        v = int(float(r.get(col) or 0))
                    except (TypeError, ValueError):
                        v = 0
                    if col == "Hitcount":
                        h += v
                    elif col == "Latency":
                        l += v
                    elif col == "Stall":
                        s += v
                    else:
                        i += v
            return h, l, s, i

        ta, tb = totals(ra), totals(rb)
        print(
            f"  Sum Hitcount / Latency / Stall / Idle — "
            f"{label_a}: {ta}  {label_b}: {tb}"
        )
        print(
            f"  Δ Latency (B−A): {tb[1] - ta[1]:+d}  Δ Stall: {tb[2] - ta[2]:+d}  "
            f"Δ Idle: {tb[3] - ta[3]:+d}"
        )

        ag_a, ag_b = aggregate_stats(ra), aggregate_stats(rb)
        keys = set(ag_a) & set(ag_b)
        deltas: list[tuple[int, str, str, str]] = []
        for k in keys:
            la, lb = ag_a[k]["Latency"], ag_b[k]["Latency"]
            if la != lb:
                deltas.append((lb - la, k[0], k[1][:60], f"{la}->{lb}"))
        deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        if deltas:
            print(f"  Top |ΔLatency| (merged by Vaddr+Instruction), up to 12:")
            for dv, vaddr, insn, pair in deltas[:12]:
                print(f"    {dv:+8}  vaddr={vaddr}  {insn}  ({pair})")
        only_a = set(ag_a) - set(ag_b)
        only_b = set(ag_b) - set(ag_a)
        if only_a or only_b:
            print(
                f"  Keys only in {label_a}: {len(only_a)}  only in {label_b}: {len(only_b)}"
                " (after merge by Vaddr+Instruction)"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workload_a", type=Path, help="First workload directory")
    p.add_argument("workload_b", type=Path, help="Second workload directory")
    p.add_argument(
        "--label-a",
        default="A",
        help="Label for first directory (default: A)",
    )
    p.add_argument(
        "--label-b",
        default="B",
        help="Label for second directory (default: B)",
    )
    p.add_argument(
        "--dispatch",
        type=int,
        default=None,
        help="Only compare this dispatch id for ATT stats (default: all common dispatches)",
    )
    args = p.parse_args()
    a, b = args.workload_a.resolve(), args.workload_b.resolve()
    if not a.is_dir() or not b.is_dir():
        print("error: both arguments must be existing directories", file=sys.stderr)
        return 1

    print(f"A ({args.label_a}): {a}")
    print(f"B ({args.label_b}): {b}")

    print_kernel_top(a, b, args.label_a, args.label_b)
    print_summary_diff(a, b, args.label_a, args.label_b)
    print_stats_pair(a, b, args.label_a, args.label_b, args.dispatch)

    print(
        "\nNote: If host-side or torch timers show a larger gap than pmc_kernel_top, "
        "they include sync/launch overhead; ATT row lists are not a full dynamic trace."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
