#!/usr/bin/env python3
"""
Merge rocprof-compute analyze CSV output into one summary with:
  - VALU / VMEM / MFMA / SALU / Branch utilization (%), IPC (from Pipeline Stats)
  - Overall instruction mix as % of total dynamic instructions (from Overall Instruction Mix)

Usage:
  1) Generate focused CSVs (optional, fewer files + smaller stdout):
       rocprof-compute analyze -p <workload_dir> -d <dispatch_id> \\
         -b 2 10 11 --output-format csv --output-name analyze_out -q

  2) Merge:
       python rocprof_merge_percentage_summary.py analyze_out summary_percentages.csv

  If you omit step 1, point the first arg at any folder that contains
  2.1_System_Speed-of-Light.csv, 11.2_Pipeline_Statistics.csv, 10.1_Overall_Instruction_Mix.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_pipeline_stats(path: Path) -> list[tuple[str, str, str]]:
    """Metric, value (Avg), Unit for utilization-style rows."""
    rows: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get("Metric", "")
            if "Utilization" in m or m in ("IPC", "IPC (Issued)", "VALU Active Threads"):
                rows.append((m, row.get("Avg", ""), row.get("Unit", "")))
    return rows


def instruction_mix_percentages(path: Path) -> list[tuple[str, str]]:
    """Each instruction class -> pct of total dynamic SQ insts."""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        data: list[tuple[str, float]] = []
        for row in r:
            name = row.get("Metric", "")
            try:
                val = float(row.get("Avg", "nan"))
            except ValueError:
                continue
            data.append((name, val))
    total = sum(v for _, v in data if v == v)  # skip NaN
    if total <= 0:
        return [(n, "") for n, _ in data]
    return [(n, f"{100.0 * v / total:.2f}") for n, v in data]


def read_system_sol_util(path: Path) -> list[tuple[str, str]]:
    """Subset of System Speed-of-Light: utilization + cache hit rates as Avg."""
    want = (
        "VALU Utilization",
        "VMEM Utilization",
        "MFMA Utilization",
        "SALU Utilization",
        "Branch Utilization",
        "CU Utilization",
        "vL1D Cache Hit Rate",
        "L2 Cache Hit Rate",
    )
    out: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get("Metric", "")
            if m in want:
                out.append((m, row.get("Avg", "")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analyze_dir", type=Path, help="Directory with rocprof-compute analyze CSVs")
    ap.add_argument(
        "out_csv",
        type=Path,
        nargs="?",
        default=Path("summary_percentages.csv"),
        help="Output CSV path (default: summary_percentages.csv)",
    )
    args = ap.parse_args()
    d = args.analyze_dir
    p_pipe = d / "11.2_Pipeline_Statistics.csv"
    p_mix = d / "10.1_Overall_Instruction_Mix.csv"
    p_sol = d / "2.1_System_Speed-of-Light.csv"

    missing = [str(p) for p in (p_pipe, p_mix, p_sol) if not p.is_file()]
    if missing:
        print("Missing required inputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    with args.out_csv.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["Section", "Name", "Value", "Unit_or_note"])

        w.writerow(["system_speed_of_light", "", "", "from 2.1_System_Speed-of-Light.csv"])
        for name, val in read_system_sol_util(p_sol):
            w.writerow(["system_speed_of_light", name, val, "Avg"])

        w.writerow([])
        w.writerow(["pipeline_statistics", "", "", "from 11.2_Pipeline_Statistics.csv"])
        for name, val, unit in read_pipeline_stats(p_pipe):
            w.writerow(["pipeline_statistics", name, val, unit])

        w.writerow([])
        w.writerow(
            [
                "instruction_mix_pct",
                "",
                "",
                "% of total dynamic insts; from 10.1_Overall_Instruction_Mix.csv",
            ]
        )
        for name, pct in instruction_mix_percentages(p_mix):
            w.writerow(["instruction_mix_pct", name, pct, "pct_of_total_insts"])

    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
