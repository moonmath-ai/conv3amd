#!/usr/bin/env python3
"""
Build a colorized ISA table HTML from rocprof Advanced Thread Trace stats CSV.

Input: stats_ui_output_agent_*_dispatch_*.csv (Instruction, Source, Hitcount, Latency, …).

Output (same directory): <input_stem>_isa.html only.
  Columns: ISA line, hitcount (Hitcount / W per traced wave), latency (shows log10(cycles); cold green → warm red by min/max log10 in column), ISA, HIP line, HIP source.
  W = count of se*_sm*_sl*_wv*.json under sibling ui_output_agent_*.

Usage:
  python rocprof_att_stats_to_isa_html.py /path/to/stats_ui_output_agent_48937_dispatch_6.csv
  python rocprof_att_stats_to_isa_html.py stats_*.csv

  --no-html   skip writing HTML (rarely useful)
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import math
import re
import sys
from pathlib import Path

# Per-wave SQTT JSON files next to stats_*.csv (same ui_thread_trace folder).
_WAVE_TRACE_JSON = re.compile(r"^se\d+_sm\d+_sl\d+_wv\d+\.json$", re.IGNORECASE)


def count_trace_waves(stats_path: Path) -> int:
    """Count se*_sm*_sl*_wv*.json under ui_output_agent_* matching this stats file stem."""
    stem = stats_path.stem
    if not stem.startswith("stats_"):
        return 0
    agent_dir = stats_path.parent / stem[6:]  # strip "stats_"
    if not agent_dir.is_dir():
        return 0
    return sum(
        1 for p in agent_dir.iterdir() if p.is_file() and _WAVE_TRACE_JSON.match(p.name)
    )


def warm_cold_spectrum_color(log_v: float, log_lo: float, log_hi: float) -> str:
    """
    Cold (green) at lowest log, warm (red) at highest; HSV hue sweep in between.
    log_* are base-10 logs of latency (cycles).
    """
    if log_hi <= log_lo:
        t = 0.5
    else:
        t = (log_v - log_lo) / (log_hi - log_lo)
        t = max(0.0, min(1.0, t))
    # Hue 120° (green) at t=0 → 0° (red) at t=1
    h = (120.0 / 360.0) * (1.0 - t)
    s = 0.58 + 0.22 * t
    v = 0.50 + 0.18 * (1.0 - t)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def parse_hip_location(ref: str) -> tuple[str | None, int | None]:
    """If ref is like '/abs/path/file.hip:42', return (path, 42). Else (None, None)."""
    if not isinstance(ref, str) or ".hip:" not in ref:
        return None, None
    before, _, after = ref.rpartition(".hip:")
    hip_path = before + ".hip"
    after = after.strip()
    if not after.isdigit():
        return None, None
    return hip_path, int(after)


def isa_mnemonic(isa: str) -> str:
    """First token of the instruction (handles leading ';')."""
    s = isa.strip()
    if not s:
        return ""
    if s.startswith(";"):
        return ";"
    m = re.match(r"(\S+)", s)
    return m.group(1) if m else s


def isa_wait_kind(isa: str) -> str | None:
    """
    s_waitcnt / barrier style ops: color by what they wait on (bold in CSS).
    vmcnt  -> same palette as global/flat VMEM; lgkmcnt -> LDS (incl. scalar loads to LDS path).
    """
    m = isa_mnemonic(isa)
    low = isa.lower()
    if m == "s_waitcnt" or m.startswith("s_waitcnt_"):
        has_vm = "vmcnt" in low
        has_lgkm = "lgkmcnt" in low
        has_store = "storecnt" in low or "expcnt" in low
        if has_vm and has_lgkm:
            return "wait-mixed"
        if has_vm:
            return "wait-vmem"
        if has_lgkm:
            return "wait-lds"
        if has_store:
            return "wait-store"
        return "wait-generic"
    if m.startswith("s_wait_"):
        return "wait-generic"
    return None


def isa_kind(isa: str) -> str:
    """
    Rough GCN/RDNA-style family for coloring.
    Order: waits before generic s_*; memory prefixes before v_/s_.
    """
    m = isa_mnemonic(isa)
    if m == ";":
        return "label"
    wk = isa_wait_kind(isa)
    if wk is not None:
        return wk
    if m.startswith("global_") or m.startswith("scratch_") or m.startswith("flat_"):
        return "vmem"
    if m.startswith("buffer_"):
        return "buffer"
    if m.startswith("ds_"):
        return "lds"
    if m.startswith("v_"):
        return "vector"
    if m.startswith("s_"):
        return "scalar"
    return "other"


ISA_KIND_CSS = """
:root {
  --bg: #1a1b26;
  --fg: #c0caf5;
  --muted: #565f89;
  --border: #3b4261;
}
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       font-size: 13px; background: var(--bg); color: var(--fg);
       margin: 1rem 1.5rem; line-height: 1.45; }
h1 { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
.legend { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; margin: 0.75rem 0 1rem;
          padding: 0.6rem 0.85rem; background: #24283b; border-radius: 6px;
          border: 1px solid var(--border); }
.legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
.swatch { width: 0.85rem; height: 0.85rem; border-radius: 3px; display: inline-block; }
/* Single-line rows: no wrap; scroll horizontally if needed */
.table-wrap { overflow-x: auto; max-width: 100%; margin-bottom: 2rem;
              border: 1px solid var(--border); border-radius: 6px; }
table { border-collapse: collapse; width: max-content; min-width: 100%; }
th, td { border: 1px solid var(--border); padding: 0.2rem 0.5rem; text-align: left;
         vertical-align: middle; white-space: nowrap; line-height: 1.2; }
th { background: #24283b; color: #a9b1d6; position: sticky; top: 0; z-index: 1; }
/* Sticky first column: stays visible when scrolling wide ISA / source rows */
th.isa-line-col {
  position: sticky; left: 0; top: 0; z-index: 4; text-align: right; min-width: 4.5rem;
  box-shadow: 1px 0 0 var(--border);
}
td.num.isa-idx {
  position: sticky; left: 0; z-index: 2;
  background: #1a1b26; color: #a9b1d6; font-weight: 600;
  box-shadow: 1px 0 0 var(--border);
}
td.num { text-align: right; color: #9aa5ce; }
td.src { color: #73daca; font-size: 12px; }
.isa-scalar  { color: #7aa2f7; }
.isa-vector  { color: #bb9af7; }
.isa-vmem    { color: #e0af68; }
.isa-buffer  { color: #ff9e64; }
.isa-lds     { color: #7dcfff; }
.isa-label   { color: #9ece6a; font-style: italic; }
.isa-other   { color: #f7768e; }
/* waits: same hues as VMEM / LDS / etc., bold */
.isa-wait-vmem   { color: #e0af68; font-weight: 700; }
.isa-wait-lds    { color: #7dcfff; font-weight: 700; }
.isa-wait-mixed  { color: #ffc777; font-weight: 700;
                   box-shadow: inset 0 -2px 0 #7dcfff; }
.isa-wait-store  { color: #ff9e64; font-weight: 700; }
.isa-wait-generic { color: #7aa2f7; font-weight: 700; font-style: italic; }
"""


def format_isa_colored(isa: str) -> str:
    kind = isa_kind(isa)
    esc = html.escape(isa, quote=True)
    return f'<span class="isa-{kind}">{esc}</span>'


Row = tuple[int, float, int, str, str, str]


def write_isa_html(out_path: Path, title_stem: str, table: list[Row]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos_lat = [r[2] for r in table if r[2] > 0]
    logs = [math.log10(x) for x in pos_lat]
    log_lo = min(logs) if logs else 0.0
    log_hi = max(logs) if logs else 1.0

    rows_html: list[str] = []
    for isa_idx, norm_hit, latency, isa, hip_ln, src in table:
        norm_s = f"{norm_hit:.4f}".rstrip("0").rstrip(".")
        if latency <= 0:
            lat_cell = '<td class="num" style="color:#565f89"></td>'
        else:
            lv = math.log10(latency)
            col = warm_cold_spectrum_color(lv, log_lo, log_hi)
            lat_txt = f"{lv:.3f}"
            lat_cell = (
                f'<td class="num" style="color:{col};font-weight:600">'
                f"{html.escape(lat_txt, quote=True)}</td>"
            )
        rows_html.append(
            "<tr>"
            f'<td class="num isa-idx">{isa_idx}</td>'
            f'<td class="num">{html.escape(norm_s, quote=True)}</td>'
            f"{lat_cell}"
            f"<td>{format_isa_colored(isa)}</td>"
            f'<td class="num">{html.escape(hip_ln, quote=True)}</td>'
            f'<td class="src">{html.escape(src, quote=True)}</td>'
            "</tr>"
        )

    legend = """
<div class="legend">
  <span><i class="swatch" style="background:#7aa2f7"></i> scalar <code>s_*</code></span>
  <span><i class="swatch" style="background:#bb9af7"></i> vector <code>v_*</code></span>
  <span><i class="swatch" style="background:#e0af68"></i> global / flat / scratch</span>
  <span><i class="swatch" style="background:#ff9e64"></i> buffer</span>
  <span><i class="swatch" style="background:#7dcfff"></i> LDS <code>ds_*</code></span>
  <span><i class="swatch" style="background:#9ece6a"></i> label <code>;</code></span>
  <span><i class="swatch" style="background:#f7768e"></i> other</span>
  <span><b style="color:#e0af68">wait</b> <code>s_waitcnt vmcnt</code> (VMEM hue, bold)</span>
  <span><b style="color:#7dcfff">wait</b> <code>s_waitcnt lgkmcnt</code> (LDS hue, bold)</span>
  <span><b style="color:#ffc777">wait</b> both counters (mixed)</span>
  <span><b style="color:#7aa2f7;font-style:italic">wait</b> generic / packed <code>s_waitcnt</code></span>
</div>
"""
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title_stem, quote=True)} — ISA</title>
<style>{ISA_KIND_CSS}</style>
</head>
<body>
<h1>{html.escape(title_stem, quote=True)}</h1>
{legend}
<div class="table-wrap">
<table>
<thead><tr><th class="num isa-line-col">ISA line</th><th class="num">hitcount</th><th class="num">latency</th><th>ISA</th><th>HIP line</th><th>HIP source line</th></tr></thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</div>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def process_stats_csv(
    stats_path: Path,
    *,
    html_out: bool,
    file_cache: dict[Path, list[str]],
) -> Path | None:
    stats_path = stats_path.resolve()
    if not stats_path.is_file():
        print(f"error: not a file: {stats_path}", file=sys.stderr)
        return None

    def line_from_hip(hip_path_str: str, line_no: int) -> str:
        pth = Path(hip_path_str).resolve()
        if not pth.is_file():
            return f"<missing file: {pth}>"
        if pth not in file_cache:
            file_cache[pth] = pth.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = file_cache[pth]
        if line_no < 1 or line_no > len(lines):
            return f"<line {line_no} out of range ({len(lines)} lines)>"
        return lines[line_no - 1].rstrip()

    table: list[Row] = []
    with stats_path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"error: empty or unreadable CSV: {stats_path}", file=sys.stderr)
            return None
        fields = {h.strip().strip('"') for h in reader.fieldnames}
        if "Instruction" not in fields or "Source" not in fields:
            print(
                f"error: expected Instruction and Source columns in {stats_path}",
                file=sys.stderr,
            )
            return None
        if "Hitcount" not in fields or "Latency" not in fields:
            print(
                f"error: expected Hitcount and Latency columns in {stats_path}",
                file=sys.stderr,
            )
            return None
        key_map = {h.strip().strip('"'): h for h in reader.fieldnames}
        ik, sk = key_map.get("Instruction"), key_map.get("Source")
        hk, lk = key_map.get("Hitcount"), key_map.get("Latency")

        W = count_trace_waves(stats_path)
        if W <= 0:
            agent_guess = stats_path.parent / (
                stats_path.stem[6:] if stats_path.stem.startswith("stats_") else stats_path.stem
            )
            print(
                f"warning: no se*_sm*_sl*_wv*.json under {agent_guess}; using W=1 for hitcount",
                file=sys.stderr,
            )
            W = 1

        for isa_line, row in enumerate(reader, start=1):
            isa = (row.get(ik) or "").strip()
            src = (row.get(sk) or "").strip()
            try:
                hitcount = int(float(row.get(hk) or 0))
            except (TypeError, ValueError):
                hitcount = 0
            try:
                latency = int(float(row.get(lk) or 0))
            except (TypeError, ValueError):
                latency = 0
            norm_hit = hitcount / W
            hip_line_str = ""
            source_line = ""
            hip_path, line_no = parse_hip_location(src)
            if hip_path is not None and line_no is not None:
                hip_line_str = str(line_no)
                source_line = line_from_hip(hip_path, line_no)
            table.append((isa_line, norm_hit, latency, isa, hip_line_str, source_line))

    html_path = stats_path.parent / f"{stats_path.stem}_isa.html"
    if html_out:
        write_isa_html(html_path, stats_path.stem, table)
        print(f"wrote {html_path}")
    else:
        print(f"skipped HTML (--no-html): would write {html_path}", file=sys.stderr)

    return stats_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stats_csv",
        nargs="+",
        type=Path,
        help="One or more stats_ui_output_agent_*_dispatch_*.csv paths",
    )
    p.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        help="Do not write <stem>_isa.html",
    )
    p.set_defaults(html=True)
    args = p.parse_args()

    file_cache: dict[Path, list[str]] = {}
    failures = 0
    for path in args.stats_csv:
        if process_stats_csv(path, html_out=args.html, file_cache=file_cache) is None:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
