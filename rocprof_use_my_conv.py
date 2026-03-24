#!/usr/bin/env python3
"""Run rocprof_use_my_conv.sh (profile + summary_percentages.csv under rocprof_out)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sh = root / "rocprof_use_my_conv.sh"
    if not sh.is_file():
        print(f"error: missing {sh}", file=sys.stderr)
        return 1
    return subprocess.call(["bash", str(sh), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
