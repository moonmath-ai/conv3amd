#!/usr/bin/env bash
# rocprof-compute profile over one conv3amd.py --use-my-conv process:
#   Collects all hardware counter / panel sections for your GPU arch (default when
#   you do NOT pass -b / --set). rocprof-compute replays the app in multiple passes
#   (PMC groups) — expect a longer run than a single rocprofv3 trace.
#
#   --warmup-iters N runs N extra forwards inside that process before the final
#   forward (each replay is a fresh process; warmups run inside each capture).
#
# Outputs: workload dir under rocprof_out/<WORKLOAD_NAME>/ (perfmon/, pmc CSVs,
# profiling_config.yaml, summary_percentages.csv, analyze_tables/*.csv (kept by default), etc.).
# Default WORKLOAD_NAME is <YYYYMMDD_HHMMSS> (local time). Override, e.g.:
#   WORKLOAD_NAME=conv3_use_my_conv ./rocprof_use_my_conv.sh
# After profile, runs rocprof-compute analyze (-b 2 10 11) and merges into summary_percentages.csv.
#
# Advanced Thread Trace for ROCprof Compute Viewer (ui_output_agent_*_dispatch_* dirs) runs after PMC profile:
#   https://github.com/ROCm/rocprof-compute-viewer
#   ROCPROF_NO_UI_TRACE=1 ./rocprof_use_my_conv.sh   # skip ATT pass (faster)
#   ROCPROF_UI_TRACE_DIR=/path ./rocprof_use_my_conv.sh   # default: WORKLOAD_PATH/ui_thread_trace
#   ROCPROF_UI_ATT_TARGET_CU=0 ROCPROF_UI_ATT_GPU_INDEX=0 ./rocprof_use_my_conv.sh   # optional ATT tuning
#   ROCPROF_UI_ATT_LIBRARY_PATH=/path/to/decoder/dir ./rocprof_use_my_conv.sh   # override auto-detect (see below)
#   ROCPROF_UI_ATT_SERIALIZE_ALL=1 ./rocprof_use_my_conv.sh   # optional; can help capture kernels on some systems
#
# Usage:
#   conda activate <env with torch + rocprof-compute deps>   # important
#   ./rocprof_use_my_conv.sh
#   python3 rocprof_use_my_conv.py          # same as the shell script
#   NO_CLEAN_ROCPROF_OUT=1 ./rocprof_use_my_conv.sh
#   ROCPROF_NO_SUMMARY=1 ./rocprof_use_my_conv.sh   # skip analyze + summary CSV
#   ROCPROF_NO_UI_TRACE=1 ./rocprof_use_my_conv.sh   # skip rocprofv3 ATT (ROCprof Compute Viewer data)
#   ROCPROF_SUMMARY_DISPATCH=8 ./rocprof_use_my_conv.sh   # force dispatch id for summary
#   ROCPROF_RM_ANALYZE_TABLES=1 ./rocprof_use_my_conv.sh   # delete analyze_tables/ after merge (default: keep)
#   WARMUP_ITERS=5 WORKLOAD_NAME=conv3_use_my_conv ./rocprof_use_my_conv.sh   # fixed name
#   PYTHON=/path/to/conda/env/bin/python ./rocprof_use_my_conv.sh
#
# Requires: rocprof-compute on PATH (or set ROCPROF_COMPUTE), rocprofv3 (used
# internally), PyTorch HIP. rocprof-compute deps MUST be importable by the SAME
# interpreter as PyTorch — use python -m pip (not bare pip), e.g.
#   "$(command -v python)" -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
# If pip says "site-packages is not writeable" / installs to /usr/lib, conda’s
# python still won’t see those packages: conda install pyyaml pandas pytz … or
# fix env permissions so pip writes into $CONDA_PREFIX/lib/python*/site-packages.
#
# CXXABI / libstdc++: conda’s lib is prepended in the shell below, but rocprofiler-sdk
# replaces LD_LIBRARY_PATH with ROCm-only for the profiled process. We run the app via
# rocprof_conv3amd_app.sh so conda lib is prepended *inside* that process after ROCm’s env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# rocprofv3 ATT: librocprof-trace-decoder.so (not always in /opt/rocm/lib). Auto-pick if ROCPROF_UI_ATT_LIBRARY_PATH unset:
#   1) /opt/rocm/lib   2) sibling ../rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 (common checkout layout)
if [[ -z "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
  if [[ -f /opt/rocm/lib/librocprof-trace-decoder.so ]]; then
    ROCPROF_UI_ATT_LIBRARY_PATH="/opt/rocm/lib"
  else
    _att_default="$(cd "${SCRIPT_DIR}/.." && pwd)/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"
    if [[ -f "${_att_default}/librocprof-trace-decoder.so" ]]; then
      ROCPROF_UI_ATT_LIBRARY_PATH="${_att_default}"
    fi
  fi
fi

PYTHON="${PYTHON:-python}"
# Use this env’s python explicitly (avoids a different `python` first on PATH).
if [[ "${PYTHON}" == "python" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
if ! PYTHON="$(command -v "${PYTHON}")"; then
  echo "error: PYTHON not found (${PYTHON:-python})" >&2
  exit 1
fi

# requirements.txt next to rocprof-compute (for error messages / optional install).
ROCM_REQ="${ROCM_REQ:-}"
if [[ -z "${ROCM_REQ}" ]]; then
  for _base in "${ROCM_PATH:-}" "/opt/rocm" "/opt/rocm-7.2.0"; do
    [[ -n "${_base}" ]] || continue
    _f="${_base}/libexec/rocprofiler-compute/requirements.txt"
    if [[ -f "${_f}" ]]; then
      ROCM_REQ="${_f}"
      break
    fi
  done
fi

ROC_OUT_DIR="${ROC_OUT_DIR:-${SCRIPT_DIR}/rocprof_out}"
# Timestamp avoids clobbering a fixed dirname (NO_CLEAN_ROCPROF_OUT still controls full-tree rm).
WORKLOAD_NAME="${WORKLOAD_NAME:-$(date +%Y%m%d_%H%M%S)}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
WORKLOAD_PATH="${ROC_OUT_DIR}/${WORKLOAD_NAME}"
export WORKLOAD_PATH
# Set early + export so `set -u` is safe; any rocprof-compute continuation subshell still sees paths.
PMC_DISPATCH_INFO="${WORKLOAD_PATH}/pmc_dispatch_info.csv"
PMC_PERF="${WORKLOAD_PATH}/pmc_perf.csv"
export PMC_DISPATCH_INFO PMC_PERF

ROCPROF_COMPUTE="${ROCPROF_COMPUTE:-}"
if [[ -z "${ROCPROF_COMPUTE}" ]]; then
  if command -v rocprof-compute &>/dev/null; then
    ROCPROF_COMPUTE="$(command -v rocprof-compute)"
  elif [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocprof-compute" ]]; then
    ROCPROF_COMPUTE="${ROCM_PATH}/bin/rocprof-compute"
  else
    echo "error: rocprof-compute not found (PATH or ROCPROF_COMPUTE or ROCM_PATH/bin)" >&2
    exit 1
  fi
fi

if ! command -v rocprofv3 &>/dev/null; then
  echo "error: rocprofv3 not found (ROCm bin on PATH; rocprof-compute invokes it)" >&2
  exit 1
fi

mkdir -p "${ROC_OUT_DIR}"

# rocprof-compute verifies importlib.metadata for names in requirements.txt;
# packages must be installed for THIS interpreter (conda), not only /usr/lib.
if ! "${PYTHON}" - <<'PY'
from importlib import metadata
for name in ("pyyaml", "PyYAML"):
    try:
        metadata.distribution(name)
        break
    except metadata.PackageNotFoundError:
        continue
else:
    raise SystemExit(1)
import pandas  # pulls in pytz for many builds
import pytz
PY
then
  echo "error: rocprof-compute Python deps missing for:" >&2
  echo "  ${PYTHON}" >&2
  if [[ -n "${ROCM_REQ}" && -f "${ROCM_REQ}" ]]; then
    echo "Install into this env (use -m pip so it targets the interpreter above):" >&2
    echo "  ${PYTHON} -m pip install -r \"${ROCM_REQ}\"" >&2
  else
    echo "  ${PYTHON} -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt" >&2
  fi
  echo "If you see 'site-packages is not writeable', fix env permissions or run:" >&2
  echo "  conda install -c conda-forge pyyaml pandas pytz  # then pip install the rest from requirements.txt" >&2
  exit 1
fi

export ROC_PROFILE_PYTHON="${PYTHON}"
export ROC_PROFILE_SCRIPT="${SCRIPT_DIR}/conv3amd.py"
# Optional: export ROC_PROFILE_LD_PREFIX yourself; else default to conda lib when present.
if [[ -z "${ROC_PROFILE_LD_PREFIX:-}" && -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export ROC_PROFILE_LD_PREFIX="${CONDA_PREFIX}/lib"
fi

echo "rocprof-compute profile (all counter sections; no -b / --set)"
echo "  workload: -n ${WORKLOAD_NAME}  -p ${WORKLOAD_PATH}"
echo "  runner:   ${PYTHON} ${ROCPROF_COMPUTE}"
echo "  app:      bash rocprof_conv3amd_app.sh …  (prepends conda lib for .so load)"
echo "  warmups:  --warmup-iters ${WARMUP_ITERS} (per replay process)"
# Do not pass -b or --set: empty filter_blocks => all YAML panels / counters for this arch.
"${PYTHON}" "${ROCPROF_COMPUTE}" profile \
  -n "${WORKLOAD_NAME}" \
  -p "${WORKLOAD_PATH}" \
  -- \
  bash "${SCRIPT_DIR}/rocprof_conv3amd_app.sh" --use-my-conv --warmup-iters "${WARMUP_ITERS}"

echo "done. Results under: ${WORKLOAD_PATH}"
echo "  e.g. rocprof-compute analyze -p ${WORKLOAD_PATH}"

# rocprofv3 Advanced Thread Trace → ui_output_agent_*_dispatch_* (ROCprof Compute Viewer).
# https://github.com/ROCm/rocprof-compute-viewer — Import → Rocprofv3 UI (pick dir that contains ui_output_*).
if [[ -z "${ROCPROF_NO_UI_TRACE:-}" ]]; then
  RCV_UI_PARENT="${ROCPROF_UI_TRACE_DIR:-${WORKLOAD_PATH}/ui_thread_trace}"
  mkdir -p "${RCV_UI_PARENT}"
  echo "rocprofv3 Advanced Thread Trace (ROCprof Compute Viewer) -> ${RCV_UI_PARENT}"
  _rv3=(rocprofv3 --advanced-thread-trace 1 --kernel-trace 1 -d "${RCV_UI_PARENT}")
  if [[ -n "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
    # One or more dirs to search for librocprof-trace-decoder.so (see ROCprof Trace Decoder releases).
    read -r -a _att_lib_dirs <<<"${ROCPROF_UI_ATT_LIBRARY_PATH//:/ }"
    _rv3+=(--att-library-path "${_att_lib_dirs[@]}")
  fi
  if [[ -n "${ROCPROF_UI_ATT_TARGET_CU:-}" ]]; then
    _rv3+=(--att-target-cu "${ROCPROF_UI_ATT_TARGET_CU}")
  fi
  if [[ -n "${ROCPROF_UI_ATT_GPU_INDEX:-}" ]]; then
    _rv3+=(--att-gpu-index "${ROCPROF_UI_ATT_GPU_INDEX}")
  fi
  if [[ -n "${ROCPROF_UI_ATT_SERIALIZE_ALL:-}" ]]; then
    _rv3+=(--att-serialize-all 1)
  fi
  _rv3_ok=0
  if "${_rv3[@]}" -- bash "${SCRIPT_DIR}/rocprof_conv3amd_app.sh" --use-my-conv --warmup-iters "${WARMUP_ITERS}"; then
    _rv3_ok=1
  else
    echo "warning: rocprofv3 thread trace failed; continuing (PMC + summary unchanged)" >&2
    echo "  Common cause: missing librocprof-trace-decoder.so — install ROCprof Trace Decoder (RCV README) or set ROCPROF_UI_ATT_LIBRARY_PATH." >&2
  fi
  _ui_dirs="$(find "${RCV_UI_PARENT}" -type d -name 'ui_output_agent_*' 2>/dev/null | head -5 || true)"
  if [[ -n "${_ui_dirs}" ]]; then
    echo "  RCV: Import → Rocprofv3 UI → pick a folder below (or its parent if RCV wants the dispatch parent):"
    while IFS= read -r _d; do [[ -n "${_d}" ]] && echo "    ${_d}"; done <<<"${_ui_dirs}"
  else
    echo "warning: no ui_output_agent_* under ${RCV_UI_PARENT} (dir empty or only nested files)." >&2
    echo "  Look for stderr: rocprof-trace-decoder library path not found" >&2
    echo "  Or list: find ${RCV_UI_PARENT} -type f | head" >&2
    if [[ "${_rv3_ok}" -eq 1 ]]; then
      echo "  ATT exited 0 but no UI tree: try ROCPROF_UI_ATT_TARGET_CU=0 ROCPROF_UI_ATT_SERIALIZE_ALL=1 or another GPU index (ROCPROF_UI_ATT_GPU_INDEX)." >&2
    fi
  fi
fi

if [[ -n "${ROCPROF_NO_SUMMARY:-}" ]]; then
  exit 0
fi

MERGE_PY="${SCRIPT_DIR}/rocprof_merge_percentage_summary.py"
if [[ ! -f "${MERGE_PY}" ]]; then
  echo "warning: missing ${MERGE_PY}; skipping summary_percentages.csv" >&2
  exit 0
fi
if [[ ! -f "${PMC_DISPATCH_INFO}" && ! -f "${PMC_PERF}" ]]; then
  echo "warning: no pmc_dispatch_info.csv or pmc_perf.csv under ${WORKLOAD_PATH}; skipping summary_percentages.csv" >&2
  exit 0
fi

DISPATCH_ID="${ROCPROF_SUMMARY_DISPATCH:-}"
if [[ -z "${DISPATCH_ID}" ]]; then
  DISPATCH_ID="$(
    "${PYTHON}" <<'PY'
import csv
import os
from pathlib import Path

wp = Path(os.environ["WORKLOAD_PATH"])


def from_dispatch_info() -> str:
    p = wp / "pmc_dispatch_info.csv"
    if not p.is_file():
        return ""
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    conv = []
    for r in rows:
        try:
            did = int(r["Dispatch_ID"])
        except (KeyError, ValueError):
            continue
        name = r.get("Kernel_Name") or ""
        if "conv3d_my" in name:
            conv.append(did)
    if conv:
        return str(max(conv))
    if rows:
        try:
            return str(int(rows[-1]["Dispatch_ID"]))
        except (KeyError, ValueError):
            pass
    return ""


def from_pmc_perf() -> str:
    p = wp / "pmc_perf.csv"
    if not p.is_file():
        return ""
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    kernel_by_dispatch: dict[int, str] = {}
    for r in rows:
        try:
            did = int(r["Dispatch_ID"])
        except (KeyError, ValueError):
            continue
        kernel_by_dispatch[did] = r.get("Kernel_Name") or ""
    conv = [d for d, n in kernel_by_dispatch.items() if "conv3d_my" in n]
    if conv:
        return str(max(conv))
    if kernel_by_dispatch:
        return str(max(kernel_by_dispatch.keys()))
    return ""


out = from_dispatch_info() or from_pmc_perf()
print(out, end="")
PY
  )"
fi

if [[ -z "${DISPATCH_ID}" ]]; then
  echo "warning: could not determine dispatch id (conv3d_my or last dispatch); skipping summary" >&2
  exit 0
fi

ANALYZE_TABLES_DIR="${WORKLOAD_PATH}/analyze_tables"
rm -rf "${ANALYZE_TABLES_DIR}"
# --output-name allows only [A-Za-z0-9_-]+ (no path); create CSVs under workload via cwd.
echo "rocprof-compute analyze (dispatch ${DISPATCH_ID}, blocks 2 10 11) -> ${ANALYZE_TABLES_DIR}"
if ! (
  cd "${WORKLOAD_PATH}"
  "${PYTHON}" "${ROCPROF_COMPUTE}" analyze \
    -p . \
    -d "${DISPATCH_ID}" \
    -b 2 10 11 \
    --output-format csv \
    --output-name analyze_tables \
    -q
)
then
  echo "warning: rocprof-compute analyze failed; summary_percentages.csv not generated" >&2
  exit 0
fi

SUMMARY_CSV="${WORKLOAD_PATH}/summary_percentages.csv"
"${PYTHON}" "${MERGE_PY}" "${ANALYZE_TABLES_DIR}" "${SUMMARY_CSV}"
echo "  summary: ${SUMMARY_CSV}"
echo "  analyze tables (rocprof-compute CSVs): ${ANALYZE_TABLES_DIR}"

if [[ -n "${ROCPROF_RM_ANALYZE_TABLES:-}" ]]; then
  rm -rf "${ANALYZE_TABLES_DIR}"
  echo "  (removed analyze_tables/ — unset ROCPROF_RM_ANALYZE_TABLES to keep next time)"
fi
