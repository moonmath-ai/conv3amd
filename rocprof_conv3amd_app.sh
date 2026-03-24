#!/usr/bin/env bash
# Launched by rocprof-compute as the profiled app. rocprofiler-sdk sets
# LD_LIBRARY_PATH to ROCm lib only (see AMD rocprofiler_sdk profiler), which drops
# conda’s libstdc++ and breaks .so extensions (CXXABI_1.3.15). Prepend the path
# the parent exports before exec’ing Python.
set -euo pipefail
if [[ -n "${ROC_PROFILE_LD_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${ROC_PROFILE_LD_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
: "${ROC_PROFILE_PYTHON:?ROC_PROFILE_PYTHON not set}"
: "${ROC_PROFILE_SCRIPT:?ROC_PROFILE_SCRIPT not set}"
exec "${ROC_PROFILE_PYTHON}" "${ROC_PROFILE_SCRIPT}" "$@"
