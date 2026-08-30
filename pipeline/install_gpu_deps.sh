#!/usr/bin/env bash
# Install the pinned GPU/runtime dependencies for CUPID-G Phase 1.
#
# Usage:
#   bash install_gpu_deps.sh
#   PYTHON=/path/to/python bash install_gpu_deps.sh
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash install_gpu_deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-/usr/bin/python3}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.mpl_cache}"

echo "Python: ${PYTHON_BIN}"
echo "Pip index: ${PIP_INDEX}"

"${PYTHON_BIN}" -m pip install -i "${PIP_INDEX}" --prefer-binary -r requirements.txt
"${PYTHON_BIN}" 00_preflight.py --config config.yaml --require-gpu
