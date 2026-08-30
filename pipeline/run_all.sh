#!/usr/bin/env bash
# ============================================================
# CUPID-G Phase 1: Master run script
# Usage: bash run_all.sh [--resume] [--mock-prm] [--dry-run]
#
# Steps:
#   01  Download + sample datasets
#   02  Run inference (all 4 budget levels)
#   06  Plan B-1 self-correction (parallel, same inference engine call)
#   03  PRM scoring + ORM verification + utility labels
#   04  Build G1-G4 feature matrix
#   05  Train classifiers, compute AUC, print Go/No-Go verdict
#
# Requires: vllm, transformers, datasets, lightgbm, scikit-learn
# See requirements.txt
# ============================================================

set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.mpl_cache}"

CONFIG="config.yaml"
RESUME_FLAG=""
MOCK_PRM_FLAG=""
DRY_RUN_FLAG=""
LOG_DIR="logs"
PYTHON_BIN="${PYTHON:-}"

# Parse args
for arg in "$@"; do
    case $arg in
        --resume)   RESUME_FLAG="--resume" ;;
        --mock-prm) MOCK_PRM_FLAG="--mock-prm" ;;
        --dry-run)  DRY_RUN_FLAG="--dry-run" ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
done

if [[ "${CONFIG}" != /* && ! -f "${CONFIG}" && -f "${ORIGINAL_CWD}/${CONFIG}" ]]; then
    CONFIG="${ORIGINAL_CWD}/${CONFIG}"
fi

if [ -z "${PYTHON_BIN}" ]; then
    if [ -x /usr/bin/python3 ]; then
        PYTHON_BIN="/usr/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[FAIL] Neither python nor python3 was found in PATH."
        exit 1
    fi
fi

# Read output/log dirs from config (fallback to local dirs if PyYAML is unavailable)
OUT_DIR=$("${PYTHON_BIN}" -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['paths']['output_dir'])" 2>/dev/null || echo "../outputs/qwen_main")
LOG_DIR=$("${PYTHON_BIN}" -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['paths'].get('log_dir', 'logs'))" 2>/dev/null || echo "${LOG_DIR}")
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "============================================" | tee -a "${LOG_FILE}"
echo "CUPID-G Phase 1 — $(date)"                   | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}"                            | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON_BIN}"                        | tee -a "${LOG_FILE}"
echo "Output: ${OUT_DIR}"                           | tee -a "${LOG_FILE}"
echo "Log:    ${LOG_FILE}"                          | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

PREFLIGHT_ARGS=(--config "${CONFIG}")
if [ -z "${DRY_RUN_FLAG}" ]; then
    PREFLIGHT_ARGS+=(--require-gpu)
fi

echo "" | tee -a "${LOG_FILE}"
echo ">>> Step 00: preflight" | tee -a "${LOG_FILE}"
if "${PYTHON_BIN}" 00_preflight.py "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/step00_${TIMESTAMP}.log" "${LOG_FILE}"; then
    echo "    [OK] Step 00 complete" | tee -a "${LOG_FILE}"
else
    echo "    [FAIL] Step 00 failed. Check ${LOG_DIR}/step00_${TIMESTAMP}.log" | tee -a "${LOG_FILE}"
    exit 1
fi

run_step() {
    local step_num="$1"
    local script="$2"
    local extra_args="${3:-}"
    local step_log="${LOG_DIR}/step${step_num}_${TIMESTAMP}.log"

    echo "" | tee -a "${LOG_FILE}"
    echo ">>> Step ${step_num}: ${script} ${extra_args}" | tee -a "${LOG_FILE}"
    echo "    Log: ${step_log}" | tee -a "${LOG_FILE}"

    if "${PYTHON_BIN}" "${script}" --config "${CONFIG}" ${extra_args} 2>&1 | tee -a "${step_log}" "${LOG_FILE}"; then
        echo "    [OK] Step ${step_num} complete" | tee -a "${LOG_FILE}"
    else
        echo "    [FAIL] Step ${step_num} failed. Check ${step_log}" | tee -a "${LOG_FILE}"
        exit 1
    fi
}

# ---- Step 01: Download data ----
run_step "01" "01_download_data.py" "${DRY_RUN_FLAG}"

if [ -n "${DRY_RUN_FLAG}" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "Dry run complete. No inference or output files were generated." | tee -a "${LOG_FILE}"
    exit 0
fi

# ---- Step 02: Inference (all budgets) ----
# Note: this is the GPU-intensive step (~2-3 hours on A100 for 800 problems)
run_step "02" "02_run_inference.py" "${RESUME_FLAG}"

# ---- Step 06: Plan B-1 CHR (uses plan_b1.initial_budget from config) ----
# Parallel track — uses same problems, minimal extra cost
echo "" | tee -a "${LOG_FILE}"
echo ">>> Step 06 (Plan B-1 parallel): 06_planb1_chr.py" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" 06_planb1_chr.py --config "${CONFIG}" ${RESUME_FLAG} \
    2>&1 | tee -a "${LOG_DIR}/step06_${TIMESTAMP}.log" "${LOG_FILE}" || {
    echo "    [WARN] Plan B-1 CHR step failed (non-fatal, continuing CUPID-G track)" | tee -a "${LOG_FILE}"
}

# ---- Step 03: PRM scoring + ORM ----
run_step "03" "03_prm_scoring.py" "${MOCK_PRM_FLAG}"

# ---- Step 04: Build features ----
run_step "04" "04_build_features.py" ""

# ---- Step 05: Train + AUC evaluation ----
run_step "05" "05_train_evaluate.py" ""

# ---- Final summary ----
echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "PHASE 1 COMPLETE — $(date)"                  | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

# Print Go/No-Go from the verdict file
VERDICT_FILE="${OUT_DIR}/auc_results.txt"
if [ -f "${VERDICT_FILE}" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "=== AUC VERDICT ===" | tee -a "${LOG_FILE}"
    cat "${VERDICT_FILE}" | tee -a "${LOG_FILE}"
fi

# Print CHR result
CHR_FILE="${OUT_DIR}/planb1_chr_result.txt"
if [ -f "${CHR_FILE}" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "=== PLAN B-1 CHR ===" | tee -a "${LOG_FILE}"
    cat "${CHR_FILE}" | tee -a "${LOG_FILE}"
fi

echo ""
echo "Full log: ${LOG_FILE}"
