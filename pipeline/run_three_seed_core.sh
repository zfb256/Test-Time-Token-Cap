#!/usr/bin/env bash
# Reproducible three-arm rerun:
# A = independent 768-token sample, B = exact 768-token prefix of C,
# C = independent 1536-token sample.
set -euo pipefail

MODE="${1:-smoke}"
PY="${PYTHON_BIN:-python}"
CORE_ROOT="${THREE_SEED_ROOT:-../outputs/three_arm_core}"
EXPANDED_ROOT="${THREE_ARM_EXPANDED_ROOT:-../outputs/three_arm_expanded}"
HAD_TIMEOUTS=0

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "expanded" ]]; then
    echo "Usage: $0 [smoke|full|expanded]" >&2
    exit 2
fi

"$PY" preflight_v2.py

if [[ "$MODE" == "smoke" ]]; then
    SEEDS=(42 314159 271828)
    LIMIT_ARGS=(--limit 8)
    DATA_ARGS=()
    RUN_ROOT="${CORE_ROOT}_smoke"
    EXPECTED=8
elif [[ "$MODE" == "full" ]]; then
    SEEDS=(42 314159 271828)
    LIMIT_ARGS=()
    DATA_ARGS=()
    RUN_ROOT="$CORE_ROOT"
    EXPECTED=800
else
    # The full split defaults to one seed because each endpoint is roughly
    # eight times a core-split endpoint. Set EXPANDED_SEEDS to extend it.
    IFS=' ' read -r -a SEEDS <<< "${EXPANDED_SEEDS:-42}"
    LIMIT_ARGS=()
    DATA_ARGS=(--data-dir ../data/expanded)
    RUN_ROOT="$EXPANDED_ROOT"
    if [[ ! -f ../data/expanded/all_problems.jsonl ]]; then
        echo "Missing ../data/expanded/all_problems.jsonl; run 01_download_data.py --full first" >&2
        exit 2
    fi
    EXPECTED=$(wc -l < ../data/expanded/all_problems.jsonl)
fi

run_endpoint () {
    local model="$1"
    local config="$2"
    local tokenizer="$3"
    local seed="$4"
    local out="$RUN_ROOT/${model}_seed${seed}"
    local score_cache="$out/.math_verify_cache.sqlite"

    "$PY" 02_run_inference.py \
        --config "$config" \
        --budget long_sample8 xlong_sample8 \
        --output-dir "$out" \
        --engine-seed "$seed" \
        --resume \
        "${DATA_ARGS[@]}" \
        "${LIMIT_ARGS[@]}"

    "$PY" ../analysis/47_reconstruct_lower_cap.py \
        --results-dir "$out" \
        --source-name xlong_sample8 \
        --output-name long_reconstructed \
        --cap 768 \
        --tokenizer "$tokenizer" \
        --rebuild-existing

    MATH_VERIFY_CACHE="$score_cache" MATH_VERIFY_TIMEOUT=10 \
        "$PY" ../analysis/39_sampling_repair_eval.py \
        --results-dir "$out" \
        --base-name long_reconstructed \
        --base-budget 768 \
        --ext-name xlong_sample8 \
        --ext-budget 1536
    MATH_VERIFY_CACHE="$score_cache" MATH_VERIFY_TIMEOUT=10 \
        "$PY" ../analysis/48_expanded_repair_eval.py \
        --results-dir "$out" \
        --base-name long_reconstructed \
        --base-budget 768 \
        --ext-name xlong_sample8 \
        --ext-budget 1536
    MATH_VERIFY_CACHE="$score_cache" MATH_VERIFY_TIMEOUT=10 \
        "$PY" ../analysis/53_three_arm_repair_decomposition.py \
        --base-independent "$out/inference_long_sample8.jsonl" \
        --samecap-control "$out/inference_long_reconstructed.jsonl" \
        --long-control "$out/inference_xlong_sample8.jsonl" \
        --output-prefix "$out/three_arm"
    local timeout_count
    timeout_count=$("$PY" -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("SELECT COUNT(*) FROM timeouts").fetchone()[0])' "$score_cache")
    if (( timeout_count == 0 )); then
        rm -f "$score_cache" "$score_cache-wal" "$score_cache-shm"
    else
        HAD_TIMEOUTS=1
        echo "Kept $score_cache: $timeout_count verifier timeout(s) need review" >&2
    fi
}

for seed in "${SEEDS[@]}"; do
    run_endpoint qwen config.yaml ../models/Qwen2.5-Math-7B-Instruct "$seed"
    run_endpoint llama config_llama_validation.yaml ../models/Llama-3.1-8B-Instruct "$seed"
done

"$PY" ../analysis/45_validate_rerun.py "$RUN_ROOT" --expected-rows "$EXPECTED"
if (( ${#SEEDS[@]} > 1 )); then
    mkdir -p "$RUN_ROOT/aggregate"
    "$PY" ../analysis/52_aggregate_three_seed_core.py \
        --input-root "$RUN_ROOT" \
        --output-dir "$RUN_ROOT/aggregate" \
        --seeds "${SEEDS[@]}"
    "$PY" ../analysis/56_aggregate_three_arm.py \
        --root "$RUN_ROOT" \
        --seeds "${SEEDS[@]}" \
        --output "$RUN_ROOT/aggregate/three_arm_cluster_summary.csv"
fi
"$PY" ../analysis/28_reproducibility_manifest.py \
    --pipeline-config ../pipeline/config.yaml \
    --results-dir "$RUN_ROOT"
if (( HAD_TIMEOUTS )); then
    echo "Run completed with verifier timeouts; retained checkpoints for review" >&2
    exit 1
fi
echo "Three-seed ${MODE} run validated: $RUN_ROOT"
