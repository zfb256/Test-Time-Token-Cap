#!/usr/bin/env bash
# Three-arm control for DeepSeek-R1-Distill-Qwen-7B at 2048 -> 4096 tokens.
#
# The 1024-token independent-budget run leaves one complete-but-wrong set,
# which cannot support a repair estimate. Raising the base cap to 2048 is the
# smallest change that puts R1 in the regime where the estimand is defined.
set -euo pipefail

MODE="${1:-smoke}"
PY="${PYTHON_BIN:-python}"
RUN_ROOT="${R1_THREE_ARM_ROOT:-../outputs/three_arm_r1}"
CONFIG="config_r1_three_arm.yaml"
TOKENIZER="../models/DeepSeek-R1-Distill-Qwen-7B"
BASE_CAP=2048
HAD_TIMEOUTS=0

if [[ "$MODE" == "smoke" ]]; then
    SEEDS=(42)
    LIMIT_ARGS=(--limit 8)
    RUN_ROOT="${RUN_ROOT}_smoke"
    EXPECTED=8
elif [[ "$MODE" == "full" ]]; then
    SEEDS=(42 314159 271828)
    LIMIT_ARGS=()
    EXPECTED=800
else
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
fi

"$PY" preflight_v2.py

run_endpoint () {
    local seed="$1"
    local out="$RUN_ROOT/r1_seed${seed}"
    local score_cache="$out/.math_verify_cache.sqlite"

    "$PY" 02_run_inference.py \
        --config "$CONFIG" \
        --budget long_sample4 xlong_sample4 \
        --output-dir "$out" \
        --engine-seed "$seed" \
        --resume \
        "${LIMIT_ARGS[@]}"

    "$PY" ../analysis/47_reconstruct_lower_cap.py \
        --results-dir "$out" \
        --source-name xlong_sample4 \
        --output-name long_reconstructed \
        --cap "$BASE_CAP" \
        --tokenizer "$TOKENIZER" \
        --rebuild-existing

    MATH_VERIFY_CACHE="$score_cache" MATH_VERIFY_TIMEOUT=10 \
        "$PY" ../analysis/53_three_arm_repair_decomposition.py \
        --base-independent "$out/inference_long_sample4.jsonl" \
        --samecap-control "$out/inference_long_reconstructed.jsonl" \
        --long-control "$out/inference_xlong_sample4.jsonl" \
        --base-budget "$BASE_CAP" \
        --long-budget 4096 \
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
    run_endpoint "$seed"
done

"$PY" ../analysis/45_validate_rerun.py "$RUN_ROOT" --expected-rows "$EXPECTED"
if [[ "${#SEEDS[@]}" -gt 1 ]]; then
    mkdir -p "$RUN_ROOT/aggregate"
    "$PY" ../analysis/56_aggregate_three_arm.py \
        --root "$RUN_ROOT" \
        --models r1 \
        --seeds "${SEEDS[@]}" \
        --output "$RUN_ROOT/aggregate/three_arm_cluster_summary.csv"
fi
"$PY" ../analysis/28_reproducibility_manifest.py \
    --pipeline-config "../pipeline/$CONFIG" \
    --results-dir "$RUN_ROOT"

if (( HAD_TIMEOUTS )); then
    echo "Run completed with verifier timeouts; retained checkpoints for review" >&2
    exit 1
fi
echo "R1 three-arm ${MODE} run validated: $RUN_ROOT"
