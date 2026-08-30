#!/usr/bin/env bash
# Temperature sensitivity for the three-arm protocol.
set -euo pipefail

MODE="${1:-smoke}"
PY="${PYTHON_BIN:-python}"
ROOT="${TEMPERATURE_ROOT:-../outputs/temperature_sensitivity}"
LIMIT_ARGS=()
EXPECTED=800
HAD_TIMEOUTS=0
read -r -a SEEDS <<< "${TEMPERATURE_SEEDS:-42}"
if (( ${#SEEDS[@]} == 0 )); then
    echo "TEMPERATURE_SEEDS must contain at least one seed" >&2
    exit 2
fi
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || {
        echo "Invalid seed in TEMPERATURE_SEEDS: $seed" >&2
        exit 2
    }
done
if [[ "$MODE" == "smoke" ]]; then
    ROOT="${ROOT}_smoke"
    LIMIT_ARGS=(--limit 5)
    EXPECTED=5
elif [[ "$MODE" != "full" ]]; then
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
fi

run_endpoint () {
    local tag="$1" temperature="$2" model="$3" config="$4" tokenizer="$5" seed="$6"
    local out="$ROOT/$tag/${model}_seed${seed}"

    "$PY" 02_run_inference.py \
        --config "$config" \
        --budget long_sample8 xlong_sample8 \
        --output-dir "$out" \
        --engine-seed "$seed" \
        --temperature "$temperature" \
        --resume \
        "${LIMIT_ARGS[@]}"
    "$PY" ../analysis/47_reconstruct_lower_cap.py \
        --results-dir "$out" \
        --source-name xlong_sample8 \
        --output-name long_reconstructed \
        --cap 768 \
        --tokenizer "$tokenizer" \
        --rebuild-existing
    MATH_VERIFY_CACHE="$out/.math_verify_cache.sqlite" MATH_VERIFY_TIMEOUT=10 \
        "$PY" ../analysis/53_three_arm_repair_decomposition.py \
        --base-independent "$out/inference_long_sample8.jsonl" \
        --samecap-control "$out/inference_long_reconstructed.jsonl" \
        --long-control "$out/inference_xlong_sample8.jsonl" \
        --output-prefix "$out/three_arm"
    local timeout_count
    timeout_count=$("$PY" -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("SELECT COUNT(*) FROM timeouts").fetchone()[0])' "$out/.math_verify_cache.sqlite")
    if (( timeout_count == 0 )); then
        rm -f "$out/.math_verify_cache.sqlite"{,-wal,-shm}
    else
        HAD_TIMEOUTS=1
        echo "Kept $out/.math_verify_cache.sqlite: $timeout_count verifier timeout(s) need review" >&2
    fi
}

"$PY" preflight_v2.py
for spec in t060:0.6 t100:1.0; do
    IFS=: read -r tag temperature <<< "$spec"
    for seed in "${SEEDS[@]}"; do
        run_endpoint "$tag" "$temperature" qwen config.yaml \
            ../models/Qwen2.5-Math-7B-Instruct "$seed"
        run_endpoint "$tag" "$temperature" llama config_llama_validation.yaml \
            ../models/Llama-3.1-8B-Instruct "$seed"
    done
    "$PY" ../analysis/45_validate_rerun.py "$ROOT/$tag" \
        --expected-rows "$EXPECTED"
    "$PY" ../analysis/28_reproducibility_manifest.py \
        --pipeline-config ../pipeline/config.yaml \
        --results-dir "$ROOT/$tag"
done
if (( HAD_TIMEOUTS )); then
    echo "Run completed with verifier timeouts; retained checkpoints for review" >&2
    exit 1
fi
echo "Temperature sensitivity complete: $ROOT"
