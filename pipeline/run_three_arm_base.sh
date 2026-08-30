#!/usr/bin/env bash
# Future GPU add-on: generate only independent 768-token arm A and reuse the
# existing exact-prefix B/C artifacts from run_three_seed_core.sh.
set -euo pipefail

PY="${PYTHON_BIN:-python}"
ROOT="${THREE_SEED_ROOT:-../outputs/three_arm_core}"
SEEDS=(42 314159 271828)

run_arm_a () {
    local model="$1"
    local config="$2"
    local tokenizer="$3"
    local seed="$4"
    local out="$ROOT/${model}_seed${seed}"

    test -f "$out/inference_long_reconstructed.jsonl"
    test -f "$out/inference_xlong_sample8.jsonl"
    "$PY" ../analysis/47_reconstruct_lower_cap.py \
        --results-dir "$out" \
        --source-name xlong_sample8 \
        --output-name long_reconstructed \
        --cap 768 \
        --tokenizer "$tokenizer" \
        --rebuild-existing
    "$PY" 02_run_inference.py \
        --config "$config" \
        --budget long_sample8 \
        --output-dir "$out" \
        --engine-seed "$seed" \
        --resume
    "$PY" ../analysis/53_three_arm_repair_decomposition.py \
        --base-independent "$out/inference_long_sample8.jsonl" \
        --samecap-control "$out/inference_long_reconstructed.jsonl" \
        --long-control "$out/inference_xlong_sample8.jsonl" \
        --output-prefix "$out/three_arm"
}

"$PY" preflight_v2.py
for seed in "${SEEDS[@]}"; do
    run_arm_a qwen config.yaml \
        ../models/Qwen2.5-Math-7B-Instruct "$seed"
    run_arm_a llama config_llama_validation.yaml \
        ../models/Llama-3.1-8B-Instruct "$seed"
done

"$PY" ../analysis/56_aggregate_three_arm.py \
    --root "$ROOT" \
    --output "$ROOT/aggregate/three_arm_cluster_summary.csv"
"$PY" ../analysis/28_reproducibility_manifest.py \
    --pipeline-config ../pipeline/config.yaml \
    --results-dir "$ROOT"
echo "Three-arm add-on complete: $ROOT"
