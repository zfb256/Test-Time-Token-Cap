#!/usr/bin/env bash
# ============================================================
# v2 review-response experiments - single GPU rental, run once.
#
# Prereqs on the server:
#   - repo uploaded with data/ and outputs/qwen_main/ intact
#     (outputs/qwen_main/inference_medium.jsonl is required by step 08)
#   - models downloaded into models/:
#       Qwen2.5-Math-7B-Instruct
#       Llama-3.1-8B-Instruct
#       DeepSeek-R1-Distill-Qwen-7B
#   - pip install -r requirements.txt && pip install "vllm>=0.6.3"
#     (recent vLLM is required for Llama-3.1 RoPE metadata)
#
# Run from pipeline/:  bash run_v2_experiments.sh 2>&1 | tee v2_run.log
# Every step is resumable; rerun the script after any crash.
# ============================================================
set -uo pipefail

PY="${PYTHON_BIN:-python3}"

step () {
    echo ""
    echo "=== [$(date +%H:%M:%S)] $1 ==="
}

fail=0
run () {
    if ! "$@"; then
        echo "[FAIL] $*"
        fail=1
    fi
}

# Fail before loading any model or generating any token. Unlike later steps,
# preflight failure is fatal because continuing would only waste GPU rental.
"$PY" preflight_v2.py || exit 1

# ---- 1. Qwen main model: sampled controls ------------------
step "Qwen medium_sample8 (384 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config.yaml --budget medium_sample8 --resume

step "Qwen long_sample8 (768 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config.yaml --budget long_sample8 --resume

step "Qwen xlong_sample8 (1536 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config.yaml --budget xlong_sample8 --resume

# ---- 2. Qwen resume-from-truncation ------------------------
step "Qwen continuation run (resume truncated medium traces to 768)"
run "$PY" 08_run_continuation.py --config config.yaml

# ---- 3. Llama second model: matched sampled control --------
step "Llama medium_sample8 (384 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config_llama_validation.yaml --budget medium_sample8 --resume

step "Llama long_sample8 (768 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config_llama_validation.yaml --budget long_sample8 --resume

step "Llama xlong_sample8 (1536 tok x 8 chains, T=0.75)"
run "$PY" 02_run_inference.py --config config_llama_validation.yaml --budget xlong_sample8 --resume

# ---- 4. R1-distill reasoning-tuned contrast ----------------
step "R1-distill budget ladder (1024/2048/4096 x 4 chains, T=0.6)"
run "$PY" 02_run_inference.py --config config_r1_distill.yaml --budget all --resume

# ---- 5. AIME hard-task validation (review 3) ----------------
# data/aime/all_problems.jsonl is committed; no download needed.
step "AIME x Qwen2.5-Math (768/1536/3072, sampled x8 + greedy singles)"
run "$PY" 02_run_inference.py --config config_aime_qwen.yaml --budget all --resume

step "AIME x R1-distill (2048/4096/8192 x 4 chains, T=0.6)"
run "$PY" 02_run_inference.py --config config_aime_r1.yaml --budget all --resume

# ---- 6. CPU analyses (also runnable locally afterwards) ----
step "Analyses"
cd ../analysis
run "$PY" 38_trace_prefix_overlap.py --a-name long --b-name long_resume \
    --a-budget 768 --split-name medium --split-budget 384
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/qwen_main \
    --base-name medium --base-budget 384 --ext-name long_sample8 --ext-budget 768
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/qwen_main \
    --base-name medium_sample8 --base-budget 384 --ext-name long_sample8 --ext-budget 768
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/qwen_main \
    --base-name long_sample8 --base-budget 768 --ext-name xlong_sample8 --ext-budget 1536
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/llama_validation \
    --base-name medium --base-budget 384 --ext-name long_sample8 --ext-budget 768
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/llama_validation \
    --base-name medium_sample8 --base-budget 384 --ext-name long_sample8 --ext-budget 768
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/llama_validation \
    --base-name long_sample8 --base-budget 768 --ext-name xlong_sample8 --ext-budget 1536
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/r1_distill \
    --base-name base1024 --base-budget 1024 --ext-name ext2048 --ext-budget 2048
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/r1_distill \
    --base-name base1024 --base-budget 1024 --ext-name ext4096 --ext-budget 4096
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/r1_distill \
    --base-name ext2048 --base-budget 2048 --ext-name ext4096 --ext-budget 4096
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_qwen \
    --base-name base768 --base-budget 768 --ext-name ext1536 --ext-budget 1536
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_qwen \
    --base-name base768 --base-budget 768 --ext-name ext3072 --ext-budget 3072
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_qwen \
    --base-name base768_greedy --base-budget 768 --ext-name ext3072_greedy --ext-budget 3072
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_r1 \
    --base-name base2048 --base-budget 2048 --ext-name ext4096 --ext-budget 4096
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_r1 \
    --base-name base2048 --base-budget 2048 --ext-name ext8192 --ext-budget 8192
run "$PY" 39_sampling_repair_eval.py --results-dir ../outputs/aime_r1 \
    --base-name ext4096 --base-budget 4096 --ext-name ext8192 --ext-budget 8192
# Independent resampling is a confound even when k/temperature match.
# Censor the extended samples at the base cap to identify repairs and
# aggregate gains that did not require extra tokens.
run "$PY" 44_same_budget_control.py --results-dir ../outputs/qwen_main \
    --base-name medium_sample8 --base-budget 384 \
    --ext-name long_sample8 --ext-budget 768
run "$PY" 44_same_budget_control.py --results-dir ../outputs/qwen_main \
    --base-name long_sample8 --base-budget 768 \
    --ext-name xlong_sample8 --ext-budget 1536
run "$PY" 44_same_budget_control.py --results-dir ../outputs/llama_validation \
    --base-name medium_sample8 --base-budget 384 \
    --ext-name long_sample8 --ext-budget 768
run "$PY" 44_same_budget_control.py --results-dir ../outputs/llama_validation \
    --base-name long_sample8 --base-budget 768 \
    --ext-name xlong_sample8 --ext-budget 1536
run "$PY" 44_same_budget_control.py --results-dir ../outputs/r1_distill \
    --base-name base1024 --base-budget 1024 \
    --ext-name ext2048 --ext-budget 2048
run "$PY" 44_same_budget_control.py --results-dir ../outputs/r1_distill \
    --base-name base1024 --base-budget 1024 \
    --ext-name ext4096 --ext-budget 4096
run "$PY" 44_same_budget_control.py --results-dir ../outputs/r1_distill \
    --base-name ext2048 --base-budget 2048 \
    --ext-name ext4096 --ext-budget 4096
run "$PY" 44_same_budget_control.py --results-dir ../outputs/aime_qwen \
    --base-name base768 --base-budget 768 \
    --ext-name ext1536 --ext-budget 1536
run "$PY" 44_same_budget_control.py --results-dir ../outputs/aime_qwen \
    --base-name base768 --base-budget 768 \
    --ext-name ext3072 --ext-budget 3072
run "$PY" 44_same_budget_control.py --results-dir ../outputs/aime_r1 \
    --base-name base2048 --base-budget 2048 \
    --ext-name ext4096 --ext-budget 4096
run "$PY" 44_same_budget_control.py --results-dir ../outputs/aime_r1 \
    --base-name base2048 --base-budget 2048 \
    --ext-name ext8192 --ext-budget 8192
run "$PY" 44_same_budget_control.py --results-dir ../outputs/aime_r1 \
    --base-name ext4096 --base-budget 4096 \
    --ext-name ext8192 --ext-budget 8192
run "$PY" 38_trace_prefix_overlap.py --results-dir ../outputs/aime_qwen \
    --a-name base768_greedy --b-name ext3072_greedy --a-budget 768

echo ""
if [ "$fail" -eq 0 ]; then
    echo "=== v2 experiments complete. Download outputs/ back to the workstation. ==="
else
    echo "=== v2 experiments finished WITH FAILURES - check the log above. ==="
    exit 1
fi
