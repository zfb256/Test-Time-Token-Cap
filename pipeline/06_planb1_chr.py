"""
Plan B-1 parallel track: Self-Correction CHR (Correction Harm Rate).

Runs SIMULTANEOUSLY with Days 1-7 CUPID-G track (zero extra cost since
it uses the same GSM8K + MATH problems).

Protocol:
  1. For each problem, the "initial answer" = plan_b1.initial_budget greedy chain
     (default: medium, 384 tokens; configurable in config.yaml).
  2. Generate one "self-corrected answer" by prompting the model:
     "Review your solution and correct any errors."
  3. Compute CHR = (# correct→wrong transitions) / (# initially correct problems)

CHR > 5% threshold (from config) -> Plan B-1 viable for AAAI
CHR < 2%                          -> Plan B-2 (data filtering, ACL/EMNLP)

Outputs:
  {output_dir}/planb1_corrections.jsonl   (initial + corrected pairs)
  {output_dir}/planb1_chr_result.txt      (CHR value + Plan B decision)

Run:
  python 06_planb1_chr.py --config config.yaml
  python 06_planb1_chr.py --config config.yaml --resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from utils.math_verify import MathVerifier

# Self-correction prompt template (Qwen-Math chat format)
SELF_CORRECT_PROMPT = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{initial_answer}\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Please review your solution carefully. If there are any errors, correct them. "
    "Provide your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSONL line {lineno} in {path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load initial answers (budget controlled by config) ----
    initial_budget = cfg["plan_b1"].get("initial_budget", "medium")
    initial_path = out_dir / f"inference_{initial_budget}.jsonl"
    if not initial_path.exists():
        logger.error(f"Missing: {initial_path}. Run 02_run_inference.py first.")
        sys.exit(1)

    initial_records = load_jsonl(initial_path)
    logger.info(f"Loaded {len(initial_records)} initial answers from inference_{initial_budget}.jsonl")
    if not initial_records:
        logger.error(f"No records found in inference_{initial_budget}.jsonl.")
        sys.exit(1)

    # ---- Check for already-processed records ----
    out_path = out_dir / "planb1_corrections.jsonl"
    done_ids = set()
    if args.resume and out_path.exists():
        existing = load_jsonl(out_path)
        done_ids = {
            r["problem_id"] for r in existing
            if r.get("corrected_chain") is not None
        }
        logger.info(f"Resuming: {len(done_ids)} already processed")

    todo = [r for r in initial_records if r["problem_id"] not in done_ids]
    logger.info(f"Generating self-corrections for {len(todo)} problems...")

    if not todo:
        logger.info("All done. Computing CHR from existing file.")
    else:
        # ---- Load vLLM ----
        from vllm import LLM, SamplingParams
        planb_cfg = cfg["plan_b1"]["self_correction"]
        inf_cfg = cfg["inference"]

        logger.info(f"Loading model: {inf_cfg['model']}")
        llm = LLM(
            model=inf_cfg["model"],
            tensor_parallel_size=inf_cfg["tensor_parallel_size"],
            dtype=inf_cfg["dtype"],
            gpu_memory_utilization=inf_cfg["gpu_memory_utilization"],
            max_model_len=inf_cfg.get("max_model_len", 2048),
            trust_remote_code=inf_cfg.get("trust_remote_code", False),
        )
        sampling = SamplingParams(
            max_tokens=planb_cfg["max_new_tokens"],
            temperature=planb_cfg["temperature"],
            n=1,
            seed=planb_cfg.get("seed", cfg["inference"].get("seed")),
        )

        batch_size = inf_cfg.get("batch_size", 32)

        with open(out_path, "a", encoding="utf-8") as fout:
            for i in range(0, len(todo), batch_size):
                batch = todo[i: i + batch_size]
                prompts = [
                    SELF_CORRECT_PROMPT.format(
                        question=r["question"],
                        initial_answer=r["chains"][0]["text"] if r["chains"] else "",
                    )
                    for r in batch
                ]
                outputs = llm.generate(prompts, sampling)

                for rec, out in zip(batch, outputs):
                    correction_text = out.outputs[0].text
                    record = {
                        "problem_id": rec["problem_id"],
                        "dataset": rec["dataset"],
                        "question": rec["question"],
                        "answer": rec["answer"],
                        "initial_chain": rec["chains"][0]["text"] if rec["chains"] else "",
                        "corrected_chain": correction_text,
                        "sampling": {
                            "max_new_tokens": planb_cfg["max_new_tokens"],
                            "temperature": planb_cfg["temperature"],
                            "seed": planb_cfg.get("seed", cfg["inference"].get("seed")),
                        },
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved corrections -> {out_path}")
        del llm
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Compute CHR ----
    logger.info("\nComputing CHR...")
    if not out_path.exists():
        logger.error(f"No correction file found: {out_path}")
        sys.exit(1)
    all_corrections = load_jsonl(out_path)
    if not all_corrections:
        logger.error("No correction records found.")
        sys.exit(1)
    verifier = MathVerifier(numeric_tolerance=cfg["orm"]["numeric_tolerance"])

    n_initially_correct = 0
    n_correct_to_wrong = 0
    n_wrong_to_correct = 0
    n_both_correct = 0
    n_both_wrong = 0

    transition_records = []

    for rec in all_corrections:
        dataset = rec["dataset"]
        gt = rec["answer"]

        initial_correct = verifier.verify(rec["initial_chain"], gt, dataset)
        corrected_correct = verifier.verify(rec["corrected_chain"], gt, dataset)

        if initial_correct:
            n_initially_correct += 1
            if not corrected_correct:
                n_correct_to_wrong += 1
            else:
                n_both_correct += 1
        else:
            if corrected_correct:
                n_wrong_to_correct += 1
            else:
                n_both_wrong += 1

        transition_records.append({
            "problem_id": rec["problem_id"],
            "initial_correct": initial_correct,
            "corrected_correct": corrected_correct,
            "transition": (
                "C→C" if initial_correct and corrected_correct else
                "C→W" if initial_correct and not corrected_correct else
                "W→C" if not initial_correct and corrected_correct else
                "W→W"
            ),
        })

    chr_value = n_correct_to_wrong / max(n_initially_correct, 1)
    chr_pct = 100 * chr_value

    logger.info(f"\n{'='*50}")
    logger.info("PLAN B-1: SELF-CORRECTION CHR RESULTS")
    logger.info("="*50)
    logger.info(f"Total problems:       {len(all_corrections)}")
    logger.info(f"Initially correct:    {n_initially_correct}")
    logger.info(f"Transitions:")
    logger.info(f"  C→C (both correct): {n_both_correct}")
    logger.info(f"  C→W (harm):         {n_correct_to_wrong}  <- CHR numerator")
    logger.info(f"  W→C (benefit):      {n_wrong_to_correct}")
    logger.info(f"  W→W (both wrong):   {n_both_wrong}")
    logger.info(f"\nCHR = {n_correct_to_wrong}/{n_initially_correct} = {chr_pct:.2f}%")

    chr_threshold = cfg["plan_b1"]["chr_threshold"] * 100
    if chr_pct > chr_threshold:
        decision = f"✅ Plan B-1 VIABLE (CHR {chr_pct:.1f}% > threshold {chr_threshold:.0f}%)"
        decision += "\n   Self-correction gate is a meaningful problem. Proceed with Plan B-1."
    elif chr_pct < 2.0:
        decision = f"⬇️  Plan B-2 (CHR {chr_pct:.1f}% < 2%). Self-correction rarely hurts. -> Data filtering only."
    else:
        decision = f"🟡 Borderline (CHR {chr_pct:.1f}%). Monitor; may still work with better correction protocol."

    logger.info(f"\nDecision: {decision}")

    # Save result
    result_txt = out_dir / "planb1_chr_result.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("Plan B-1 CHR Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total: {len(all_corrections)}\n")
        f.write(f"Initially correct: {n_initially_correct}\n")
        f.write(f"C->W (harm): {n_correct_to_wrong}\n")
        f.write(f"W->C (benefit): {n_wrong_to_correct}\n")
        f.write(f"CHR: {chr_pct:.2f}%\n\n")
        f.write(f"Decision: {decision}\n")
    logger.info(f"\nSaved: {result_txt}")
    logger.info("\nStep 6 (Plan B-1 CHR) complete.")


if __name__ == "__main__":
    main()
