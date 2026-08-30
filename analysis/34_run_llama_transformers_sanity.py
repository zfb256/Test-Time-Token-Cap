"""
Run the Llama-3.1 second-model sanity check with Transformers.

The installed vLLM (0.4.2) does not support Llama-3.1's llama3 RoPE
configuration. This script keeps the output schema identical to
pipeline/02_run_inference.py while using Transformers, which supports that
configuration in the current environment.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as hf_modeling_utils

if hf_modeling_utils.ALL_PARALLEL_STYLES is None:
    # Transformers >=4.52 exposes Llama tensor-parallel metadata even when
    # torch<2.5 cannot use TP. We run single-GPU inference, so the metadata is
    # only passing a post_init validation check.
    hf_modeling_utils.ALL_PARALLEL_STYLES = {
        "colwise",
        "rowwise",
        "colwise_rep",
    }


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent / "pipeline"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def resolve_path(path: str, base: Path = PHASE1_DIR) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(out_path: Path, expected_n_chains: int = 1) -> set:
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(row.get("chains", [])) >= expected_n_chains:
                done.add(row["problem_id"])
    return done


def budget_seed(base_seed: int, budget_name: str, budget_cfg: Dict) -> int:
    if "seed" in budget_cfg:
        return int(budget_cfg["seed"])
    offsets = {"short": 0, "medium": 1000, "long": 2000, "sample": 3000}
    if budget_name in offsets:
        return int(base_seed) + offsets[budget_name]
    digest = hashlib.sha256(budget_name.encode("utf-8")).hexdigest()
    return int(base_seed) + 4000 + int(digest[:8], 16) % 100000


def build_prompts(tokenizer, problems: List[Dict]) -> List[str]:
    prompts = []
    for problem in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem["question"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def generate_batch(tokenizer, model, prompts: List[str], budget_cfg: Dict) -> List[Dict]:
    n_chains = int(budget_cfg["n_chains"])
    max_new_tokens = int(budget_cfg["max_new_tokens"])
    temperature = float(budget_cfg["temperature"])
    top_p = float(budget_cfg["top_p"])
    do_sample = temperature > 0.0 and n_chains > 1

    expanded_prompts = []
    owner_idx = []
    for i, prompt in enumerate(prompts):
        for _ in range(n_chains):
            expanded_prompts.append(prompt)
            owner_idx.append(i)

    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    prompt_width = int(inputs["input_ids"].shape[1])

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    grouped = [[] for _ in prompts]
    for out_idx, output_ids in enumerate(outputs):
        new_ids = output_ids[prompt_width:]
        if tokenizer.eos_token_id in new_ids:
            eos_positions = (new_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                new_ids = new_ids[: int(eos_positions[0])]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        grouped[owner_idx[out_idx]].append(
            {
                "text": text,
                "token_count": int(new_ids.numel()),
            }
        )
    return grouped


def write_records(out_path: Path, problems: List[Dict], chains_by_problem: List[List[Dict]], budget_cfg: Dict):
    with open(out_path, "a", encoding="utf-8") as f:
        for problem, chains in zip(problems, chains_by_problem):
            record = {
                "problem_id": problem["problem_id"],
                "dataset": problem["dataset"],
                "question": problem["question"],
                "answer": problem["answer"],
                "math_level": problem.get("math_level", 0),
                "chains": chains,
                "budget": {
                    "max_new_tokens": budget_cfg["max_new_tokens"],
                    "n_chains": budget_cfg["n_chains"],
                    "temperature": budget_cfg["temperature"],
                    "top_p": budget_cfg["top_p"],
                    "seed": budget_cfg.get("seed"),
                    "engine": "transformers",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHASE1_DIR / "config_llama_validation.yaml"))
    parser.add_argument("--budget", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inf_cfg = cfg["inference"]
    if args.budget not in inf_cfg["budgets"]:
        raise SystemExit(f"Unknown budget {args.budget}; available: {list(inf_cfg['budgets'])}")

    model_path = resolve_path(inf_cfg["model"])
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_cfg = dict(inf_cfg["budgets"][args.budget])
    if inf_cfg.get("seed") is not None and "seed" not in budget_cfg:
        budget_cfg["seed"] = budget_seed(int(inf_cfg["seed"]), args.budget, budget_cfg)

    torch.manual_seed(int(budget_cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(budget_cfg.get("seed", 42)))

    problems = load_jsonl(data_dir / "all_problems.jsonl")
    out_path = out_dir / f"inference_{args.budget}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = load_done_ids(out_path, int(budget_cfg["n_chains"])) if args.resume else set()
    todo = [p for p in problems if p["problem_id"] not in done]
    logger.info("Loaded %d problems; running %d, skipping %d", len(problems), len(todo), len(done))
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    batch_size = args.batch_size or int(inf_cfg.get("batch_size", 4))
    batch_size = max(1, min(batch_size, 16))
    logger.info("Model loaded. batch_size=%d budget=%s", batch_size, args.budget)

    start = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i: i + batch_size]
        prompts = build_prompts(tokenizer, batch)
        chains = generate_batch(tokenizer, model, prompts, budget_cfg)
        write_records(out_path, batch, chains, budget_cfg)
        logger.info("Saved %d/%d -> %s", min(i + len(batch), len(todo)), len(todo), out_path)

    logger.info("Done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
