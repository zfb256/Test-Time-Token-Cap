#!/usr/bin/env python3
"""
Step 8 (v2): Resume-from-truncation continuation runs.

Review response (cost-accounting concern): the paper's savings formula
    avg budget = B_m + r * (B_l - B_m)
is only honest if the upgrade action *continues* the truncated medium trace
instead of rerunning from scratch. This script implements that action:

  - For medium traces that hit the token cap, re-prompt with
    (original prompt + medium text) and generate at most (B_l - B_m)
    additional tokens under the same greedy decoding.
  - For medium traces that terminated naturally (EOS), the resume action
    is a no-op: the medium answer is kept at zero extra cost.

Output: inference_long_resume.jsonl in the config's output_dir, with the
same record schema as 02_run_inference.py (chains[0].text holds the full
medium+continuation text) plus a "resume" block with continuation stats.

Comparing these resumed traces against the independent fixed-long rerun
(inference_long.jsonl) also quantifies how close "rerun" and "resume" are
under greedy decoding (see analysis/38_trace_prefix_overlap.py).

Note: re-tokenizing (prompt + generated text) may split the boundary
differently than the original token sequence; continuation token budgets
are therefore approximate at the +/- 1-2 token level.

Usage:
    python 08_run_continuation.py --config config.yaml \
        --medium-budget medium --long-budget long
"""

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_continuation")

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_inference_module():
    spec = importlib.util.spec_from_file_location(
        "run_inference", SCRIPT_DIR / "02_run_inference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path):
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                record = json.loads(line)
                pid = record["problem_id"]
                if pid in seen:
                    raise ValueError(f"{path}:{line_no}: duplicate problem_id={pid}")
                seen.add(pid)
                records.append(record)
    return records


def medium_hit_length(record, medium_cap):
    chains = record.get("chains", [])
    if len(chains) != 1:
        raise ValueError(
            f"problem_id={record.get('problem_id')}: expected one medium chain"
        )
    chain = chains[0]
    token_count = chain.get("token_count")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or not 0 <= token_count <= medium_cap
    ):
        raise ValueError(
            f"problem_id={record.get('problem_id')}: invalid medium token_count "
            f"{token_count!r}"
        )
    finish_reason = chain.get("finish_reason")
    if finish_reason not in {"stop", "length", None}:
        raise ValueError(
            f"problem_id={record.get('problem_id')}: invalid finish_reason "
            f"{finish_reason!r}"
        )
    return (
        finish_reason == "length"
        if finish_reason is not None
        else token_count >= medium_cap
    )


def validate_resume_record(source, resumed, medium_cap, long_cap, signature):
    problem_id = source["problem_id"]
    for field in ("problem_id", "dataset", "question", "answer"):
        if resumed.get(field) != source.get(field):
            raise ValueError(
                f"resume {field} mismatch for problem_id={problem_id}"
            )
    budget = resumed.get("budget", {})
    if (
        budget.get("max_new_tokens") != long_cap
        or budget.get("n_chains") != 1
        or budget.get("resume_signature") != signature
    ):
        raise ValueError(
            f"resume budget/signature mismatch for problem_id={problem_id}"
        )
    chains = resumed.get("chains", [])
    if len(chains) != 1:
        raise ValueError(
            f"resume chain-count mismatch for problem_id={problem_id}"
        )
    chain = chains[0]
    token_count = chain.get("token_count")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or not 0 <= token_count <= long_cap
        or not isinstance(chain.get("text"), str)
    ):
        raise ValueError(
            f"invalid resumed chain for problem_id={problem_id}"
        )
    if "token_ids" in chain and len(chain["token_ids"]) != token_count:
        raise ValueError(
            f"resumed token_ids mismatch for problem_id={problem_id}"
        )

    source_chain = source["chains"][0]
    info = resumed.get("resume", {})
    expected_continued = medium_hit_length(source, medium_cap)
    if info.get("continued") is not expected_continued:
        raise ValueError(
            f"resume continued flag mismatch for problem_id={problem_id}"
        )
    if info.get("medium_tokens") != source_chain["token_count"]:
        raise ValueError(
            f"resume medium token count mismatch for problem_id={problem_id}"
        )
    continuation_tokens = info.get("continuation_tokens")
    if (
        not isinstance(continuation_tokens, int)
        or isinstance(continuation_tokens, bool)
        or continuation_tokens < 0
        or continuation_tokens > long_cap - medium_cap
    ):
        raise ValueError(
            f"invalid continuation token count for problem_id={problem_id}"
        )
    if expected_continued:
        if (
            token_count != source_chain["token_count"] + continuation_tokens
            or not chain["text"].startswith(source_chain.get("text", ""))
        ):
            raise ValueError(
                f"resume is not a valid source continuation for "
                f"problem_id={problem_id}"
            )
        if "token_ids" in source_chain and (
            "token_ids" not in chain
            or chain["token_ids"][: len(source_chain["token_ids"])]
            != source_chain["token_ids"]
        ):
            raise ValueError(
                f"resumed token IDs do not preserve the source prefix for "
                f"problem_id={problem_id}"
            )
    elif continuation_tokens != 0 or chain != source_chain:
        raise ValueError(
            f"natural completion was not preserved for problem_id={problem_id}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--medium-budget", default="medium",
                        help="Budget name whose traces are resumed")
    parser.add_argument("--long-budget", default="long",
                        help="Budget name defining the extended token cap")
    parser.add_argument("--output-name", default="long_resume",
                        help="Writes inference_{output-name}.jsonl")
    parser.add_argument("--output-dir", default=None,
                        help="Override paths.output_dir without editing config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir

    out_dir = Path(cfg["paths"]["output_dir"])
    ri = _load_inference_module()
    medium_cfg = dict(cfg["inference"]["budgets"][args.medium_budget])
    long_cfg = dict(cfg["inference"]["budgets"][args.long_budget])
    base_seed = cfg["inference"].get("seed")
    if base_seed is None:
        parser.error("inference.seed is required for reproducible continuation")
    medium_cfg["seed"] = ri.effective_budget_seed(
        base_seed, args.medium_budget, medium_cfg
    )
    long_cfg["seed"] = ri.effective_budget_seed(
        base_seed, args.long_budget, long_cfg
    )
    b_m = medium_cfg["max_new_tokens"]
    b_l = long_cfg["max_new_tokens"]
    if b_l <= b_m:
        logger.error(f"Long budget ({b_l}) must exceed medium budget ({b_m}).")
        sys.exit(1)
    for name, budget in (
        (args.medium_budget, medium_cfg),
        (args.long_budget, long_cfg),
    ):
        if budget.get("n_chains") != 1 or float(budget.get("temperature", 0)) != 0:
            parser.error(
                f"{name} must use one greedy chain for deterministic continuation"
            )

    medium_path = out_dir / f"inference_{args.medium_budget}.jsonl"
    if not medium_path.exists():
        logger.error(f"Missing {medium_path}. Run 02_run_inference.py first.")
        sys.exit(1)
    records = load_jsonl(medium_path)
    for record in records:
        budget = record.get("budget", {})
        policy = budget.get("seed_policy", {})
        expected = (
            b_m,
            medium_cfg["n_chains"],
            medium_cfg["temperature"],
            medium_cfg["top_p"],
            medium_cfg["seed"],
        )
        actual = (
            budget.get("max_new_tokens"),
            budget.get("n_chains"),
            budget.get("temperature"),
            budget.get("top_p"),
            budget.get("seed"),
        )
        if actual != expected or len(record.get("chains", [])) != 1:
            raise ValueError(
                f"stale or inconsistent medium record "
                f"problem_id={record.get('problem_id')}: {actual} != {expected}"
            )
        if (
            policy.get("engine_seed") != base_seed
            or policy.get("request_seed") != medium_cfg["seed"]
            or policy.get("mode") != "per_request_single_chain"
        ):
            raise ValueError(
                f"stale or inconsistent medium seed policy "
                f"problem_id={record.get('problem_id')}: {policy}"
            )
        chain = record["chains"][0]
        if not isinstance(chain.get("text"), str):
            raise ValueError(
                f"medium text is not a string "
                f"problem_id={record.get('problem_id')}"
            )
        medium_hit_length(record, b_m)
        if chain.get("chain_index") not in (None, 0):
            raise ValueError(
                f"invalid medium chain_index "
                f"problem_id={record.get('problem_id')}"
            )
        if chain.get("request_seed") not in (None, medium_cfg["seed"]):
            raise ValueError(
                f"invalid medium request_seed "
                f"problem_id={record.get('problem_id')}"
            )
        if "token_ids" in chain and (
            not isinstance(chain["token_ids"], list)
            or len(chain["token_ids"]) != chain["token_count"]
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool)
                for token_id in chain["token_ids"]
            )
        ):
            raise ValueError(
                f"invalid medium token_ids "
                f"problem_id={record.get('problem_id')}"
            )
    logger.info(f"Loaded {len(records)} medium records from {medium_path}")
    medium_ids = {record["problem_id"] for record in records}
    resume_signature = {
        "source_budget": args.medium_budget,
        "target_budget": args.long_budget,
        "source_cap": b_m,
        "target_cap": b_l,
        "model": str(cfg["inference"]["model"]),
        "prompt_template": cfg["inference"].get(
            "prompt_template", "qwen_math"
        ),
        "engine_seed": base_seed,
        "continuation_seed": long_cfg["seed"],
    }

    out_path = out_dir / f"inference_{args.output_name}.jsonl"
    if out_path.exists():
        existing = load_jsonl(out_path)
        existing_map = {record["problem_id"]: record for record in existing}
        if set(existing_map) == medium_ids and len(existing) == len(records):
            try:
                for source in records:
                    validate_resume_record(
                        source,
                        existing_map[source["problem_id"]],
                        b_m,
                        b_l,
                        resume_signature,
                    )
            except ValueError as error:
                raise SystemExit(
                    f"{out_path} exists but failed validation: {error}"
                ) from error
            logger.info(f"{out_path} is complete and valid; skipping.")
            return
        raise SystemExit(
            f"{out_path} exists but is incomplete or invalid "
            f"({len(existing)} rows, {len(existing_map)} unique IDs). "
            "Refusing to silently skip it."
        )

    template = cfg["inference"].get("prompt_template", "qwen_math")

    truncated = [rec for rec in records if medium_hit_length(rec, b_m)]
    logger.info(f"Truncated medium traces: {len(truncated)} / {len(records)}")

    results = {}
    if truncated:
        from vllm import LLM, SamplingParams
        inf_cfg = cfg["inference"]
        llm = LLM(
            model=inf_cfg["model"],
            seed=int(base_seed),
            tensor_parallel_size=inf_cfg["tensor_parallel_size"],
            dtype=inf_cfg["dtype"],
            gpu_memory_utilization=inf_cfg["gpu_memory_utilization"],
            max_model_len=inf_cfg.get("max_model_len", 2048),
            trust_remote_code=inf_cfg.get("trust_remote_code", False),
        )
        sampling = SamplingParams(
            max_tokens=b_l - b_m,
            n=1,
            temperature=long_cfg.get("temperature", 0.0),
            top_p=long_cfg.get("top_p", 1.0),
            seed=long_cfg["seed"],
        )
        prompts = [
            ri.build_prompt(rec["question"], template) + rec["chains"][0]["text"]
            for rec in truncated
        ]
        batch_size = inf_cfg.get("batch_size", 32)
        t0 = time.time()
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            batch_records = truncated[i: i + batch_size]
            outputs = llm.generate(batch_prompts, sampling)
            if len(outputs) != len(batch_records):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for "
                    f"{len(batch_records)} continuation prompts"
                )
            for rec, out in zip(batch_records, outputs):
                if len(out.outputs) != 1:
                    raise RuntimeError(
                        f"problem_id={rec['problem_id']}: expected one continuation, "
                        f"received {len(out.outputs)}"
                    )
                comp = out.outputs[0]
                results[rec["problem_id"]] = {
                    "text": comp.text,
                    "token_count": len(comp.token_ids),
                    "finish_reason": comp.finish_reason,
                    "stop_reason": comp.stop_reason,
                    "token_ids": [int(token_id) for token_id in comp.token_ids],
                }
        logger.info(f"Continuation done in {time.time() - t0:.1f}s")
    if set(results) != {record["problem_id"] for record in truncated}:
        raise RuntimeError("continuation results do not match truncated inputs")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            chain = dict(rec["chains"][0])
            cont = results.get(rec["problem_id"])
            if cont is not None:
                full_text = chain["text"] + cont["text"]
                total_tokens = chain["token_count"] + cont["token_count"]
                resume_info = {
                    "continued": True,
                    "medium_tokens": chain["token_count"],
                    "continuation_tokens": cont["token_count"],
                }
                chain = {
                    "text": full_text,
                    "token_count": total_tokens,
                    "finish_reason": cont.get("finish_reason"),
                    "stop_reason": cont.get("stop_reason"),
                    "chain_index": 0,
                    "request_seed": long_cfg["seed"],
                }
                if "token_ids" in rec["chains"][0]:
                    chain["token_ids"] = (
                        list(rec["chains"][0]["token_ids"])
                        + cont["token_ids"]
                    )
            else:
                resume_info = {
                    "continued": False,
                    "medium_tokens": chain["token_count"],
                    "continuation_tokens": 0,
                }
            out_rec = {
                "problem_id": rec["problem_id"],
                "dataset": rec["dataset"],
                "question": rec["question"],
                "answer": rec["answer"],
                "math_level": rec.get("math_level", 0),
                "chains": [chain],
                "budget": {
                    "max_new_tokens": b_l,
                    "n_chains": 1,
                    "temperature": long_cfg.get("temperature", 0.0),
                    "top_p": long_cfg.get("top_p", 1.0),
                    "seed": long_cfg["seed"],
                    "mode": f"resume_{args.medium_budget}_to_{args.long_budget}",
                    "resume_signature": resume_signature,
                },
                "resume": resume_info,
            }
            validate_resume_record(
                rec, out_rec, b_m, b_l, resume_signature
            )
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)

    logger.info(f"Saved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
