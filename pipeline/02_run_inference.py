"""
Step 2: Run inference at all 4 budget levels using vLLM.

For each problem, generates:
  - short:  3 chains × 128 tokens (temp=0.75)  -> answer consistency + quick ORM
  - medium: 1 chain  × 384 tokens (greedy)
  - long:   1 chain  × 768 tokens (greedy)      -> PRIMARY utility signal
  - sample: 4 chains × 256 tokens (temp=0.75)   -> for later diversity analysis

All outputs saved as JSONL (one record per problem). Resumable: skips already-done
problems based on existing output files.

Also collects initial answer for Plan B-1 (self-correction) at no extra cost —
just saves the "long" greedy output as the initial answer for CHR computation.

Outputs:
  {output_dir}/inference_short.jsonl
  {output_dir}/inference_medium.jsonl
  {output_dir}/inference_long.jsonl
  {output_dir}/inference_sample.jsonl

Run:
  python 02_run_inference.py --config config.yaml
  python 02_run_inference.py --config config.yaml --budget short   # one budget only
  python 02_run_inference.py --config config.yaml --resume         # skip done
"""

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Dict

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from reproducibility import (
    deterministic_chain_seed,
    model_snapshot_sha256,
    runtime_versions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MAX_IDENTICAL_CHAIN_FRACTION = 0.20
CHECKPOINT_PROBLEMS = 10


def request_seed(seed, n_chains, temperature):
    """Legacy single-request seed policy.

    Multi-chain sampled runs use ``deterministic_chain_seed`` instead: every
    chain is issued as its own request with a stable request-specific seed.
    """
    sampled = int(n_chains) > 1 and float(temperature) > 0
    return None if sampled else seed


def effective_budget_seed(base_seed, budget_name, budget_cfg):
    """Return the explicit or stable name-derived seed for one budget."""
    if budget_cfg.get("seed") is not None:
        return int(budget_cfg["seed"])
    if base_seed is None:
        return None
    default_offsets = {
        "short": 0,
        "medium": 1000,
        "long": 2000,
        "sample": 3000,
    }
    if budget_name in default_offsets:
        budget_offset = default_offsets[budget_name]
    else:
        digest = hashlib.sha256(budget_name.encode("utf-8")).hexdigest()
        budget_offset = 4000 + int(digest[:8], 16) % 100000
    return int(base_seed) + budget_offset


def seed_policy(
    seed,
    n_chains,
    temperature,
    engine_seed=None,
    deterministic_sampling=True,
):
    sampled = int(n_chains) > 1 and float(temperature) > 0
    if sampled and deterministic_sampling:
        mode = "per_problem_per_chain_sha256"
        request_seed = "derived"
    elif sampled:
        mode = "engine_only_multi_sample"
        request_seed = None
    else:
        mode = "per_request_single_chain"
        request_seed = seed
    return {
        "engine_seed": seed if engine_seed is None else engine_seed,
        "request_seed": request_seed,
        "base_seed": seed,
        "mode": mode,
    }


def assert_chain_diversity(results, n_chains, temperature, threshold=MAX_IDENTICAL_CHAIN_FRACTION):
    """Reject sampled artifacts whose within-record chains have collapsed."""
    if int(n_chains) <= 1 or float(temperature) <= 0 or not results:
        return
    identical = sum(
        len({chain.get("text", "") for chain in row.get("chains", [])}) == 1
        for row in results
    )
    fraction = identical / len(results)
    if fraction > threshold:
        raise RuntimeError(
            "sampled-chain diversity check failed: "
            f"{identical}/{len(results)} ({fraction:.1%}) records have all "
            f"{n_chains} chains identical; maximum allowed is {threshold:.0%}"
        )

# ---- Prompt templates ----
QWEN_MATH_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

QWEN_MATH_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

LLAMA3_MATH_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
# Intentionally explicit instead of tokenizer.apply_chat_template: the local
# Llama tokenizer injects a fixed "Today Date: 26 Jul 2024" preamble. The
# existing greedy medium/long artifacts were generated without that preamble,
# so v2 sampled controls must preserve this exact template for comparability.


R1_MATH_USER_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# DeepSeek-R1-Distill chat format: no system prompt (per model card), the
# assistant turn starts with an opening <think> tag.
R1_MATH_CHAT_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>{question}"
    "<｜Assistant｜><think>\n"
)


def build_prompt(question: str, template: str = "qwen_math") -> str:
    if template == "qwen_math":
        return QWEN_MATH_CHAT_TEMPLATE.format(
            system=QWEN_MATH_SYSTEM, question=question
        )
    elif template == "llama3_math":
        return LLAMA3_MATH_CHAT_TEMPLATE.format(
            system=QWEN_MATH_SYSTEM, question=question
        )
    elif template == "r1_math":
        return R1_MATH_CHAT_TEMPLATE.format(question=question + R1_MATH_USER_SUFFIX)
    else:
        return f"Question: {question}\nAnswer:"


# ------------------------------------------------------------------
# vLLM inference engine
# ------------------------------------------------------------------

class VLLMInferenceEngine:
    def __init__(self, cfg: dict):
        from vllm import LLM, SamplingParams
        self.LLM = LLM
        self.SamplingParams = SamplingParams
        inf_cfg = cfg["inference"]

        self.model_id = str(inf_cfg["model"])
        logger.info("Fingerprinting model snapshot before GPU initialization.")
        self.model_snapshot_sha256 = model_snapshot_sha256(self.model_id)
        self.runtime_versions = runtime_versions()
        logger.info(f"Loading model: {self.model_id}")
        self.dtype = inf_cfg["dtype"]
        self.max_model_len = inf_cfg.get("max_model_len", 2048)
        self.trust_remote_code = inf_cfg.get("trust_remote_code", False)
        self.llm = LLM(
            model=inf_cfg["model"],
            seed=int(inf_cfg.get("seed", 0)),
            tensor_parallel_size=inf_cfg["tensor_parallel_size"],
            dtype=self.dtype,
            gpu_memory_utilization=inf_cfg["gpu_memory_utilization"],
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
        )
        self.engine_seed = int(inf_cfg.get("seed", 0))
        self.template = inf_cfg.get("prompt_template", "qwen_math")
        self.batch_size = inf_cfg.get("batch_size", 32)
        self.collect_logprobs = inf_cfg.get("collect_logprobs", False)
        self.num_logprobs = inf_cfg.get("num_logprobs", 2)
        self.store_token_ids = bool(inf_cfg.get("store_token_ids", False))
        self.deterministic_sampling = bool(
            inf_cfg.get("deterministic_sampling", True)
        )
        logger.info("Model loaded.")

    @staticmethod
    def _logprob_value(item) -> float:
        if hasattr(item, "logprob"):
            return float(item.logprob)
        return float(item)

    @classmethod
    def _summarize_logprobs(cls, comp) -> Dict:
        token_logprobs = getattr(comp, "logprobs", None) or []
        margins = []
        chosen_logprobs = []

        for token_lp in token_logprobs:
            if not token_lp:
                continue
            vals = sorted(
                [cls._logprob_value(v) for v in token_lp.values()],
                reverse=True,
            )
            if vals:
                chosen_logprobs.append(vals[0])
            if len(vals) >= 2:
                margins.append(vals[0] - vals[1])

        avg_logprob = (
            float(getattr(comp, "cumulative_logprob") / max(len(comp.token_ids), 1))
            if getattr(comp, "cumulative_logprob", None) is not None
            else (sum(chosen_logprobs) / len(chosen_logprobs) if chosen_logprobs else None)
        )
        avg_margin = sum(margins) / len(margins) if margins else None
        return {
            "avg_logprob": avg_logprob,
            "avg_logprob_margin": avg_margin,
        }

    def generate_batch(
        self,
        problems: List[Dict],
        max_new_tokens: int,
        n_chains: int,
        temperature: float,
        top_p: float,
        collect_logprobs: bool,
        seed: int = None,
    ) -> List[Dict]:
        """
        Generate n_chains completions per problem.
        Returns list of dicts with keys:
          problem_id, chains: [{"text": ..., "token_count": ...}, ...]
        """
        prompts = [build_prompt(p["question"], self.template) for p in problems]
        base_sampling_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if collect_logprobs:
            base_sampling_kwargs["logprobs"] = self.num_logprobs

        sampled = int(n_chains) > 1 and float(temperature) > 0
        deterministic_multi = sampled and self.deterministic_sampling
        common_request_seed = request_seed(seed, n_chains, temperature)
        if deterministic_multi and seed is None:
            raise ValueError(
                "deterministic_sampling requires a configured budget seed"
            )

        # vLLM processes in one shot but we batch for memory safety
        all_results = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]
            batch_problems = problems[i: i + self.batch_size]
            if deterministic_multi:
                request_prompts = []
                request_params = []
                for prompt, problem in zip(batch_prompts, batch_problems):
                    for chain_index in range(n_chains):
                        kwargs = dict(base_sampling_kwargs)
                        kwargs["n"] = 1
                        kwargs["seed"] = deterministic_chain_seed(
                            seed, problem["problem_id"], chain_index
                        )
                        request_prompts.append(prompt)
                        request_params.append(self.SamplingParams(**kwargs))
                flat_outputs = self.llm.generate(request_prompts, request_params)
                expected = len(batch_problems) * n_chains
                if len(flat_outputs) != expected:
                    raise RuntimeError(
                        "vLLM returned an unexpected number of deterministic "
                        f"chain outputs: {len(flat_outputs)} for {expected} requests"
                    )
                outputs = [
                    flat_outputs[j : j + n_chains]
                    for j in range(0, len(flat_outputs), n_chains)
                ]
            else:
                sampling_kwargs = dict(base_sampling_kwargs)
                sampling_kwargs["n"] = n_chains
                if common_request_seed is not None:
                    sampling_kwargs["seed"] = common_request_seed
                sampling = self.SamplingParams(**sampling_kwargs)
                outputs = self.llm.generate(batch_prompts, sampling)
            if len(outputs) != len(batch_problems):
                raise RuntimeError(
                    "vLLM returned an unexpected number of request outputs: "
                    f"{len(outputs)} for {len(batch_problems)} prompts"
                )
            for prob, out in zip(batch_problems, outputs):
                completions = (
                    [request_output.outputs[0] for request_output in out]
                    if deterministic_multi
                    else out.outputs
                )
                if len(completions) != n_chains:
                    raise RuntimeError(
                        f"problem_id={prob['problem_id']}: expected {n_chains} "
                        f"chains, received {len(completions)}"
                    )
                chains = []
                for chain_index, comp in enumerate(completions):
                    chain_seed = (
                        deterministic_chain_seed(
                            seed, prob["problem_id"], chain_index
                        )
                        if deterministic_multi
                        else common_request_seed
                    )
                    chain_record = {
                        "text": comp.text,
                        "token_count": len(comp.token_ids),
                        # token_count alone cannot distinguish an EOS that
                        # happens exactly at the cap from length truncation.
                        # Persist vLLM's termination metadata for all future
                        # mechanism analyses.
                        "finish_reason": comp.finish_reason,
                        "stop_reason": comp.stop_reason,
                        "chain_index": chain_index,
                        "request_seed": chain_seed,
                    }
                    if self.store_token_ids:
                        chain_record["token_ids"] = [
                            int(token_id) for token_id in comp.token_ids
                        ]
                    if collect_logprobs:
                        chain_record.update(self._summarize_logprobs(comp))
                    chains.append(chain_record)
                all_results.append({
                    "problem_id": prob["problem_id"],
                    "chains": chains,
                })
        return all_results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def load_problems(data_dir: Path) -> List[Dict]:
    combined = data_dir / "all_problems.jsonl"
    if not combined.exists():
        logger.error(f"File not found: {combined}. Run 01_download_data.py first.")
        sys.exit(1)
    problems = []
    seen = set()
    with open(combined, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            problem = json.loads(line)
            missing = {
                field
                for field in ("problem_id", "dataset", "question", "answer")
                if field not in problem
            }
            if missing:
                raise ValueError(
                    f"{combined}:{line_no}: missing required fields "
                    f"{sorted(missing)}"
                )
            problem_id = problem["problem_id"]
            if problem_id in seen:
                raise ValueError(
                    f"{combined}:{line_no}: duplicate problem_id={problem_id}"
                )
            seen.add(problem_id)
            problems.append(problem)
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def run_signature(engine) -> Dict:
    """Metadata that must match before an existing row is safe to resume."""
    return {
        "model": str(engine.model_id),
        "model_snapshot_sha256": engine.model_snapshot_sha256,
        "runtime_versions": engine.runtime_versions,
        "prompt_template": engine.template,
        "prompt_sha256": hashlib.sha256(
            build_prompt("", engine.template).encode("utf-8")
        ).hexdigest(),
        "store_token_ids": bool(engine.store_token_ids),
        "dtype": engine.dtype,
        "max_model_len": engine.max_model_len,
        "trust_remote_code": engine.trust_remote_code,
    }


def validate_budget_config(budget_cfg: Dict) -> None:
    """Reject malformed decoding budgets before launching an expensive run."""
    max_tokens = budget_cfg.get("max_new_tokens")
    n_chains = budget_cfg.get("n_chains")
    temperature = budget_cfg.get("temperature")
    top_p = budget_cfg.get("top_p")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        not isinstance(n_chains, int)
        or isinstance(n_chains, bool)
        or n_chains < 1
    ):
        raise ValueError("n_chains must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise ValueError("temperature must be a finite non-negative number")
    if (
        not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("top_p must be finite and in (0, 1]")
    seed = budget_cfg.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer or null")


def valid_chain_record(
    chain: Dict,
    chain_index: int,
    problem_id,
    budget_cfg: Dict,
    seed_policy_metadata: Dict,
    signature: Dict,
) -> bool:
    """Return whether one persisted chain is safe to reuse."""
    if not isinstance(chain, dict) or not isinstance(chain.get("text"), str):
        return False
    token_count = chain.get("token_count")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or not 0 <= token_count <= budget_cfg["max_new_tokens"]
        or chain.get("chain_index") != chain_index
    ):
        return False
    if chain.get("finish_reason") not in {"stop", "length"}:
        return False

    token_ids = chain.get("token_ids")
    if token_ids is None:
        if signature.get("store_token_ids"):
            return False
    elif (
        not isinstance(token_ids, list)
        or len(token_ids) != token_count
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in token_ids
        )
    ):
        return False

    mode = seed_policy_metadata.get("mode")
    if mode == "per_problem_per_chain_sha256":
        base_seed = seed_policy_metadata.get("base_seed")
        if base_seed is None:
            return False
        expected_seed = deterministic_chain_seed(
            base_seed, problem_id, chain_index
        )
    elif mode == "per_request_single_chain":
        expected_seed = seed_policy_metadata.get("request_seed")
    elif mode == "engine_only_multi_sample":
        expected_seed = None
    else:
        return False
    return chain.get("request_seed") == expected_seed


def load_completed_records(
    out_path: Path,
    expected_n_chains: int = 1,
    expected_budget_cfg: Dict = None,
    expected_engine_seed: int = None,
    expected_problems: Dict[str, Dict] = None,
    expected_seed_policy: Dict = None,
    expected_run_signature: Dict = None,
) -> Dict[str, Dict]:
    """Load valid completed records, deduplicated by problem_id.

    Interrupted runs can leave malformed lines or records with fewer chains
    than requested. Keeping only complete records prevents --resume from
    appending a second row for the same problem and silently violating the
    one-record-per-problem schema.
    """
    completed = {}
    rejected = 0
    malformed = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                    chains = r.get("chains", [])
                    budget = r.get("budget", {})
                    policy = budget.get("seed_policy", {})
                    budget_matches = True
                    if expected_budget_cfg is not None:
                        budget_matches = all(
                            budget.get(field) == expected_budget_cfg.get(field)
                            for field in (
                                "max_new_tokens",
                                "n_chains",
                                "temperature",
                                "top_p",
                                "seed",
                            )
                        )
                    seed_matches = True
                    if expected_engine_seed is not None:
                        seed_matches = (
                            policy.get("engine_seed") == expected_engine_seed
                        )
                    if expected_seed_policy is not None:
                        seed_matches = seed_matches and all(
                            policy.get(field)
                            == expected_seed_policy.get(field)
                            for field in (
                                "engine_seed",
                                "request_seed",
                                "base_seed",
                                "mode",
                            )
                        )
                    signature_matches = True
                    if expected_run_signature is not None:
                        actual_signature = budget.get("run_signature", {})
                        signature_matches = all(
                            actual_signature.get(field) == value
                            for field, value in expected_run_signature.items()
                        )
                    problem_matches = True
                    if expected_problems is not None:
                        expected = expected_problems.get(r.get("problem_id"))
                        problem_matches = expected is not None and all(
                            r.get(field) == expected.get(field)
                            for field in ("dataset", "question", "answer")
                        )
                        problem_matches = problem_matches and (
                            r.get("math_level", 0)
                            == expected.get("math_level", 0)
                        )
                    chains_match = True
                    if (
                        expected_budget_cfg is not None
                        and expected_seed_policy is not None
                        and expected_run_signature is not None
                    ):
                        chains_match = all(
                            valid_chain_record(
                                chain,
                                chain_index,
                                r.get("problem_id"),
                                expected_budget_cfg,
                                expected_seed_policy,
                                expected_run_signature,
                            )
                            for chain_index, chain in enumerate(chains)
                        )
                    if (
                        len(chains) == expected_n_chains
                        and budget_matches
                        and seed_matches
                        and signature_matches
                        and problem_matches
                        and chains_match
                    ):
                        completed[r["problem_id"]] = r
                    else:
                        rejected += 1
                except (
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    malformed += 1
                    logger.warning(
                        "Ignoring malformed resume row %s:%d: %s",
                        out_path,
                        line_no,
                        exc,
                    )
    if rejected:
        logger.warning(
            "Rejected %d incompatible or incomplete resume rows from %s",
            rejected,
            out_path,
        )
    if malformed:
        logger.warning(
            "Ignored %d malformed resume rows from %s",
            malformed,
            out_path,
        )
    return completed


def load_done_ids(out_path: Path, expected_n_chains: int = 1) -> set:
    """Backward-compatible helper used by external checks."""
    return set(load_completed_records(out_path, expected_n_chains))


def write_completed_records(out_path, problems, completed):
    """Atomically persist completed rows in canonical dataset order."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            record = completed.get(problem["problem_id"])
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)


def run_budget(
    engine: "VLLMInferenceEngine",
    problems: List[Dict],
    budget_cfg: Dict,
    out_path: Path,
    resume: bool,
):
    """Run inference for one budget level, save JSONL."""
    validate_budget_config(budget_cfg)
    problem_ids = [p.get("problem_id") for p in problems]
    if None in problem_ids or len(set(problem_ids)) != len(problem_ids):
        raise ValueError("problems must have unique, non-missing problem_id values")
    expected_problems = {p["problem_id"]: p for p in problems}
    effective_seed_policy = seed_policy(
        budget_cfg.get("seed"),
        budget_cfg["n_chains"],
        budget_cfg["temperature"],
        engine_seed=getattr(engine, "engine_seed", None),
        deterministic_sampling=getattr(engine, "deterministic_sampling", True),
    )
    signature = run_signature(engine)
    completed = (
        load_completed_records(
            out_path,
            budget_cfg["n_chains"],
            expected_budget_cfg=budget_cfg,
            expected_problems=expected_problems,
            expected_seed_policy=effective_seed_policy,
            expected_run_signature=signature,
        )
        if resume
        else {}
    )
    done_ids = set(completed)
    todo = [p for p in problems if p["problem_id"] not in done_ids]

    if not todo:
        # Canonicalize even a complete resume file: malformed/stale/duplicate
        # lines may coexist with one valid row for every problem, in which
        # case merely skipping would preserve a structurally invalid JSONL.
        assert_chain_diversity(
            list(completed.values()),
            budget_cfg["n_chains"],
            budget_cfg["temperature"],
        )
        write_completed_records(out_path, problems, completed)
        logger.info(
            f"  All {len(problems)} problems validated; canonicalized output."
        )
        return

    logger.info(f"  Running {len(todo)} problems (skipping {len(done_ids)} done)")

    for start in range(0, len(todo), CHECKPOINT_PROBLEMS):
        batch = todo[start : start + CHECKPOINT_PROBLEMS]
        results = engine.generate_batch(
            batch,
            max_new_tokens=budget_cfg["max_new_tokens"],
            n_chains=budget_cfg["n_chains"],
            temperature=budget_cfg["temperature"],
            top_p=budget_cfg["top_p"],
            collect_logprobs=budget_cfg.get(
                "collect_logprobs", engine.collect_logprobs
            ),
            seed=budget_cfg.get("seed"),
        )
        result_ids = [result.get("problem_id") for result in results]
        batch_ids = [problem["problem_id"] for problem in batch]
        if (
            len(result_ids) != len(set(result_ids))
            or set(result_ids) != set(batch_ids)
        ):
            raise RuntimeError(
                "inference results do not match requested problem IDs"
            )
        prob_map = {problem["problem_id"]: problem for problem in batch}
        for result in results:
            pid = result["problem_id"]
            problem = prob_map[pid]
            record = {
                "problem_id": pid,
                "dataset": problem["dataset"],
                "question": problem["question"],
                "answer": problem["answer"],
                "math_level": problem.get("math_level", 0),
                "chains": result["chains"],
                "budget": {
                    "max_new_tokens": budget_cfg["max_new_tokens"],
                    "n_chains": budget_cfg["n_chains"],
                    "temperature": budget_cfg["temperature"],
                    "top_p": budget_cfg["top_p"],
                    "seed": budget_cfg.get("seed"),
                    "seed_policy": effective_seed_policy,
                    "deterministic_sampling": getattr(
                        engine, "deterministic_sampling", True
                    ),
                    "run_signature": signature,
                },
            }
            if len(record["chains"]) != budget_cfg["n_chains"] or not all(
                valid_chain_record(
                    chain,
                    chain_index,
                    pid,
                    budget_cfg,
                    effective_seed_policy,
                    signature,
                )
                for chain_index, chain in enumerate(record["chains"])
            ):
                raise RuntimeError(
                    f"engine returned an invalid chain record for problem_id={pid}"
                )
            completed[pid] = record
        write_completed_records(out_path, problems, completed)
        logger.info("  Checkpointed %d/%d records", len(completed), len(problems))

    assert_chain_diversity(
        list(completed.values()),
        budget_cfg["n_chains"],
        budget_cfg["temperature"],
    )
    logger.info(f"  Saved {len(todo)} new records -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--budget",
        nargs="+",
        default=["all"],
        help="Which budget level(s) to run. Use 'all' or any key under inference.budgets.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-done problems")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N problems (intended for protocol smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override paths.output_dir without editing the experiment config.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override paths.data_dir without editing the experiment config.",
    )
    parser.add_argument(
        "--engine-seed",
        type=int,
        default=None,
        help="Override inference.seed (useful for an independent same-cap control).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature for the selected budget(s).",
    )
    args = parser.parse_args()

    if args.temperature is not None and (
        not math.isfinite(args.temperature) or args.temperature < 0
    ):
        parser.error("--temperature must be finite and non-negative")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.data_dir is not None:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.engine_seed is not None:
        cfg["inference"]["seed"] = args.engine_seed

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(data_dir)
    if not problems:
        logger.error("No problems found in all_problems.jsonl.")
        sys.exit(1)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        problems = problems[: args.limit]
        logger.info("Smoke-test limit active: first %d problems", len(problems))

    configured_budgets = list(cfg["inference"]["budgets"].keys())
    if "all" in args.budget:
        if len(args.budget) != 1:
            parser.error("'all' cannot be combined with named budgets")
        budgets_to_run = configured_budgets
    else:
        budgets_to_run = args.budget
    unknown = set(budgets_to_run) - set(configured_budgets)
    if unknown:
        parser.error(
            "unknown budget(s): "
            + ", ".join(sorted(unknown))
            + "; available: "
            + ", ".join(configured_budgets)
        )
    if args.temperature is not None:
        for budget_name in budgets_to_run:
            cfg["inference"]["budgets"][budget_name]["temperature"] = (
                args.temperature
            )

    # Load the model once for every requested budget.
    engine = VLLMInferenceEngine(cfg)

    for budget_name in budgets_to_run:
        logger.info(f"\n=== Budget: {budget_name} ===")
        budget_cfg = dict(cfg["inference"]["budgets"][budget_name])
        base_seed = cfg["inference"].get("seed")
        budget_seed = effective_budget_seed(base_seed, budget_name, budget_cfg)
        if budget_seed is not None:
            budget_cfg["seed"] = budget_seed
        out_path = out_dir / f"inference_{budget_name}.jsonl"
        t0 = time.time()
        run_budget(engine, problems, budget_cfg, out_path, resume=args.resume)
        elapsed = time.time() - t0
        logger.info(f"  Budget '{budget_name}' done in {elapsed:.1f}s")

    logger.info("\nStep 2 complete.")


if __name__ == "__main__":
    main()
