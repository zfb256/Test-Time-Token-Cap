"""
Step 39 (v2): Repair-vs-completion evaluation under non-greedy decoding.

Review response (the central objection): under deterministic greedy decoding
"reasoning repair" is impossible by construction, so the original taxonomy
cannot falsify the completion-recovery hypothesis. This script evaluates the
mechanism question on sampled runs, where the model CAN take a different
reasoning path.

Inputs (produced on the GPU server):
  - inference_{base}.jsonl      base-budget run (n_chains >= 1)
  - inference_{ext}.jsonl       extended-budget sampled run (n_chains >= 1)

Per problem and run we compute chain-level strict correctness, cap-hit and
boxed rates, plus pass@1 (mean over chains), any@k, and majority-vote
correctness. The report answers, with exact binomial CIs:

  1. Does extra budget still help under sampling (helpful rate, maj/pass@1)?
  2. Are complete-but-wrong base cases now repairable? (the key number the
     greedy protocol could not measure)
  3. Among helpful cases, what fraction had truncated vs complete base
     traces (taxonomy WITHOUT the subtraction shortcut: both classes are
     defined by independent, observable predicates)?

Usage examples:
  # Qwen main: greedy medium vs sampled long
  python 39_sampling_repair_eval.py --results-dir ../outputs/qwen_main \
      --base-name medium --base-budget 384 \
      --ext-name long_sample8 --ext-budget 768

  # R1-distill: sampled base vs sampled extended
  python 39_sampling_repair_eval.py --results-dir ../outputs/r1_distill \
      --base-name base1024 --base-budget 1024 \
      --ext-name ext4096 --ext-budget 4096
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier  # noqa: E402


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                pid = r["problem_id"]
                if pid in out:
                    raise ValueError(f"{path}:{line_no}: duplicate problem_id={pid}")
                out[pid] = r
    return out


def validate_run(records: Dict[str, Dict], label: str, expected_budget: int) -> None:
    """Reject mislabeled or structurally inconsistent inference artifacts."""
    if not records:
        raise SystemExit(f"{label} run is empty")
    signatures = set()
    for pid, rec in records.items():
        metadata = rec.get("budget", {})
        signature = (
            metadata.get("max_new_tokens"),
            metadata.get("n_chains"),
            metadata.get("temperature"),
            metadata.get("top_p"),
        )
        signatures.add(signature)
        chains = rec.get("chains", [])
        if len(chains) != metadata.get("n_chains"):
            raise SystemExit(
                f"{label} problem_id={pid}: {len(chains)} chains != "
                f"declared {metadata.get('n_chains')}"
            )
    if len(signatures) != 1:
        raise SystemExit(f"{label} run has inconsistent decoding metadata")
    signature = next(iter(signatures))
    if signature[0] != expected_budget:
        raise SystemExit(
            f"{label} metadata cap {signature[0]} != requested {expected_budget}"
        )


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def paired_bootstrap_ci(values, seed: int = 42, n_boot: int = 10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def eval_run(rec: Dict, budget: int, verifier: MathVerifier) -> Dict:
    """Chain-level stats for one problem in one run."""
    dataset = rec["dataset"]
    answer = rec["answer"]
    correct, hit_max, boxed, answers = [], [], [], []
    for chain in rec["chains"]:
        text = chain.get("text", "")
        correct.append(bool(verifier.verify(text, answer, dataset)))
        # New artifacts preserve the engine's actual stopping decision.
        # Fall back to the legacy token-count heuristic only for old JSONL
        # files that predate finish_reason.
        finish_reason = chain.get("finish_reason")
        hit_max.append(
            finish_reason == "length"
            if finish_reason is not None
            else int(chain.get("token_count", 0)) >= budget
        )
        boxed.append("\\boxed" in text)
        answers.append(verifier.extract_prediction_answer(text, dataset))
    k = len(correct)
    # Majority vote over extracted answers (None answers never win).
    maj_correct = False
    maj_tied = False
    maj_correct_any_tie = False
    maj_correct_all_tie = False
    valid = [(a, c) for a, c in zip(answers, correct) if a is not None]
    if valid:
        counts = collections.Counter(a for a, _ in valid)
        top_count = max(counts.values())
        top_answers = sorted(a for a, count in counts.items() if count == top_count)
        maj_tied = len(top_answers) > 1
        # Lexicographic tie-breaking is deterministic and independent of vLLM
        # output order. Bounds below expose whether this choice affects accuracy.
        maj_answer = top_answers[0]
        maj_correct = any(c for a, c in valid if a == maj_answer)
        tie_correctness = [
            any(c for a, c in valid if a == answer) for answer in top_answers
        ]
        maj_correct_any_tie = any(tie_correctness)
        maj_correct_all_tie = all(tie_correctness)
    all_complete = bool(hit_max) and not any(hit_max)
    all_answered = bool(answers) and all(a is not None for a in answers)
    complete_wrong = all_complete and all_answered and not any(correct)
    if hit_max and all(hit_max):
        completion_state = "all_truncated"
    elif hit_max and any(hit_max):
        completion_state = "mixed_truncated_complete"
    elif complete_wrong:
        completion_state = "all_complete_answered_wrong"
    else:
        # Includes naturally terminated chains with missing/unparseable answers
        # and sets containing a correct chain whose aggregate vote is wrong.
        completion_state = "complete_other"

    return {
        "k": k,
        "pass1": sum(correct) / k if k else 0.0,
        "any_correct": any(correct),
        "maj_correct": maj_correct,
        "maj_tied": maj_tied,
        "maj_correct_any_tie": maj_correct_any_tie,
        "maj_correct_all_tie": maj_correct_all_tie,
        "all_hit_max": all(hit_max) if hit_max else False,
        "any_complete": any(not h for h in hit_max),
        "all_complete": all_complete,
        "hit_max_rate": sum(hit_max) / k if k else 0.0,
        "boxed_rate": sum(boxed) / k if k else 0.0,
        # A problem is "complete but wrong" only when every sampled chain
        # terminated naturally, every chain yielded a parseable answer, and
        # none was correct. Mixed truncated/complete sets are kept separate.
        "complete_wrong": complete_wrong,
        "completion_state": completion_state,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--base-name", default="medium")
    parser.add_argument("--base-budget", type=int, default=384)
    parser.add_argument("--ext-name", default="long_sample8")
    parser.add_argument("--ext-budget", type=int, default=768)
    args = parser.parse_args()
    if not 0 < args.base_budget <= args.ext_budget:
        parser.error("budgets must satisfy 0 < base-budget <= ext-budget")

    out_dir = resolve_path(args.results_dir)
    base_map = load_jsonl_map(out_dir / f"inference_{args.base_name}.jsonl")
    ext_map = load_jsonl_map(out_dir / f"inference_{args.ext_name}.jsonl")
    validate_run(base_map, "base", args.base_budget)
    validate_run(ext_map, "extended", args.ext_budget)
    if set(base_map) != set(ext_map):
        missing_ext = sorted(set(base_map) - set(ext_map))
        missing_base = sorted(set(ext_map) - set(base_map))
        raise SystemExit(
            "Input problem_id sets differ: "
            f"{len(missing_ext)} missing from extended run "
            f"(examples: {missing_ext[:5]}), "
            f"{len(missing_base)} missing from base run "
            f"(examples: {missing_base[:5]}). "
            "Refusing to report silently intersected results."
        )
    verifier = MathVerifier()

    rows = []
    for pid, base_rec in base_map.items():
        if base_rec["dataset"] != ext_map[pid]["dataset"]:
            raise SystemExit(f"dataset mismatch for problem_id={pid}")
        if base_rec.get("question") != ext_map[pid].get("question"):
            raise SystemExit(f"question mismatch for problem_id={pid}")
        if base_rec.get("answer") != ext_map[pid].get("answer"):
            raise SystemExit(f"ground-truth answer mismatch for problem_id={pid}")
        b = eval_run(base_rec, args.base_budget, verifier)
        e = eval_run(ext_map[pid], args.ext_budget, verifier)
        rows.append({
            "problem_id": pid,
            "dataset": base_rec["dataset"],
            "math_level": base_rec.get("math_level", 0),
            **{f"base_{k}": v for k, v in b.items()},
            **{f"ext_{k}": v for k, v in e.items()},
            # Independent, observable taxonomy predicates (review response:
            # not defined by subtraction):
            "base_truncated": b["all_hit_max"],
            "base_complete_wrong": b["complete_wrong"],
            "base_completion_state": b["completion_state"],
            "helpful_maj": (not b["maj_correct"]) and e["maj_correct"],
            "helpful_any": (not b["any_correct"]) and e["any_correct"],
            "harm_maj": b["maj_correct"] and (not e["maj_correct"]),
            "repaired_maj": b["complete_wrong"] and e["maj_correct"],
            "repaired_any": b["complete_wrong"] and e["any_correct"],
        })

    df = pd.DataFrame(rows)
    tag = f"{args.base_name}_to_{args.ext_name}"
    detail_path = out_dir / f"sampling_repair_{tag}_detail.csv"
    df.to_csv(detail_path, index=False)

    summary_rows = []
    for dataset, sub in [("all", df), *df.groupby("dataset")]:
        n = len(sub)
        cw = sub[sub["base_complete_wrong"]]
        helpful = sub[sub["helpful_maj"]]
        rep_maj_k = int(cw["repaired_maj"].sum())
        rep_any_k = int(cw["repaired_any"].sum())
        rep_tie_lower_k = int(cw["ext_maj_correct_all_tie"].sum())
        rep_tie_upper_k = int(cw["ext_maj_correct_any_tie"].sum())
        lo_maj, hi_maj = clopper_pearson(rep_maj_k, len(cw))
        lo_any, hi_any = clopper_pearson(rep_any_k, len(cw))
        helpful_k = int(sub["helpful_maj"].sum())
        harm_k = int(sub["harm_maj"].sum())
        helpful_lo, helpful_hi = clopper_pearson(helpful_k, n)
        harm_lo, harm_hi = clopper_pearson(harm_k, n)
        pass1_delta = sub["ext_pass1"] - sub["base_pass1"]
        pass1_lo, pass1_hi = paired_bootstrap_ci(pass1_delta)
        maj_delta = (
            sub["ext_maj_correct"].astype(float)
            - sub["base_maj_correct"].astype(float)
        )
        maj_lo, maj_hi = paired_bootstrap_ci(maj_delta)
        helpful_states = helpful["base_completion_state"].value_counts()
        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "base_pass1": float(sub["base_pass1"].mean()),
            "base_maj_acc": float(sub["base_maj_correct"].mean()),
            "base_maj_tie_rate": float(sub["base_maj_tied"].mean()),
            "ext_pass1": float(sub["ext_pass1"].mean()),
            "pass1_delta": float(pass1_delta.mean()),
            "pass1_delta_ci95": f"[{pass1_lo:.3f},{pass1_hi:.3f}]",
            "ext_maj_acc": float(sub["ext_maj_correct"].mean()),
            "maj_acc_delta": float(maj_delta.mean()),
            "maj_acc_delta_ci95": f"[{maj_lo:.3f},{maj_hi:.3f}]",
            "ext_maj_tie_rate": float(sub["ext_maj_tied"].mean()),
            "ext_maj_acc_tie_lower": float(sub["ext_maj_correct_all_tie"].mean()),
            "ext_maj_acc_tie_upper": float(sub["ext_maj_correct_any_tie"].mean()),
            "helpful_maj_rate": float(sub["helpful_maj"].mean()),
            "helpful_maj_ci95": f"[{helpful_lo:.3f},{helpful_hi:.3f}]",
            "harm_maj_rate": float(sub["harm_maj"].mean()),
            "harm_maj_ci95": f"[{harm_lo:.3f},{harm_hi:.3f}]",
            "base_trunc_rate": float(sub["base_truncated"].mean()),
            "n_complete_wrong": len(cw),
            "repair_rate_maj": rep_maj_k / len(cw) if len(cw) else float("nan"),
            "repair_count_maj": rep_maj_k,
            "repair_count_maj_tie_lower": rep_tie_lower_k,
            "repair_count_maj_tie_upper": rep_tie_upper_k,
            "repair_rate_maj_tie_lower": (
                rep_tie_lower_k / len(cw) if len(cw) else float("nan")
            ),
            "repair_rate_maj_tie_upper": (
                rep_tie_upper_k / len(cw) if len(cw) else float("nan")
            ),
            "repair_rate_maj_ci95": f"[{lo_maj:.3f},{hi_maj:.3f}]",
            "repair_rate_any": rep_any_k / len(cw) if len(cw) else float("nan"),
            "repair_rate_any_ci95": f"[{lo_any:.3f},{hi_any:.3f}]",
            "helpful_from_truncated": float(helpful["base_truncated"].mean()) if len(helpful) else float("nan"),
            "helpful_from_complete_wrong": float(helpful["base_complete_wrong"].mean()) if len(helpful) else float("nan"),
            "helpful_from_mixed": (
                float((helpful["base_completion_state"] == "mixed_truncated_complete").mean())
                if len(helpful) else float("nan")
            ),
            "helpful_from_complete_other": (
                float((helpful["base_completion_state"] == "complete_other").mean())
                if len(helpful) else float("nan")
            ),
            "helpful_taxonomy_n": int(helpful_states.sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"sampling_repair_{tag}_summary.csv"
    summary.to_csv(summary_path, index=False)

    txt_path = out_dir / f"sampling_repair_{tag}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Repair vs Completion under Sampling: {tag}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nKey quantities:\n")
        f.write("- repair_rate_*: among base complete-but-wrong problems, how often the\n")
        f.write("  sampled extended run produces a correct majority (or any) answer.\n")
        f.write("  Under the old greedy protocol this was 0 by construction. Under\n")
        f.write("  independent sampling it is an observable transition, but not by\n")
        f.write("  itself a token-budget effect: a same-budget resampling control is\n")
        f.write("  required. Tie-lower/upper repair counts are reported explicitly.\n")
        f.write("- helpful_from_truncated vs helpful_from_complete_wrong: taxonomy of\n")
        f.write("  helpful cases via independent observable predicates. Mixed sampled\n")
        f.write("  sets and complete-but-unparseable/aggregate-error sets are reported\n")
        f.write("  separately, so the four helpful_from_* columns partition all helpful\n")
        f.write("  cases instead of hiding a residual category.\n")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
