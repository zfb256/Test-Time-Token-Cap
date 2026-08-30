"""Audit repair rates under several explicit base-case definitions.

This is a sensitivity analysis, not a replacement for the paper's strict
complete-but-wrong (CBW) definition.  It separates:

* strict_all_wrong: all chains stop naturally, parse, and are incorrect;
* plurality_wrong: all chains stop naturally and parse, but the plurality is
  wrong (a correct minority is allowed);
* first_chain_wrong: the first chain stops naturally, parses, and is wrong.

The extended run is scored by deterministic plurality, any-correct, and
tie-breaking lower/upper bounds.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier  # noqa: E402


def load_map(path):
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record["problem_id"]
            if pid in records:
                raise ValueError(f"{path}:{line_no}: duplicate problem_id={pid}")
            records[pid] = record
    return records


def validate_run(records, label, expected_budget):
    if not records:
        raise ValueError(f"{label} run is empty")
    signatures = set()
    for problem_id, record in records.items():
        budget = record.get("budget", {})
        signature = (
            budget.get("max_new_tokens"),
            budget.get("n_chains"),
            budget.get("temperature"),
            budget.get("top_p"),
        )
        signatures.add(signature)
        if len(record.get("chains", [])) != budget.get("n_chains"):
            raise ValueError(
                f"{label} chain-count mismatch for problem_id={problem_id}"
            )
    if len(signatures) != 1:
        raise ValueError(f"{label} run has inconsistent decoding metadata")
    if next(iter(signatures))[0] != expected_budget:
        raise ValueError(
            f"{label} metadata cap does not match {expected_budget}"
        )


def chain_values(record, budget, verifier):
    values = []
    for chain in record.get("chains", []):
        text = chain.get("text", "")
        finish = chain.get("finish_reason")
        hit_max = (
            finish == "length"
            if finish is not None
            else int(chain.get("token_count", 0)) >= budget
        )
        values.append(
            {
                "complete": not hit_max,
                "answer": verifier.extract_prediction_answer(
                    text, record["dataset"]
                ),
                "correct": bool(
                    verifier.verify(text, record["answer"], record["dataset"])
                ),
            }
        )
    return values


def plurality(values):
    valid = [value for value in values if value["answer"] is not None]
    if not valid:
        return {
            "correct": False,
            "tie": False,
            "lower": False,
            "upper": False,
        }
    counts = collections.Counter(value["answer"] for value in valid)
    top_count = max(counts.values())
    winners = sorted(a for a, count in counts.items() if count == top_count)
    winner_correct = [
        any(v["correct"] for v in valid if v["answer"] == answer)
        for answer in winners
    ]
    return {
        "correct": winner_correct[0],
        "tie": len(winners) > 1,
        "lower": all(winner_correct),
        "upper": any(winner_correct),
    }


def eligibility(values):
    all_complete_parseable = bool(values) and all(
        value["complete"] and value["answer"] is not None for value in values
    )
    vote = plurality(values)
    return {
        "strict_all_wrong": all_complete_parseable
        and not any(value["correct"] for value in values),
        "plurality_wrong": all_complete_parseable and not vote["correct"],
        "first_chain_wrong": bool(values)
        and values[0]["complete"]
        and values[0]["answer"] is not None
        and not values[0]["correct"],
    }


def analyze(base_map, ext_map, base_budget, ext_budget):
    if set(base_map) != set(ext_map):
        raise ValueError("base and extended problem_id sets differ")
    rows = []
    for pid, base_record in base_map.items():
        ext_record = ext_map[pid]
        for field in ("dataset", "question", "answer"):
            if base_record.get(field) != ext_record.get(field):
                raise ValueError(f"{field} mismatch for problem_id={pid}")
        base = chain_values(base_record, base_budget, VERIFIER)
        ext = chain_values(ext_record, ext_budget, VERIFIER)
        eligible = eligibility(base)
        base_vote = plurality(base)
        ext_vote = plurality(ext)
        base_any_correct = any(value["correct"] for value in base)
        ext_any_correct = any(value["correct"] for value in ext)
        common = {
            "problem_id": pid,
            "dataset": base_record["dataset"],
            "base_plurality_correct": base_vote["correct"],
            "base_any_correct": base_any_correct,
            "ext_plurality_correct": ext_vote["correct"],
            "ext_plurality_tied": ext_vote["tie"],
            "ext_tie_lower_correct": ext_vote["lower"],
            "ext_tie_upper_correct": ext_vote["upper"],
            "ext_any_correct": ext_any_correct,
        }
        for definition, is_eligible in eligible.items():
            if definition == "first_chain_wrong":
                # The comparator is the declared first (single) base chain.
                repair_plurality = ext_vote["correct"]
                repair_any = ext_any_correct
                repair_tie_lower = ext_vote["lower"]
                repair_tie_upper = ext_vote["upper"]
            else:
                # For set-level definitions, require a false->true transition;
                # a correct base minority is not a new any-chain repair.
                repair_plurality = (
                    not base_vote["correct"] and ext_vote["correct"]
                )
                repair_any = not base_any_correct and ext_any_correct
                repair_tie_lower = (
                    not base_vote["correct"] and ext_vote["lower"]
                )
                repair_tie_upper = (
                    not base_vote["correct"] and ext_vote["upper"]
                )
            rows.append(
                {
                    **common,
                    "definition": definition,
                    "eligible": is_eligible,
                    "repair_plurality": repair_plurality,
                    "repair_any": repair_any,
                    "repair_tie_lower": repair_tie_lower,
                    "repair_tie_upper": repair_tie_upper,
                }
            )
    return pd.DataFrame(rows)


def summarize(detail):
    rows = []
    for (definition, dataset), group in detail.groupby(
        ["definition", "dataset"], sort=False
    ):
        eligible = group[group["eligible"]]
        n = len(eligible)
        rows.append(
            {
                "definition": definition,
                "dataset": dataset,
                "n_problems": len(group),
                "n_eligible": n,
                "eligible_rate": n / len(group),
                "repair_plurality_count": int(
                    eligible["repair_plurality"].sum()
                ),
                "repair_plurality_rate": (
                    float(eligible["repair_plurality"].mean())
                    if n
                    else float("nan")
                ),
                "repair_any_rate": (
                    float(eligible["repair_any"].mean())
                    if n
                    else float("nan")
                ),
                "repair_tie_lower_rate": (
                    float(eligible["repair_tie_lower"].mean())
                    if n
                    else float("nan")
                ),
                "repair_tie_upper_rate": (
                    float(eligible["repair_tie_upper"].mean())
                    if n
                    else float("nan")
                ),
            }
        )
    all_rows = []
    for definition, group in detail.groupby("definition", sort=False):
        eligible = group[group["eligible"]]
        n = len(eligible)
        all_rows.append(
            {
                "definition": definition,
                "dataset": "all",
                "n_problems": len(group),
                "n_eligible": n,
                "eligible_rate": n / len(group),
                "repair_plurality_count": int(
                    eligible["repair_plurality"].sum()
                ),
                "repair_plurality_rate": (
                    float(eligible["repair_plurality"].mean())
                    if n
                    else float("nan")
                ),
                "repair_any_rate": (
                    float(eligible["repair_any"].mean())
                    if n
                    else float("nan")
                ),
                "repair_tie_lower_rate": (
                    float(eligible["repair_tie_lower"].mean())
                    if n
                    else float("nan")
                ),
                "repair_tie_upper_rate": (
                    float(eligible["repair_tie_upper"].mean())
                    if n
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(all_rows + rows)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--base-budget", required=True, type=int)
    parser.add_argument("--ext-name", required=True)
    parser.add_argument("--ext-budget", required=True, type=int)
    args = parser.parse_args()
    if not 0 < args.base_budget < args.ext_budget:
        parser.error("budgets must satisfy 0 < base-budget < ext-budget")
    out_dir = resolve(args.results_dir)
    base = load_map(out_dir / f"inference_{args.base_name}.jsonl")
    ext = load_map(out_dir / f"inference_{args.ext_name}.jsonl")
    validate_run(base, "base", args.base_budget)
    validate_run(ext, "extended", args.ext_budget)
    detail = analyze(base, ext, args.base_budget, args.ext_budget)
    summary = summarize(detail)
    tag = f"{args.base_name}_to_{args.ext_name}"
    detail_path = out_dir / f"expanded_repair_{tag}_detail.csv"
    summary_path = out_dir / f"expanded_repair_{tag}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


VERIFIER = MathVerifier()

if __name__ == "__main__":
    main()
