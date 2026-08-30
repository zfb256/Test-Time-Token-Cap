"""
Step 30: Budget scaling validation.

After running additional inference budgets, evaluate whether medium->long
utility persists as the budget scale increases. Example usage:

  python 02_run_inference.py --config config.yaml --budget medium512 --resume
  python 02_run_inference.py --config config.yaml --budget long1024 --resume
  python 30_budget_scaling_validation.py --pairs medium:long medium512:long1024

This script only evaluates existing inference files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    out = {}
    for record in load_jsonl(path):
        pid = record["problem_id"]
        if pid in out:
            raise ValueError(f"{path}: duplicate problem_id={pid}")
        out[pid] = record
    return out


def first_chain(record: Dict) -> Dict:
    chains = record.get("chains") or []
    return chains[0] if chains else {"text": "", "token_count": 0}


def parse_pair(pair: str) -> Tuple[str, str]:
    if ":" not in pair:
        raise ValueError(f"Budget pair must be formatted as medium_budget:long_budget, got {pair}")
    left, right = pair.split(":", 1)
    return left.strip(), right.strip()


def classify_utility(medium_correct: bool, long_correct: bool, medium_hit_max: bool, medium_has_boxed: bool) -> str:
    if medium_correct and long_correct:
        return "already_solved"
    if medium_correct and not long_correct:
        return "long_harm"
    if not medium_correct and not long_correct:
        return "no_utility"
    if medium_hit_max or not medium_has_boxed:
        return "completion_recovery"
    return "reasoning_recovery"


def evaluate_pair(out_dir: Path, pair: Tuple[str, str], verifier: MathVerifier) -> Tuple[pd.DataFrame, pd.DataFrame]:
    medium_name, long_name = pair
    medium_path = out_dir / f"inference_{medium_name}.jsonl"
    long_path = out_dir / f"inference_{long_name}.jsonl"
    if not medium_path.exists() or not long_path.exists():
        missing = [str(p) for p in [medium_path, long_path] if not p.exists()]
        raise FileNotFoundError("Missing inference files: " + ", ".join(missing))

    medium_map = load_jsonl_map(medium_path)
    long_map = load_jsonl_map(long_path)
    if set(medium_map) != set(long_map):
        raise ValueError(f"{medium_path} and {long_path} problem_id sets differ")
    common_ids = sorted(medium_map)
    rows = []
    for pid in common_ids:
        m = medium_map[pid]
        l = long_map[pid]
        m_chain = first_chain(m)
        l_chain = first_chain(l)
        dataset = m["dataset"]
        answer = m["answer"]
        medium_correct = verifier.verify(m_chain.get("text", ""), answer, dataset)
        long_correct = verifier.verify(l_chain.get("text", ""), answer, dataset)
        medium_budget = int(m.get("budget", {}).get("max_new_tokens", m_chain.get("token_count", 0)))
        medium_hit_max = (
            m_chain.get("finish_reason") == "length"
            if m_chain.get("finish_reason") is not None
            else int(m_chain.get("token_count", 0)) == medium_budget
        )
        medium_has_boxed = "\\boxed" in m_chain.get("text", "")
        rows.append({
            "pair": f"{medium_name}:{long_name}",
            "problem_id": pid,
            "dataset": dataset,
            "math_level": m.get("math_level", 0),
            "medium_budget_name": medium_name,
            "long_budget_name": long_name,
            "medium_budget_tokens": medium_budget,
            "long_budget_tokens": int(l.get("budget", {}).get("max_new_tokens", l_chain.get("token_count", 0))),
            "medium_tokens": int(m_chain.get("token_count", 0)),
            "long_tokens": int(l_chain.get("token_count", 0)),
            "medium_hit_max": medium_hit_max,
            "medium_has_boxed": medium_has_boxed,
            "medium_correct": medium_correct,
            "long_correct": long_correct,
            "helpful_ml": (not medium_correct) and long_correct,
            "utility_taxonomy": classify_utility(
                medium_correct,
                long_correct,
                medium_hit_max,
                medium_has_boxed,
            ),
        })
    detail = pd.DataFrame(rows)
    summary_rows = []
    for dataset, sub in [("all", detail), *detail.groupby("dataset")]:
        helpful = sub[sub["helpful_ml"]]
        summary_rows.append({
            "pair": f"{medium_name}:{long_name}",
            "dataset": dataset,
            "n": len(sub),
            "medium_acc": float(sub["medium_correct"].mean()),
            "long_acc": float(sub["long_correct"].mean()),
            "helpful_ml_rate": float(sub["helpful_ml"].mean()),
            "long_harm_rate": float((sub["medium_correct"] & ~sub["long_correct"]).mean()),
            "medium_hit_max_rate": float(sub["medium_hit_max"].mean()),
            "completion_recovery_fraction_of_helpful": float(
                (helpful["utility_taxonomy"] == "completion_recovery").mean()
            ) if len(helpful) else 0.0,
            "reasoning_recovery_fraction_of_helpful": float(
                (helpful["utility_taxonomy"] == "reasoning_recovery").mean()
            ) if len(helpful) else 0.0,
        })
    return detail, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--pairs", nargs="+", default=["medium:long"])
    args = parser.parse_args()

    out_dir = resolve_path(args.results_dir)
    verifier = MathVerifier()
    all_details = []
    all_summaries = []
    for raw_pair in args.pairs:
        detail, summary = evaluate_pair(out_dir, parse_pair(raw_pair), verifier)
        all_details.append(detail)
        all_summaries.append(summary)

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    detail_path = out_dir / "budget_scaling_validation_detail.csv"
    summary_path = out_dir / "budget_scaling_validation_summary.csv"
    txt_path = out_dir / "budget_scaling_validation_summary.txt"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Budget Scaling Validation\n")
        f.write("=========================\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
