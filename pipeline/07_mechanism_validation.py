"""
Mechanism-only validation for a medium-to-long inference run.

This script avoids PRM scoring and learned routing. It verifies medium/long
outputs, classifies helpful examples by completion state, and reports whether
complete-but-wrong traces are repaired by longer inference.

Run after:
  python 01_download_data.py --config <config>.yaml
  python 02_run_inference.py --config <config>.yaml --budget medium --resume
  python 02_run_inference.py --config <config>.yaml --budget long --resume

Example:
  python 07_mechanism_validation.py --config config.yaml
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.math_verify import MathVerifier


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_chain_text(row: Dict) -> str:
    chains = row.get("chains", [])
    return chains[0].get("text", "") if chains else ""


def first_chain_tokens(row: Dict) -> int:
    chains = row.get("chains", [])
    return int(chains[0].get("token_count", 0)) if chains else 0


def first_chain_finish_reason(row: Dict) -> Optional[str]:
    chains = row.get("chains", [])
    reason = chains[0].get("finish_reason") if chains else None
    return str(reason).lower() if reason is not None else None


def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\s*\{", text) or re.search(r"\\boxed\s+", text))


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.3f}"


def routed_accuracy(details: List[Dict], route_key: str) -> float:
    correct = []
    for row in details:
        use_long = bool(row[route_key])
        correct.append(row["long_correct"] if use_long else row["medium_correct"])
    return sum(correct) / len(correct) if correct else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    medium_path = out_dir / "inference_medium.jsonl"
    long_path = out_dir / "inference_long.jsonl"
    if not medium_path.exists() or not long_path.exists():
        raise SystemExit(
            f"Missing inference outputs. Expected {medium_path} and {long_path}."
        )

    medium_records = read_jsonl(medium_path)
    long_records = read_jsonl(long_path)
    medium_rows = {r["problem_id"]: r for r in medium_records}
    long_rows = {r["problem_id"]: r for r in long_records}
    if len(medium_rows) != len(medium_records) or len(long_rows) != len(long_records):
        raise SystemExit("Duplicate problem_id in medium or long outputs.")
    if set(medium_rows) != set(long_rows):
        raise SystemExit("Medium and long problem_id sets differ.")
    common_ids = sorted(medium_rows)
    if not common_ids:
        raise SystemExit("Medium and long outputs are empty.")

    medium_budget = int(cfg["inference"]["budgets"]["medium"]["max_new_tokens"])
    long_budget = int(cfg["inference"]["budgets"]["long"]["max_new_tokens"])
    verifier = MathVerifier(cfg.get("orm", {}).get("numeric_tolerance", 1e-4))

    details = []
    for pid in common_ids:
        med = medium_rows[pid]
        lng = long_rows[pid]
        for field in ("dataset", "question", "answer", "math_level"):
            default = 0 if field == "math_level" else None
            if med.get(field, default) != lng.get(field, default):
                raise SystemExit(
                    f"Medium/long {field} mismatch for problem_id={pid}"
                )
        dataset = med["dataset"]
        answer = med["answer"]
        med_text = first_chain_text(med)
        lng_text = first_chain_text(lng)
        med_tokens = first_chain_tokens(med)
        lng_tokens = first_chain_tokens(lng)
        med_finish_reason = first_chain_finish_reason(med)
        lng_finish_reason = first_chain_finish_reason(lng)
        med_correct = bool(verifier.verify(med_text, answer, dataset))
        lng_correct = bool(verifier.verify(lng_text, answer, dataset))
        med_hit_max = (
            med_finish_reason == "length"
            if med_finish_reason is not None
            else med_tokens >= medium_budget
        )
        lng_hit_max = (
            lng_finish_reason == "length"
            if lng_finish_reason is not None
            else lng_tokens >= long_budget
        )
        med_has_boxed = has_boxed(med_text)
        med_incomplete = med_hit_max or (not med_has_boxed)
        helpful = (not med_correct) and lng_correct
        complete_but_wrong = (not med_correct) and (not med_hit_max) and med_has_boxed

        details.append({
            "problem_id": pid,
            "dataset": dataset,
            "math_level": med.get("math_level", 0),
            "medium_correct": med_correct,
            "long_correct": lng_correct,
            "helpful": helpful,
            "medium_tokens": med_tokens,
            "long_tokens": lng_tokens,
            "medium_finish_reason": med_finish_reason,
            "long_finish_reason": lng_finish_reason,
            "medium_hit_max": med_hit_max,
            "long_hit_max": lng_hit_max,
            "medium_has_boxed": med_has_boxed,
            "medium_incomplete": med_incomplete,
            "complete_but_wrong": complete_but_wrong,
            "complete_but_wrong_fixed": complete_but_wrong and lng_correct,
        })

    n = len(details)
    med_acc = mean([r["medium_correct"] for r in details])
    long_acc = mean([r["long_correct"] for r in details])
    helpful_rows = [r for r in details if r["helpful"]]
    cbw_rows = [r for r in details if r["complete_but_wrong"]]
    cbw_fixed = [r for r in cbw_rows if r["complete_but_wrong_fixed"]]
    hitmax_rows = [r for r in details if r["medium_hit_max"]]

    helpful_rate = len(helpful_rows) / n
    hitmax_rate = len(hitmax_rows) / n
    helpful_incomplete_frac = (
        sum(r["medium_incomplete"] for r in helpful_rows) / len(helpful_rows)
        if helpful_rows else None
    )
    helpful_hitmax_frac = (
        sum(r["medium_hit_max"] for r in helpful_rows) / len(helpful_rows)
        if helpful_rows else None
    )
    cbw_rate = len(cbw_rows) / n
    cbw_fix_rate = len(cbw_fixed) / len(cbw_rows) if cbw_rows else None
    route_hitmax_acc = routed_accuracy(details, "medium_hit_max")
    random_same_fraction = med_acc + hitmax_rate * (long_acc - med_acc)

    summary_rows = [
        ("n", n),
        ("medium_budget", medium_budget),
        ("long_budget", long_budget),
        ("medium_acc", fmt(med_acc)),
        ("long_acc", fmt(long_acc)),
        ("gain", fmt(long_acc - med_acc)),
        ("helpful_count", len(helpful_rows)),
        ("helpful_rate", fmt(helpful_rate)),
        ("medium_hit_max_count", len(hitmax_rows)),
        ("medium_hit_max_rate", fmt(hitmax_rate)),
        ("helpful_incomplete_frac", fmt(helpful_incomplete_frac)),
        ("helpful_hitmax_frac", fmt(helpful_hitmax_frac)),
        ("complete_but_wrong_count", len(cbw_rows)),
        ("complete_but_wrong_rate", fmt(cbw_rate)),
        ("complete_but_wrong_fixed", len(cbw_fixed)),
        ("complete_but_wrong_fix_rate", fmt(cbw_fix_rate)),
        ("hitmax_rule_routed", fmt(hitmax_rate)),
        ("hitmax_rule_acc", fmt(route_hitmax_acc)),
        ("random_same_fraction_acc_expected", fmt(random_same_fraction)),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "large_model_mechanism_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    summary_path = out_dir / "large_model_mechanism_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    table_path = out_dir / "paper_table_large_model_validation.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Data", "N", "Med. acc.", "Long acc.", "Helpful",
            "Hit max", "Helpful incomplete", "CBW fixed",
        ])
        writer.writerow([
            "hard MATH", n, fmt(med_acc), fmt(long_acc), fmt(helpful_rate),
            fmt(hitmax_rate), fmt(helpful_incomplete_frac),
            f"{len(cbw_fixed)}/{len(cbw_rows)}",
        ])

    md_path = out_dir / "large_model_mechanism_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Large-Model Hard-Math Mechanism Validation\n\n")
        f.write(f"- N: {n}\n")
        f.write(f"- Medium/long budgets: {medium_budget} -> {long_budget}\n")
        f.write(f"- Accuracy: {fmt(med_acc)} -> {fmt(long_acc)} (gain {fmt(long_acc - med_acc)})\n")
        f.write(f"- Helpful: {len(helpful_rows)} ({fmt(helpful_rate)})\n")
        f.write(f"- Medium hit-max: {len(hitmax_rows)} ({fmt(hitmax_rate)})\n")
        f.write(f"- Helpful incomplete fraction: {fmt(helpful_incomplete_frac)}\n")
        f.write(f"- Complete-but-wrong fixed: {len(cbw_fixed)}/{len(cbw_rows)}\n")
        f.write(f"- Hit-max rule: routed {fmt(hitmax_rate)}, accuracy {fmt(route_hitmax_acc)}\n")
        f.write(f"- Random same-fraction expected accuracy: {fmt(random_same_fraction)}\n")

    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paper table: {table_path}")
    print(f"Saved markdown summary: {md_path}")
    print("\nKey summary:")
    for k, v in summary_rows:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
