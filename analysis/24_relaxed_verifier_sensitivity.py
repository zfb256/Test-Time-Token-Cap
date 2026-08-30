"""
Step 24: Strict-vs-relaxed verifier sensitivity.

The main labels intentionally use strict MATH verification: predicted answers
must include \\boxed{...}. This script does not replace those labels. It audits
how much of the medium->long utility signal is driven by formatting/completion
requirements by recomputing a conservative relaxed numeric verifier for MATH
outputs without boxed answers.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDIUM_TOKENS = 384
LONG_TOKENS = 768
DELTA_TOKENS = LONG_TOKENS - MEDIUM_TOKENS


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    records = {}
    for row in load_jsonl(path):
        problem_id = row["problem_id"]
        if problem_id in records:
            raise ValueError(f"{path}: duplicate problem_id={problem_id}")
        records[problem_id] = row
    return records


def first_chain_text(record: Dict) -> str:
    chains = record.get("chains") or []
    return chains[0].get("text", "") if chains else ""


def first_chain_tokens(record: Dict) -> int:
    chains = record.get("chains") or []
    return int(chains[0].get("token_count", 0)) if chains else 0


def normalize_numeric_text(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = re.sub(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = text.replace("$", "")
    return text


def strip_math_units(expr: str) -> str:
    """Remove presentation-only unit suffixes without changing math content."""
    expr = re.sub(r"\\(?:,|;|!|\s)+", "", expr)
    expr = re.sub(r"\^\s*\\circ", "", expr)
    expr = re.sub(
        r"\\text\{(?:degrees?|edges?|square inches?|sq\.?\s*cm)\}",
        "",
        expr,
        flags=re.IGNORECASE,
    )
    return expr.strip()


def extract_last_numberish(text: str) -> Optional[str]:
    candidates = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+(?:,\d{3})*(?:\.\d+)?)?",
        text,
    )
    return candidates[-1] if candidates else None


def try_numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = normalize_numeric_text(value)
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def relaxed_verify_math(
    prediction: str,
    ground_truth: str,
    verifier: MathVerifier,
    allow_unboxed: bool = True,
) -> bool:
    if verifier.verify_math(prediction, ground_truth):
        return True

    gt_expr = verifier._extract_boxed(ground_truth)
    pred_expr = verifier._extract_boxed(prediction)
    if gt_expr is not None and pred_expr is not None:
        clean_gt = strip_math_units(gt_expr)
        clean_pred = strip_math_units(pred_expr)
        if verifier.verify_math(
            rf"\boxed{{{clean_pred}}}", rf"\boxed{{{clean_gt}}}"
        ):
            return True

    gt_num = try_numeric(gt_expr)
    # Ground-truth MATH records are boxed. If that expression is not a simple
    # scalar, falling back to the final digit can turn unequal fractions,
    # tuples, or matrices into false matches.
    if gt_num is None:
        return False

    pred_num = try_numeric(pred_expr)
    if pred_expr is None and allow_unboxed:
        pred_num = try_numeric(extract_last_numberish(prediction))
    if pred_num is None:
        return False

    return abs(pred_num - gt_num) <= verifier.tol


def relaxed_verify(
    prediction: str,
    ground_truth: str,
    dataset: str,
    verifier: MathVerifier,
    allow_unboxed: bool = True,
) -> bool:
    if dataset.lower() == "math":
        return relaxed_verify_math(
            prediction, ground_truth, verifier, allow_unboxed=allow_unboxed
        )
    return verifier.verify(prediction, ground_truth, dataset)


def eval_mask(mask, medium_correct, long_correct):
    mask = np.array(mask, dtype=bool)
    correct = np.where(mask, long_correct, medium_correct)
    return {
        "routed": float(mask.mean()),
        "accuracy": float(correct.mean()),
        "avg_budget_tokens": float(MEDIUM_TOKENS + mask.mean() * DELTA_TOKENS),
        "budget_savings_vs_long_pct": float(
            100.0 * (1.0 - (MEDIUM_TOKENS + mask.mean() * DELTA_TOKENS) / LONG_TOKENS)
        ),
    }


def random_accuracy(frac, medium_correct, long_correct, n_seeds=500):
    rng = np.random.default_rng(0)
    n = len(medium_correct)
    if not n:
        return float("nan")
    k = min(n, max(0, int(round(frac * n))))
    if k == 0:
        return float(np.mean(medium_correct))
    if k == n:
        return float(np.mean(long_correct))
    accs = []
    for _ in range(n_seeds):
        idx = rng.choice(n, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        accs.append(eval_mask(mask, medium_correct, long_correct)["accuracy"])
    return float(np.mean(accs))


def summarize_labels(df: pd.DataFrame, prefix: str, medium_col: str, long_col: str) -> List[Dict]:
    rows = []
    for dataset, sub in [("all", df), *df.groupby("dataset")]:
        rows.append({
            "setting": prefix,
            "dataset": dataset,
            "n": len(sub),
            "medium_acc": float(sub[medium_col].mean()),
            "long_acc": float(sub[long_col].mean()),
            "helpful_ml_rate": float((sub[long_col] & ~sub[medium_col]).mean()),
            "long_harm_rate": float((sub[medium_col] & ~sub[long_col]).mean()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    args = parser.parse_args()

    out_dir = resolve_path(args.results_dir)
    medium_map = load_jsonl_map(out_dir / "inference_medium.jsonl")
    long_map = load_jsonl_map(out_dir / "inference_long.jsonl")
    if set(medium_map) != set(long_map):
        raise SystemExit("medium and long problem_id sets differ")
    verifier = MathVerifier()

    rows = []
    for pid, medium_rec in medium_map.items():
        long_rec = long_map[pid]
        for field in ("dataset", "question", "answer"):
            if medium_rec.get(field) != long_rec.get(field):
                raise SystemExit(f"{field} mismatch for problem_id={pid}")
        if not medium_rec.get("chains") or not long_rec.get("chains"):
            raise ValueError(f"missing inference chain for problem_id={pid}")
        medium_text = first_chain_text(medium_rec)
        long_text = first_chain_text(long_rec)
        medium_chain = medium_rec["chains"][0]
        medium_finish = medium_chain.get("finish_reason")
        medium_tokens = first_chain_tokens(medium_rec)
        medium_hit_max = (
            medium_finish == "length"
            if medium_finish is not None
            else medium_tokens >= MEDIUM_TOKENS
        )
        long_chain = long_rec["chains"][0]
        long_finish = long_chain.get("finish_reason")
        long_tokens = first_chain_tokens(long_rec)
        long_hit_max = (
            long_finish == "length"
            if long_finish is not None
            else long_tokens >= LONG_TOKENS
        )
        answer = medium_rec.get("answer", "")
        dataset = medium_rec["dataset"]
        strict_medium = verifier.verify(medium_text, answer, dataset)
        strict_long = verifier.verify(long_text, answer, dataset)
        relaxed_medium = relaxed_verify(
            medium_text,
            answer,
            dataset,
            verifier,
            allow_unboxed=not medium_hit_max,
        )
        relaxed_long = relaxed_verify(
            long_text,
            answer,
            dataset,
            verifier,
            allow_unboxed=not long_hit_max,
        )
        rows.append({
            "problem_id": pid,
            "dataset": dataset,
            "math_level": medium_rec.get("math_level", 0),
            "strict_medium_correct": strict_medium,
            "strict_long_correct": strict_long,
            "relaxed_medium_correct": relaxed_medium,
            "relaxed_long_correct": relaxed_long,
            "strict_helpful_ml": strict_long and not strict_medium,
            "relaxed_helpful_ml": relaxed_long and not relaxed_medium,
            "medium_changed_by_relaxed": relaxed_medium != strict_medium,
            "long_changed_by_relaxed": relaxed_long != strict_long,
            "medium_has_boxed": "\\boxed" in medium_text,
            "long_has_boxed": "\\boxed" in long_text,
            "medium_tokens": medium_tokens,
            "medium_hit_max": medium_hit_max,
            "long_tokens": long_tokens,
            "long_hit_max": long_hit_max,
            "strict_medium_prediction": verifier.extract_prediction_answer(
                medium_text, dataset
            ),
            "strict_long_prediction": verifier.extract_prediction_answer(
                long_text, dataset
            ),
            "ground_truth": answer,
            "medium_text": medium_text,
            "long_text": long_text,
        })

    df = pd.DataFrame(rows)
    detail_path = out_dir / "relaxed_verifier_sensitivity_detail.csv"
    df.to_csv(detail_path, index=False)
    disagreement_cols = [
        "problem_id",
        "dataset",
        "math_level",
        "strict_medium_correct",
        "relaxed_medium_correct",
        "strict_long_correct",
        "relaxed_long_correct",
        "medium_has_boxed",
        "long_has_boxed",
        "strict_medium_prediction",
        "strict_long_prediction",
        "ground_truth",
        "medium_text",
        "long_text",
    ]
    disagreements = df[
        df["medium_changed_by_relaxed"] | df["long_changed_by_relaxed"]
    ][disagreement_cols]
    disagreement_path = out_dir / "relaxed_verifier_disagreement_cases.csv"
    disagreements.to_csv(disagreement_path, index=False)

    summary_rows = []
    summary_rows.extend(summarize_labels(df, "strict", "strict_medium_correct", "strict_long_correct"))
    summary_rows.extend(summarize_labels(df, "relaxed", "relaxed_medium_correct", "relaxed_long_correct"))

    for name, mask in {
        "rule_hit_max_tokens": df["medium_hit_max"].values.astype(bool),
        "rule_no_boxed": ~df["medium_has_boxed"].values,
        "rule_hit_max_or_no_boxed": df["medium_hit_max"].values.astype(bool) | (~df["medium_has_boxed"].values),
    }.items():
        for setting, medium_col, long_col in [
            ("strict", "strict_medium_correct", "strict_long_correct"),
            ("relaxed", "relaxed_medium_correct", "relaxed_long_correct"),
        ]:
            medium_correct = df[medium_col].values.astype(bool)
            long_correct = df[long_col].values.astype(bool)
            row = eval_mask(mask, medium_correct, long_correct)
            row.update({
                "setting": setting,
                "strategy": name,
                "random_accuracy_same_fraction": random_accuracy(row["routed"], medium_correct, long_correct),
            })
            row["gain_over_random"] = row["accuracy"] - row["random_accuracy_same_fraction"]
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "relaxed_verifier_sensitivity.csv"
    summary.to_csv(summary_path, index=False)

    txt_path = out_dir / "relaxed_verifier_sensitivity_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Strict-vs-relaxed verifier sensitivity\n")
        f.write("=" * 39 + "\n\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Medium labels changed by relaxed verifier: {int(df['medium_changed_by_relaxed'].sum())}\n")
        f.write(f"Long labels changed by relaxed verifier: {int(df['long_changed_by_relaxed'].sum())}\n")
        f.write(f"MATH medium labels changed: {int(df.loc[df['dataset'] == 'math', 'medium_changed_by_relaxed'].sum())}\n")
        f.write(f"MATH long labels changed: {int(df.loc[df['dataset'] == 'math', 'long_changed_by_relaxed'].sum())}\n\n")
        label_summary = summary[summary["setting"].isin(["strict", "relaxed"]) & summary["dataset"].notna()]
        f.write("Label summary:\n")
        f.write(label_summary[[
            "setting",
            "dataset",
            "n",
            "medium_acc",
            "long_acc",
            "helpful_ml_rate",
            "long_harm_rate",
        ]].to_string(index=False))
        f.write("\n\nRule sensitivity:\n")
        rule_summary = summary[summary["strategy"].notna()] if "strategy" in summary.columns else pd.DataFrame()
        f.write(rule_summary[[
            "setting",
            "strategy",
            "routed",
            "accuracy",
            "budget_savings_vs_long_pct",
            "gain_over_random",
        ]].to_string(index=False))
        f.write("\n")

    logger.info("Saved: %s", detail_path)
    logger.info("Saved: %s", disagreement_path)
    logger.info("Saved: %s", summary_path)
    logger.info("Saved: %s", txt_path)


if __name__ == "__main__":
    main()
