"""
Step 35: Complete-but-wrong subset analysis.

This tests whether medium traces that appear complete but are wrong are repaired
by the long budget. It directly addresses the reviewer question: is
medium->long utility only completion recovery, or is there a separate reasoning
recovery mechanism?
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier

relaxed_path = SCRIPT_DIR / "24_relaxed_verifier_sensitivity.py"
relaxed_spec = importlib.util.spec_from_file_location(
    "relaxed_verifier_sensitivity", relaxed_path
)
if relaxed_spec is None or relaxed_spec.loader is None:
    raise ImportError(f"Could not load relaxed verifier module from {relaxed_path}")
relaxed_mod = importlib.util.module_from_spec(relaxed_spec)
relaxed_spec.loader.exec_module(relaxed_mod)
relaxed_verify = relaxed_mod.relaxed_verify


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


def first_chain(record: Dict) -> Dict:
    chains = record.get("chains") or []
    return chains[0] if chains else {"text": "", "token_count": 0}


def extract_last_numberish(text: str) -> Optional[str]:
    matches = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+(?:,\d{3})*(?:\.\d+)?)?",
        text,
    )
    return matches[-1] if matches else None


def compact_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize(df: pd.DataFrame, mask_col: str) -> List[Dict]:
    rows = []
    for dataset, sub in [("all", df), *df.groupby("dataset")]:
        cand = sub[sub[mask_col]]
        n = len(cand)
        rows.append({
            "subset": mask_col,
            "dataset": dataset,
            "n_total": len(sub),
            "n_subset": n,
            "subset_rate": n / len(sub) if len(sub) else 0.0,
            "strict_long_fix_rate": float(
                (
                    ~cand["strict_medium_correct"]
                    & cand["strict_long_correct"]
                ).mean()
            )
            if n
            else 0.0,
            "relaxed_long_fix_rate": float(
                (
                    ~cand["relaxed_medium_correct"]
                    & cand["relaxed_long_correct"]
                ).mean()
            )
            if n
            else 0.0,
            "medium_hit_max_rate": float(cand["medium_hit_max"].mean()) if n else 0.0,
            "medium_boxed_rate": float(cand["medium_has_boxed"].mean()) if n else 0.0,
            "medium_final_answer_rate": float(cand["medium_has_final_answer"].mean()) if n else 0.0,
            "mean_medium_tokens": float(cand["medium_tokens"].mean()) if n else 0.0,
            "mean_long_tokens": float(cand["long_tokens"].mean()) if n else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--medium-budget", type=int, default=384)
    parser.add_argument("--long-budget", type=int, default=768)
    args = parser.parse_args()

    out_dir = resolve_path(args.results_dir)
    medium_map = load_jsonl_map(out_dir / "inference_medium.jsonl")
    long_map = load_jsonl_map(out_dir / "inference_long.jsonl")
    if set(medium_map) != set(long_map):
        raise ValueError("medium and long problem_id sets differ")
    if not 0 < args.medium_budget < args.long_budget:
        parser.error("budgets must satisfy 0 < medium-budget < long-budget")
    verifier = MathVerifier()

    rows = []
    for pid, medium_rec in medium_map.items():
        long_rec = long_map[pid]
        for field in ("dataset", "question", "answer"):
            if medium_rec.get(field) != long_rec.get(field):
                raise ValueError(f"{field} mismatch for problem_id={pid}")
        if len(medium_rec.get("chains", [])) != 1 or len(
            long_rec.get("chains", [])
        ) != 1:
            raise ValueError(
                f"expected exactly one chain for problem_id={pid}"
            )
        m_chain = first_chain(medium_rec)
        l_chain = first_chain(long_rec)
        dataset = medium_rec["dataset"]
        answer = medium_rec["answer"]
        m_text = m_chain.get("text", "")
        l_text = l_chain.get("text", "")
        m_tokens = int(m_chain.get("token_count", 0))
        l_tokens = int(l_chain.get("token_count", 0))
        medium_has_boxed = "\\boxed" in m_text
        medium_has_number = extract_last_numberish(m_text) is not None
        # Dataset-specific final-answer signal. The prompt asks for boxed
        # answers for both datasets, but GSM8K verification is numeric, so keep
        # a numeric final-answer fallback to avoid defining the subset as MATH-only.
        medium_has_final_answer = medium_has_boxed or (
            dataset == "gsm8k" and medium_has_number
        )
        strict_medium = verifier.verify(m_text, answer, dataset)
        strict_long = strict_medium if l_text == m_text else verifier.verify(l_text, answer, dataset)
        medium_finish = m_chain.get("finish_reason")
        long_finish = l_chain.get("finish_reason")
        medium_hit_max = (
            medium_finish == "length"
            if medium_finish is not None
            else m_tokens >= args.medium_budget
        )
        long_hit_max = (
            long_finish == "length"
            if long_finish is not None
            else l_tokens >= args.long_budget
        )
        relaxed_medium = relaxed_verify(
            m_text,
            answer,
            dataset,
            verifier,
            allow_unboxed=not medium_hit_max,
        )
        relaxed_long = (
            relaxed_medium
            if l_text == m_text
            else relaxed_verify(
                l_text,
                answer,
                dataset,
                verifier,
                allow_unboxed=not long_hit_max,
            )
        )
        complete_medium = medium_has_final_answer and not medium_hit_max
        row = {
            "problem_id": pid,
            "dataset": dataset,
            "math_level": medium_rec.get("math_level", 0),
            "medium_tokens": m_tokens,
            "long_tokens": l_tokens,
            "medium_hit_max": medium_hit_max,
            "long_hit_max": long_hit_max,
            "medium_has_boxed": medium_has_boxed,
            "medium_has_number": medium_has_number,
            "medium_has_final_answer": medium_has_final_answer,
            "complete_medium": complete_medium,
            "strict_medium_correct": strict_medium,
            "strict_long_correct": strict_long,
            "relaxed_medium_correct": relaxed_medium,
            "relaxed_long_correct": relaxed_long,
            "strict_complete_wrong": complete_medium and not strict_medium,
            "relaxed_complete_wrong": complete_medium and not relaxed_medium,
            "strict_reasoning_recovery": complete_medium and (not strict_medium) and strict_long,
            "relaxed_reasoning_recovery": complete_medium and (not relaxed_medium) and relaxed_long,
            "question": medium_rec.get("question", ""),
            "answer": answer,
            "medium_answer": verifier.extract_prediction_answer(m_text, dataset),
            "long_answer": verifier.extract_prediction_answer(l_text, dataset),
            "medium_text_snippet": compact_text(m_text),
            "long_text_snippet": compact_text(l_text),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    detail_path = out_dir / "complete_but_wrong_detail.csv"
    df.to_csv(detail_path, index=False)

    summary_rows = []
    for mask_col in ["strict_complete_wrong", "relaxed_complete_wrong"]:
        summary_rows.extend(summarize(df, mask_col))
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "complete_but_wrong_summary.csv"
    summary.to_csv(summary_path, index=False)

    strict_cases = df[df["strict_reasoning_recovery"]].copy()
    relaxed_cases = df[df["relaxed_reasoning_recovery"]].copy()
    strict_cases.to_csv(out_dir / "complete_but_wrong_strict_recoveries.csv", index=False)
    relaxed_cases.to_csv(out_dir / "complete_but_wrong_relaxed_recoveries.csv", index=False)

    txt_path = out_dir / "complete_but_wrong_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Complete-But-Wrong Subset Analysis\n")
        f.write("==================================\n\n")
        f.write(
            "Definition: medium output has a dataset-specific final-answer signal "
            "(boxed answer, or numeric answer for GSM8K), does not hit the medium "
            "token cap, and is incorrect under the chosen verifier.\n\n"
        )
        f.write("Summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nStrict complete-wrong cases fixed by long:\n")
        if strict_cases.empty:
            f.write("None.\n")
        else:
            f.write(strict_cases[[
                "problem_id",
                "dataset",
                "math_level",
                "medium_tokens",
                "long_tokens",
                "medium_answer",
                "long_answer",
            ]].to_string(index=False))
            f.write("\n")
        f.write("\nRelaxed complete-wrong cases fixed by long:\n")
        if relaxed_cases.empty:
            f.write("None.\n")
        else:
            f.write(relaxed_cases[[
                "problem_id",
                "dataset",
                "math_level",
                "medium_tokens",
                "long_tokens",
                "medium_answer",
                "long_answer",
            ]].to_string(index=False))
            f.write("\n")

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {out_dir / 'complete_but_wrong_strict_recoveries.csv'}")
    print(f"Saved: {out_dir / 'complete_but_wrong_relaxed_recoveries.csv'}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
