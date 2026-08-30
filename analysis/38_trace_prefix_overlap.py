"""
Step 38 (v2): Trace prefix-overlap audit between two budget runs.

Review response (both reviews: "under greedy decoding the long trace should
be a token-level continuation of the medium trace, so complete-but-wrong
fixes are impossible by construction"). This script measures that claim
empirically instead of leaving it implicit:

  - For every problem, compare the serialized medium and long text:
    exact equality, prefix relation, and longest-common-prefix ratio.
  - Split by whether the medium trace hit the token cap (truncated) or
    terminated naturally (complete).

Expected under deterministic greedy decoding:
  - complete medium traces: long text identical -> repair impossible.
  - truncated medium traces: long text extends the medium prefix; any
    divergence indicates numeric nondeterminism worth reporting.

Also supports comparing the independent long rerun against the resumed
run (inference_long_resume.jsonl) once 08_run_continuation.py has run:
    python 38_trace_prefix_overlap.py --a-name long --b-name long_resume

Runs on existing outputs; no GPU needed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


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


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def validate_run(records, label, expected_budget):
    if not records:
        raise SystemExit(f"{label} run is empty")
    for problem_id, record in records.items():
        budget = record.get("budget", {})
        if budget.get("max_new_tokens") != expected_budget:
            raise SystemExit(
                f"{label} cap mismatch for problem_id={problem_id}: "
                f"{budget.get('max_new_tokens')} != {expected_budget}"
            )
        if len(record.get("chains", [])) != 1:
            raise SystemExit(
                f"{label} must contain one chain for problem_id={problem_id}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--a-name", default="medium",
                        help="Reference run (prefix side)")
    parser.add_argument("--b-name", default="long",
                        help="Comparison run (extension side)")
    parser.add_argument("--a-budget", type=int, default=384,
                        help="Token cap of the reference run (for hit-max split)")
    parser.add_argument(
        "--split-name",
        default=None,
        help="Optional run whose cap-hit state defines the complete/truncated split",
    )
    parser.add_argument(
        "--split-budget",
        type=int,
        default=None,
        help="Token cap for --split-name (required when --split-name is used)",
    )
    args = parser.parse_args()
    if (args.split_name is None) != (args.split_budget is None):
        parser.error("--split-name and --split-budget must be provided together")

    out_dir = resolve_path(args.results_dir)
    a_map = load_jsonl_map(out_dir / f"inference_{args.a_name}.jsonl")
    b_map = load_jsonl_map(out_dir / f"inference_{args.b_name}.jsonl")
    validate_run(a_map, args.a_name, args.a_budget)
    if set(a_map) != set(b_map):
        missing_b = sorted(set(a_map) - set(b_map))
        missing_a = sorted(set(b_map) - set(a_map))
        raise SystemExit(
            "Input problem_id sets differ: "
            f"{len(missing_b)} missing from '{args.b_name}' "
            f"(examples: {missing_b[:5]}), "
            f"{len(missing_a)} missing from '{args.a_name}' "
            f"(examples: {missing_a[:5]})."
        )
    split_map = a_map
    split_label = args.a_name
    split_budget = args.a_budget
    if args.split_name is not None:
        split_map = load_jsonl_map(out_dir / f"inference_{args.split_name}.jsonl")
        if set(split_map) != set(a_map):
            raise SystemExit(
                f"Split run '{args.split_name}' does not contain exactly the "
                "same problem_id set as the compared runs."
            )
        split_label = args.split_name
        split_budget = args.split_budget
    validate_run(split_map, split_label, split_budget)
    if not b_map:
        raise SystemExit(f"{args.b_name} run is empty")
    for problem_id, record in b_map.items():
        if len(record.get("chains", [])) != 1:
            raise SystemExit(
                f"{args.b_name} must contain one chain for "
                f"problem_id={problem_id}"
            )

    rows = []
    for pid, a_rec in a_map.items():
        if a_rec["dataset"] != b_map[pid]["dataset"]:
            raise SystemExit(f"dataset mismatch for problem_id={pid}")
        if a_rec.get("question") != b_map[pid].get("question"):
            raise SystemExit(f"question mismatch for problem_id={pid}")
        if a_rec.get("answer") != b_map[pid].get("answer"):
            raise SystemExit(f"ground-truth answer mismatch for problem_id={pid}")
        for field in ("dataset", "question", "answer"):
            if split_map[pid].get(field) != a_rec.get(field):
                raise SystemExit(
                    f"split-run {field} mismatch for problem_id={pid}"
                )
        a_text = a_rec["chains"][0]["text"]
        b_text = b_map[pid]["chains"][0]["text"]
        a_chain = a_rec["chains"][0]
        split_chain = split_map[pid]["chains"][0]
        a_tokens = int(a_chain.get("token_count", 0))
        split_tokens = int(split_chain.get("token_count", 0))
        a_finish = a_chain.get("finish_reason")
        split_finish = split_chain.get("finish_reason")
        lcp = common_prefix_len(a_text, b_text)
        rows.append({
            "problem_id": pid,
            "dataset": a_rec["dataset"],
            "a_tokens": a_tokens,
            "a_hit_max": (
                a_finish == "length"
                if a_finish is not None
                else a_tokens >= args.a_budget
            ),
            "split_run": split_label,
            "split_tokens": split_tokens,
            "split_hit_max": (
                split_finish == "length"
                if split_finish is not None
                else split_tokens >= split_budget
            ),
            "a_chars": len(a_text),
            "b_chars": len(b_text),
            "exact_equal": a_text == b_text,
            "a_is_prefix_of_b": b_text.startswith(a_text),
            "lcp_chars": lcp,
            "lcp_ratio_of_a": lcp / len(a_text) if a_text else 1.0,
        })

    df = pd.DataFrame(rows)
    detail_path = out_dir / f"prefix_overlap_{args.a_name}_vs_{args.b_name}_detail.csv"
    df.to_csv(detail_path, index=False)

    def agg(sub: pd.DataFrame) -> Dict:
        return {
            "n": len(sub),
            "exact_equal_rate": float(sub["exact_equal"].mean()),
            "a_is_prefix_rate": float(sub["a_is_prefix_of_b"].mean()),
            "mean_lcp_ratio": float(sub["lcp_ratio_of_a"].mean()),
            "min_lcp_ratio": float(sub["lcp_ratio_of_a"].min()),
            "n_divergent": int((sub["lcp_ratio_of_a"] < 0.999).sum()),
        }

    summary_rows = []
    for group_name, sub in [
        ("all", df),
        (f"{split_label}_complete", df[~df["split_hit_max"]]),
        (f"{split_label}_truncated", df[df["split_hit_max"]]),
        *[(f"dataset={d}", g) for d, g in df.groupby("dataset")],
    ]:
        row = {"group": group_name}
        row.update(agg(sub))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"prefix_overlap_{args.a_name}_vs_{args.b_name}_summary.csv"
    summary.to_csv(summary_path, index=False)

    txt_path = out_dir / f"prefix_overlap_{args.a_name}_vs_{args.b_name}_summary.txt"
    complete = df[~df["split_hit_max"]]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Prefix Overlap Audit: {args.a_name} vs {args.b_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        if len(complete):
            eq = complete["exact_equal"].mean()
            f.write(
                f"- Among {len(complete)} complete (non-cap-hit) '{split_label}' traces, "
                f"{eq:.1%} of '{args.b_name}' outputs are byte-identical.\n"
            )
            f.write(
                "- If this rate is ~100%, 'no repair of complete-but-wrong traces'\n"
                "  is a property of the deterministic decoding protocol, and must be\n"
                "  reported as such rather than as an empirical finding about the model.\n"
            )
        div = df[df["lcp_ratio_of_a"] < 0.999]
        f.write(f"- Divergent pairs (LCP < 99.9% of '{args.a_name}'): {len(div)}\n")
        if len(div):
            f.write("  These indicate numeric nondeterminism in batched inference;\n")
            f.write("  inspect the detail CSV before making determinism claims.\n")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
