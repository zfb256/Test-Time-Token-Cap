#!/usr/bin/env python3
"""Audit matched-budget repair against an independent same-cap resample.

The extended run is an independent sample, so an apparent repair can occur
without using any token above the base cap.  This script evaluates the same
extended records twice:

1. at their real extended cap; and
2. as a *completed-chain-only* base-cap control, retaining only chains that
   terminated naturally below the base cap.

The second evaluation exactly identifies repairs supported entirely by
naturally completed base-length chains.  It is deliberately not called an
exact reconstruction of a base-cap run: legacy artifacts do not contain
token IDs or finish reasons, so the prefix of a longer chain cannot be
recovered exactly.  Chains whose stored token_count is at or above the base
cap are censored rather than scored using text generated after the cap.
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import importlib

sampling_eval = importlib.import_module("39_sampling_repair_eval")
MathVerifier = sampling_eval.MathVerifier
clopper_pearson = sampling_eval.clopper_pearson
eval_run = sampling_eval.eval_run
load_jsonl_map = sampling_eval.load_jsonl_map
resolve_path = sampling_eval.resolve_path
validate_run = sampling_eval.validate_run


def censor_to_completed_chains(rec, cap):
    """Return a record containing only chains known to finish below ``cap``.

    An empty chain placeholder preserves k, so pass@1 remains conservative.
    Plurality ignores its missing answer, matching the evaluator's existing
    missing-answer semantics.
    """
    out = copy.deepcopy(rec)
    def completed_by_cap(chain):
        token_count = int(chain.get("token_count", 0))
        if token_count < cap:
            return True
        # New artifacts distinguish an EOS exactly on the boundary from a
        # length stop. Legacy artifacts remain conservatively censored.
        return token_count == cap and chain.get("finish_reason") == "stop"

    out["chains"] = [
        chain
        if completed_by_cap(chain)
        else {
            "text": "",
            "token_count": cap,
            "finish_reason": "length",
            "control_censored": True,
        }
        for chain in rec["chains"]
    ]
    return out


def validate_pair(base_map, ext_map):
    if set(base_map) != set(ext_map):
        raise SystemExit("base and extended problem_id sets differ")
    for pid, base in base_map.items():
        ext = ext_map[pid]
        for field in ("dataset", "question", "answer"):
            if base.get(field) != ext.get(field):
                raise SystemExit(f"{field} mismatch for problem_id={pid}")


def summarize(df):
    rows = []
    for dataset, sub in [("all", df), *df.groupby("dataset")]:
        cbw = sub[sub["base_complete_wrong"]]
        n_cbw = len(cbw)
        real_k = int(cbw["real_repair_point"].sum())
        control_k = int(cbw["control_repair_point"].sum())
        real_lo, real_hi = clopper_pearson(real_k, n_cbw)
        control_lo, control_hi = clopper_pearson(control_k, n_cbw)
        rows.append(
            {
                "dataset": dataset,
                "n": len(sub),
                "n_complete_wrong": n_cbw,
                "base_plurality_accuracy": sub["base_maj_correct"].mean(),
                "real_extended_plurality_accuracy": sub["real_maj_correct"].mean(),
                "completed_only_control_accuracy": sub[
                    "control_maj_correct"
                ].mean(),
                "real_repairs_point": real_k,
                "real_repairs_tie_lower": int(
                    cbw["real_repair_tie_lower"].sum()
                ),
                "real_repairs_tie_upper": int(
                    cbw["real_repair_tie_upper"].sum()
                ),
                "real_repair_ci95": f"[{real_lo:.3f},{real_hi:.3f}]",
                "control_repairs_point": control_k,
                "control_repairs_tie_lower": int(
                    cbw["control_repair_tie_lower"].sum()
                ),
                "control_repairs_tie_upper": int(
                    cbw["control_repair_tie_upper"].sum()
                ),
                "control_repair_ci95": f"[{control_lo:.3f},{control_hi:.3f}]",
                "real_repairs_using_no_extra_tokens": int(
                    cbw["real_repair_point_and_control_point"].sum()
                ),
                "control_censored_chain_rate": sub[
                    "control_censored_chain_rate"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--base-budget", type=int, required=True)
    parser.add_argument("--ext-name", required=True)
    parser.add_argument("--ext-budget", type=int, required=True)
    args = parser.parse_args()

    if args.ext_budget <= args.base_budget:
        parser.error("--ext-budget must exceed --base-budget")
    out_dir = resolve_path(args.results_dir)
    base_map = load_jsonl_map(out_dir / f"inference_{args.base_name}.jsonl")
    ext_map = load_jsonl_map(out_dir / f"inference_{args.ext_name}.jsonl")
    validate_run(base_map, "base", args.base_budget)
    validate_run(ext_map, "extended", args.ext_budget)
    validate_pair(base_map, ext_map)
    verifier = MathVerifier()

    rows = []
    for pid, base_rec in base_map.items():
        ext_rec = ext_map[pid]
        base = eval_run(base_rec, args.base_budget, verifier)
        real = eval_run(ext_rec, args.ext_budget, verifier)
        control_rec = censor_to_completed_chains(ext_rec, args.base_budget)
        control = eval_run(control_rec, args.base_budget, verifier)
        rows.append(
            {
                "problem_id": pid,
                "dataset": base_rec["dataset"],
                "base_complete_wrong": base["complete_wrong"],
                **{f"base_{k}": v for k, v in base.items()},
                **{f"real_{k}": v for k, v in real.items()},
                **{f"control_{k}": v for k, v in control.items()},
                "real_repair_point": base["complete_wrong"]
                and real["maj_correct"],
                "real_repair_tie_lower": base["complete_wrong"]
                and real["maj_correct_all_tie"],
                "real_repair_tie_upper": base["complete_wrong"]
                and real["maj_correct_any_tie"],
                "control_repair_point": base["complete_wrong"]
                and control["maj_correct"],
                "control_repair_tie_lower": base["complete_wrong"]
                and control["maj_correct_all_tie"],
                "control_repair_tie_upper": base["complete_wrong"]
                and control["maj_correct_any_tie"],
                "real_repair_point_and_control_point": base["complete_wrong"]
                and real["maj_correct"]
                and control["maj_correct"],
                "control_censored_chain_rate": sum(
                    not (
                        int(c.get("token_count", 0)) < args.base_budget
                        or (
                            int(c.get("token_count", 0)) == args.base_budget
                            and c.get("finish_reason") == "stop"
                        )
                    )
                    for c in ext_rec["chains"]
                )
                / len(ext_rec["chains"]),
            }
        )

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    tag = f"{args.base_name}_to_{args.ext_name}"
    detail_path = out_dir / f"same_budget_control_{tag}_detail.csv"
    summary_path = out_dir / f"same_budget_control_{tag}_summary.csv"
    text_path = out_dir / f"same_budget_control_{tag}_summary.txt"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    text_path.write_text(
        "Completed-chain-only same-budget resampling control\n"
        "===================================================\n\n"
        + summary.to_string(index=False)
        + "\n\n"
        "Interpretation: chains at or above the base cap are censored because "
        "legacy JSONL files do not preserve their exact base-cap prefixes. "
        "This is an exact audit of repairs supported by naturally completed "
        "base-length chains, not a full reconstruction of a same-cap run.\n",
        encoding="utf-8",
    )
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
