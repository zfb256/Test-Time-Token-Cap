"""
Step 40 (v2): Honest cost accounting - rerun vs resume.

Review response (review 2, concern 2): the paper's savings formula
    avg budget = B_m + r * (B_l - B_m)
assumes the upgrade CONTINUES the medium trace, while the text describes a
RERUN. The two differ materially:

  rerun accounting:  routed example costs B_m (routing evidence) + B_l
                     -> avg = B_m + r * B_l
  resume accounting: routed example costs B_m + (B_l - B_m)
                     -> avg = B_m + r * (B_l - B_m)

This script reports both accountings side by side for every routing
strategy, against the honest deployment baseline "fixed long WITHOUT a
medium pass" (cost B_l). It also computes actual-generated-token versions
for the transparent rules using real token counts.

Runs on existing outputs; no GPU needed. After 08_run_continuation.py has
produced inference_long_resume.jsonl, rerun with --resume-name to verify
that resumed answers match the independent long rerun for truncated cases.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def boolean_array(series, label):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains missing or non-boolean values")
    return mapped.to_numpy(dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--medium-budget", type=int, default=384)
    parser.add_argument("--long-budget", type=int, default=768)
    args = parser.parse_args()

    out_dir = resolve_path(args.results_dir)
    detail = pd.read_csv(out_dir / "complete_but_wrong_detail.csv")
    n = len(detail)
    b_m, b_l = float(args.medium_budget), float(args.long_budget)
    if n == 0:
        raise ValueError("complete_but_wrong_detail.csv is empty")
    if not 0 < b_m < b_l:
        parser.error("budgets must satisfy 0 < medium-budget < long-budget")

    hit_max = boolean_array(detail["medium_hit_max"], "medium_hit_max")
    no_boxed = ~boolean_array(detail["medium_has_boxed"], "medium_has_boxed")
    med_ok = boolean_array(
        detail["strict_medium_correct"], "strict_medium_correct"
    )
    long_ok = boolean_array(
        detail["strict_long_correct"], "strict_long_correct"
    )
    med_tokens = detail["medium_tokens"].values.astype(float)
    long_tokens = detail["long_tokens"].values.astype(float)
    if (
        not np.isfinite(med_tokens).all()
        or not np.isfinite(long_tokens).all()
        or (med_tokens < 0).any()
        or (long_tokens < 0).any()
    ):
        raise ValueError("token counts must be finite and nonnegative")

    def strategy_row(name: str, mask) -> Dict:
        r = float(mask.mean())
        acc = float((long_ok[mask].sum() + med_ok[~mask].sum()) / n)
        cap_rerun = b_m + r * b_l
        cap_resume = b_m + r * (b_l - b_m)
        # actual token accounting (per-example):
        actual_rerun = float((med_tokens + mask * long_tokens).mean())
        # resume: truncated routed examples pay the continuation only;
        # approximate continuation cost as (long_tokens - medium_tokens)
        # clipped at 0 (valid when long extends the medium prefix).
        cont = (long_tokens - med_tokens).clip(min=0.0)
        # The continuation runner is a no-op after a natural EOS, even if a
        # routing rule selects that example. Only length-truncated traces can
        # incur continuation tokens.
        cont = cont * hit_max
        actual_resume = float((med_tokens + mask * cont).mean())
        return {
            "strategy": name,
            "route_fraction": r,
            "accuracy": acc,
            "cap_budget_rerun": cap_rerun,
            "savings_rerun_pct": 100.0 * (1 - cap_rerun / b_l),
            "cap_budget_resume": cap_resume,
            "savings_resume_pct": 100.0 * (1 - cap_resume / b_l),
            "actual_tokens_rerun": actual_rerun,
            "estimated_actual_tokens_resume": actual_resume,
        }

    rows: List[Dict] = []
    rows.append({
        "strategy": "fixed_medium",
        "route_fraction": 0.0,
        "accuracy": float(med_ok.mean()),
        "cap_budget_rerun": b_m, "savings_rerun_pct": 100.0 * (1 - b_m / b_l),
        "cap_budget_resume": b_m, "savings_resume_pct": 100.0 * (1 - b_m / b_l),
        "actual_tokens_rerun": float(med_tokens.mean()),
        "estimated_actual_tokens_resume": float(med_tokens.mean()),
    })
    rows.append({
        "strategy": "fixed_long_no_medium_pass",
        "route_fraction": 1.0,
        "accuracy": float(long_ok.mean()),
        "cap_budget_rerun": b_l, "savings_rerun_pct": 0.0,
        "cap_budget_resume": b_l, "savings_resume_pct": 0.0,
        "actual_tokens_rerun": float(long_tokens.mean()),
        "estimated_actual_tokens_resume": float(long_tokens.mean()),
    })
    rows.append(strategy_row("rule_hit_max", hit_max))
    rows.append(strategy_row("rule_no_boxed", no_boxed))
    # random / learned routers at fixed fraction: cap accounting only
    for r in (0.2, 0.3, 0.4):
        k = int(np.ceil(r * n))
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True  # placeholder mask: cap-based numbers depend only on r
        row = strategy_row(f"any_router_top{int(r*100)}pct_cap_only", mask)
        row["accuracy"] = float("nan")  # depends on the router; see main tables
        row["actual_tokens_rerun"] = float("nan")
        row["estimated_actual_tokens_resume"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cost_accounting_v2.csv"
    df.to_csv(csv_path, index=False)

    txt_path = out_dir / "cost_accounting_v2.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Honest Cost Accounting: rerun vs resume (v2)\n")
        f.write("=" * 55 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write(f"- B_m={int(b_m)}, B_l={int(b_l)}; savings measured against fixed long\n")
        f.write("  WITHOUT a medium pass (the honest deployment baseline).\n")
        f.write("- rerun: routed examples pay B_m + B_l (the medium trace is required\n")
        f.write("  as routing evidence and is then discarded).\n")
        f.write("- resume: routed examples pay B_m + (B_l - B_m); requires the system\n")
        f.write("  to continue the truncated trace (08_run_continuation.py validates\n")
        f.write("  and quantifies agreement with independent long reruns).\n")
        f.write("  Naturally terminated traces are a no-op and incur no continuation\n")
        f.write("  tokens even if selected by a routing rule.\n")
        f.write("- estimated_actual_tokens_resume uses the independent long trace\n")
        f.write("  length as a continuation-length proxy; it is not a measured\n")
        f.write("  serving cost from inference_long_resume.jsonl.\n")
        f.write("- The paper must state which accounting it uses; the original 35%\n")
        f.write("  figure is only valid under resume accounting.\n")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
