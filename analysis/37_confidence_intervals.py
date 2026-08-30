"""
Step 37 (v2): Statistical confidence for the main claims.

Review response (both reviews: "no CIs anywhere, 0/32 is weak evidence"):

  1. Exact Clopper-Pearson intervals for zero-event rates
     (complete-but-wrong long-fix rates), stating explicitly which true
     repair rates the data can and cannot rule out.
  2. Bootstrap CIs (B=10000) for the headline accuracies and helpful rates.
  3. Paired bootstrap comparison of the cheap learned router vs the
     transparent hit-max rule AT THE SAME route fraction, which the
     original Table 1 did not provide.

Runs on existing outputs; no GPU needed.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boolean_series(series, label):
    if pd.api.types.is_bool_dtype(series):
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains missing or non-boolean values")
    return mapped.astype(bool)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact two-sided (1-alpha) CI for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def bootstrap_ci(values: np.ndarray, b: int = 10000, seed: int = 42, alpha: float = 0.05):
    if len(values) == 0 or b < 1:
        raise ValueError("bootstrap requires nonempty values and b >= 1")
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(b, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def routed_correct(df: pd.DataFrame, route_mask: np.ndarray) -> np.ndarray:
    return np.where(route_mask, df["strict_long_correct"].values,
                    df["strict_medium_correct"].values).astype(float)


def paired_bootstrap_diff(a: np.ndarray, b_arr: np.ndarray, b: int = 10000, seed: int = 42):
    """CI for mean(a) - mean(b) with paired resampling, plus one-sided p."""
    if len(a) == 0 or len(a) != len(b_arr) or b < 1:
        raise ValueError("paired bootstrap requires equal nonempty arrays and b >= 1")
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(b, n))
    diffs = a[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    p_le_zero = float((diffs <= 0).mean())
    return lo, hi, p_le_zero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    parser.add_argument("--route-fraction", type=float, default=0.30)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.route_fraction <= 1:
        parser.error("--route-fraction must lie in [0, 1]")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")

    out_dir = resolve_path(args.results_dir)
    detail = pd.read_csv(out_dir / "complete_but_wrong_detail.csv")
    detail = detail.sort_values("problem_id").reset_index(drop=True)
    n = len(detail)
    if n == 0 or detail["problem_id"].duplicated().any():
        raise ValueError("detail table must contain unique, nonempty problem rows")
    for column in (
        "strict_complete_wrong",
        "relaxed_complete_wrong",
        "strict_medium_correct",
        "strict_long_correct",
        "relaxed_long_correct",
        "medium_hit_max",
    ):
        detail[column] = boolean_series(detail[column], column)
    rows = []

    # ---- 1. Zero-event Clopper-Pearson bounds --------------------------
    subsets = {
        "strict_complete_wrong_all": detail[detail["strict_complete_wrong"]],
        "strict_complete_wrong_gsm8k": detail[detail["strict_complete_wrong"] & (detail["dataset"] == "gsm8k")],
        "strict_complete_wrong_math": detail[detail["strict_complete_wrong"] & (detail["dataset"] == "math")],
        "relaxed_complete_wrong_all": detail[detail["relaxed_complete_wrong"]],
    }
    for name, sub in subsets.items():
        col = "strict_long_correct" if name.startswith("strict") else "relaxed_long_correct"
        k = int(sub[col].sum())
        lo, hi = clopper_pearson(k, len(sub))
        rows.append({
            "quantity": f"long_fix_rate[{name}]",
            "estimate": k / len(sub) if len(sub) else float("nan"),
            "n": len(sub), "events": k,
            "ci_low": lo, "ci_high": hi,
            "method": "clopper_pearson_95",
        })

    # ---- 2. Bootstrap CIs for headline rates ---------------------------
    med = detail["strict_medium_correct"].values.astype(float)
    lng = detail["strict_long_correct"].values.astype(float)
    helpful = ((detail["strict_medium_correct"] == False) & detail["strict_long_correct"]).values.astype(float)  # noqa: E712
    for name, vals in [("fixed_medium_acc", med), ("fixed_long_acc", lng),
                       ("helpful_rate", helpful)]:
        lo, hi = bootstrap_ci(vals, args.bootstrap, args.seed)
        rows.append({"quantity": name, "estimate": float(vals.mean()), "n": n,
                     "events": int(vals.sum()), "ci_low": lo, "ci_high": hi,
                     "method": f"bootstrap_{args.bootstrap}"})

    # ---- 3. Routers at the same operating point ------------------------
    # Historical router features are optional in the canonical mechanism
    # artifact bundle. Keep the primary confidence report usable when those
    # frozen feature files are intentionally absent.
    router_inputs = (
        out_dir / "features_medium.jsonl",
        out_dir / "inference_medium.jsonl",
    )
    router_available = all(path.is_file() for path in router_inputs)
    if router_available:
        mod22 = _load_module("cheap_router_analysis", "22_cheap_router_analysis.py")
        feature_rows = load_jsonl(router_inputs[0])
        if len({row["problem_id"] for row in feature_rows}) != len(feature_rows):
            raise ValueError("features_medium.jsonl has duplicate problem_id rows")
        feats = {row["problem_id"]: row for row in feature_rows}
        medium_rows = load_jsonl(router_inputs[1])
        if len({row["problem_id"] for row in medium_rows}) != len(medium_rows):
            raise ValueError("inference_medium.jsonl has duplicate problem_id rows")
        medium_map = {row["problem_id"]: row for row in medium_rows}
        missing = set(detail["problem_id"]) - set(feats)
        if missing:
            raise ValueError(
                f"features_medium.jsonl lacks {len(missing)} evaluated problems"
            )
        records = [feats[pid] for pid in detail["problem_id"]]
        y = pd.Series([int(record["label"]) for record in records])
        X = mod22.build_feature_sets(records, medium_map)["cheap"]
        probs_lr, _ = mod22.oof_probs(
            X, y, mod22.build_lr(), folds=5, random_state=args.seed
        )
        probs_rf, _ = mod22.oof_probs(
            X, y, mod22.build_rf(), folds=5, random_state=args.seed
        )

        r = args.route_fraction
        k_route = int(np.ceil(r * n))

        def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
            order = np.argsort(-scores, kind="stable")
            mask = np.zeros(len(scores), dtype=bool)
            mask[order[:k]] = True
            return mask

        hit_max = detail["medium_hit_max"].to_numpy(dtype=bool)
        rule_scores = (
            hit_max.astype(float) * 1e6
            + detail["medium_tokens"].to_numpy(dtype=float)
        )
        strategies = {
            f"cheap_LR_top{int(r*100)}": topk_mask(probs_lr, k_route),
            f"cheap_RF_top{int(r*100)}": topk_mask(probs_rf, k_route),
            f"rule_hitmax_rank_top{int(r*100)}": topk_mask(
                rule_scores, k_route
            ),
            "rule_hit_max_natural": hit_max,
        }
        outcomes = {}
        for name, mask in strategies.items():
            vals = routed_correct(detail, mask)
            outcomes[name] = vals
            lo, hi = bootstrap_ci(vals, args.bootstrap, args.seed)
            rows.append({
                "quantity": f"routed_acc[{name}]",
                "estimate": float(vals.mean()),
                "n": n,
                "events": int(vals.sum()),
                "ci_low": lo,
                "ci_high": hi,
                "method": f"bootstrap_{args.bootstrap}",
                "route_fraction": float(mask.mean()),
            })

        comparisons = [
            (
                f"cheap_LR_top{int(r*100)}",
                f"rule_hitmax_rank_top{int(r*100)}",
            ),
            (
                f"cheap_RF_top{int(r*100)}",
                f"rule_hitmax_rank_top{int(r*100)}",
            ),
        ]
        for a_name, b_name in comparisons:
            lo, hi, p = paired_bootstrap_diff(
                outcomes[a_name],
                outcomes[b_name],
                args.bootstrap,
                args.seed,
            )
            rows.append({
                "quantity": f"diff[{a_name} - {b_name}]",
                "estimate": float(
                    outcomes[a_name].mean() - outcomes[b_name].mean()
                ),
                "n": n,
                "events": None,
                "ci_low": lo,
                "ci_high": hi,
                "method": f"paired_bootstrap_{args.bootstrap}",
                "p_diff_le_0": p,
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "confidence_intervals.csv"
    df.to_csv(csv_path, index=False)

    txt_path = out_dir / "confidence_intervals.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Statistical Confidence Summary (v2)\n")
        f.write("===================================\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nReading guide:\n")
        f.write("- Zero-event rows: the 95% upper bound is the largest true fix rate\n")
        f.write("  still compatible with observing 0 fixes; claims must be phrased as\n")
        f.write("  'we cannot rule out repair rates below this bound'.\n")
        f.write("- diff rows compare routers at the SAME route fraction; p_diff_le_0 is\n")
        f.write("  the one-sided bootstrap probability that the difference is <= 0.\n")
        if not router_available:
            f.write(
                "- Router rows were skipped because the optional frozen "
                "features_medium.jsonl artifact is absent.\n"
            )
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
