"""Post-trace budget allocation evaluated inside the three-arm design.

The routing study in earlier drafts used separately generated greedy long
outputs, so its routed accuracy was not a counterfactual for the same
trajectory. Arms B and C share a request seed and a token prefix, which makes
"extend this record" an exactly observed intervention: routing a record swaps
its arm-B aggregate for the arm-C aggregate, and leaving it alone keeps arm B.
Routing every record that still has a chain running at the base cap therefore
reproduces fixed-long accuracy exactly, because every other record is
token-identical in the two arms. That value is a reference point rather than an
upper bound: continuation also reverses some already-correct records, so a
router that avoids those can finish above fixed-long accuracy.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CAP = 768
EXTENDED_CAP = 1536
ROUTE_FRACTIONS = tuple(round(0.05 * i, 2) for i in range(21))
CV_FOLDS = 5
RANDOM_SEED = 0
RANDOM_BASELINE_DRAWS = 1000
BOXED = re.compile(r"\\boxed\{")

SCOPES = (
    ("core_3seed", "three_arm_core", ("qwen", "llama"),
     ("seed42", "seed314159", "seed271828")),
    ("full_split", "three_arm_expanded", ("qwen", "llama"), ("seed42",)),
    ("r1_core_3seed", "three_arm_r1", ("r1",),
     ("seed42", "seed314159", "seed271828")),
)
FEATURE_NAMES = (
    "n_cap_hits",
    "frac_cap_hits",
    "max_token_count",
    "mean_token_count",
    "min_token_count",
    "std_token_count",
    "frac_boxed",
    "question_chars",
    "math_level",
    "is_math",
)


def load_scope(outputs_root, model, directory, seeds):
    """Return per-record features, arm-B/arm-C correctness, and identifiers."""
    features, base_correct, extended_correct, keys = [], [], [], []
    for seed in seeds:
        root = outputs_root / directory / f"{model}_{seed}"
        detail = {
            row["problem_id"]: row
            for row in csv.DictReader((root / "three_arm_detail.csv").open(encoding="utf-8"))
            if row["definition"] == "strict_all_wrong"
        }
        with (root / "inference_long_reconstructed.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                row = detail[record["problem_id"]]
                chains = record["chains"]
                tokens = np.array([chain["token_count"] for chain in chains], dtype=float)
                cap_hits = sum(chain["finish_reason"] == "length" for chain in chains)
                boxed = sum(bool(BOXED.search(chain["text"])) for chain in chains)
                features.append([
                    cap_hits,
                    cap_hits / len(chains),
                    tokens.max(),
                    tokens.mean(),
                    tokens.min(),
                    tokens.std(),
                    boxed / len(chains),
                    len(record["question"]),
                    float(record.get("math_level") or 0),
                    float(record["dataset"] != "gsm8k"),
                ])
                base_correct.append(row["samecap_plurality_correct"] == "True")
                extended_correct.append(row["long_plurality_correct"] == "True")
                keys.append((seed, record["problem_id"]))
    return (np.array(features, dtype=float),
            np.array(base_correct, dtype=bool),
            np.array(extended_correct, dtype=bool),
            keys)


def routed_accuracy(base_correct, extended_correct, routed):
    return float(np.where(routed, extended_correct, base_correct).mean())


def top_fraction_mask(scores, fraction, tie_breaker):
    """Route the highest-scoring ceil(fraction * n) records, ties broken explicitly."""
    n = len(scores)
    k = math.ceil(fraction * n)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    order = np.lexsort((-tie_breaker, -scores))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def random_accuracy(base_correct, extended_correct, fraction, rng):
    n = len(base_correct)
    k = math.ceil(fraction * n)
    if k == 0:
        return float(base_correct.mean())
    if k == n:
        return float(extended_correct.mean())
    draws = np.empty(RANDOM_BASELINE_DRAWS)
    for i in range(RANDOM_BASELINE_DRAWS):
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=k, replace=False)] = True
        draws[i] = routed_accuracy(base_correct, extended_correct, mask)
    return float(draws.mean())


def out_of_fold_scores(features, labels, groups, model_name, rng_seed=RANDOM_SEED):
    """Cross-validated routing scores.

    Returns the scores and whether a classifier was actually fitted. Too few
    positives to stratify leaves the scores flat, which makes the learned rows
    collapse onto the tie-breaker; the caller records that so a flat curve is
    never read as a trained router that failed to separate.
    """
    if (len(set(groups[labels])) < CV_FOLDS
            or len(set(groups[~labels])) < CV_FOLDS):
        return np.zeros(len(labels)), False
    if model_name == "logistic":
        build = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))
    else:
        build = lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=rng_seed, n_jobs=-1)
    scores = np.zeros(len(labels))
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=rng_seed
    )
    for train_idx, test_idx in splitter.split(features, labels, groups):
        estimator = build()
        estimator.fit(features[train_idx], labels[train_idx])
        scores[test_idx] = estimator.predict_proba(features[test_idx])[:, 1]
    return scores, True


def cost_savings(fraction):
    """Cap-accounting projections for the two execution models."""
    fixed_long = EXTENDED_CAP
    resume = BASE_CAP + fraction * (EXTENDED_CAP - BASE_CAP)
    rerun = BASE_CAP + fraction * EXTENDED_CAP
    return 1.0 - resume / fixed_long, 1.0 - rerun / fixed_long


def evaluate(model, scope_name, features, base_correct, extended_correct, keys):
    rng = np.random.default_rng(RANDOM_SEED)
    labels = (~base_correct) & extended_correct
    cap_hits = features[:, FEATURE_NAMES.index("n_cap_hits")]
    max_tokens = features[:, FEATURE_NAMES.index("max_token_count")]
    cap_hit_fraction = float((cap_hits > 0).mean())
    if np.any(base_correct[cap_hits == 0] != extended_correct[cap_hits == 0]):
        raise ValueError(f"{model}/{scope_name}: uncapped B/C records differ")
    groups = np.array([problem_id for _, problem_id in keys])

    logistic_scores, logistic_fitted = out_of_fold_scores(
        features, labels, groups, "logistic"
    )
    forest_scores, forest_fitted = out_of_fold_scores(
        features, labels, groups, "forest"
    )
    methods = {
        "cap_hit_rank": (cap_hits, max_tokens),
        "learned_logistic": (logistic_scores, cap_hits),
        "learned_forest": (forest_scores, cap_hits),
    }

    rows = []
    def emit(method, fraction, accuracy, gain):
        resume, rerun = cost_savings(fraction)
        rows.append({
            "model": model,
            "scope": scope_name,
            "method": method,
            "route_fraction": f"{fraction:.4f}",
            "accuracy": f"{accuracy:.4f}",
            "gain_vs_random": "" if gain is None else f"{gain:+.4f}",
            "resume_saving": f"{resume:.4f}",
            "rerun_saving": f"{rerun:.4f}",
        })

    emit("fixed_base", 0.0, float(base_correct.mean()), None)
    emit("fixed_extended", 1.0, float(extended_correct.mean()), None)
    emit("route_every_cap_hit", cap_hit_fraction,
         routed_accuracy(base_correct, extended_correct, cap_hits > 0), None)

    for fraction in ROUTE_FRACTIONS:
        reference = random_accuracy(base_correct, extended_correct, fraction, rng)
        emit("random", fraction, reference, 0.0)
        for method, (scores, tie_breaker) in methods.items():
            mask = top_fraction_mask(scores, fraction, tie_breaker)
            accuracy = routed_accuracy(base_correct, extended_correct, mask)
            emit(method, fraction, accuracy, accuracy - reference)

    summary = {
        "model": model,
        "scope": scope_name,
        "n_records": len(base_correct),
        "base_accuracy": f"{base_correct.mean():.4f}",
        "extended_accuracy": f"{extended_correct.mean():.4f}",
        "cap_hit_fraction": f"{cap_hit_fraction:.4f}",
        "n_routable_flips": int(labels.sum()),
        "n_routable_reversals": int((base_correct & ~extended_correct).sum()),
        "fixed_long_gain": f"{extended_correct.mean() - base_correct.mean():+.4f}",
        "learned_router_fitted": logistic_fitted and forest_fitted,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path,
                        default=SCRIPT_DIR.parent / "outputs")
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR.parent / "outputs" / "three_arm_routing")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these models; default runs every scope.")
    args = parser.parse_args()

    curve_rows, summaries = [], []
    for scope_name, directory, models, seeds in SCOPES:
        for model in models:
            if args.models and model not in args.models:
                continue
            missing = [seed for seed in seeds
                       if not (args.outputs_root / directory / f"{model}_{seed}").is_dir()]
            if missing:
                print(f"skip {model:6} {scope_name:13} "
                      f"(no artifacts for {', '.join(missing)})")
                continue
            features, base_correct, extended_correct, keys = load_scope(
                args.outputs_root, model, directory, seeds)
            rows, summary = evaluate(model, scope_name, features,
                                     base_correct, extended_correct, keys)
            curve_rows.extend(rows)
            summaries.append(summary)
            print(f"{model:6} {scope_name:13} n={summary['n_records']:>5} "
                  f"base={summary['base_accuracy']} ext={summary['extended_accuracy']} "
                  f"cap_hit={summary['cap_hit_fraction']} "
                  f"flips={summary['n_routable_flips']} "
                  f"learned={summary['learned_router_fitted']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "routing_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    with (args.out_dir / "routing_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {args.out_dir / 'routing_curve.csv'}")


if __name__ == "__main__":
    main()
