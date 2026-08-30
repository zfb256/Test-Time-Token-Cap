"""CPU-only diagnostics requested by the latest paper review."""

import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "expanded_repair", ROOT / "48_expanded_repair_eval.py"
)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


def clopper_pearson(k, n, alpha=0.05):
    if not n:
        return 0.0, 1.0
    low = 0.0 if not k else stats.beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def build_cases(base_map, ext_map, base_budget, ext_budget):
    if set(base_map) != set(ext_map):
        raise ValueError("base and extended problem_id sets differ")
    cases = []
    for problem_id, base_record in base_map.items():
        ext_record = ext_map[problem_id]
        for field in ("dataset", "question", "answer"):
            if base_record.get(field) != ext_record.get(field):
                raise ValueError(f"{field} mismatch for problem_id={problem_id}")
        cases.append(
            {
                "problem_id": problem_id,
                "dataset": base_record["dataset"],
                "base": REPAIR.chain_values(
                    base_record, base_budget, REPAIR.VERIFIER
                ),
                "extended": REPAIR.chain_values(
                    ext_record, ext_budget, REPAIR.VERIFIER
                ),
            }
        )
    return cases


def complete_correct_definitions(values):
    complete_parseable = bool(values) and all(
        value["complete"] and value["answer"] is not None for value in values
    )
    vote = REPAIR.plurality(values)
    return {
        "complete_plurality_correct": complete_parseable and vote["correct"],
        "complete_all_correct": complete_parseable
        and all(value["correct"] for value in values),
    }


def symmetric_harm(cases, eligibility_values=None):
    rows = []
    for case in cases:
        source_vote = REPAIR.plurality(case["base"])
        ext_vote = REPAIR.plurality(case["extended"])
        selector = (
            case["base"]
            if eligibility_values is None
            else eligibility_values[case["problem_id"]]
        )
        for definition, eligible in complete_correct_definitions(
            selector
        ).items():
            rows.append(
                {
                    "problem_id": case["problem_id"],
                    "dataset": case["dataset"],
                    "definition": definition,
                    "eligible": eligible,
                    "harm": (
                        eligible
                        and source_vote["correct"]
                        and not ext_vote["correct"]
                    ),
                    "certain_harm": (
                        eligible
                        and source_vote["lower"]
                        and not ext_vote["upper"]
                    ),
                    "possible_harm": (
                        eligible
                        and source_vote["upper"]
                        and not ext_vote["lower"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_harm(detail):
    rows = []
    groups = [("all", detail)]
    groups.extend((name, group) for name, group in detail.groupby("dataset"))
    for dataset, dataset_frame in groups:
        for definition, group in dataset_frame.groupby("definition", sort=False):
            eligible = group[group["eligible"]]
            n = len(eligible)
            k = int(eligible["harm"].sum())
            low, high = clopper_pearson(k, n)
            rows.append(
                {
                    "dataset": dataset,
                    "definition": definition,
                    "n_eligible": n,
                    "harm_count": k,
                    "harm_rate": k / n if n else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                    "certain_harm_count": int(eligible["certain_harm"].sum()),
                    "possible_harm_count": int(eligible["possible_harm"].sum()),
                }
            )
    return pd.DataFrame(rows)


def split_null(cases, k=4):
    expected = 2 * k
    if any(len(case["base"]) != expected for case in cases):
        raise ValueError(f"4+4 null requires exactly {expected} base chains")
    assignments = list(itertools.combinations(range(expected), k))
    rows = []
    all_indices = set(range(expected))
    for assignment_id, selected_indices in enumerate(assignments):
        selected = set(selected_indices)
        control_indices = sorted(all_indices - selected)
        eligible_count = repair_count = 0
        for case in cases:
            selector = [case["base"][index] for index in selected_indices]
            if not REPAIR.eligibility(selector)["strict_all_wrong"]:
                continue
            eligible_count += 1
            control = [case["base"][index] for index in control_indices]
            repair_count += int(REPAIR.plurality(control)["correct"])
        rows.append(
            {
                "assignment": assignment_id,
                "selector_indices": ",".join(map(str, selected_indices)),
                "n_eligible": eligible_count,
                "repair_count": repair_count,
                "repair_rate": (
                    repair_count / eligible_count
                    if eligible_count
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def exact_accuracy(values, k):
    votes = (
        REPAIR.plurality([values[index] for index in indices])["correct"]
        for indices in itertools.combinations(range(len(values)), k)
    )
    return float(np.mean(list(votes)))


def factorial(cases):
    rows = []
    datasets = ["all", *sorted({case["dataset"] for case in cases})]
    for dataset in datasets:
        selected = (
            cases
            if dataset == "all"
            else [case for case in cases if case["dataset"] == dataset]
        )
        for budget in ("base", "extended"):
            chain_count = {len(case[budget]) for case in selected}
            if len(chain_count) != 1:
                raise ValueError(f"inconsistent {budget} chain counts")
            max_k = next(iter(chain_count))
            for k in (1, max_k):
                rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "k": k,
                        "accuracy": float(
                            np.mean(
                                [
                                    exact_accuracy(case[budget], k)
                                    for case in selected
                                ]
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def factorial_effects(cells):
    rows = []
    for dataset, group in cells.groupby("dataset", sort=False):
        values = {
            (row["budget"], int(row["k"])): row["accuracy"]
            for _, row in group.iterrows()
        }
        ks = sorted({key[1] for key in values})
        low_k, high_k = ks[0], ks[-1]
        token_low = values[("extended", low_k)] - values[("base", low_k)]
        token_high = values[("extended", high_k)] - values[("base", high_k)]
        chain_base = values[("base", high_k)] - values[("base", low_k)]
        chain_extended = (
            values[("extended", high_k)] - values[("extended", low_k)]
        )
        rows.append(
            {
                "dataset": dataset,
                "token_effect_k1": token_low,
                "token_effect_kmax": token_high,
                "chain_effect_base": chain_base,
                "chain_effect_extended": chain_extended,
                "interaction": token_high - token_low,
            }
        )
    return pd.DataFrame(rows)


def transition_counts(cases):
    base = np.array(
        [REPAIR.plurality(case["base"])["correct"] for case in cases]
    )
    extended = np.array(
        [REPAIR.plurality(case["extended"])["correct"] for case in cases]
    )
    return {
        "n": len(cases),
        "helpful": int((~base & extended).sum()),
        "harmful": int((base & ~extended).sum()),
        "net": int(extended.sum() - base.sum()),
    }


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

    results_dir = Path(args.results_dir)
    base = REPAIR.load_map(
        results_dir / f"inference_{args.base_name}.jsonl"
    )
    extended = REPAIR.load_map(
        results_dir / f"inference_{args.ext_name}.jsonl"
    )
    REPAIR.validate_run(base, "base", args.base_budget)
    REPAIR.validate_run(extended, "extended", args.ext_budget)
    cases = build_cases(base, extended, args.base_budget, args.ext_budget)

    tag = f"{args.base_name}_to_{args.ext_name}"
    prefix = results_dir / f"review_diagnostics_{tag}"
    harm = summarize_harm(symmetric_harm(cases))
    null = split_null(cases)
    cells = factorial(cases)
    effects = factorial_effects(cells)
    harm.to_csv(f"{prefix}_harm.csv", index=False)
    null.to_csv(f"{prefix}_split4_null.csv", index=False)
    cells.to_csv(f"{prefix}_factorial_cells.csv", index=False)
    effects.to_csv(f"{prefix}_factorial_effects.csv", index=False)

    transitions = transition_counts(cases)
    eligible_assignments = int(null["n_eligible"].sum())
    repair_assignments = int(null["repair_count"].sum())
    pooled_rate = (
        repair_assignments / eligible_assignments
        if eligible_assignments
        else float("nan")
    )
    with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("CPU Review Diagnostics\n")
        handle.write("=" * 40 + "\n\n")
        handle.write(f"Transitions: {transitions}\n\n")
        handle.write("Symmetric complete-correct -> wrong:\n")
        handle.write(harm.to_string(index=False))
        handle.write("\n\nExact 4+4 same-budget split null:\n")
        handle.write(
            f"assignments={len(null)}, "
            f"eligible_assignments={eligible_assignments}, "
            f"repair_assignments={repair_assignments}, "
            f"pooled_rate={pooled_rate:.6f}\n"
        )
        handle.write(
            "This is a k=4 diagnostic over correlated exact assignments, "
            "not an independent-sample confidence interval.\n\n"
        )
        handle.write("Budget x chain-count cells:\n")
        handle.write(cells.to_string(index=False))
        handle.write("\n\nEffects:\n")
        handle.write(effects.to_string(index=False))
        handle.write("\n")
    print(f"Saved diagnostics with prefix: {prefix}")


if __name__ == "__main__":
    main()
