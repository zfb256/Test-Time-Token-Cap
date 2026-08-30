"""Plan additional seeds from observed eligible/repair counts.

Reports expected eligible cases and the exact 95% upper repair-rate bound when
no repairs are observed.  This is a planning calculation; final inference must
cluster by problem because seeds repeat the same benchmark questions.
"""

import argparse
import math

import pandas as pd


def zero_event_upper(n, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return 1.0 if n <= 0 else 1.0 - (alpha / 2.0) ** (1.0 / n)


def make_plan(n_problems, eligible_rate, seeds):
    if n_problems < 1:
        raise ValueError("n_problems must be positive")
    if not 0 <= eligible_rate <= 1:
        raise ValueError("eligible_rate must lie in [0, 1]")
    if not seeds or any(seed_count < 1 for seed_count in seeds):
        raise ValueError("seed counts must be positive")
    rows = []
    for seed_count in seeds:
        expected = n_problems * eligible_rate * seed_count
        conservative_n = math.floor(expected)
        rows.append(
            {
                "n_problems_per_seed": n_problems,
                "eligible_rate": eligible_rate,
                "n_seeds": seed_count,
                "expected_eligible": expected,
                "zero_repairs_upper_95": zero_event_upper(conservative_n),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, required=True)
    parser.add_argument("--eligible-rate", type=float, required=True)
    parser.add_argument("--seeds", default="1,3,5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",")]
        plan = make_plan(args.n_problems, args.eligible_rate, seeds)
    except ValueError as error:
        parser.error(str(error))
    plan.to_csv(args.output, index=False)
    print(plan.to_string(index=False))
    print(
        "\nCaution: this binomial calculation is for operational planning only; "
        "final uncertainty must resample problems as clusters."
    )


if __name__ == "__main__":
    main()
