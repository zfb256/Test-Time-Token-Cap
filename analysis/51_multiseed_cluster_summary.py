"""Aggregate repeated-seed detail CSVs with problem-cluster uncertainty."""

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


def coerce_numeric(series, label):
    """Parse numeric/boolean CSV columns without treating "False" as true."""
    if pd.api.types.is_bool_dtype(series):
        values = series.astype(int)
    else:
        def normalize(value):
            if isinstance(value, (bool, np.bool_)):
                return int(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "true":
                    return 1
                if lowered == "false":
                    return 0
            return value

        normalized = series.map(normalize)
        try:
            values = pd.to_numeric(normalized, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric or boolean") from error
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    return values.astype(float)


def wilson_interval(successes, trials, alpha=0.05):
    """Two-sided Wilson score interval for an auxiliary count-based bound."""
    if trials <= 0 or successes < 0 or successes > trials:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _estimate(frame, metric, denominator=None):
    if denominator is None:
        return float(frame[metric].mean())
    denominator_sum = float(frame[denominator].sum())
    if denominator_sum == 0:
        return float("nan")
    return float(frame[metric].sum() / denominator_sum)


def cluster_bootstrap(frame, metric, denominator=None, seed=42, n_boot=10000):
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    problem_ids = frame["problem_id"].drop_duplicates().to_numpy()
    if not len(problem_ids):
        return float("nan"), float("nan")
    columns = [metric] if denominator is None else [metric, denominator]
    grouped = frame.groupby("problem_id")[columns].sum().reindex(problem_ids)
    metric_values = grouped[metric].to_numpy(dtype=float)
    if denominator is None:
        # Seeds are balanced by the validation in aggregate().
        metric_values /= frame["seed_label"].nunique()
    else:
        denominator_values = grouped[denominator].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        sampled = rng.integers(0, len(problem_ids), size=(size, len(problem_ids)))
        numerators = metric_values[sampled].sum(axis=1)
        if denominator is None:
            means[start : start + size] = numerators / len(problem_ids)
        else:
            denominators = denominator_values[sampled].sum(axis=1)
            means[start : start + size] = np.divide(
                numerators,
                denominators,
                out=np.full(size, np.nan),
                where=denominators != 0,
            )
    if np.isnan(means).all():
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.nanquantile(means, [0.025, 0.975]))


def aggregate(
    frames,
    labels,
    metric,
    denominator=None,
    seed=42,
    n_boot=10000,
    binary_metric=None,
):
    if len(frames) != len(labels):
        raise ValueError("one label is required for each input")
    if not frames:
        raise ValueError("at least one input frame is required")
    if len(set(labels)) != len(labels):
        raise ValueError("seed labels must be unique")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    tagged = []
    reference_ids = None
    for frame, label in zip(frames, labels):
        required = {"problem_id", metric}
        if denominator is not None:
            required.add(denominator)
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} missing columns {required - set(frame.columns)}")
        if frame["problem_id"].duplicated().any():
            raise ValueError(f"{label} has duplicate problem_id rows")
        ids = set(frame["problem_id"])
        if reference_ids is not None and ids != reference_ids:
            raise ValueError("seed files have different problem_id sets")
        reference_ids = ids
        columns = ["problem_id", metric]
        if denominator is not None:
            columns.append(denominator)
        copy = frame[columns].copy()
        copy[metric] = coerce_numeric(copy[metric], f"{label}.{metric}")
        if denominator is not None:
            copy[denominator] = coerce_numeric(
                copy[denominator], f"{label}.{denominator}"
            )
            if (copy[denominator] < 0).any():
                raise ValueError(f"{label}.{denominator} must be nonnegative")
        copy["seed_label"] = label
        tagged.append(copy)
    pooled = pd.concat(tagged, ignore_index=True)
    observed_binary = pooled[metric].isin([0.0, 1.0]).all()
    if binary_metric is None:
        binary_metric = bool(observed_binary)
    elif binary_metric and not observed_binary:
        raise ValueError(f"{metric} was declared binary but contains other values")
    if binary_metric and denominator is not None:
        if (pooled[metric] > pooled[denominator]).any():
            raise ValueError(
                f"{metric} contains successes outside or above {denominator}"
            )
    lo, hi = cluster_bootstrap(pooled, metric, denominator, seed, n_boot)

    def result_row(scope, subset, n_seeds):
        denominator_count = (
            float(subset[denominator].sum())
            if denominator is not None
            else float(len(subset))
        )
        numerator_count = float(subset[metric].sum())
        wilson_lo, wilson_hi = (
            wilson_interval(numerator_count, denominator_count)
            if binary_metric
            else (float("nan"), float("nan"))
        )
        return {
            "scope": scope,
            "n_seeds": n_seeds,
            "n_problems": len(reference_ids or []),
            "numerator_count": numerator_count,
            "denominator_count": denominator_count,
            "estimate": _estimate(subset, metric, denominator),
            "binomial_wilson95_lower": wilson_lo,
            "binomial_wilson95_upper": wilson_hi,
        }

    overall = result_row("all_seeds", pooled, len(labels))
    overall["cluster_ci95_lower"] = lo
    overall["cluster_ci95_upper"] = hi
    rows = [
        overall
    ]
    if len(labels) > 1:
        for omitted in labels:
            subset = pooled[pooled["seed_label"] != omitted]
            omit_lo, omit_hi = cluster_bootstrap(
                subset, metric, denominator, seed, n_boot
            )
            row = result_row(f"leave_out:{omitted}", subset, len(labels) - 1)
            row["cluster_ci95_lower"] = omit_lo
            row["cluster_ci95_upper"] = omit_hi
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument(
        "--denominator",
        help="optional indicator/count column for a pooled conditional rate",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--metric-type",
        choices=("auto", "binary", "nonbinary"),
        default="auto",
        help="Declare metric semantics; use nonbinary for signed effects.",
    )
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    summary = aggregate(
        frames,
        args.labels,
        args.metric,
        args.denominator,
        args.bootstrap_seed,
        args.n_boot,
        None
        if args.metric_type == "auto"
        else args.metric_type == "binary",
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
