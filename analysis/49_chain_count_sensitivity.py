"""Exact k-of-8 subsampling audit for plurality accuracy and tie sensitivity."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier  # noqa: E402

from importlib.util import module_from_spec, spec_from_file_location

_spec = spec_from_file_location("expanded_repair", SCRIPT_DIR / "48_expanded_repair_eval.py")
_repair = module_from_spec(_spec)
_spec.loader.exec_module(_repair)


def exact_subsample_record(record, budget, ks, verifier):
    values = _repair.chain_values(record, budget, verifier)
    rows = []
    for k in ks:
        if k > len(values):
            continue
        votes = [
            _repair.plurality([values[index] for index in combo])
            for combo in itertools.combinations(range(len(values)), k)
        ]
        rows.append(
            {
                "k": k,
                "n_subsets": len(votes),
                "accuracy": np.mean([vote["correct"] for vote in votes]),
                "tie_rate": np.mean([vote["tie"] for vote in votes]),
                "tie_lower_accuracy": np.mean([vote["lower"] for vote in votes]),
                "tie_upper_accuracy": np.mean([vote["upper"] for vote in votes]),
            }
        )
    return rows


def parse_ks(value):
    try:
        ks = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--ks must be a comma-separated integer list") from error
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("--ks values must be unique positive integers")
    return ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", default="1,2,4,8")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.budget < 1:
        parser.error("--budget must be positive")
    try:
        ks = parse_ks(args.ks)
    except ValueError as error:
        parser.error(str(error))
    verifier = MathVerifier()
    detail = []
    seen = set()
    declared_n_chains = None
    with open(input_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record["problem_id"]
            if problem_id in seen:
                raise ValueError(
                    f"{input_path}:{line_no}: duplicate problem_id={problem_id}"
                )
            seen.add(problem_id)
            metadata = record.get("budget", {})
            if metadata.get("max_new_tokens") != args.budget:
                raise ValueError(
                    f"{input_path}:{line_no}: metadata cap "
                    f"{metadata.get('max_new_tokens')} != {args.budget}"
                )
            n_chains = metadata.get("n_chains")
            if len(record.get("chains", [])) != n_chains:
                raise ValueError(
                    f"{input_path}:{line_no}: actual/declared chain counts differ"
                )
            if declared_n_chains is None:
                declared_n_chains = n_chains
                too_large = [k for k in ks if k > declared_n_chains]
                if too_large:
                    raise ValueError(
                        f"--ks values {too_large} exceed n_chains="
                        f"{declared_n_chains}"
                    )
            elif n_chains != declared_n_chains:
                raise ValueError(f"{input_path} has inconsistent chain counts")
            for result in exact_subsample_record(
                record, args.budget, ks, verifier
            ):
                detail.append(
                    {
                        "problem_id": problem_id,
                        "dataset": record["dataset"],
                        **result,
                    }
                )
    if not detail:
        raise ValueError(
            "no subsamples were produced; ensure the input is nonempty and "
            "--ks does not exceed n_chains"
        )
    frame = pd.DataFrame(detail)
    summary = (
        frame.groupby(["dataset", "k"], as_index=False)
        .agg(
            n_problems=("problem_id", "size"),
            n_subsets=("n_subsets", "first"),
            accuracy=("accuracy", "mean"),
            tie_rate=("tie_rate", "mean"),
            tie_lower_accuracy=("tie_lower_accuracy", "mean"),
            tie_upper_accuracy=("tie_upper_accuracy", "mean"),
        )
    )
    overall = (
        frame.groupby("k", as_index=False)
        .agg(
            n_problems=("problem_id", "size"),
            n_subsets=("n_subsets", "first"),
            accuracy=("accuracy", "mean"),
            tie_rate=("tie_rate", "mean"),
            tie_lower_accuracy=("tie_lower_accuracy", "mean"),
            tie_upper_accuracy=("tie_upper_accuracy", "mean"),
        )
    )
    overall.insert(0, "dataset", "all")
    summary = pd.concat([overall, summary], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path.with_name(output_path.stem + "_detail.csv"), index=False)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
