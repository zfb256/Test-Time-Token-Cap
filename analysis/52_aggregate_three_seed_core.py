"""Aggregate the deterministic three-seed core experiment.

The script reports both unconditional endpoint accuracy and repair rates
conditioned on each pre-declared eligibility definition. Confidence intervals
resample problem IDs, keeping all seed observations for a problem together.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pipeline"))
from reproducibility import REQUIRED_RUN_SIGNATURE

SPEC = importlib.util.spec_from_file_location(
    "multiseed_cluster_summary", ROOT / "51_multiseed_cluster_summary.py"
)
MULTISEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MULTISEED)

MODELS = ("qwen", "llama")
SEEDS = ("42", "314159", "271828")
DEFINITIONS = ("strict_all_wrong", "plurality_wrong", "first_chain_wrong")
REPAIR_OUTCOMES = {
    "repair_plurality": "repair_plurality",
    "repair_any": "repair_any",
    "repair_tie_lower": "repair_tie_lower",
    "repair_tie_upper": "repair_tie_upper",
}
ENDPOINT_OUTCOMES = {
    "base_plurality_accuracy": "base_maj_correct",
    "extended_plurality_accuracy": "ext_maj_correct",
    "base_pass_at_1": "base_pass1",
    "extended_pass_at_1": "ext_pass1",
}
BINARY_ENDPOINTS = {
    "base_plurality_accuracy",
    "extended_plurality_accuracy",
}
EXPECTED_MODEL_NAMES = {
    "qwen": "Qwen2.5-Math-7B-Instruct",
    "llama": "Llama-3.1-8B-Instruct",
}


def read_inference_identity(path):
    signatures = {}
    problem_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            record = json.loads(line)
            problem_id = record["problem_id"]
            if problem_id in problem_ids:
                raise ValueError(f"{path}:{line_no}: duplicate problem_id")
            problem_ids.add(problem_id)
            signature = record.get("budget", {}).get("run_signature")
            if (
                not isinstance(signature, dict)
                or not REQUIRED_RUN_SIGNATURE.issubset(signature)
            ):
                raise ValueError(f"{path}:{line_no}: incomplete run signature")
            signatures[json.dumps(signature, sort_keys=True)] = signature
    if len(signatures) != 1 or not problem_ids:
        raise ValueError(f"{path}: inconsistent signature or empty inference file")
    return next(iter(signatures.values())), problem_ids


def validate_model_signatures(model, signatures):
    serialized = {json.dumps(signature, sort_keys=True) for signature in signatures}
    if len(serialized) != 1:
        raise ValueError(f"{model} seeds use different model/runtime signatures")
    signature = signatures[0]
    expected = EXPECTED_MODEL_NAMES.get(model)
    if expected and Path(signature["model"]).name != expected:
        raise ValueError(
            f"{model} directory contains model {signature['model']!r}, "
            f"expected {expected!r}"
        )
    return signature


def read_seed_files(input_root, model, seeds):
    sampling_frames = []
    expanded_frames = []
    signatures = []
    for seed in seeds:
        directory = input_root / f"{model}_seed{seed}"
        inference_path = directory / "inference_xlong_sample8.jsonl"
        sampling_path = (
            directory
            / "sampling_repair_long_reconstructed_to_xlong_sample8_detail.csv"
        )
        expanded_path = (
            directory
            / "expanded_repair_long_reconstructed_to_xlong_sample8_detail.csv"
        )
        if not all(
            path.is_file()
            for path in (inference_path, sampling_path, expanded_path)
        ):
            raise FileNotFoundError(f"incomplete seed output: {directory}")
        signature, problem_ids = read_inference_identity(inference_path)
        sampling = pd.read_csv(sampling_path)
        expanded = pd.read_csv(expanded_path)
        if (
            set(sampling["problem_id"]) != problem_ids
            or set(expanded["problem_id"]) != problem_ids
        ):
            raise ValueError(f"derived CSV problem IDs differ from {inference_path}")
        signatures.append(signature)
        sampling_frames.append(sampling)
        expanded_frames.append(expanded)
    return sampling_frames, expanded_frames, validate_model_signatures(
        model, signatures
    )


def datasets_with_all(frames):
    names = sorted(set(frames[0]["dataset"]))
    return ["all", *names]


def select_dataset(frame, dataset):
    return frame if dataset == "all" else frame[frame["dataset"] == dataset]


def validate_seed_frames(frames, labels, key_columns):
    reference = None
    for frame, label in zip(frames, labels):
        required = {*key_columns, "dataset"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{label} missing columns {required - set(frame.columns)}"
            )
        if frame.duplicated(key_columns).any():
            raise ValueError(f"{label} has duplicate rows for keys {key_columns}")
        mapping = {
            tuple(row[column] for column in key_columns): row["dataset"]
            for _, row in frame.iterrows()
        }
        if reference is not None and mapping != reference:
            raise ValueError(
                "seed files have different problem/definition/dataset mappings"
            )
        reference = mapping


def binary_values(series, label):
    values = MULTISEED.coerce_numeric(series, label)
    if not values.isin([0.0, 1.0]).all():
        raise ValueError(f"{label} must contain only binary values")
    return values.astype(int)


def unit_interval_values(series, label):
    values = MULTISEED.coerce_numeric(series, label)
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{label} must lie in [0, 1]")
    return values


def aggregate_core(input_root, models, seeds, bootstrap_seed=42, n_boot=10000):
    summary_rows = []
    per_seed_rows = []
    model_fingerprints = {}
    for model in models:
        sampling_frames, expanded_frames, signature = read_seed_files(
            input_root, model, seeds
        )
        fingerprint = signature["model_snapshot_sha256"]
        if fingerprint in model_fingerprints:
            raise ValueError(
                f"{model} and {model_fingerprints[fingerprint]} use the same model"
            )
        model_fingerprints[fingerprint] = model
        validate_seed_frames(sampling_frames, seeds, ["problem_id"])
        validate_seed_frames(
            expanded_frames, seeds, ["problem_id", "definition"]
        )

        for dataset in datasets_with_all(sampling_frames):
            frames = [select_dataset(frame, dataset) for frame in sampling_frames]
            for outcome, column in ENDPOINT_OUTCOMES.items():
                prepared_endpoints = []
                for label, frame in zip(seeds, frames):
                    copy = frame[["problem_id", column]].copy()
                    copy[column] = unit_interval_values(
                        copy[column],
                        f"{model}/{label}/{column}",
                    )
                    prepared_endpoints.append(copy)
                result = MULTISEED.aggregate(
                    prepared_endpoints,
                    seeds,
                    column,
                    seed=bootstrap_seed,
                    n_boot=n_boot,
                    binary_metric=outcome in BINARY_ENDPOINTS,
                )
                result.insert(0, "outcome", outcome)
                result.insert(0, "definition", "unconditional")
                result.insert(0, "dataset", dataset)
                result.insert(0, "model", model)
                summary_rows.extend(result.to_dict("records"))
                for label, frame in zip(seeds, prepared_endpoints):
                    per_seed_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "definition": "unconditional",
                            "outcome": outcome,
                            "seed": label,
                            "numerator_count": float(frame[column].sum()),
                            "denominator_count": len(frame),
                            "estimate": float(frame[column].mean()),
                        }
                    )

        for dataset in datasets_with_all(expanded_frames):
            for definition in DEFINITIONS:
                frames = []
                for frame in expanded_frames:
                    selected = select_dataset(frame, dataset)
                    selected = selected[selected["definition"] == definition].copy()
                    if selected["problem_id"].duplicated().any():
                        raise ValueError(
                            f"duplicate problem rows for {model}/{dataset}/{definition}"
                        )
                    frames.append(selected)
                for outcome, column in REPAIR_OUTCOMES.items():
                    prepared = []
                    for label, frame in zip(seeds, frames):
                        copy = frame[["problem_id", "eligible", column]].copy()
                        copy["eligible"] = binary_values(
                            copy["eligible"],
                            f"{model}/{label}/{definition}/eligible",
                        )
                        outcome_values = binary_values(
                            copy[column],
                            f"{model}/{label}/{definition}/{column}",
                        )
                        copy["repair_event"] = copy["eligible"] * outcome_values
                        prepared.append(copy)
                    result = MULTISEED.aggregate(
                        prepared,
                        seeds,
                        "repair_event",
                        denominator="eligible",
                        seed=bootstrap_seed,
                        n_boot=n_boot,
                        binary_metric=True,
                    )
                    result.insert(0, "outcome", outcome)
                    result.insert(0, "definition", definition)
                    result.insert(0, "dataset", dataset)
                    result.insert(0, "model", model)
                    summary_rows.extend(result.to_dict("records"))
                    for label, frame in zip(seeds, prepared):
                        denominator = int(frame["eligible"].sum())
                        numerator = int(frame["repair_event"].sum())
                        per_seed_rows.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "definition": definition,
                                "outcome": outcome,
                                "seed": label,
                                "numerator_count": numerator,
                                "denominator_count": denominator,
                                "estimate": (
                                    numerator / denominator
                                    if denominator
                                    else float("nan")
                                ),
                            }
                        )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_seed_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    summary, per_seed = aggregate_core(
        Path(args.input_root),
        args.models,
        args.seeds,
        args.bootstrap_seed,
        args.n_boot,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "three_seed_cluster_summary.csv", index=False)
    per_seed.to_csv(output_dir / "three_seed_per_seed.csv", index=False)
    print(
        f"Saved {len(summary)} aggregate rows and {len(per_seed)} per-seed rows "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
