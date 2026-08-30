"""Aggregate repeated-seed three-arm decomposition outputs."""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "multiseed", SCRIPT_DIR / "51_multiseed_cluster_summary.py"
)
MULTISEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MULTISEED)

OUTCOMES = (
    "samecap_repair_plurality",
    "long_repair_plurality",
    "samecap_repair_any",
    "long_repair_any",
    "incremental_plurality_helpful",
    "incremental_plurality_harmful",
    "incremental_net",
)


def _binary(series, label):
    values = MULTISEED.coerce_numeric(series, label)
    if not values.isin([0.0, 1.0]).all():
        raise ValueError(f"{label} must contain only binary values")
    return values.astype(int)


def _validate_frames(frames, labels):
    reference = None
    for frame, label in zip(frames, labels):
        required = {"problem_id", "dataset", "definition", "eligible"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{label} missing columns {required - set(frame.columns)}"
            )
        if frame.duplicated(["problem_id", "definition"]).any():
            raise ValueError(f"{label} has duplicate problem/definition rows")
        mapping = {
            (row["problem_id"], row["definition"]): row["dataset"]
            for _, row in frame.iterrows()
        }
        if reference is not None and mapping != reference:
            raise ValueError(
                "three-arm seed files have different "
                "problem/definition/dataset mappings"
            )
        reference = mapping


def _load_harm_frames(root, model, seeds):
    frames = []
    reference = None
    for seed in seeds:
        path = root / f"{model}_seed{seed}" / "three_arm_harm_detail.csv"
        frame = pd.read_csv(path)
        required = {
            "problem_id",
            "dataset",
            "contrast",
            "definition",
            "eligible",
            "harm",
            "certain_harm",
            "possible_harm",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} missing columns {required - set(frame.columns)}")
        if frame.duplicated(["problem_id", "contrast", "definition"]).any():
            raise ValueError(f"{path} has duplicate problem/contrast/definition rows")
        mapping = {
            (row["problem_id"], row["contrast"], row["definition"]): row["dataset"]
            for _, row in frame.iterrows()
        }
        if reference is not None and mapping != reference:
            raise ValueError("three-arm harm seed files have different mappings")
        reference = mapping
        for column in ("eligible", "harm", "certain_harm", "possible_harm"):
            frame[column] = _binary(frame[column], f"{model}/{seed}/{column}")
        frames.append(frame)
    return frames


def aggregate_root(root, models, seeds, bootstrap_seed=42, n_boot=10000):
    rows = []
    for model in models:
        frames = []
        for seed in seeds:
            path = root / f"{model}_seed{seed}" / "three_arm_detail.csv"
            frame = pd.read_csv(path)
            frame["eligible"] = _binary(
                frame["eligible"], f"{model}/{seed}/eligible"
            )
            for outcome in OUTCOMES:
                if outcome == "incremental_net":
                    continue
                if outcome not in frame:
                    raise ValueError(f"{path} missing column {outcome}")
                frame[outcome] = _binary(
                    frame[outcome], f"{model}/{seed}/{outcome}"
                )
            frame["incremental_net"] = (
                frame["incremental_plurality_helpful"]
                - frame["incremental_plurality_harmful"]
            )
            frames.append(frame)
        _validate_frames(frames, seeds)
        datasets = ["all", *sorted(set(frames[0]["dataset"]))]
        definitions = list(frames[0]["definition"].drop_duplicates())
        for dataset in datasets:
            for definition in definitions:
                selected = []
                for frame in frames:
                    subset = frame[frame["definition"] == definition]
                    if dataset != "all":
                        subset = subset[subset["dataset"] == dataset]
                    selected.append(subset)
                for outcome in OUTCOMES:
                    result = MULTISEED.aggregate(
                        selected,
                        seeds,
                        outcome,
                        denominator="eligible",
                        seed=bootstrap_seed,
                        n_boot=n_boot,
                        binary_metric=outcome != "incremental_net",
                    )
                    result.insert(0, "outcome", outcome)
                    result.insert(0, "definition", definition)
                    result.insert(0, "dataset", dataset)
                    result.insert(0, "model", model)
                    rows.extend(result.to_dict("records"))
        harm_frames = _load_harm_frames(root, model, seeds)
        datasets = ["all", *sorted(set(harm_frames[0]["dataset"]))]
        contrasts = list(harm_frames[0]["contrast"].drop_duplicates())
        definitions = list(harm_frames[0]["definition"].drop_duplicates())
        for dataset in datasets:
            for contrast in contrasts:
                for definition in definitions:
                    selected = []
                    for frame in harm_frames:
                        subset = frame[
                            (frame["contrast"] == contrast)
                            & (frame["definition"] == definition)
                        ]
                        if dataset != "all":
                            subset = subset[subset["dataset"] == dataset]
                        selected.append(subset)
                    for outcome in ("harm", "certain_harm", "possible_harm"):
                        result = MULTISEED.aggregate(
                            selected,
                            seeds,
                            outcome,
                            denominator="eligible",
                            seed=bootstrap_seed,
                            n_boot=n_boot,
                            binary_metric=True,
                        )
                        result.insert(0, "contrast", contrast)
                        result.insert(0, "outcome", outcome)
                        result.insert(0, "definition", definition)
                        result.insert(0, "dataset", dataset)
                        result.insert(0, "model", model)
                        rows.extend(result.to_dict("records"))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--models", nargs="+", default=["qwen", "llama"])
    parser.add_argument("--seeds", nargs="+", default=["42", "314159", "271828"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    result = aggregate_root(
        Path(args.root),
        args.models,
        args.seeds,
        args.bootstrap_seed,
        args.n_boot,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} rows -> {output}")


if __name__ == "__main__":
    main()
