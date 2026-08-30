#!/usr/bin/env python3
"""Refresh cached verifier labels without rerunning the expensive PRM."""

import collections
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "pipeline"))
from utils.math_verify import MathVerifier  # noqa: E402


def load_map(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out = {}
    for row in rows:
        pid = row["problem_id"]
        if pid in out:
            raise ValueError(f"{path}: duplicate problem_id={pid}")
        out[pid] = row
    return out


def evaluate_chains(record, verifier):
    dataset = record["dataset"]
    truth = record["answer"]
    evaluated = []
    for chain in record.get("chains", []):
        text = chain.get("text", "")
        evaluated.append(
            {
                "correct": bool(verifier.verify(text, truth, dataset)),
                "answer": verifier.extract_prediction_answer(text, dataset),
            }
        )
    return evaluated


def plurality_correct(evaluated):
    valid = [item for item in evaluated if item["answer"] is not None]
    if not valid:
        return False
    counts = collections.Counter(item["answer"] for item in valid)
    max_count = max(counts.values())
    winners = sorted(answer for answer, n in counts.items() if n == max_count)
    winner = winners[0]
    return any(item["correct"] for item in valid if item["answer"] == winner)


def main():
    out_dir = REPO / "outputs/qwen_main"
    paths = {
        name: out_dir / f"inference_{name}.jsonl"
        for name in ("short", "medium", "long", "sample")
    }
    maps = {name: load_map(path) for name, path in paths.items()}
    cached_path = out_dir / "prm_scores.jsonl"
    cached = load_map(cached_path)
    expected = set(cached)
    for name, records in maps.items():
        if set(records) != expected:
            raise SystemExit(f"inference_{name} IDs do not match prm_scores IDs")

    verifier = MathVerifier()
    changed = collections.Counter()
    output = []
    for pid, row in cached.items():
        short_eval = evaluate_chains(maps["short"][pid], verifier)
        medium_eval = evaluate_chains(maps["medium"][pid], verifier)
        long_eval = evaluate_chains(maps["long"][pid], verifier)
        sample_eval = evaluate_chains(maps["sample"][pid], verifier)

        if len(row["short_chains"]) != len(short_eval):
            raise SystemExit(f"{pid}: cached and inference short-chain counts differ")
        for cached_chain, fresh in zip(row["short_chains"], short_eval):
            cached_chain["orm_correct"] = fresh["correct"]
            cached_chain["extracted_answer"] = fresh["answer"]

        values = {
            "short_orm_majority": plurality_correct(short_eval),
            "medium_orm_correct": medium_eval[0]["correct"],
            "long_orm_correct": long_eval[0]["correct"],
            "sample_orm_correct": [item["correct"] for item in sample_eval],
            "sample_any_correct": any(item["correct"] for item in sample_eval),
            "sample_majority_correct": plurality_correct(sample_eval),
        }
        values["utility_label"] = int(
            not values["short_orm_majority"] and values["long_orm_correct"]
        )
        for key, value in values.items():
            if row.get(key) != value:
                changed[key] += 1
            row[key] = value
        row["action_labels"] = {
            "stop_correct": values["short_orm_majority"],
            "medium_correct": values["medium_orm_correct"],
            "long_correct": values["long_orm_correct"],
            "sample_any_correct": values["sample_any_correct"],
            "sample_majority_correct": values["sample_majority_correct"],
        }
        output.append(row)

    tmp_path = cached_path.with_suffix(cached_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(cached_path)
    print(f"Relabeled {len(output)} cached PRM records: {dict(changed)}")

    # Medium-trace PRM features are expensive and remain valid, but their
    # cached verifier labels must be refreshed as well.
    medium_features_path = out_dir / "features_medium.jsonl"
    medium_features = load_map(medium_features_path)
    if set(medium_features) != expected:
        raise SystemExit("features_medium IDs do not match prm_scores IDs")
    for pid, row in medium_features.items():
        refreshed = cached[pid]
        medium_correct = bool(refreshed["medium_orm_correct"])
        long_correct = bool(refreshed["long_orm_correct"])
        row["medium_orm_correct"] = medium_correct
        row["long_orm_correct"] = long_correct
        row["short_orm_majority"] = bool(refreshed["short_orm_majority"])
        row["label"] = int((not medium_correct) and long_correct)
    tmp_features = medium_features_path.with_suffix(
        medium_features_path.suffix + ".tmp"
    )
    with open(tmp_features, "w", encoding="utf-8") as handle:
        for row in medium_features.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_features.replace(medium_features_path)

    csv_path = out_dir / "features_medium_G4.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path, index_col=0)
        ordered = list(medium_features)
        if list(frame.index) != ordered:
            raise SystemExit(
                "features_medium_G4.csv row order does not match "
                "features_medium.jsonl problem_id order"
            )
        frame.loc[ordered, "label"] = [
            medium_features[pid]["label"] for pid in ordered
        ]
        frame.loc[ordered, "dataset"] = [
            medium_features[pid]["dataset"] for pid in ordered
        ]
        frame.loc[ordered, "math_level"] = [
            medium_features[pid]["math_level"] for pid in ordered
        ]
        tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
        frame.to_csv(tmp_csv)
        tmp_csv.replace(csv_path)
    print(f"Relabeled cached medium features: {len(medium_features)} records")


if __name__ == "__main__":
    main()
