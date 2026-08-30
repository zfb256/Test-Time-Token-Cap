"""Compile the one-seed three-arm temperature sensitivity table."""

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPECS = (
    ("Qwen2.5-Math-7B-Instruct", 0.60, "temperature_sensitivity/t060/qwen_seed42"),
    ("Qwen2.5-Math-7B-Instruct", 0.75, "three_arm_core/qwen_seed42"),
    ("Qwen2.5-Math-7B-Instruct", 1.00, "temperature_sensitivity/t100/qwen_seed42"),
    ("Llama-3.1-8B-Instruct", 0.60, "temperature_sensitivity/t060/llama_seed42"),
    ("Llama-3.1-8B-Instruct", 0.75, "three_arm_core/llama_seed42"),
    ("Llama-3.1-8B-Instruct", 1.00, "temperature_sensitivity/t100/llama_seed42"),
)


def select(path, **criteria):
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if all(row.get(key) == value for key, value in criteria.items())
        ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def compile_rows(outputs_root):
    rows = []
    for model, temperature, relative in SPECS:
        root = outputs_root / relative
        with (root / "inference_long_sample8.jsonl").open(encoding="utf-8") as handle:
            observed = float(json.loads(next(handle))["budget"]["temperature"])
        if observed != temperature:
            raise ValueError(f"{root}: observed temperature {observed} != {temperature}")
        repair = select(
            root / "three_arm_summary.csv",
            dataset="all",
            definition="strict_all_wrong",
        )
        harm = select(
            root / "three_arm_harm_summary.csv",
            contrast="B_to_C_tokens",
            dataset="all",
            definition="complete_plurality_correct",
        )
        samecap = int(repair["samecap_repair_count"])
        long = int(repair["long_repair_count"])
        helpful = int(repair["incremental_helpful_count"])
        harmful = int(repair["incremental_harmful_count"])
        if long - samecap != helpful - harmful:
            raise ValueError(f"{root}: inconsistent incremental repair counts")
        rows.append(
            {
                "model": model,
                "temperature": f"{temperature:.2f}",
                "n_problems": int(repair["n_problems"]),
                "n_strict_cbw": int(repair["n_eligible"]),
                "samecap_repair_count": samecap,
                "long_repair_count": long,
                "incremental_helpful_count": helpful,
                "incremental_harmful_count": harmful,
                "token_harm_eligible": int(harm["n_eligible"]),
                "token_harm_count": int(harm["harm_count"]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=(SCRIPT_DIR / "../outputs").resolve(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            SCRIPT_DIR
            / "../outputs/temperature_sensitivity/temperature_sensitivity_summary.csv"
        ).resolve(),
    )
    args = parser.parse_args()
    rows = compile_rows(args.outputs_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
