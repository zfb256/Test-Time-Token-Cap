"""Pool the temperature sweep over whatever seeds are present on disk.

Script 58 compiles the one-seed table. This one pools every seed directory it
finds for each model--temperature cell, reports the pooled counts with exact
one-sided bounds, and records which seeds contributed. Running it before the
extra seeds land reproduces the one-seed numbers, so the paper table and the
handover run share a single code path.
"""

import argparse
import csv
import json
from pathlib import Path

from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
SEEDS = ("seed42", "seed314159", "seed271828")
CELLS = (
    ("Qwen2.5-Math-7B-Instruct", "qwen", 0.60, "temperature_sensitivity/t060"),
    ("Qwen2.5-Math-7B-Instruct", "qwen", 0.75, "three_arm_core"),
    ("Qwen2.5-Math-7B-Instruct", "qwen", 1.00, "temperature_sensitivity/t100"),
    ("Llama-3.1-8B-Instruct", "llama", 0.60, "temperature_sensitivity/t060"),
    ("Llama-3.1-8B-Instruct", "llama", 0.75, "three_arm_core"),
    ("Llama-3.1-8B-Instruct", "llama", 1.00, "temperature_sensitivity/t100"),
)
FIELDS = (
    "model",
    "temperature",
    "n_seeds",
    "seeds",
    "cbw_n",
    "samecap_repair",
    "long_repair",
    "incremental_helpful",
    "incremental_harmful",
    "incremental_upper95",
    "samecap_any_repair",
    "long_any_repair",
    "harm_n",
    "harm_incremental",
)


def select(path, **criteria):
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def upper_bound(k, n, alpha=ALPHA):
    if n == 0:
        return float("nan")
    if k == n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def check_temperature(root, expected):
    with (root / "inference_long_sample8.jsonl").open(encoding="utf-8") as handle:
        observed = float(json.loads(next(handle))["budget"]["temperature"])
    if observed != expected:
        raise ValueError(f"{root}: observed temperature {observed} != {expected}")


def compile_cell(outputs_root, model_name, model_key, temperature, relative):
    totals = dict.fromkeys(
        ("cbw_n", "samecap_repair", "long_repair", "incremental_helpful",
         "incremental_harmful", "samecap_any_repair", "long_any_repair",
         "harm_n", "harm_incremental"), 0)
    present = []
    for seed in SEEDS:
        root = outputs_root / relative / f"{model_key}_{seed}"
        if not (root / "three_arm_summary.csv").exists():
            continue
        check_temperature(root, temperature)
        present.append(seed.replace("seed", ""))
        repair = select(root / "three_arm_summary.csv",
                        dataset="all", definition="strict_all_wrong")
        harm = select(root / "three_arm_harm_summary.csv",
                      contrast="B_to_C_tokens", dataset="all",
                      definition="complete_plurality_correct")
        totals["cbw_n"] += int(repair["n_eligible"])
        totals["samecap_repair"] += int(repair["samecap_repair_count"])
        totals["long_repair"] += int(repair["long_repair_count"])
        totals["incremental_helpful"] += int(repair["incremental_helpful_count"])
        totals["incremental_harmful"] += int(repair["incremental_harmful_count"])
        totals["samecap_any_repair"] += int(repair["samecap_any_repair_count"])
        totals["long_any_repair"] += int(repair["long_any_repair_count"])
        totals["harm_n"] += int(harm["n_eligible"])
        totals["harm_incremental"] += int(harm["harm_count"])

    if not present:
        raise FileNotFoundError(f"no seed directories under {outputs_root / relative}")

    delta = totals["long_repair"] - totals["samecap_repair"]
    if delta != totals["incremental_helpful"] - totals["incremental_harmful"]:
        raise ValueError(f"{relative}/{model_key}: inconsistent incremental counts")

    row = {
        "model": model_name,
        "temperature": f"{temperature:.2f}",
        "n_seeds": len(present),
        "seeds": "+".join(present),
        "incremental_upper95": f"{upper_bound(totals['incremental_helpful'], totals['cbw_n']):.4f}",
    }
    row.update({key: value for key, value in totals.items()})
    return row


def format_table(rows):
    lines = ["Temperature sweep pooled over available seeds", ""]
    header = (f"{'model':30} {'T':>5} {'seeds':>5} {'CBW n':>6} "
              f"{'A->B':>5} {'A->C':>5} {'B->C +/-':>9} {'upper95':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['model']:30} {row['temperature']:>5} {row['n_seeds']:>5} "
            f"{row['cbw_n']:>6} {row['samecap_repair']:>5} {row['long_repair']:>5} "
            f"{str(row['incremental_helpful']) + '/' + str(row['incremental_harmful']):>9} "
            f"{row['incremental_upper95']:>8}"
        )
    lines.append("")
    lines.append("upper95 is the exact one-sided Clopper-Pearson bound on the")
    lines.append("incremental (B->C) plurality repair rate within the CBW subset.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path,
                        default=SCRIPT_DIR.parent / "outputs")
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR.parent / "outputs" / "temperature_sensitivity")
    args = parser.parse_args()

    rows = [compile_cell(args.outputs_root, *cell) for cell in CELLS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "temperature_multiseed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    text = format_table(rows)
    (args.out_dir / "temperature_multiseed_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
