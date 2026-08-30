"""Equivalence bounds, detection limits, and paired tests for the B->C contrast.

The three-arm design reports several near-zero incremental effects. A raw zero
count is not evidence of absence on its own, so this script converts each of
them into (a) an exact one-sided upper confidence bound, (b) the smallest
equivalence margin the observation rejects, (c) the smallest true rate the
design could have detected, and (d) where the paired records are available, an
exact McNemar comparison of same-budget resampling against continuation.
"""

import argparse
import csv
import json
from pathlib import Path

from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
POWER = 0.80
MARGINS = (0.10, 0.05, 0.02, 0.01)

CORE_SEEDS = ("seed42", "seed314159", "seed271828")
SCOPES = (
    ("core_3seed", "three_arm_core", CORE_SEEDS, ("qwen", "llama")),
    ("full_split", "three_arm_expanded", ("seed42",), ("qwen", "llama")),
    ("r1_3seed", "three_arm_r1", CORE_SEEDS, ("r1",)),
)
TEMPERATURE_CELLS = (
    ("qwen", 0.60, "temperature_sensitivity/t060"),
    ("qwen", 0.75, "three_arm_core"),
    ("qwen", 1.00, "temperature_sensitivity/t100"),
    ("llama", 0.60, "temperature_sensitivity/t060"),
    ("llama", 0.75, "three_arm_core"),
    ("llama", 1.00, "temperature_sensitivity/t100"),
)

FIELDS = (
    "model",
    "scope",
    "contrast",
    "population",
    "n",
    "k",
    "rate",
    "ci95_upper_onesided",
    "equivalence_margin_rejected",
    "p_equivalence_margin_10",
    "p_equivalence_margin_05",
    "p_equivalence_margin_02",
    "p_equivalence_margin_01",
    "detectable_rate_power80",
    "reference_contrast",
    "reference_k",
    "mcnemar_discordant",
    "mcnemar_p_exact",
    "paired_sets_identical",
)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select(rows, **criteria):
    matches = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def upper_bound(k, n, alpha=ALPHA):
    """Exact one-sided Clopper-Pearson upper bound on a binomial rate."""
    if n == 0:
        return float("nan")
    if k == n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def equivalence_p(k, n, margin):
    """One-sided exact p-value for H0: p >= margin against H1: p < margin."""
    if n == 0:
        return float("nan")
    return float(stats.binom.cdf(k, n, margin))


def smallest_rejected_margin(k, n, alpha=ALPHA):
    """Smallest equivalence margin the observation rejects at level alpha.

    This coincides with the exact one-sided upper confidence bound; reporting
    both makes the duality explicit for readers who prefer one framing.
    """
    return upper_bound(k, n, alpha)


def detectable_rate(n, power=POWER):
    """Smallest true rate at which n draws yield at least one event with the
    stated probability. Solving 1 - (1 - p)^n = power gives the bound below.
    Rates under it are the ones this design could plausibly have missed."""
    if n == 0:
        return float("nan")
    return 1.0 - (1.0 - power) ** (1.0 / n)


def mcnemar_exact(both, only_reference, only_target):
    """Exact (binomial) McNemar test on the discordant cells."""
    discordant = only_reference + only_target
    if discordant == 0:
        return discordant, None
    p = float(stats.binomtest(only_target, discordant, 0.5).pvalue)
    return discordant, p


def paired_counts(detail_rows, definition, reference_field, target_field):
    both = ref_only = tgt_only = neither = 0
    for row in detail_rows:
        if row["definition"] != definition or row["eligible"] != "True":
            continue
        reference = row[reference_field] == "True"
        target = row[target_field] == "True"
        if reference and target:
            both += 1
        elif reference:
            ref_only += 1
        elif target:
            tgt_only += 1
        else:
            neither += 1
    return both, ref_only, tgt_only, neither


def make_row(**kwargs):
    row = {field: "" for field in FIELDS}
    row.update(kwargs)
    for key in ("rate", "ci95_upper_onesided", "equivalence_margin_rejected",
                "detectable_rate_power80"):
        if isinstance(row[key], float):
            row[key] = f"{row[key]:.6f}"
    for key in [f"p_equivalence_margin_{int(m * 100):02d}" for m in MARGINS] + ["mcnemar_p_exact"]:
        if isinstance(row[key], float):
            row[key] = f"{row[key]:.6g}"
        elif row[key] is None:
            row[key] = ""
    return row


def repair_rows(outputs_root, model, scope_name, directory, seeds):
    summaries = []
    details = []
    for seed in seeds:
        root = outputs_root / directory / f"{model}_{seed}"
        summaries.append(select(read_rows(root / "three_arm_summary.csv"),
                                dataset="all", definition="strict_all_wrong"))
        details.extend(read_rows(root / "three_arm_detail.csv"))

    n = sum(int(row["n_eligible"]) for row in summaries)
    helpful = sum(int(row["incremental_helpful_count"]) for row in summaries)
    harmful = sum(int(row["incremental_harmful_count"]) for row in summaries)
    samecap = sum(int(row["samecap_repair_count"]) for row in summaries)

    both, ref_only, tgt_only, neither = paired_counts(
        details, "strict_all_wrong", "samecap_repair_plurality", "long_repair_plurality")
    discordant, mcnemar_p = mcnemar_exact(both, ref_only, tgt_only)

    rows = [make_row(
        model=model,
        scope=scope_name,
        contrast="B_to_C_incremental_repair",
        population="strict_CBW_on_arm_A",
        n=n,
        k=helpful,
        rate=helpful / n if n else float("nan"),
        ci95_upper_onesided=upper_bound(helpful, n),
        equivalence_margin_rejected=smallest_rejected_margin(helpful, n),
        p_equivalence_margin_10=equivalence_p(helpful, n, 0.1),
        p_equivalence_margin_05=equivalence_p(helpful, n, 0.05),
        p_equivalence_margin_02=equivalence_p(helpful, n, 0.02),
        p_equivalence_margin_01=equivalence_p(helpful, n, 0.01),
        detectable_rate_power80=detectable_rate(n),
        reference_contrast="A_to_B_same_budget_repair",
        reference_k=samecap,
        mcnemar_discordant=discordant,
        mcnemar_p_exact=mcnemar_p,
        paired_sets_identical=(discordant == 0),
    ), make_row(
        model=model,
        scope=scope_name,
        contrast="B_to_C_incremental_harm_within_CBW",
        population="strict_CBW_on_arm_A",
        n=n,
        k=harmful,
        rate=harmful / n if n else float("nan"),
        ci95_upper_onesided=upper_bound(harmful, n),
        equivalence_margin_rejected=smallest_rejected_margin(harmful, n),
        p_equivalence_margin_10=equivalence_p(harmful, n, 0.1),
        p_equivalence_margin_05=equivalence_p(harmful, n, 0.05),
        p_equivalence_margin_02=equivalence_p(harmful, n, 0.02),
        p_equivalence_margin_01=equivalence_p(harmful, n, 0.01),
        detectable_rate_power80=detectable_rate(n),
    )]
    return rows


def harm_rows(outputs_root, model, scope_name, directory, seeds):
    n = k_target = k_reference = 0
    both = ref_only = tgt_only = 0
    for seed in seeds:
        root = outputs_root / directory / f"{model}_{seed}"
        summary = read_rows(root / "three_arm_harm_summary.csv")
        target = select(summary, contrast="B_to_C_tokens", dataset="all",
                        definition="complete_plurality_correct")
        reference = select(summary, contrast="A_to_B_resampling", dataset="all",
                           definition="complete_plurality_correct")
        n += int(target["n_eligible"])
        k_target += int(target["harm_count"])
        k_reference += int(reference["harm_count"])

        flags = {}
        for row in read_rows(root / "three_arm_harm_detail.csv"):
            if row["definition"] != "complete_plurality_correct" or row["eligible"] != "True":
                continue
            if row["contrast"] not in ("A_to_B_resampling", "B_to_C_tokens"):
                continue
            flags.setdefault(row["problem_id"], {})[row["contrast"]] = row["harm"] == "True"
        for pair in flags.values():
            reference_hit = pair.get("A_to_B_resampling", False)
            target_hit = pair.get("B_to_C_tokens", False)
            if reference_hit and target_hit:
                both += 1
            elif reference_hit:
                ref_only += 1
            elif target_hit:
                tgt_only += 1

    discordant, mcnemar_p = mcnemar_exact(both, ref_only, tgt_only)
    return [make_row(
        model=model,
        scope=scope_name,
        contrast="B_to_C_harm",
        population="complete_and_plurality_correct_on_arm_A",
        n=n,
        k=k_target,
        rate=k_target / n if n else float("nan"),
        ci95_upper_onesided=upper_bound(k_target, n),
        equivalence_margin_rejected=smallest_rejected_margin(k_target, n),
        p_equivalence_margin_10=equivalence_p(k_target, n, 0.1),
        p_equivalence_margin_05=equivalence_p(k_target, n, 0.05),
        p_equivalence_margin_02=equivalence_p(k_target, n, 0.02),
        p_equivalence_margin_01=equivalence_p(k_target, n, 0.01),
        detectable_rate_power80=detectable_rate(n),
        reference_contrast="A_to_B_resampling_harm",
        reference_k=k_reference,
        mcnemar_discordant=discordant,
        mcnemar_p_exact=mcnemar_p,
        paired_sets_identical=(discordant == 0),
    )]


def temperature_rows(outputs_root):
    rows = []
    for model, temperature, relative in TEMPERATURE_CELLS:
        summaries = []
        for seed in CORE_SEEDS:
            root = outputs_root / relative / f"{model}_{seed}"
            if not (root / "three_arm_summary.csv").exists():
                continue
            with (root / "inference_long_sample8.jsonl").open(encoding="utf-8") as handle:
                observed = float(json.loads(next(handle))["budget"]["temperature"])
            if observed != temperature:
                raise ValueError(f"{root}: observed temperature {observed} != {temperature}")
            summaries.append(select(read_rows(root / "three_arm_summary.csv"),
                                    dataset="all", definition="strict_all_wrong"))
        if not summaries:
            raise FileNotFoundError(f"no seed directories under {outputs_root / relative}")
        n = sum(int(summary["n_eligible"]) for summary in summaries)
        k = sum(int(summary["incremental_helpful_count"]) for summary in summaries)
        samecap = sum(int(summary["samecap_repair_count"]) for summary in summaries)
        rows.append(make_row(
            model=model,
            scope=f"core_T{temperature:.2f}_{len(summaries)}seed",
            contrast="B_to_C_incremental_repair",
            population="strict_CBW_on_arm_A",
            n=n,
            k=k,
            rate=k / n if n else float("nan"),
            ci95_upper_onesided=upper_bound(k, n),
            equivalence_margin_rejected=smallest_rejected_margin(k, n),
            p_equivalence_margin_10=equivalence_p(k, n, 0.1),
            p_equivalence_margin_05=equivalence_p(k, n, 0.05),
            p_equivalence_margin_02=equivalence_p(k, n, 0.02),
            p_equivalence_margin_01=equivalence_p(k, n, 0.01),
            detectable_rate_power80=detectable_rate(n),
            reference_contrast="A_to_B_same_budget_repair",
            reference_k=samecap,
        ))
    return rows


def format_table(rows):
    lines = ["Equivalence bounds and detection limits for the B->C contrast", ""]
    header = f"{'model':6} {'scope':14} {'contrast':34} {'n':>5} {'k':>4} {'upper95':>8} {'mde80':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['model']:6} {row['scope']:14} {row['contrast']:34} "
            f"{row['n']:>5} {row['k']:>4} {float(row['ci95_upper_onesided']):>8.4f} "
            f"{float(row['detectable_rate_power80']):>8.4f}"
        )
    lines.append("")
    lines.append(f"Equivalence margins tested: {', '.join(f'{m:.2f}' for m in MARGINS)}")
    lines.append("upper95 is the exact one-sided Clopper-Pearson bound and equals the")
    lines.append("smallest equivalence margin the observation rejects at alpha=0.05.")
    lines.append("mde80 is the smallest true rate the design detects with 80% power.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path,
                        default=SCRIPT_DIR.parent / "outputs")
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR.parent / "outputs" / "equivalence")
    args = parser.parse_args()

    rows = []
    for scope_name, directory, seeds, models in SCOPES:
        for model in models:
            rows.extend(repair_rows(args.outputs_root, model, scope_name, directory, seeds))
            rows.extend(harm_rows(args.outputs_root, model, scope_name, directory, seeds))
    rows.extend(temperature_rows(args.outputs_root))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "equivalence_power.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    text = format_table(rows)
    (args.out_dir / "equivalence_power.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
