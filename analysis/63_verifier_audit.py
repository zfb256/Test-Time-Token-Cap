"""
Step 63: Can strict answer matching be hiding repairs?

The headline result is a zero count, so the instrument that produced it deserves
its own check. A missed repair requires the extended arm (arm C) to be scored
wrong when its answer was in fact acceptable. This script re-scores every
eligible complete-but-wrong record under a deliberately permissive answer rule
and reports how many additional plurality repairs appear.

Two guards make the result trustworthy rather than merely reassuring:

* the strict pass must reproduce the labels already recorded in
  ``three_arm_detail.csv``; a mismatch means this audit is wired wrong, not
  that the pipeline is, and the script says so instead of reporting a rate;
* the relaxed rule only ever *adds* accepted answers, so it can uncover a
  hidden repair but never erase a recorded one.

CPU only. No inference is run.
"""

import argparse
import collections
import csv
import importlib.util
import io
import json
import logging
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "expanded_repair_eval", SCRIPT_DIR / "48_expanded_repair_eval.py"
)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def one_sided_upper(n, k):
    """Exact one-sided 95% Clopper-Pearson upper bound, matching the paper."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 1 - 0.05 ** (1 / n)
    from scipy.stats import beta

    return float(beta.ppf(0.95, k + 1, n - k))


def relaxed_answer(text, dataset, verifier):
    """Strict extraction, then a numeric fallback for answers that never got boxed."""
    strict = verifier.extract_prediction_answer(text, dataset)
    if strict is not None:
        return strict
    hits = NUMBER.findall(text.replace(",", ""))
    return hits[-1] if hits else None


def gold_number(gold, dataset, verifier):
    """Numeric value of the reference answer, or None when it is not a plain number.

    Naively taking the last number in the reference is wrong for MATH, where a
    reference such as ``\\boxed{\\frac{1}{2}}`` would yield 2. The fallback is
    therefore only offered when the reference really is a bare number.
    """
    if dataset.lower() == "gsm8k":
        return verifier._extract_gsm8k_answer(gold)
    boxed = verifier._extract_boxed(gold)
    if boxed is None:
        return None
    try:
        return float(boxed.replace(",", "").strip())
    except ValueError:
        return None


def relaxed_correct(text, gold, dataset, verifier):
    """Strict verdict, widened only by accepting an unboxed numeric final answer."""
    if verifier.verify(text, gold, dataset):
        return True
    if verifier.extract_prediction_answer(text, dataset) is not None:
        return False          # it had a parseable answer; strict already judged it
    gold_val = gold_number(gold, dataset, verifier)
    if gold_val is None:
        return False          # reference is not a bare number, no safe relaxation
    hits = NUMBER.findall(text.replace(",", ""))
    if not hits:
        return False
    try:
        return abs(float(hits[-1]) - gold_val) <= 1e-6
    except ValueError:
        return False


def chain_values(record, budget, verifier, relaxed):
    values = []
    for chain in record.get("chains", []):
        text = chain.get("text", "")
        finish = chain.get("finish_reason")
        hit_max = (
            finish == "length"
            if finish is not None
            else int(chain.get("token_count", 0)) >= budget
        )
        if relaxed:
            answer = relaxed_answer(text, record["dataset"], verifier)
            correct = relaxed_correct(text, record["answer"], record["dataset"], verifier)
        else:
            answer = verifier.extract_prediction_answer(text, record["dataset"])
            correct = bool(verifier.verify(text, record["answer"], record["dataset"]))
        values.append({"complete": not hit_max, "answer": answer, "correct": correct})
    return values


def load_jsonl(path):
    with io.open(path, encoding="utf-8") as fh:
        return {json.loads(line)["problem_id"]: json.loads(line)
                for line in fh if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../outputs/three_arm_core")
    ap.add_argument("--extended-budget", type=int, default=1536)
    ap.add_argument("--output", default="../outputs/equivalence/verifier_audit.csv")
    args = ap.parse_args()

    root = (SCRIPT_DIR / args.root).resolve()
    verifier = MathVerifier()
    rows, mismatches = [], 0

    for run in sorted(p for p in root.iterdir() if p.is_dir() and "_seed" in p.name):
        detail = run / "three_arm_detail.csv"
        long_arm = run / "inference_xlong_sample8.jsonl"
        if not detail.exists() or not long_arm.exists():
            logger.info("skip %s (missing inputs)", run.name)
            continue
        model = "qwen" if "qwen" in run.name else "llama"
        recorded = {
            r["problem_id"]: r
            for r in csv.DictReader(io.open(detail, encoding="utf-8"))
            if r["definition"] == "strict_all_wrong" and r["eligible"] == "True"
        }
        records = load_jsonl(long_arm)

        strict_repair = relaxed_repair = 0
        for pid, meta in recorded.items():
            rec = records.get(pid)
            if rec is None:
                continue
            s = REPAIR.plurality(chain_values(rec, args.extended_budget, verifier, False))
            r = REPAIR.plurality(chain_values(rec, args.extended_budget, verifier, True))
            if s["correct"] != (meta["long_plurality_correct"] == "True"):
                mismatches += 1
            strict_repair += bool(s["correct"])
            relaxed_repair += bool(r["correct"])

        rows.append({
            "model": model, "run": run.name, "n_eligible": len(recorded),
            "strict_plurality_repair": strict_repair,
            "relaxed_plurality_repair": relaxed_repair,
            "hidden_by_strict_matching": relaxed_repair - strict_repair,
        })
        logger.info("%-22s n=%-4d strict=%d relaxed=%d",
                    run.name, len(recorded), strict_repair, relaxed_repair)

    if mismatches:
        logger.error(
            "strict re-scoring disagrees with recorded labels on %d records; "
            "this audit is mis-wired and no rate is reported", mismatches)
        return 1

    total_n = sum(r["n_eligible"] for r in rows)
    total_hidden = sum(r["hidden_by_strict_matching"] for r in rows)
    out = (SCRIPT_DIR / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bound = one_sided_upper(total_n, total_hidden)
    print(f"\n  strict re-scoring reproduced every recorded label ({total_n} records)")
    print(f"  additional repairs under a relaxed answer rule: {total_hidden} of {total_n}")
    print(f"  one-sided 95% upper bound on hidden repair rate: {bound:.4f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
