"""
Step 3: Score inference chains with PRM (Process Reward Model).

Reads inference_short.jsonl (and optionally other budgets).
Writes PRM scores for each chain into:
  {output_dir}/prm_scores_short.jsonl   (one record per problem)

Also computes ORM (programmatic) correctness for each chain via math_verify.py,
and saves those results in the same file.

Output record format:
  {
    "problem_id": "gsm8k_0001",
    "dataset": "gsm8k",
    "answer": "...",
    "short_chains_prm": [         # one entry per chain
      {"step_scores": [...], "mean": 0.72, "min": 0.55, "var": 0.03, "chain_score": 0.72,
       "orm_correct": true, "extracted_answer": "42"}
    ],
    "short_orm_majority": true,   # majority vote across short chains
    "long_orm_correct": true,     # from inference_long.jsonl
    "medium_orm_correct": true,
    "utility_label": 1,           # helpful=1 if short_wrong and long_right
  }

Run:
  python 03_prm_scoring.py --config config.yaml
  python 03_prm_scoring.py --config config.yaml --mock-prm   # use mock PRM (no GPU)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.math_verify import MathVerifier
from utils.prm_utils import PRMScorer
from utils.feature_utils import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path, strict: bool = False) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    msg = f"Skipping malformed JSONL line {lineno} in {path}"
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
    return records


def load_jsonl_as_map(path: Path) -> Dict[str, Dict]:
    """Load JSONL and index by problem_id."""
    out = {}
    for record in load_jsonl(path, strict=True):
        pid = record["problem_id"]
        if pid in out:
            raise ValueError(f"{path}: duplicate problem_id={pid}")
        out[pid] = record
    return out


def plurality_correct(answer_correct_pairs) -> bool:
    """Verify the deterministic plurality answer, not a majority of ORM labels."""
    valid = [(answer, correct) for answer, correct in answer_correct_pairs if answer]
    if not valid:
        return False
    counts = Counter(answer for answer, _ in valid)
    max_count = max(counts.values())
    winner = sorted(answer for answer, n in counts.items() if n == max_count)[0]
    return any(correct for answer, correct in valid if answer == winner)


def score_chain_orm(rec: Optional[Dict], verifier: MathVerifier, gt_answer: str, dataset: str) -> Optional[bool]:
    if not rec or not rec.get("chains"):
        return None
    return verifier.verify(rec["chains"][0]["text"], gt_answer, dataset)


def score_sample_orm(rec: Optional[Dict], verifier: MathVerifier, gt_answer: str, dataset: str) -> Dict:
    if not rec:
        return {
            "sample_orm_correct": None,
            "sample_any_correct": None,
            "sample_majority_correct": None,
        }
    votes = []
    answers = []
    for chain in rec.get("chains", []):
        text = chain.get("text", "")
        votes.append(bool(verifier.verify(text, gt_answer, dataset)))
        answers.append(verifier.extract_prediction_answer(text, dataset))
    if not votes:
        return {
            "sample_orm_correct": [],
            "sample_any_correct": False,
            "sample_majority_correct": False,
        }
    return {
        "sample_orm_correct": votes,
        "sample_any_correct": any(votes),
        "sample_majority_correct": plurality_correct(zip(answers, votes)),
    }


def extract_answer(text: str, dataset: str, verifier: MathVerifier) -> Optional[str]:
    """
    Extract the answer string from model output for consistency comparison.
    For GSM8K: last number in text.
    For MATH: content of \\boxed{...}.
    """
    return verifier.extract_prediction_answer(text, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mock-prm", action="store_true",
                        help="Use mock PRM scores (for testing without GPU)")
    parser.add_argument("--budgets", nargs="+",
                        default=["short", "medium", "long", "sample"],
                        choices=["short", "medium", "long", "sample"],
                        help="Which inference budgets to load for ORM")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load inference results ----
    def load_budget(budget_name):
        p = out_dir / f"inference_{budget_name}.jsonl"
        if not p.exists():
            logger.error(f"Missing: {p}. Run 02_run_inference.py first.")
            sys.exit(1)
        return load_jsonl_as_map(p)

    short_map = load_budget("short")
    long_map  = load_budget("long")
    medium_map = load_budget("medium") if "medium" in args.budgets else {}
    sample_map = load_budget("sample") if "sample" in args.budgets else {}

    required_maps = {"long": long_map}
    if medium_map:
        required_maps["medium"] = medium_map
    if sample_map:
        required_maps["sample"] = sample_map
    reference_ids = short_map.keys()
    for name, records in required_maps.items():
        record_ids = records.keys()
        if record_ids != reference_ids:
            missing = sorted(reference_ids - record_ids)
            extra = sorted(record_ids - reference_ids)
            raise SystemExit(
                f"inference_{name} problem IDs differ from inference_short: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        for pid in reference_ids:
            for field in ("dataset", "question", "answer", "math_level"):
                default = 0 if field == "math_level" else None
                if records[pid].get(field, default) != short_map[pid].get(
                    field, default
                ):
                    raise SystemExit(
                        f"inference_{name} {field} differs from "
                        f"inference_short for problem_id={pid}"
                    )
    all_problem_ids = sorted(reference_ids)
    logger.info(f"Problems present in every requested budget: {len(all_problem_ids)}")
    if not all_problem_ids:
        logger.error("No overlapping problem_ids between short and long inference files.")
        sys.exit(1)

    # ---- Init verifier + PRM ----
    verifier = MathVerifier(numeric_tolerance=cfg["orm"]["numeric_tolerance"])
    prm_scorer = PRMScorer(
        model_name=cfg["prm"]["model"],
        device=cfg["prm"].get("device_map", "auto"),
        dtype=cfg["prm"]["dtype"],
        aggregation=cfg["prm"]["score_aggregation"],
        mock=args.mock_prm,
    )

    if args.mock_prm:
        logger.warning("Using MOCK PRM — scores are random. Results are for pipeline testing only.")

    # ---- Process each problem ----
    output_records = []
    for pid in tqdm(all_problem_ids, desc="Scoring"):
        short_rec = short_map[pid]
        long_rec  = long_map[pid]
        dataset   = short_rec["dataset"]
        gt_answer = short_rec["answer"]

        # ---- Score short chains (PRM + ORM) ----
        short_chains_scored = []

        for chain_data in short_rec["chains"]:
            chain_text = chain_data["text"]

            # PRM scoring
            prm_result = prm_scorer.score_chain(
                question=short_rec["question"],
                chain=chain_text,
            )

            # ORM scoring
            orm_correct = verifier.verify(chain_text, gt_answer, dataset)

            # Extract answer for consistency check
            extracted = extract_answer(chain_text, dataset, verifier)

            short_chains_scored.append({
                **prm_result,
                "orm_correct": orm_correct,
                "extracted_answer": extracted,
                "token_count": chain_data.get("token_count", 0),
            })

        # Majority vote ORM for short chains. Empty generations are treated as
        # wrong so the label remains conservative instead of crashing.
        short_orm_majority = plurality_correct(
            (chain["extracted_answer"], chain["orm_correct"])
            for chain in short_chains_scored
        )

        # ---- ORM for long chain ----
        long_chain_text = long_rec["chains"][0]["text"] if long_rec["chains"] else ""
        long_orm_correct = bool(
            score_chain_orm(long_rec, verifier, gt_answer, dataset)
        )

        # ---- ORM for medium chain (optional) ----
        medium_orm_correct = score_chain_orm(
            medium_map.get(pid), verifier, gt_answer, dataset
        )
        sample_stats = score_sample_orm(
            sample_map.get(pid), verifier, gt_answer, dataset
        )

        # ---- Utility label ----
        utility_label = FeatureBuilder.compute_utility_label(
            short_correct=short_orm_majority,
            long_correct=long_orm_correct,
        )

        record = {
            "problem_id": pid,
            "dataset": dataset,
            "question": short_rec["question"],
            "answer": gt_answer,
            "math_level": short_rec.get("math_level", 0),
            "short_chains": short_chains_scored,
            "short_orm_majority": bool(short_orm_majority),
            "long_orm_correct": bool(long_orm_correct),
            "medium_orm_correct": medium_orm_correct,
            **sample_stats,
            "action_labels": {
                "stop_correct": bool(short_orm_majority),
                "medium_correct": medium_orm_correct,
                "long_correct": bool(long_orm_correct),
                "sample_any_correct": sample_stats["sample_any_correct"],
                "sample_majority_correct": sample_stats["sample_majority_correct"],
            },
            "long_chain_text": long_chain_text,  # saved for Plan B-1 CHR
            "utility_label": utility_label,
        }
        output_records.append(record)

    # ---- Save ----
    out_path = out_dir / "prm_scores.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(output_records)} records -> {out_path}")

    # ---- Quick stats ----
    n_helpful = sum(r["utility_label"] for r in output_records)
    n_total = len(output_records)
    if n_total == 0:
        logger.error("No scored records produced. Check inference_short/long inputs.")
        sys.exit(1)
    logger.info(f"\nUtility label distribution:")
    logger.info(f"  helpful=1 (short wrong, long right): {n_helpful} ({100*n_helpful/n_total:.1f}%)")
    logger.info(f"  helpful=0: {n_total - n_helpful} ({100*(n_total-n_helpful)/n_total:.1f}%)")
    logger.info(f"\nShort ORM accuracy: {sum(r['short_orm_majority'] for r in output_records)/n_total:.3f}")
    logger.info(f"Long ORM accuracy:  {sum(r['long_orm_correct'] for r in output_records)/n_total:.3f}")
    medium_vals = [r["medium_orm_correct"] for r in output_records if r["medium_orm_correct"] is not None]
    sample_vals = [r["sample_any_correct"] for r in output_records if r["sample_any_correct"] is not None]
    if medium_vals:
        logger.info(f"Medium ORM accuracy: {sum(medium_vals)/len(medium_vals):.3f}")
    if sample_vals:
        logger.info(f"Sample any-correct rate: {sum(sample_vals)/len(sample_vals):.3f}")

    if n_helpful < 20:
        logger.warning("WARNING: very few helpful=1 examples. AUC will be unreliable.")
        logger.warning("Consider increasing n_samples or using a harder dataset subset.")

    logger.info("\nStep 3 complete.")


if __name__ == "__main__":
    main()
