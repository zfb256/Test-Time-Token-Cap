'Step 4: Build G1-G4 feature matrix + utility labels.'

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.feature_utils import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def process_record(rec: Dict, fb: FeatureBuilder) -> Dict:
    'Convert one prm_scores record into a feature record.'
    pid = rec["problem_id"]
    dataset = rec["dataset"]
    question = rec["question"]
    math_level = rec.get("math_level", 0)


    g1 = fb.build_g1(question, dataset)


    short_chains = rec.get("short_chains", [])
    if short_chains:

        all_steps = [
            step
            for chain in short_chains
            for step in chain.get("step_scores", [])
        ]
        if all_steps:
            prm_mean = float(np.mean(all_steps))
            prm_min = float(np.min(all_steps))
            prm_var = float(np.var(all_steps))
        else:
            prm_mean = float(np.mean([c["mean"] for c in short_chains]))
            prm_min = float(np.mean([c["min"] for c in short_chains]))
            prm_var = float(np.mean([c["var"] for c in short_chains]))
    else:
        prm_mean = prm_min = 0.5
        prm_var = 0.0
        all_steps = []

    g2 = fb.build_g2(
        prm_scores=all_steps,
        prm_mean=prm_mean,
        prm_min=prm_min,
        prm_var=prm_var,
    )


    extracted_answers = [c.get("extracted_answer") for c in short_chains]
    g3 = fb.build_g3(extracted_answers)
    uncertainty = fb.build_uncertainty(short_chains)


    groups = fb.build_feature_groups(
        g1, g2, g3, math_level=math_level, uncertainty=uncertainty
    )

    return {
        "problem_id": pid,
        "dataset": dataset,
        "math_level": math_level,
        "label": rec["utility_label"],
        "short_orm_majority": rec.get("short_orm_majority", False),
        "long_orm_correct": rec.get("long_orm_correct", False),
        "medium_orm_correct": rec.get("medium_orm_correct"),
        "sample_any_correct": rec.get("sample_any_correct"),
        "sample_majority_correct": rec.get("sample_majority_correct"),
        "features": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    in_path = out_dir / "prm_scores.jsonl"

    if not in_path.exists():
        logger.error(f"Missing: {in_path}. Run 03_prm_scoring.py first.")
        sys.exit(1)


    records = []
    seen = set()
    with open(in_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                record = json.loads(line)
                problem_id = record.get("problem_id")
                if problem_id is None:
                    raise ValueError(
                        f"{in_path}:{line_no}: missing problem_id"
                    )
                if problem_id in seen:
                    raise ValueError(
                        f"{in_path}:{line_no}: duplicate "
                        f"problem_id={problem_id}"
                    )
                seen.add(problem_id)
                records.append(record)
    logger.info(f"Loaded {len(records)} records from {in_path}")
    if not records:
        logger.error("No records found in prm_scores.jsonl.")
        sys.exit(1)

    fb = FeatureBuilder()


    feature_records = [
        process_record(rec, fb)
        for rec in tqdm(records, desc="Building features")
    ]


    out_jsonl = out_dir / "features.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in feature_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved features.jsonl: {out_jsonl}")


    for group_name in ["G1", "G2", "G3", "G4"]:
        X, y = FeatureBuilder.records_to_dataframe(feature_records, group_name)
        df_out = X.copy()
        df_out["label"] = y.values
        df_out["dataset"] = [r["dataset"] for r in feature_records]
        df_out["math_level"] = [r["math_level"] for r in feature_records]

        csv_path = out_dir / f"features_{group_name}.csv"
        df_out.to_csv(csv_path, index=True)
        logger.info(f"Saved {group_name} CSV: {csv_path} ({len(df_out)} rows × {len(X.columns)} features)")


    labels = [r["label"] for r in feature_records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    logger.info(f"\nLabel distribution:")
    logger.info(f"  helpful=1: {n_pos} ({100*n_pos/len(labels):.1f}%)")
    logger.info(f"  helpful=0: {n_neg} ({100*n_neg/len(labels):.1f}%)")
    logger.info(f"  Imbalance ratio: {n_neg/max(n_pos,1):.1f}:1")

    if n_pos < 30:
        logger.warning("WARNING: <30 positive examples. Cross-validation AUC will be noisy.")

    logger.info("\nStep 4 complete.")


if __name__ == "__main__":
    main()
