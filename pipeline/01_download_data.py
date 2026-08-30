"""
Step 1: Download and sample GSM8K + MATH datasets.

Outputs:
  {data_dir}/gsm8k_samples.jsonl
  {data_dir}/math_samples.jsonl
  {data_dir}/all_problems.jsonl   (combined, with dataset tag)

Run:
  python 01_download_data.py --config config.yaml
  python 01_download_data.py --config config.yaml --dry-run
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and sample math datasets")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write files")
    parser.add_argument("--data-dir", help="Override paths.data_dir")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use every example in the configured GSM8K and MATH splits",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data_dir:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.full:
        cfg["dataset"]["gsm8k"]["n_samples"] = None
        cfg["dataset"]["math"]["n_per_level"] = None

    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    id_prefix = "full_" if args.full else ""

    all_problems = []

    # ------ GSM8K ------
    if cfg["dataset"]["gsm8k"]["enabled"]:
        logger.info("Loading GSM8K...")
        gsm8k = load_dataset("openai/gsm8k", "main", split=cfg["dataset"]["gsm8k"]["split"])
        n = cfg["dataset"]["gsm8k"]["n_samples"]
        seed = cfg["dataset"]["gsm8k"]["seed"]

        if n and n < len(gsm8k):
            random.seed(seed)
            indices = random.sample(range(len(gsm8k)), n)
            gsm8k_samples = [gsm8k[i] for i in sorted(indices)]
        else:
            gsm8k_samples = list(gsm8k)

        logger.info(f"GSM8K: {len(gsm8k_samples)} samples selected")

        gsm8k_records = []
        for i, item in enumerate(gsm8k_samples):
            record = {
                "problem_id": f"{id_prefix}gsm8k_{i:04d}",
                "dataset": "gsm8k",
                "source": "openai/gsm8k",
                "source_split": cfg["dataset"]["gsm8k"]["split"],
                "question": item["question"],
                "answer": item["answer"],   # full answer with #### at end
                "math_level": 0,
            }
            gsm8k_records.append(record)
            all_problems.append(record)

        if not args.dry_run:
            out_path = data_dir / "gsm8k_samples.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in gsm8k_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"Saved: {out_path}")

    # ------ MATH ------
    if cfg["dataset"]["math"]["enabled"]:
        logger.info("Loading MATH dataset...")
        math_ds = None
        dataset_names = cfg["dataset"]["math"].get(
            "hf_names",
            ["lighteval/MATH", "hendrycks/competition_math", "competition_math"],
        )
        for ds_name in dataset_names:
            try:
                math_ds = load_dataset(ds_name, split=cfg["dataset"]["math"]["split"])
                logger.info(f"Loaded MATH from: {ds_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {ds_name}: {e}")

        if math_ds is None:
            logger.error("Could not load MATH dataset. Try: pip install datasets and check HF access.")
            sys.exit(1)

        n_per_level = cfg["dataset"]["math"]["n_per_level"]
        levels = cfg["dataset"]["math"]["levels"]
        seed = cfg["dataset"]["math"]["seed"]
        random.seed(seed)

        math_records = []
        for level in levels:
            level_str = str(level)
            # Different datasets use different field names for difficulty
            level_items = [
                item for item in math_ds
                if str(item.get("level", item.get("difficulty", "0"))).replace("Level ", "") == level_str
            ]

            if not level_items:
                logger.warning(f"No items found for MATH level {level}")
                continue

            sample_size = len(level_items) if n_per_level is None else min(n_per_level, len(level_items))
            sampled = random.sample(level_items, sample_size)
            logger.info(f"MATH level {level}: {len(level_items)} total, {len(sampled)} sampled")

            for i, item in enumerate(sampled):
                problem_id = f"{id_prefix}math_l{level}_{i:04d}"
                record = {
                    "problem_id": problem_id,
                    "dataset": "math",
                    "source": ds_name,
                    "source_split": cfg["dataset"]["math"]["split"],
                    "question": item.get("problem", item.get("question", "")),
                    "answer": item.get("solution", item.get("answer", "")),
                    "math_level": level,
                    "math_type": item.get("type", ""),
                }
                math_records.append(record)
                all_problems.append(record)

        logger.info(f"MATH: {len(math_records)} samples total")

        if not args.dry_run:
            out_path = data_dir / "math_samples.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in math_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"Saved: {out_path}")

    # ------ Combined ------
    logger.info(f"Total problems: {len(all_problems)}")
    logger.info(f"  GSM8K: {sum(1 for p in all_problems if p['dataset']=='gsm8k')}")
    logger.info(f"  MATH:  {sum(1 for p in all_problems if p['dataset']=='math')}")
    if not all_problems:
        logger.error("No problems selected. Check dataset.enabled settings and MATH level filters.")
        sys.exit(1)

    if not args.dry_run:
        out_path = data_dir / "all_problems.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_problems:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved combined: {out_path}")
    else:
        logger.info("[DRY RUN] No files written.")

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()
