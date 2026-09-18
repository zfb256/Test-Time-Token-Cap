#!/usr/bin/env python3
'Step 9 (v2): Download AIME 2024 + 2025 problems.'

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


CANDIDATES_2024 = [
    ("Maxwell-Jia/AIME_2024", "train", "Problem", "Answer", "aime24"),
    ("HuggingFaceH4/aime_2024", "train", "problem", "answer", "aime24"),
]
CANDIDATES_2025 = [
    ("yentinglin/aime_2025", "train", "problem", "answer", "aime25"),
    ("math-ai/aime25", "test", "problem", "answer", "aime25"),
    ("opencompass/AIME2025", "test", "question", "answer", "aime25"),
]


def try_load(candidates):
    from datasets import load_dataset
    last_err = None
    for hf_id, split, qf, af, tag in candidates:
        try:
            ds = load_dataset(hf_id, split=split)
            if qf not in ds.column_names or af not in ds.column_names:

                cols = {c.lower(): c for c in ds.column_names}
                qf = cols.get(qf.lower(), qf)
                af = cols.get(af.lower(), af)
            rows = []
            for i, r in enumerate(ds):
                q = str(r[qf]).strip()
                raw_answer = str(r[af]).strip()
                boxed = re.fullmatch(r"\\boxed\{\s*(\d{1,3})\s*\}", raw_answer)
                plain = re.fullmatch(r"(\d{1,3})", raw_answer)
                match = boxed or plain
                if not match:
                    raise ValueError(
                        f"{hf_id} row {i} has a non-integer AIME answer: "
                        f"{raw_answer!r}"
                    )

                a = int(match.group(1))
                if not 0 <= a <= 999:
                    raise ValueError(f"{hf_id} row {i} answer out of range: {a}")

                if not q:
                    continue
                rows.append({
                    "problem_id": f"{tag}_{i:03d}",
                    "dataset": "aime",
                    "question": q,
                    "answer": f"\\boxed{{{a}}}",
                    "math_level": 6,  
                    "source": hf_id,
                })
            if len(rows) == 30:
                print(f"Loaded {len(rows)} problems from {hf_id}")
                return rows
            raise ValueError(f"{hf_id} returned {len(rows)} rows; expected exactly 30")
        except Exception as e:  
            last_err = e
            print(f"  {hf_id} failed: {e}")
    raise RuntimeError(f"All candidates failed; last error: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR.parent / "data" / "aime"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_problems.jsonl"

    rows = try_load(CANDIDATES_2024) + try_load(CANDIDATES_2025)
    if len({row["problem_id"] for row in rows}) != 60:
        raise RuntimeError("AIME download produced duplicate or missing problem IDs")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    print(f"Saved {len(rows)} problems -> {out_path}")


    bad = [r for r in rows if not r["answer"].startswith("\\boxed{")]
    if bad:
        print(f"WARNING: {len(bad)} records lack boxed answers")
        sys.exit(1)


if __name__ == "__main__":
    main()
