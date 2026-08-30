import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "39_sampling_repair_eval.py"


def chain(text, tokens):
    return {"text": text, "token_count": tokens}


class SamplingSummaryCliTest(unittest.TestCase):
    def test_helpful_taxonomy_is_an_exhaustive_partition(self):
        base_rows = [
            ("truncated", [chain(r"\boxed{0}", 100), chain(r"\boxed{0}", 100)]),
            ("mixed", [chain(r"\boxed{0}", 100), chain(r"\boxed{0}", 80)]),
            ("complete_wrong", [chain(r"\boxed{0}", 80), chain(r"\boxed{0}", 70)]),
            (
                "complete_other",
                [
                    chain("no parseable final answer", 70),
                    chain("still no parseable final answer", 60),
                ],
            ),
        ]
        ext_rows = [
            (pid, [chain(r"\boxed{1}", 80), chain(r"\boxed{1}", 70)])
            for pid, _ in base_rows
        ]

        def records(rows):
            return [
                {
                    "problem_id": pid,
                    "dataset": "math",
                    "question": "q",
                    "answer": r"\boxed{1}",
                    "chains": chains,
                    "budget": {
                        "max_new_tokens": 100,
                        "n_chains": len(chains),
                        "temperature": 0.75,
                        "top_p": 0.95,
                    },
                }
                for pid, chains in rows
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in (("base", records(base_rows)), ("ext", records(ext_rows))):
                with open(root / f"inference_{name}.jsonl", "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--results-dir",
                    str(root),
                    "--base-name",
                    "base",
                    "--base-budget",
                    "100",
                    "--ext-name",
                    "ext",
                    "--ext-budget",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(
                root / "sampling_repair_base_to_ext_summary.csv",
                encoding="utf-8",
            ) as f:
                row = next(csv.DictReader(f))

        fractions = [
            float(row["helpful_from_truncated"]),
            float(row["helpful_from_mixed"]),
            float(row["helpful_from_complete_wrong"]),
            float(row["helpful_from_complete_other"]),
        ]
        self.assertEqual(fractions, [0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(sum(fractions), 1.0)
        self.assertEqual(int(row["helpful_taxonomy_n"]), 4)


if __name__ == "__main__":
    unittest.main()
