import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "44_same_budget_control.py"
)
SPEC = importlib.util.spec_from_file_location("same_budget_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SameBudgetControlTest(unittest.TestCase):
    def test_censors_chain_at_or_above_cap(self):
        rec = {
            "chains": [
                {"text": "short", "token_count": 99},
                {"text": "at cap answer", "token_count": 100},
                {"text": "past cap answer", "token_count": 140},
            ]
        }
        censored = MODULE.censor_to_completed_chains(rec, 100)
        self.assertEqual(censored["chains"][0]["text"], "short")
        self.assertEqual(censored["chains"][1]["text"], "")
        self.assertEqual(censored["chains"][2]["text"], "")
        self.assertEqual(rec["chains"][1]["text"], "at cap answer")

    def test_keeps_natural_stop_exactly_at_cap(self):
        rec = {
            "chains": [
                {
                    "text": "natural boundary stop",
                    "token_count": 100,
                    "finish_reason": "stop",
                }
            ]
        }
        censored = MODULE.censor_to_completed_chains(rec, 100)
        self.assertEqual(censored["chains"][0], rec["chains"][0])

    def test_rejects_mismatched_problem_sets(self):
        with self.assertRaises(SystemExit):
            MODULE.validate_pair({"a": {}}, {"b": {}})


if __name__ == "__main__":
    unittest.main()
