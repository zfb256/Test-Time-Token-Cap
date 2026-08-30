import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "42_relabel_cached_scores.py"
SPEC = importlib.util.spec_from_file_location("cached_relabel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CachedRelabelTest(unittest.TestCase):
    def test_plurality_uses_answer_votes_not_correctness_votes(self):
        evaluated = [
            {"answer": "wrong-a", "correct": False},
            {"answer": "wrong-b", "correct": False},
            {"answer": "42", "correct": True},
            {"answer": "42", "correct": True},
        ]
        self.assertTrue(MODULE.plurality_correct(evaluated))

    def test_plurality_tie_break_is_deterministic(self):
        evaluated = [
            {"answer": "b", "correct": True},
            {"answer": "a", "correct": False},
        ]
        self.assertFalse(MODULE.plurality_correct(evaluated))


if __name__ == "__main__":
    unittest.main()
