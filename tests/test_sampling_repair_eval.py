import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "39_sampling_repair_eval.py"
SPEC = importlib.util.spec_from_file_location("sampling_repair_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeVerifier:
    def verify(self, text, answer, dataset):
        return text.endswith("CORRECT")

    def extract_prediction_answer(self, text, dataset):
        if "NOANSWER" in text:
            return None
        return text.rsplit(" ", 1)[-1]


def record(*chains):
    return {
        "dataset": "math",
        "answer": "CORRECT",
        "chains": [
            {"text": text, "token_count": token_count}
            for text, token_count in chains
        ],
    }


class SamplingRepairTaxonomyTest(unittest.TestCase):
    def test_finish_reason_overrides_legacy_token_count_heuristic(self):
        rec = record(("done WRONG", 4))
        rec["chains"][0]["finish_reason"] = "length"
        result = MODULE.eval_run(rec, 8, FakeVerifier())
        self.assertTrue(result["all_hit_max"])
        self.assertFalse(result["all_complete"])

    def test_natural_stop_at_cap_is_complete(self):
        rec = record(("done WRONG", 8))
        rec["chains"][0]["finish_reason"] = "stop"
        result = MODULE.eval_run(rec, 8, FakeVerifier())
        self.assertFalse(result["all_hit_max"])
        self.assertTrue(result["all_complete"])
        self.assertTrue(result["complete_wrong"])

    def test_legacy_chain_at_cap_remains_conservatively_truncated(self):
        result = MODULE.eval_run(record(("done WRONG", 8)), 8, FakeVerifier())
        self.assertTrue(result["all_hit_max"])
        self.assertFalse(result["all_complete"])

    def test_all_truncated_is_not_complete_wrong(self):
        result = MODULE.eval_run(
            record(("work WRONG", 100), ("more WRONG", 100)), 100, FakeVerifier()
        )
        self.assertEqual(result["completion_state"], "all_truncated")
        self.assertFalse(result["complete_wrong"])

    def test_mixed_set_is_kept_out_of_complete_wrong(self):
        result = MODULE.eval_run(
            record(("work WRONG", 100), ("done WRONG", 80)), 100, FakeVerifier()
        )
        self.assertEqual(result["completion_state"], "mixed_truncated_complete")
        self.assertFalse(result["complete_wrong"])

    def test_complete_wrong_requires_every_chain_to_be_answered(self):
        complete_wrong = MODULE.eval_run(
            record(("done WRONG", 80), ("also WRONG", 70)), 100, FakeVerifier()
        )
        missing_answer = MODULE.eval_run(
            record(("done WRONG", 80), ("NOANSWER", 70)), 100, FakeVerifier()
        )
        self.assertEqual(
            complete_wrong["completion_state"], "all_complete_answered_wrong"
        )
        self.assertTrue(complete_wrong["complete_wrong"])
        self.assertEqual(missing_answer["completion_state"], "complete_other")
        self.assertFalse(missing_answer["complete_wrong"])


if __name__ == "__main__":
    unittest.main()
