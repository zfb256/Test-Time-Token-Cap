import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "pipeline"
    / "08_run_continuation.py"
)
SPEC = importlib.util.spec_from_file_location("run_continuation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


SIGNATURE = {
    "source_budget": "medium",
    "target_budget": "long",
}


def source(finish_reason="length"):
    return {
        "problem_id": "p",
        "dataset": "math",
        "question": "q",
        "answer": "a",
        "chains": [
            {
                "text": "base",
                "token_count": 2,
                "token_ids": [1, 2],
                "finish_reason": finish_reason,
                "stop_reason": None,
                "chain_index": 0,
                "request_seed": 7,
            }
        ],
    }


def resumed(continued=True):
    chain = {
        "text": "baseplus" if continued else "base",
        "token_count": 3 if continued else 2,
        "token_ids": [1, 2, 3] if continued else [1, 2],
        "finish_reason": "stop",
        "stop_reason": None,
        "chain_index": 0,
        "request_seed": 9 if continued else 7,
    }
    return {
        "problem_id": "p",
        "dataset": "math",
        "question": "q",
        "answer": "a",
        "chains": [chain],
        "budget": {
            "max_new_tokens": 4,
            "n_chains": 1,
            "resume_signature": SIGNATURE,
        },
        "resume": {
            "continued": continued,
            "medium_tokens": 2,
            "continuation_tokens": 1 if continued else 0,
        },
    }


class ContinuationValidationTest(unittest.TestCase):
    def test_accepts_valid_truncated_continuation(self):
        MODULE.validate_resume_record(
            source("length"), resumed(True), 2, 4, SIGNATURE
        )

    def test_rejects_noop_for_truncated_source(self):
        with self.assertRaisesRegex(ValueError, "continued flag"):
            MODULE.validate_resume_record(
                source("length"), resumed(False), 2, 4, SIGNATURE
            )

    def test_requires_natural_completion_to_remain_identical(self):
        natural_source = source("stop")
        valid = resumed(False)
        MODULE.validate_resume_record(
            natural_source, valid, 2, 4, SIGNATURE
        )
        valid["chains"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "not preserved"):
            MODULE.validate_resume_record(
                natural_source, valid, 2, 4, SIGNATURE
            )


if __name__ == "__main__":
    unittest.main()
