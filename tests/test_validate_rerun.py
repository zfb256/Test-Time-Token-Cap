import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "45_validate_rerun.py"
SPEC = importlib.util.spec_from_file_location("validate_rerun", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ValidateRerunTest(unittest.TestCase):
    def test_rejects_missing_run_signature(self):
        record = {
            "problem_id": "p",
            "dataset": "math",
            "question": "q",
            "answer": "a",
            "budget": {
                "max_new_tokens": 1,
                "n_chains": 1,
                "temperature": 0,
                "top_p": 1,
                "seed_policy": {
                    "engine_seed": 1,
                    "base_seed": 1,
                    "request_seed": 1,
                    "mode": "per_request_single_chain",
                },
            },
            "chains": [{
                "text": "x",
                "token_count": 1,
                "finish_reason": "stop",
                "stop_reason": None,
                "chain_index": 0,
                "request_seed": 1,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run signature"):
                MODULE.validate_file(path, 1)
            record["budget"]["run_signature"] = {
                "model": "model",
                "model_snapshot_sha256": "model-hash",
                "runtime_versions": {
                    "python": "test",
                    "torch": "test",
                    "transformers": "test",
                    "vllm": "test",
                },
                "prompt_template": "qwen_math",
                "prompt_sha256": "prompt-hash",
                "store_token_ids": False,
                "dtype": "bfloat16",
                "max_model_len": 2048,
                "trust_remote_code": False,
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(MODULE.validate_file(path, 1), (1, 1))


if __name__ == "__main__":
    unittest.main()
