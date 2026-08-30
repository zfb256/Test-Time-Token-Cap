import copy
import importlib.util
from pathlib import Path
import unittest


RUN_SIGNATURE = {
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
    "store_token_ids": True,
    "dtype": "bfloat16",
    "max_model_len": 2048,
    "trust_remote_code": False,
}


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "47_reconstruct_lower_cap.py"
)
SPEC = importlib.util.spec_from_file_location("reconstruct_lower_cap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReconstructLowerCapTest(unittest.TestCase):
    def test_vllm_compatible_decode_suppresses_incomplete_utf8_tail(self):
        class Tokenizer:
            def decode(self, token_ids, **kwargs):
                self.kwargs = kwargs
                return "prefix\ufffd"

        tokenizer = Tokenizer()
        self.assertEqual(
            MODULE.decode_generated_tokens(tokenizer, [1, 2]), "prefix"
        )
        self.assertTrue(tokenizer.kwargs["skip_special_tokens"])
        self.assertFalse(tokenizer.kwargs["clean_up_tokenization_spaces"])

    def test_truncates_token_ids_and_marks_length_stop(self):
        chain = {
            "text": "full",
            "token_ids": [1, 2, 3, 4],
            "token_count": 4,
            "finish_reason": "stop",
            "stop_reason": "eos",
        }
        rebuilt = MODULE.reconstruct_chain(
            chain, 2, lambda ids: "-".join(map(str, ids))
        )
        self.assertEqual(rebuilt["text"], "1-2")
        self.assertEqual(rebuilt["token_ids"], [1, 2])
        self.assertEqual(rebuilt["token_count"], 2)
        self.assertEqual(rebuilt["finish_reason"], "length")
        self.assertIsNone(rebuilt["stop_reason"])

    def test_preserves_natural_completion_below_cap(self):
        chain = {
            "text": "done",
            "token_ids": [1, 2],
            "token_count": 2,
            "finish_reason": "stop",
        }
        rebuilt = MODULE.reconstruct_chain(chain, 3, lambda ids: "unused")
        self.assertEqual(rebuilt, chain)
        self.assertIsNot(rebuilt, chain)

    def test_rejects_missing_or_inconsistent_token_ids(self):
        with self.assertRaisesRegex(ValueError, "lacks token_ids"):
            MODULE.reconstruct_chain({"token_count": 2}, 1, str)
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.reconstruct_chain(
                {"token_ids": [1], "token_count": 2}, 1, str
            )

    def test_validation_detects_a_tampered_prefix(self):
        source = {
            "p": {
                "problem_id": "p",
                "dataset": "math",
                "question": "q",
                "answer": "a",
                "chains": [
                    {
                        "text": "source",
                        "token_ids": [1, 2, 3],
                        "token_count": 3,
                        "finish_reason": "length",
                        "stop_reason": None,
                        "chain_index": 0,
                        "request_seed": 7,
                    }
                ],
                "budget": {
                    "max_new_tokens": 3,
                    "n_chains": 1,
                    "seed_policy": {"base_seed": 5},
                    "run_signature": RUN_SIGNATURE,
                },
            }
        }
        rebuilt_record = copy.deepcopy(source["p"])
        rebuilt_record["chains"][0].update(
            {
                "text": "prefix",
                "token_ids": [1, 9],
                "token_count": 2,
                "finish_reason": "length",
                "stop_reason": None,
                "reconstructed_at_cap": 2,
            }
        )
        rebuilt_record["budget"].update(
            {
                "max_new_tokens": 2,
                "mode": "reconstructed_token_prefix",
                "reconstructed_from": {"source_cap": 3},
            }
        )
        with self.assertRaisesRegex(ValueError, "token-prefix mismatch"):
            MODULE.validate_reconstruction(
                source, {"p": rebuilt_record}, cap=2
            )

    def test_validation_detects_tampered_decoded_text(self):
        source = {
            "p": {
                "problem_id": "p",
                "dataset": "math",
                "question": "q",
                "answer": "a",
                "chains": [
                    {
                        "text": "full",
                        "token_ids": [1, 2, 3],
                        "token_count": 3,
                        "finish_reason": "length",
                        "stop_reason": None,
                        "chain_index": 0,
                        "request_seed": 7,
                    }
                ],
                "budget": {
                    "max_new_tokens": 3,
                    "n_chains": 1,
                    "seed_policy": {"base_seed": 5},
                    "run_signature": RUN_SIGNATURE,
                },
            }
        }
        rebuilt_record = copy.deepcopy(source["p"])
        rebuilt_record["chains"][0].update(
            {
                "text": "tampered",
                "token_ids": [1, 2],
                "token_count": 2,
                "finish_reason": "length",
                "stop_reason": None,
                "reconstructed_at_cap": 2,
            }
        )
        rebuilt_record["budget"].update(
            {
                "max_new_tokens": 2,
                "mode": "reconstructed_token_prefix",
                "reconstructed_from": {"source_cap": 3},
            }
        )
        with self.assertRaisesRegex(ValueError, "decoded prefix text mismatch"):
            MODULE.validate_reconstruction(
                source,
                {"p": rebuilt_record},
                cap=2,
                decode=lambda ids: "correct-prefix",
            )

    def test_validation_rejects_changed_run_signature(self):
        source = {
            "p": {
                "problem_id": "p",
                "dataset": "math",
                "question": "q",
                "answer": "a",
                "chains": [],
                "budget": {
                    "max_new_tokens": 3,
                    "n_chains": 0,
                    "seed_policy": {},
                    "run_signature": RUN_SIGNATURE,
                },
            }
        }
        rebuilt = copy.deepcopy(source)
        rebuilt["p"]["budget"].update(
            {
                "max_new_tokens": 2,
                "mode": "reconstructed_token_prefix",
                "reconstructed_from": {"source_cap": 3},
                "run_signature": {**RUN_SIGNATURE, "model": "other"},
            }
        )
        with self.assertRaisesRegex(ValueError, "run signature"):
            MODULE.validate_reconstruction(source, rebuilt, cap=2)


if __name__ == "__main__":
    unittest.main()
