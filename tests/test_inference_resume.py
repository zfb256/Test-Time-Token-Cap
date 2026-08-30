import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "pipeline" / "02_run_inference.py"
SPEC = importlib.util.spec_from_file_location("run_inference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeEngine:
    collect_logprobs = False
    deterministic_sampling = False
    engine_seed = 42
    model_id = "fake-model"
    model_snapshot_sha256 = "model-hash"
    runtime_versions = {
        "python": "test",
        "torch": "test",
        "transformers": "test",
        "vllm": "test",
    }
    template = "qwen_math"
    store_token_ids = False
    dtype = "bfloat16"
    max_model_len = 2048
    trust_remote_code = False

    def generate_batch(self, problems, **kwargs):
        n = kwargs["n_chains"]
        return [
            {
                "problem_id": problem["problem_id"],
                "chains": [
                    {
                        "text": f"chain-{i}",
                        "token_count": 1,
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "chain_index": i,
                        "request_seed": None,
                    }
                    for i in range(n)
                ],
            }
            for problem in problems
        ]


class InferenceResumeTest(unittest.TestCase):
    def test_run_budget_checkpoints_before_a_later_batch_fails(self):
        calls = 0

        class InterruptedEngine(FakeEngine):
            def generate_batch(self, problems, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted")
                return super().generate_batch(problems, **kwargs)

        problems = [
            {
                "problem_id": f"p{i}",
                "dataset": "math",
                "question": "q",
                "answer": "a",
            }
            for i in range(MODULE.CHECKPOINT_PROBLEMS + 1)
        ]
        budget = {
            "max_new_tokens": 10,
            "n_chains": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                MODULE.run_budget(
                    InterruptedEngine(), problems, budget, path, resume=False
                )
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), MODULE.CHECKPOINT_PROBLEMS)

    def test_sampled_multi_chain_uses_stable_per_chain_seeds(self):
        self.assertIsNone(MODULE.request_seed(42, n_chains=8, temperature=0.75))
        self.assertEqual(MODULE.request_seed(42, n_chains=1, temperature=0), 42)
        policy = MODULE.seed_policy(42, n_chains=8, temperature=0.75)
        self.assertEqual(policy["mode"], "per_problem_per_chain_sha256")
        self.assertEqual(policy["request_seed"], "derived")
        seeds = [
            MODULE.deterministic_chain_seed(42, "problem-1", i)
            for i in range(8)
        ]
        self.assertEqual(
            seeds,
            [
                MODULE.deterministic_chain_seed(42, "problem-1", i)
                for i in range(8)
            ],
        )
        self.assertEqual(len(set(seeds)), 8)
        self.assertNotEqual(
            MODULE.deterministic_chain_seed(42, "problem-1", 0),
            MODULE.deterministic_chain_seed(42, "problem-2", 0),
        )
        self.assertNotEqual(
            MODULE.deterministic_chain_seed(42, "gsm8k_0025", 1),
            MODULE.deterministic_chain_seed(42, "gsm8k_0062", 0),
        )
        self.assertTrue(all(0 <= seed < 2**64 for seed in seeds))
        self.assertEqual(
            MODULE.effective_budget_seed(42, "long", {}), 2042
        )
        self.assertNotEqual(
            MODULE.effective_budget_seed(42, "long_sample8", {}),
            MODULE.effective_budget_seed(42, "xlong_sample8", {}),
        )
        self.assertEqual(len(MODULE.run_signature(FakeEngine())["prompt_sha256"]), 64)

    def test_model_snapshot_fingerprint_tracks_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            artifact = model / "weights.bin"
            artifact.write_bytes(b"first")
            first = MODULE.model_snapshot_sha256(model)
            self.assertEqual(first, MODULE.model_snapshot_sha256(model))
            artifact.write_bytes(b"other")
            self.assertNotEqual(first, MODULE.model_snapshot_sha256(model))

    def test_deterministic_sampling_issues_individually_seeded_requests(self):
        captured = {}

        class FakeLLM:
            def generate(self, prompts, params):
                captured["prompts"] = prompts
                captured["params"] = params
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text=f"seed-{param['seed']}",
                                token_ids=[1],
                                finish_reason="stop",
                                stop_reason=None,
                                logprobs=None,
                                cumulative_logprob=None,
                            )
                        ]
                    )
                    for param in params
                ]

        engine = MODULE.VLLMInferenceEngine.__new__(MODULE.VLLMInferenceEngine)
        engine.llm = FakeLLM()
        engine.SamplingParams = lambda **kwargs: kwargs
        engine.template = "qwen_math"
        engine.batch_size = 8
        engine.num_logprobs = 2
        engine.deterministic_sampling = True
        engine.store_token_ids = True
        problems = [
            {
                "problem_id": "p1",
                "question": "1+1?",
            }
        ]
        rows = engine.generate_batch(
            problems,
            max_new_tokens=10,
            n_chains=3,
            temperature=0.75,
            top_p=0.95,
            collect_logprobs=False,
            seed=42,
        )
        seeds = [param["seed"] for param in captured["params"]]
        self.assertEqual(len(captured["prompts"]), 3)
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(
            seeds,
            [MODULE.deterministic_chain_seed(42, "p1", i) for i in range(3)],
        )
        self.assertEqual(len(rows[0]["chains"]), 3)
        self.assertEqual(
            [chain["request_seed"] for chain in rows[0]["chains"]],
            seeds,
        )
        self.assertEqual(rows[0]["chains"][0]["token_ids"], [1])

    def test_diversity_guard_rejects_collapsed_samples(self):
        collapsed = [
            {
                "problem_id": f"p{i}",
                "chains": [{"text": "same"}, {"text": "same"}],
            }
            for i in range(5)
        ]
        with self.assertRaisesRegex(RuntimeError, "diversity check failed"):
            MODULE.assert_chain_diversity(collapsed, 2, 0.75)

    def test_diversity_guard_accepts_diverse_samples(self):
        diverse = [
            {
                "problem_id": f"p{i}",
                "chains": [{"text": "a"}, {"text": f"b{i}"}],
            }
            for i in range(5)
        ]
        MODULE.assert_chain_diversity(diverse, 2, 0.75)

    def test_resume_replaces_partial_and_deduplicates_complete_records(self):
        problems = [
            {
                "problem_id": "p1",
                "dataset": "math",
                "question": "q1",
                "answer": "a1",
            },
            {
                "problem_id": "p2",
                "dataset": "math",
                "question": "q2",
                "answer": "a2",
            },
        ]
        budget = {
            "max_new_tokens": 10,
            "n_chains": 2,
            "temperature": 0.5,
            "top_p": 0.9,
            "seed": 42,
            "seed_policy": {
                "engine_seed": 42,
                "request_seed": None,
                "base_seed": 42,
                "mode": "engine_only_multi_sample",
            },
            "run_signature": MODULE.run_signature(FakeEngine()),
        }
        partial = {
            **problems[0],
            "chains": [
                {
                    "text": "partial",
                    "token_count": 1,
                    "finish_reason": "stop",
                    "chain_index": 0,
                    "request_seed": None,
                }
            ],
            "budget": budget,
        }
        complete = {
            **problems[1],
            "chains": [
                {
                    "text": "old-1",
                    "token_count": 1,
                    "finish_reason": "stop",
                    "chain_index": 0,
                    "request_seed": None,
                },
                {
                    "text": "old-2",
                    "token_count": 1,
                    "finish_reason": "stop",
                    "chain_index": 1,
                    "request_seed": None,
                },
            ],
            "budget": budget,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(partial) + "\n")
                f.write(json.dumps(complete) + "\n")
                f.write(json.dumps(complete) + "\n")
            MODULE.run_budget(FakeEngine(), problems, budget, path, resume=True)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["problem_id"] for row in rows], ["p1", "p2"])
        self.assertEqual(len(rows[0]["chains"]), 2)
        self.assertEqual(rows[1]["chains"][0]["text"], "old-1")

    def test_resume_rejects_record_with_too_many_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "problem_id": "p1",
                        "chains": [
                            {"text": "a"},
                            {"text": "b"},
                            {"text": "unexpected-extra"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = MODULE.load_completed_records(path, expected_n_chains=2)
        self.assertEqual(loaded, {})

    def test_resume_rejects_mismatched_budget_or_engine_seed(self):
        expected = {
            "max_new_tokens": 100,
            "n_chains": 2,
            "temperature": 0.75,
            "top_p": 0.95,
        }
        record = {
            "problem_id": "p1",
            "chains": [{"text": "a"}, {"text": "b"}],
            "budget": {
                **expected,
                "max_new_tokens": 200,
                "seed_policy": {"engine_seed": 42},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            wrong_budget = MODULE.load_completed_records(
                path,
                expected_n_chains=2,
                expected_budget_cfg=expected,
                expected_engine_seed=42,
            )
            record["budget"]["max_new_tokens"] = 100
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            wrong_seed = MODULE.load_completed_records(
                path,
                expected_n_chains=2,
                expected_budget_cfg=expected,
                expected_engine_seed=314159,
            )
        self.assertEqual(wrong_budget, {})
        self.assertEqual(wrong_seed, {})

    def test_resume_rejects_changed_budget_seed_and_run_signature(self):
        record = {
            "problem_id": "p1",
            "chains": [{"text": "a"}],
            "budget": {
                "max_new_tokens": 100,
                "n_chains": 1,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "run_signature": {
                    "model": "model-a",
                    "prompt_template": "qwen_math",
                    "store_token_ids": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            wrong_seed = MODULE.load_completed_records(
                path,
                expected_budget_cfg={
                    "max_new_tokens": 100,
                    "n_chains": 1,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 43,
                },
            )
            wrong_model = MODULE.load_completed_records(
                path,
                expected_run_signature={
                    "model": "model-b",
                    "prompt_template": "qwen_math",
                    "store_token_ids": True,
                },
            )
        self.assertEqual(wrong_seed, {})
        self.assertEqual(wrong_model, {})

    def test_run_budget_rejects_duplicate_problem_ids(self):
        problem = {
            "problem_id": "p1",
            "dataset": "math",
            "question": "q",
            "answer": "a",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.run_budget(
                    FakeEngine(),
                    [problem, dict(problem)],
                    {
                        "max_new_tokens": 10,
                        "n_chains": 1,
                        "temperature": 0,
                        "top_p": 1,
                    },
                    Path(tmp) / "out.jsonl",
                    resume=False,
                )

    def test_resume_rejects_stale_problem_metadata(self):
        expected_problem = {
            "problem_id": "p1",
            "dataset": "math",
            "question": "current question",
            "answer": "current answer",
        }
        stale = {
            **expected_problem,
            "question": "stale question",
            "chains": [{"text": "a"}],
            "budget": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
            loaded = MODULE.load_completed_records(
                path,
                expected_n_chains=1,
                expected_problems={"p1": expected_problem},
            )
        self.assertEqual(loaded, {})

    def test_resume_rejects_corrupt_chain_metadata(self):
        problem = {
            "problem_id": "p1",
            "dataset": "math",
            "question": "q",
            "answer": "a",
        }
        budget = {
            "max_new_tokens": 2,
            "n_chains": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": 9,
        }
        policy = MODULE.seed_policy(
            9, n_chains=1, temperature=0, engine_seed=42
        )
        signature = {
            "model": "m",
            "prompt_template": "qwen_math",
            "store_token_ids": True,
        }
        record = {
            **problem,
            "chains": [
                {
                    "text": "answer",
                    "token_count": 2,
                    "token_ids": [1],
                    "finish_reason": "stop",
                    "chain_index": 0,
                    "request_seed": 9,
                }
            ],
            "budget": {
                **budget,
                "seed_policy": policy,
                "run_signature": signature,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = MODULE.load_completed_records(
                path,
                expected_n_chains=1,
                expected_budget_cfg=budget,
                expected_engine_seed=42,
                expected_problems={"p1": problem},
                expected_seed_policy=policy,
                expected_run_signature=signature,
            )
        self.assertEqual(loaded, {})

    def test_run_budget_rejects_missing_or_duplicate_engine_results(self):
        problem = {
            "problem_id": "p1",
            "dataset": "math",
            "question": "q",
            "answer": "a",
        }

        class BadEngine(FakeEngine):
            def generate_batch(self, problems, **kwargs):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "problem IDs"):
                MODULE.run_budget(
                    BadEngine(),
                    [problem],
                    {
                        "max_new_tokens": 10,
                        "n_chains": 1,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 42,
                    },
                    Path(tmp) / "out.jsonl",
                    resume=False,
                )

    def test_complete_resume_cleans_duplicate_and_malformed_lines(self):
        problem = {
            "problem_id": "p1",
            "dataset": "math",
            "question": "q",
            "answer": "a",
        }
        budget = {
            "max_new_tokens": 10,
            "n_chains": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        }
        policy = MODULE.seed_policy(
            42,
            n_chains=1,
            temperature=0,
            engine_seed=FakeEngine.engine_seed,
            deterministic_sampling=False,
        )
        record = {
            **problem,
            "chains": [
                {
                    "text": "done",
                    "token_count": 1,
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "chain_index": 0,
                    "request_seed": 42,
                }
            ],
            "budget": {
                **budget,
                "seed_policy": policy,
                "deterministic_sampling": False,
                "run_signature": MODULE.run_signature(FakeEngine()),
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_test.jsonl"
            path.write_text(
                json.dumps(record) + "\n{broken\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            MODULE.run_budget(
                FakeEngine(), [problem], budget, path, resume=True
            )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(rows, [record])


if __name__ == "__main__":
    unittest.main()
