import importlib.util
import json
from pathlib import Path
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "three_arm", ROOT / "analysis" / "53_three_arm_repair_decomposition.py"
)
THREE_ARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THREE_ARM)


def run_signature(model="model"):
    return {
        "model": model,
        "model_snapshot_sha256": f"{model}-hash",
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


def record(pid, chains):
    return {
        "problem_id": pid,
        "dataset": "gsm8k",
        "question": "q",
        "answer": "#### 2",
        "chains": [
            {"text": f"#### {answer}", "finish_reason": "stop"}
            for answer in chains
        ],
    }


def test_three_arm_separates_resampling_from_incremental_tokens():
    base = {"p": record("p", [1, 1, 1])}
    samecap = {"p": record("p", [1, 1, 1])}
    long = {"p": record("p", [2, 2, 1])}
    detail = THREE_ARM.analyze(base, samecap, long, 768, 1536)
    row = detail[detail["definition"] == "strict_all_wrong"].iloc[0]
    assert row["eligible"]
    assert not row["samecap_repair_plurality"]
    assert row["long_repair_plurality"]
    assert row["incremental_plurality_helpful"]


def test_three_arm_subtracts_same_cap_resampling_recovery():
    base = {"p": record("p", [1, 1, 1])}
    samecap = {"p": record("p", [2, 2, 1])}
    long = {"p": record("p", [2, 2, 1])}
    summary = THREE_ARM.summarize(
        THREE_ARM.analyze(base, samecap, long, 768, 1536)
    )
    row = summary[
        (summary["dataset"] == "all")
        & (summary["definition"] == "strict_all_wrong")
    ].iloc[0]
    assert row["samecap_repair_rate"] == 1
    assert row["long_repair_rate"] == 1
    assert row["incremental_token_effect"] == 0


def test_three_arm_reports_harm_for_each_contrast():
    base = {"p": record("p", [2, 2, 2])}
    samecap = {"p": record("p", [1, 1, 1])}
    long = {"p": record("p", [2, 2, 2])}
    detail = THREE_ARM.analyze_harm(base, samecap, long, 768, 1536)
    rows = detail[
        detail["definition"] == "complete_plurality_correct"
    ].set_index("contrast")
    assert rows.loc["A_to_B_resampling", "harm"]
    assert not rows.loc["A_to_C_combined", "harm"]
    assert rows.loc["B_to_C_tokens", "eligible"]
    assert not rows.loc["B_to_C_tokens", "harm"]


def test_three_arm_token_harm_uses_arm_a_eligibility():
    base = {"p": record("p", [2, 2, 2])}
    samecap = {"p": record("p", [2, 2, 2])}
    long = {"p": record("p", [1, 1, 1])}
    detail = THREE_ARM.analyze_harm(base, samecap, long, 768, 1536)
    row = detail[
        (detail["contrast"] == "B_to_C_tokens")
        & (detail["definition"] == "complete_plurality_correct")
    ].iloc[0]
    assert row["eligible"]
    assert row["harm"]


def test_three_arm_checkpoint_resumes_completed_problems():
    base = {"p": record("p", [1, 1, 1])}
    samecap = {"p": record("p", [1, 1, 1])}
    long = {"p": record("p", [2, 2, 1])}
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "progress.sqlite"
        first = THREE_ARM.analyze_checkpointed(
            base, samecap, long, 768, 1536, checkpoint, "test"
        )
        with mock.patch.object(
            THREE_ARM, "analyze", side_effect=AssertionError("recomputed")
        ):
            second = THREE_ARM.analyze_checkpointed(
                base, samecap, long, 768, 1536, checkpoint, "test"
            )
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_three_arm_jsonl_streams_in_ten_problem_batches():
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"arm_{index}.jsonl" for index in range(3)]
        for path in paths:
            path.write_text(
                "".join(
                    json.dumps({"problem_id": f"p{index}"}) + "\n"
                    for index in range(11)
                ),
                encoding="utf-8",
            )
        batches = list(THREE_ARM._aligned_batches(paths))
    assert [len(batch[0]) for batch in batches] == [10, 1]


def test_three_arm_protocol_rejects_nonprefix_control():
    base = record("p", [1])
    samecap = record("p", [1])
    long = record("p", [1])
    for item, cap, seed in (
        (base, 2, 11),
        (samecap, 2, 22),
        (long, 4, 22),
    ):
        item["budget"] = {
            "max_new_tokens": cap,
            "n_chains": 1,
            "temperature": 0.75,
            "top_p": 0.95,
            "seed_policy": {"base_seed": seed},
            "run_signature": run_signature(),
        }
        item["chains"][0].update(
            {
                "token_ids": [1, 2] if cap == 2 else [1, 2, 3],
                "token_count": 2 if cap == 2 else 3,
                "chain_index": 0,
                "request_seed": seed,
            }
        )
    samecap["budget"].update(
        {
            "mode": "reconstructed_token_prefix",
            "reconstructed_from": {"source_cap": 4},
        }
    )
    samecap["chains"][0]["token_ids"] = [9, 2]
    try:
        THREE_ARM.validate_protocol(
            {"p": base}, {"p": samecap}, {"p": long}, 2, 4
        )
    except ValueError as error:
        assert "exact C prefix" in str(error)
    else:
        raise AssertionError("nonprefix B/C control was accepted")


def test_three_arm_protocol_rejects_cross_chain_seed_overlap():
    base = record("p", [1, 1])
    samecap = record("p", [1, 1])
    long = record("p", [1, 1])
    for item, cap, seeds in (
        (base, 2, [11, 22]),
        (samecap, 2, [22, 33]),
        (long, 4, [22, 33]),
    ):
        item["budget"] = {
            "max_new_tokens": cap,
            "n_chains": 2,
            "temperature": 0.75,
            "top_p": 0.95,
            "seed_policy": {"base_seed": seeds[0]},
            "run_signature": run_signature(),
        }
        for index, chain in enumerate(item["chains"]):
            chain.update(
                {
                    "token_ids": [1, 2] if cap == 2 else [1, 2, 3],
                    "token_count": 2 if cap == 2 else 3,
                    "chain_index": index,
                    "request_seed": seeds[index],
                }
            )
    samecap["budget"].update(
        {
            "mode": "reconstructed_token_prefix",
            "reconstructed_from": {"source_cap": 4},
        }
    )
    try:
        THREE_ARM.validate_protocol(
            {"p": base}, {"p": samecap}, {"p": long}, 2, 4
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("cross-chain A/C seed overlap was accepted")


def test_three_arm_protocol_rejects_model_mismatch():
    base = record("p", [1])
    samecap = record("p", [1])
    long = record("p", [1])
    for item, cap, seed in (
        (base, 2, 11),
        (samecap, 2, 22),
        (long, 4, 22),
    ):
        item["budget"] = {
            "max_new_tokens": cap,
            "n_chains": 1,
            "temperature": 0.75,
            "top_p": 0.95,
            "seed_policy": {"base_seed": seed},
            "run_signature": run_signature(),
        }
        item["chains"][0].update(
            {
                "token_ids": [1, 2] if cap == 2 else [1, 2, 3],
                "token_count": 2 if cap == 2 else 3,
                "chain_index": 0,
                "request_seed": seed,
            }
        )
    samecap["budget"].update(
        {
            "mode": "reconstructed_token_prefix",
            "reconstructed_from": {"source_cap": 4},
        }
    )
    long["budget"]["run_signature"] = run_signature("other-model")
    try:
        THREE_ARM.validate_protocol(
            {"p": base}, {"p": samecap}, {"p": long}, 2, 4
        )
    except ValueError as error:
        assert "decoding mismatch" in str(error)
    else:
        raise AssertionError("cross-model three-arm input was accepted")
