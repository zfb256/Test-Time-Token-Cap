import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load_script("48_expanded_repair_eval.py")
stopping = load_script("50_seed_stopping_plan.py")
multiseed = load_script("51_multiseed_cluster_summary.py")
core_aggregate = load_script("52_aggregate_three_seed_core.py")
verifier_audit = load_script("24_relaxed_verifier_sensitivity.py")
routing = load_script("61_three_arm_routing.py")


def test_repair_definitions_separate_strict_from_plurality_wrong():
    values = [
        {"complete": True, "answer": "wrong", "correct": False},
        {"complete": True, "answer": "wrong", "correct": False},
        {"complete": True, "answer": "right", "correct": True},
    ]
    result = repair.eligibility(values)
    assert not result["strict_all_wrong"]
    assert result["plurality_wrong"]
    assert result["first_chain_wrong"]


def test_relaxed_any_repair_requires_a_new_correct_chain():
    base = {
        "p": {
            "problem_id": "p",
            "dataset": "gsm8k",
            "question": "q",
            "answer": "#### 2",
            "chains": [
                {"text": "#### 1", "finish_reason": "stop"},
                {"text": "#### 1", "finish_reason": "stop"},
                {"text": "#### 2", "finish_reason": "stop"},
            ],
        }
    }
    extended = {
        "p": {
            **base["p"],
            "chains": [
                {"text": "#### 1", "finish_reason": "stop"},
                {"text": "#### 1", "finish_reason": "stop"},
                {"text": "#### 2", "finish_reason": "stop"},
            ],
        }
    }
    detail = repair.analyze(base, extended, 10, 20)
    row = detail[detail["definition"] == "plurality_wrong"].iloc[0]
    assert row["eligible"]
    assert row["ext_any_correct"]
    assert not row["repair_any"]


def test_plurality_tie_bounds():
    values = [
        {"answer": "a", "correct": False},
        {"answer": "b", "correct": True},
    ]
    vote = repair.plurality(values)
    assert vote["tie"]
    assert not vote["lower"]
    assert vote["upper"]


def test_zero_event_bound_tightens_with_more_cases():
    assert stopping.zero_event_upper(30) < stopping.zero_event_upper(10)
    assert stopping.zero_event_upper(0) == 1.0


def test_routing_fraction_rounds_up_consistently():
    mask = routing.top_fraction_mask(
        scores=routing.np.array([0.1, 0.2, 0.3]),
        fraction=0.5,
        tie_breaker=routing.np.array([0, 1, 2]),
    )
    assert mask.sum() == 2


def test_relaxed_verifier_does_not_compare_last_fraction_denominator():
    verifier = verifier_audit.MathVerifier()
    assert not verifier_audit.relaxed_verify_math(
        r"\boxed{\frac{540}{7}}",
        r"\boxed{\frac{360}{7}}",
        verifier,
    )


def test_relaxed_verifier_can_reject_unboxed_capped_output():
    verifier = verifier_audit.MathVerifier()
    assert not verifier_audit.relaxed_verify_math(
        "unfinished derivation whose last number happens to be 42",
        r"\boxed{42}",
        verifier,
        allow_unboxed=False,
    )


def test_multiseed_rejects_problem_mismatch():
    first = pd.DataFrame({"problem_id": ["a", "b"], "score": [0, 1]})
    second = pd.DataFrame({"problem_id": ["a", "c"], "score": [1, 1]})
    with pytest.raises(ValueError, match="different problem_id"):
        multiseed.aggregate([first, second], ["1", "2"], "score", n_boot=10)


def test_multiseed_clusters_repeated_problems():
    first = pd.DataFrame({"problem_id": ["a", "b"], "score": [0, 1]})
    second = pd.DataFrame({"problem_id": ["a", "b"], "score": [1, 1]})
    result = multiseed.aggregate(
        [first, second], ["1", "2"], "score", seed=7, n_boot=100
    )
    assert result.iloc[0]["estimate"] == pytest.approx(0.75)
    assert set(result["scope"]) == {"all_seeds", "leave_out:1", "leave_out:2"}


def test_multiseed_conditional_rate_uses_pooled_denominator():
    first = pd.DataFrame(
        {"problem_id": ["a", "b"], "repaired": [1, 0], "eligible": [1, 0]}
    )
    second = pd.DataFrame(
        {"problem_id": ["a", "b"], "repaired": [0, 1], "eligible": [1, 1]}
    )
    result = multiseed.aggregate(
        [first, second],
        ["1", "2"],
        "repaired",
        denominator="eligible",
        seed=7,
        n_boot=100,
    )
    assert result.iloc[0]["numerator_count"] == 2
    assert result.iloc[0]["denominator_count"] == 3
    assert result.iloc[0]["estimate"] == pytest.approx(2 / 3)


def test_zero_event_wilson_interval_is_not_degenerate():
    lower, upper = multiseed.wilson_interval(0, 55)
    assert lower == 0
    assert 0.05 < upper < 0.07


def test_multiseed_supports_signed_incremental_effect():
    first = pd.DataFrame(
        {"problem_id": ["a", "b"], "net": [1, -1], "eligible": [1, 1]}
    )
    second = pd.DataFrame(
        {"problem_id": ["a", "b"], "net": [1, 0], "eligible": [1, 1]}
    )
    result = multiseed.aggregate(
        [first, second],
        ["1", "2"],
        "net",
        denominator="eligible",
        n_boot=100,
    )
    assert result.iloc[0]["estimate"] == pytest.approx(0.25)
    assert pd.isna(result.iloc[0]["binomial_wilson95_upper"])


def test_all_zero_signed_metric_does_not_get_binomial_interval():
    frame = pd.DataFrame(
        {"problem_id": ["a", "b"], "net": [0, 0], "eligible": [1, 1]}
    )
    result = multiseed.aggregate(
        [frame],
        ["1"],
        "net",
        denominator="eligible",
        n_boot=20,
        binary_metric=False,
    )
    assert result.iloc[0]["estimate"] == 0
    assert pd.isna(result.iloc[0]["binomial_wilson95_upper"])


def test_multiseed_parses_false_strings_and_rejects_missing_values():
    frame = pd.DataFrame(
        {
            "problem_id": ["a", "b"],
            "score": ["False", "True"],
        }
    )
    result = multiseed.aggregate(
        [frame], ["1"], "score", n_boot=20, binary_metric=True
    )
    assert result.iloc[0]["estimate"] == pytest.approx(0.5)
    bad = pd.DataFrame({"problem_id": ["a"], "score": [float("nan")]})
    with pytest.raises(ValueError, match="missing or non-finite"):
        multiseed.aggregate([bad], ["1"], "score", n_boot=20)


def test_core_aggregate_rejects_silent_dataset_drift():
    first = pd.DataFrame(
        {"problem_id": ["a"], "dataset": ["gsm8k"], "score": [1]}
    )
    second = pd.DataFrame(
        {"problem_id": ["a"], "dataset": ["math"], "score": [1]}
    )
    with pytest.raises(ValueError, match="different problem"):
        core_aggregate.validate_seed_frames(
            [first, second], ["1", "2"], ["problem_id"]
        )


def test_core_aggregate_binds_directory_label_to_model():
    qwen = {
        "model": "../models/Qwen2.5-Math-7B-Instruct",
        "model_snapshot_sha256": "qwen-hash",
    }
    assert core_aggregate.validate_model_signatures("qwen", [qwen, qwen]) == qwen
    llama = {
        "model": "../models/Llama-3.1-8B-Instruct",
        "model_snapshot_sha256": "llama-hash",
    }
    with pytest.raises(ValueError, match="expected"):
        core_aggregate.validate_model_signatures("qwen", [llama, llama])
