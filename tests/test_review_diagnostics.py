import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_diagnostics", ROOT / "analysis" / "57_review_diagnostics.py"
)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


def value(answer, correct, complete=True):
    return {"answer": answer, "correct": correct, "complete": complete}


def case(base, extended):
    return {
        "problem_id": "p",
        "dataset": "gsm8k",
        "base": base,
        "extended": extended,
    }


def test_symmetric_harm_reports_strict_and_plurality_definitions():
    base = [value("2", True)] * 8
    extended = [value("1", False)] * 8
    summary = DIAGNOSTICS.summarize_harm(
        DIAGNOSTICS.symmetric_harm([case(base, extended)])
    )
    assert set(summary["definition"]) == {
        "complete_plurality_correct",
        "complete_all_correct",
    }
    assert (summary["n_eligible"] == 1).all()
    assert (summary["harm_count"] == 1).all()


def test_split_null_uses_disjoint_four_chain_groups():
    base = [value("1", False)] * 4 + [value("2", True)] * 4
    result = DIAGNOSTICS.split_null([case(base, base)])
    assert len(result) == 70
    assert result["n_eligible"].sum() == 1
    assert result["repair_count"].sum() == 1


def test_factorial_effects_include_interaction():
    wrong = value("1", False)
    correct = value("2", True)
    cells = DIAGNOSTICS.factorial(
        [case([wrong] * 8, [correct] * 8)]
    )
    effects = DIAGNOSTICS.factorial_effects(cells).iloc[0]
    assert effects["token_effect_k1"] == 1
    assert effects["token_effect_kmax"] == 1
    assert effects["chain_effect_base"] == 0
    assert effects["chain_effect_extended"] == 0
    assert effects["interaction"] == 0
