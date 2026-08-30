import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "three_arm_aggregate", ROOT / "analysis" / "56_aggregate_three_arm.py"
)
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


def test_three_arm_aggregate_reports_signed_net_effect(tmp_path):
    for seed, values in [("1", (1, 0)), ("2", (0, 1))]:
        directory = tmp_path / f"qwen_seed{seed}"
        directory.mkdir()
        helpful, harmful = values
        pd.DataFrame(
            {
                "problem_id": ["p"],
                "dataset": ["gsm8k"],
                "definition": ["strict_all_wrong"],
                "eligible": [1],
                "samecap_repair_plurality": [0],
                "long_repair_plurality": [helpful],
                "samecap_repair_any": [0],
                "long_repair_any": [helpful],
                "incremental_plurality_helpful": [helpful],
                "incremental_plurality_harmful": [harmful],
            }
        ).to_csv(directory / "three_arm_detail.csv", index=False)
        pd.DataFrame(
            {
                "problem_id": ["p"],
                "dataset": ["gsm8k"],
                "contrast": ["A_to_B_resampling"],
                "definition": ["complete_plurality_correct"],
                "eligible": [1],
                "harm": [harmful],
                "certain_harm": [harmful],
                "possible_harm": [harmful],
            }
        ).to_csv(directory / "three_arm_harm_detail.csv", index=False)
    result = AGGREGATE.aggregate_root(
        tmp_path, ["qwen"], ["1", "2"], n_boot=100
    )
    row = result[
        (result["dataset"] == "all")
        & (result["scope"] == "all_seeds")
        & (result["outcome"] == "incremental_net")
    ].iloc[0]
    assert row["estimate"] == 0
    assert pd.isna(row["binomial_wilson95_upper"])
    harm = result[
        (result["dataset"] == "all")
        & (result["scope"] == "all_seeds")
        & (result["outcome"] == "harm")
    ].iloc[0]
    assert harm["estimate"] == 0.5
