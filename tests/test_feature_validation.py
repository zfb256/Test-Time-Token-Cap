import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from utils.feature_utils import FeatureBuilder


def feature_record(problem_id="p", value=1.0, label=0):
    return {
        "problem_id": problem_id,
        "label": label,
        "features": {"G1": {"x": value}},
    }


class FeatureValidationTest(unittest.TestCase):
    def test_rejects_duplicate_problem_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate problem_id"):
            FeatureBuilder.records_to_dataframe(
                [feature_record(), feature_record()], "G1"
            )

    def test_rejects_inconsistent_or_nonfinite_features(self):
        inconsistent = feature_record("b")
        inconsistent["features"]["G1"] = {"y": 2.0}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            FeatureBuilder.records_to_dataframe(
                [feature_record("a"), inconsistent], "G1"
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FeatureBuilder.records_to_dataframe(
                [feature_record(value=float("nan"))], "G1"
            )

    def test_rejects_nonbinary_labels(self):
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            FeatureBuilder.records_to_dataframe(
                [feature_record(label=True)], "G1"
            )

if __name__ == "__main__":
    unittest.main()
