import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "28_reproducibility_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("reproducibility_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReproducibilityManifestTest(unittest.TestCase):
    def test_repo_files_use_relative_manifest_paths(self):
        record = MODULE.file_record(SCRIPT)
        self.assertEqual(
            record["path"], "analysis/28_reproducibility_manifest.py"
        )

    def test_artifact_discovery_uses_requested_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "requested"
            other = root / "other"
            (requested / "model").mkdir(parents=True)
            (other / "model").mkdir(parents=True)
            record = {
                "chains": [
                    {"request_seed": 7},
                    {"request_seed": 7},
                    {"request_seed": 9},
                ],
                "budget": {
                    "seed_policy": {
                        "mode": "per_problem_per_chain_sha256",
                    }
                }
            }
            (requested / "model" / "inference_test.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            (other / "model" / "inference_other.jsonl").write_text(
                json.dumps({"budget": {"seed_policy": {"mode": "wrong"}}}) + "\n",
                encoding="utf-8",
            )

            artifacts = MODULE.canonical_artifacts(requested)
            policies = MODULE.seed_policies(requested)
            signatures = MODULE.run_signatures(requested)
            budgets = MODULE.observed_budgets(requested)
            collisions = MODULE.request_seed_collisions(requested)

        self.assertEqual(
            [path.name for path in artifacts],
            ["inference_test.jsonl"],
        )
        self.assertEqual(len(policies), 1)
        self.assertEqual(
            next(iter(policies.values()))["mode"],
            "per_problem_per_chain_sha256",
        )
        self.assertEqual(
            next(iter(signatures.values())), "legacy_or_unrecorded"
        )
        self.assertEqual(len(budgets), 1)
        self.assertEqual(
            next(iter(budgets.values()))["seed_policy"]["mode"],
            "per_problem_per_chain_sha256",
        )
        summary = next(iter(collisions.values()))
        self.assertEqual(summary["n_duplicate_assignments"], 1)
        self.assertEqual(summary["max_seed_multiplicity"], 2)


if __name__ == "__main__":
    unittest.main()
