"""Build a reproducibility manifest for the canonical clean-rerun artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Dict

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CANONICAL_ROOT = (SCRIPT_DIR / "../outputs").resolve()

KEY_FILES = [
    "../pipeline/config.yaml",
    "../pipeline/config_llama_validation.yaml",
    "../pipeline/config_r1_distill.yaml",
    "../pipeline/config_aime_qwen.yaml",
    "../pipeline/config_aime_r1.yaml",
    "../data/all_problems.jsonl",
    "../data/expanded/all_problems.jsonl",
    "../data/llama_validation/all_problems.jsonl",
    "../data/aime/all_problems.jsonl",
    "../outputs/temperature_sensitivity/temperature_sensitivity_summary.csv",
    "../README.md",
    "../pipeline/preflight_v2.py",
    "../pipeline/02_run_inference.py",
    "../pipeline/08_run_continuation.py",
    "../pipeline/requirements.txt",
    "../pipeline/utils/math_verify.py",
    "../pipeline/run_three_seed_core.sh",
    "../pipeline/run_three_arm_base.sh",
    "28_reproducibility_manifest.py",
    "32_second_model_sanity_eval.py",
    "35_complete_but_wrong_analysis.py",
    "36_make_paper_figures.py",
    "39_sampling_repair_eval.py",
    "42_relabel_cached_scores.py",
    "44_same_budget_control.py",
    "45_validate_rerun.py",
    "47_reconstruct_lower_cap.py",
    "48_expanded_repair_eval.py",
    "49_chain_count_sensitivity.py",
    "50_seed_stopping_plan.py",
    "51_multiseed_cluster_summary.py",
    "52_aggregate_three_seed_core.py",
    "53_three_arm_repair_decomposition.py",
    "56_aggregate_three_arm.py",
    "63_verifier_audit.py",
]

PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "torch",
    "transformers",
    "vllm",
    "datasets",
    "PyYAML",
    "sympy",
    "latex2sympy2",
    "antlr4-python3-runtime",
    "pytest",
]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> Dict:
    if not path.exists():
        return {"path": manifest_path(path), "exists": False}
    lines = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
    except UnicodeDecodeError:
        pass
    return {
        "path": manifest_path(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": lines,
        "sha256": sha256_file(path),
    }


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def seed_policies(root: Path) -> Dict[str, object]:
    policies = {}
    for path in sorted(root.glob("*/inference_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            record = json.loads(next(handle))
        policies[manifest_path(path)] = record.get("budget", {}).get(
            "seed_policy", "legacy_or_unrecorded"
        )
    return policies


def run_signatures(root: Path) -> Dict[str, object]:
    signatures = {}
    for path in sorted(root.glob("*/inference_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            record = json.loads(next(handle))
        signatures[manifest_path(path)] = record.get("budget", {}).get(
            "run_signature", "legacy_or_unrecorded"
        )
    return signatures


def observed_budgets(root: Path) -> Dict[str, object]:
    budgets = {}
    for path in sorted(root.glob("*/inference_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            budgets[manifest_path(path)] = json.loads(next(handle)).get(
                "budget", {}
            )
    return budgets


def source_files():
    """Discover executable experiment source so the manifest cannot omit it."""
    repo = SCRIPT_DIR.parent
    patterns = (
        "analysis/*.py",
        "pipeline/*.py",
        "pipeline/*.sh",
        "pipeline/*.yaml",
        "pipeline/utils/*.py",
        "tests/test_*.py",
    )
    files = []
    for pattern in patterns:
        files.extend(repo.glob(pattern))
    requirements = repo / "pipeline/requirements.txt"
    if requirements.exists():
        files.append(requirements)
    return sorted(set(path.resolve() for path in files))


def request_seed_collisions(root: Path) -> Dict[str, object]:
    """Record global 64-bit hash collisions without exposing prompts/text."""
    summaries = {}
    for path in sorted(root.glob("*/inference_*.jsonl")):
        seeds = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                seeds.extend(
                    chain.get("request_seed")
                    for chain in record.get("chains", [])
                    if chain.get("request_seed") is not None
                )
        counts = {}
        for seed in seeds:
            counts[seed] = counts.get(seed, 0) + 1
        summaries[manifest_path(path)] = {
            "n_seeded_requests": len(seeds),
            "n_unique_request_seeds": len(counts),
            "n_duplicate_assignments": len(seeds) - len(counts),
            "max_seed_multiplicity": max(counts.values(), default=0),
        }
    return summaries


def canonical_artifacts(root: Path):
    excluded = {"reproducibility_manifest.json", "reproducibility_manifest.txt"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="../pipeline/config.yaml")
    parser.add_argument("--results-dir", default=str(CANONICAL_ROOT))
    args = parser.parse_args()

    cfg_path = resolve_path(args.pipeline_config)
    out_dir = resolve_path(args.results_dir)
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    paths = [resolve_path(path) for path in KEY_FILES]
    paths.extend(source_files())
    paths.extend(canonical_artifacts(out_dir))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    policies = seed_policies(out_dir)
    signatures = run_signatures(out_dir)
    collision_summary = request_seed_collisions(out_dir)
    policy_modes = sorted(
        {
            policy.get("mode", "legacy_or_unrecorded")
            if isinstance(policy, dict)
            else str(policy)
            for policy in policies.values()
        }
    )
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "notes": [
            "This manifest records the current artifact tree.",
            "Historical pre-rerun outputs are available from Git commit e89bec5.",
            "Seed-policy modes observed in the artifacts: "
            + ", ".join(policy_modes),
            "New deterministic sampled runs derive a stable request-specific "
            "seed for each problem and chain; legacy engine-only runs remain "
            "identified as such in seed_policies.",
            "Request seeds are 64-bit SHA-256 projections, so rare global "
            "collisions are possible and are counted in request_seed_collisions.",
            "Legacy artifacts without a run_signature remain explicit in "
            "run_signatures and are not silently accepted by strict resume.",
            "Every corrected sampled output passed the hard diversity guard.",
        ],
        "seeds": {
            "dataset_gsm8k": cfg["dataset"]["gsm8k"].get("seed"),
            "dataset_math": cfg["dataset"]["math"].get("seed"),
            "inference": cfg["inference"].get("seed"),
            "classifier_random_state": cfg["classifiers"].get("random_state"),
            "random_forest_random_state": cfg["classifiers"].get(
                "random_forest", {}
            ).get("random_state", cfg["classifiers"].get("random_state")),
        },
        "budgets": cfg["inference"]["budgets"],
        "observed_budgets": observed_budgets(out_dir),
        "seed_policies": policies,
        "run_signatures": signatures,
        "request_seed_collisions": collision_summary,
        "packages": package_versions(),
        "files": [file_record(path) for path in paths],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reproducibility_manifest.json"
    txt_path = out_dir / "reproducibility_manifest.txt"
    json_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    missing = [row["path"] for row in manifest["files"] if not row["exists"]]
    txt_path.write_text(
        "\n".join(
            [
                "Reproducibility Manifest",
                "==============================================",
                "",
                f"Artifacts recorded: {len(manifest['files'])}",
                f"Missing files: {len(missing)}",
                *(f"  {path}" for path in missing),
                "",
            ]
        ),
        encoding="utf-8",
    )
    if missing:
        raise SystemExit(f"Manifest has {len(missing)} missing files")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
