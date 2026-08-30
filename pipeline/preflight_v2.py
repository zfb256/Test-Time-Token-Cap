#!/usr/bin/env python3
"""Fail-fast checks for the expensive v2 experiment batch."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
CONFIGS = [
    "config.yaml",
    "config_llama_validation.yaml",
    "config_r1_distill.yaml",
    "config_aime_qwen.yaml",
    "config_aime_r1.yaml",
    "config_r1_three_arm.yaml",
]
RUNTIME_VERSIONS = {
    "vllm": "0.6.6.post1",
    "torch": "2.5.1",
    "transformers": "4.47.1",
}
VERIFIER_VERSIONS = {
    "sympy": "1.13.1",
    "latex2sympy2": "1.9.1",
    "antlr4-python3-runtime": "4.7.2",
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def check_unique(rows, path, failures):
    ids = [row.get("problem_id") for row in rows]
    if None in ids:
        failures.append(f"{path}: missing problem_id")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        failures.append(f"{path}: {duplicates} duplicate problem_id rows")


def check_matches_dataset(rows, dataset_rows, path, failures):
    reference = {row["problem_id"]: row for row in dataset_rows}
    if {row["problem_id"] for row in rows} != set(reference):
        failures.append(f"{path}: problem_id set differs from its configured dataset")
        return
    for row in rows:
        expected = reference[row["problem_id"]]
        for key in ("dataset", "question", "answer"):
            if row.get(key) != expected.get(key):
                failures.append(
                    f"{path}: {key} mismatch for problem_id={row['problem_id']}"
                )
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip CUDA/vLLM runtime checks (for workstation validation).",
    )
    args = parser.parse_args()
    failures = []
    tokenizer_cache = {}
    inference_module = None
    try:
        spec = importlib.util.spec_from_file_location(
            "v2_run_inference", SCRIPT_DIR / "02_run_inference.py"
        )
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)
        from transformers import AutoTokenizer
    except Exception as exc:
        failures.append(f"cannot load tokenizer preflight dependencies: {exc}")
        AutoTokenizer = None

    loaded = {}
    for name in CONFIGS:
        path = SCRIPT_DIR / name
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded[name] = cfg
        except Exception as exc:
            failures.append(f"{path}: cannot load config: {exc}")
            continue

        model = (SCRIPT_DIR / cfg["inference"]["model"]).resolve()
        for required in ("config.json", "tokenizer_config.json"):
            if not (model / required).exists():
                failures.append(f"{model}: missing {required}")
        if not list(model.glob("*.safetensors")):
            failures.append(f"{model}: no safetensors weights found")

        budgets = cfg["inference"]["budgets"]
        if not cfg["inference"].get("deterministic_sampling", False):
            failures.append(
                f"{name}: inference.deterministic_sampling must be true for "
                "auditable sampled reruns"
            )
        if not cfg["inference"].get("store_token_ids", False):
            failures.append(
                f"{name}: inference.store_token_ids must be true so lower "
                "token caps can be reconstructed exactly"
            )
        max_model_len = int(cfg["inference"]["max_model_len"])
        for budget_name, budget in budgets.items():
            if int(budget["max_new_tokens"]) >= max_model_len:
                failures.append(
                    f"{name}:{budget_name}: max_new_tokens must be below "
                    f"max_model_len ({max_model_len}) to leave room for the prompt"
                )
            if int(budget["n_chains"]) < 1:
                failures.append(f"{name}:{budget_name}: n_chains must be positive")

        data_path = (SCRIPT_DIR / cfg["paths"]["data_dir"] / "all_problems.jsonl").resolve()
        try:
            rows = load_jsonl(data_path)
            check_unique(rows, data_path, failures)
            expected = 60 if "aime" in name else 800
            if len(rows) != expected:
                failures.append(f"{data_path}: expected {expected} rows, found {len(rows)}")
        except Exception as exc:
            failures.append(str(exc))
            rows = []

        if rows and AutoTokenizer is not None and inference_module is not None:
            try:
                model_key = str(model)
                if model_key not in tokenizer_cache:
                    tokenizer_cache[model_key] = AutoTokenizer.from_pretrained(
                        model, local_files_only=True
                    )
                tokenizer = tokenizer_cache[model_key]
                template = cfg["inference"].get("prompt_template", "qwen_math")
                max_prompt_tokens = max(
                    len(
                        tokenizer.encode(
                            inference_module.build_prompt(row["question"], template),
                            add_special_tokens=False,
                        )
                    )
                    for row in rows
                )
                largest_budget = max(
                    int(budget["max_new_tokens"]) for budget in budgets.values()
                )
                if max_prompt_tokens + largest_budget > max_model_len:
                    failures.append(
                        f"{name}: longest prompt ({max_prompt_tokens}) + largest "
                        f"budget ({largest_budget}) exceeds max_model_len "
                        f"({max_model_len})"
                    )
            except Exception as exc:
                failures.append(f"{name}: prompt-length validation failed: {exc}")

    required_inputs = [
        (
            REPO / "outputs/qwen_main/inference_medium.jsonl",
            REPO / "data/all_problems.jsonl",
        ),
        (
            REPO / "outputs/qwen_main/inference_long.jsonl",
            REPO / "data/all_problems.jsonl",
        ),
        (
            REPO / "outputs/llama_validation/inference_medium.jsonl",
            REPO / "data/llama_validation/all_problems.jsonl",
        ),
    ]
    for input_path, dataset_path in required_inputs:
        try:
            rows = load_jsonl(input_path)
            dataset_rows = load_jsonl(dataset_path)
            check_unique(rows, input_path, failures)
            if len(rows) != 800:
                failures.append(
                    f"{input_path}: expected 800 rows, found {len(rows)}"
                )
            check_matches_dataset(rows, dataset_rows, input_path, failures)
        except Exception as exc:
            failures.append(str(exc))

    free_gib = shutil.disk_usage(REPO).free / 2**30
    if free_gib < 15:
        failures.append(f"only {free_gib:.1f} GiB free; require at least 15 GiB")

    for package, expected in VERIFIER_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
            if actual != expected:
                failures.append(
                    f"{package}=={actual}; expected validated version {expected}"
                )
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{package} is not installed")
    try:
        from utils.math_verify import MathVerifier

        verifier = MathVerifier()
        latex_fraction = verifier._canonical_math_answer(r"\frac{1}{2}")
        plain_fraction = verifier._canonical_math_answer("1/2")
        if latex_fraction != plain_fraction:
            failures.append(
                "symbolic verifier functional probe failed: "
                rf"\frac{{1}}{{2}} -> {latex_fraction!r}, "
                f"1/2 -> {plain_fraction!r}"
            )
    except Exception as exc:
        failures.append(f"symbolic verifier functional probe failed: {exc}")

    if not args.static_only:
        if not Path("/dev/nvidiactl").exists() and shutil.which("nvidia-modprobe"):
            try:
                subprocess.run(
                    ["nvidia-modprobe", "-u", "-c=0"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                failures.append(
                    "failed to create NVIDIA device nodes with nvidia-modprobe: "
                    f"{exc.stderr.strip()}"
                )
        for package, expected in RUNTIME_VERSIONS.items():
            try:
                actual = importlib.metadata.version(package)
                if actual != expected:
                    failures.append(
                        f"{package}=={actual}; expected validated version {expected}"
                    )
            except importlib.metadata.PackageNotFoundError:
                failures.append(f"{package} is not installed")
        try:
            import torch

            if not torch.cuda.is_available():
                failures.append("torch.cuda.is_available() is false")
        except Exception as exc:
            failures.append(f"cannot validate CUDA through torch: {exc}")

    if failures:
        print("v2 preflight FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"v2 preflight OK ({'static' if args.static_only else 'GPU runtime'} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
