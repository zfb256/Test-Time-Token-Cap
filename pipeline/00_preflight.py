"""
Preflight checks for CUPID-G Phase 1.

This script verifies local data/model readiness before the expensive vLLM run.
By default it checks CPU-side prerequisites and reports GPU status without
failing. Use --require-gpu on the machine that will run Step 02.
"""

import argparse
import importlib.util
import importlib.metadata as importlib_metadata
import json
import subprocess
import sys
from pathlib import Path

import yaml


MODEL_SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]

EXPECTED_VERSIONS = {
    "datasets": "2.18.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scikit-learn": "1.4.2",
    "lightgbm": "4.3.0",
    "sympy": "1.12",
    "latex2sympy2": "1.9.1",
    "antlr4-python3-runtime": "4.7.2",
    "jsonlines": "4.0.0",
    "scipy": "1.12.0",
    "transformers": "4.43.1",
    "vllm": "0.4.2",
    "torch": "2.3.0",
    "xformers": "0.0.26.post1",
}


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str, failures: list[str]) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def count_jsonl(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_jsonl_sample(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if len(rows) >= 3:
                    break
    return rows


def check_import(name: str, failures: list[str], required: bool = True) -> bool:
    found = importlib.util.find_spec(name) is not None
    if found:
        ok(f"import {name}")
    elif required:
        fail(f"missing Python package: {name}", failures)
    else:
        warn(f"optional package not importable: {name}")
    return found


def check_version(package: str, failures: list[str], required: bool = True) -> bool:
    expected = EXPECTED_VERSIONS[package]
    try:
        actual = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        if required:
            fail(f"missing Python package: {package}", failures)
        else:
            warn(f"optional package not installed: {package}")
        return False
    if actual != expected:
        if required:
            fail(f"{package} version {actual}, expected {expected}", failures)
        else:
            warn(f"{package} version {actual}, expected {expected}")
        return False
    else:
        ok(f"{package}=={expected}")
        return True


def check_model_dir(name: str, path: Path, failures: list[str]) -> None:
    if not path.exists():
        fail(f"{name} model path missing: {path}", failures)
        return
    required = ["config.json", "tokenizer.json", "model.safetensors.index.json", *MODEL_SHARDS]
    missing = [item for item in required if not (path / item).exists()]
    if missing:
        fail(f"{name} model missing files: {missing}", failures)
    else:
        ok(f"{name} model files present: {path}")

    leftovers = list(path.glob("*.aria2")) + list(path.glob("*.lock"))
    if leftovers:
        fail(f"{name} model has unfinished download markers: {[p.name for p in leftovers]}", failures)


def check_data(data_dir: Path, cfg: dict, failures: list[str]) -> None:
    expected_gsm = cfg["dataset"]["gsm8k"]["n_samples"]
    expected_math = len(cfg["dataset"]["math"]["levels"]) * cfg["dataset"]["math"]["n_per_level"]
    expected_all = expected_gsm + expected_math

    checks = [
        ("gsm8k", data_dir / "gsm8k_samples.jsonl", expected_gsm),
        ("math", data_dir / "math_samples.jsonl", expected_math),
        ("all", data_dir / "all_problems.jsonl", expected_all),
    ]
    for label, path, expected in checks:
        if not path.exists():
            fail(f"missing data file: {path}", failures)
            continue
        n = count_jsonl(path)
        if n != expected:
            fail(f"{label} data row count {n}, expected {expected}", failures)
        else:
            ok(f"{label} data rows: {n}")

    combined = data_dir / "all_problems.jsonl"
    if combined.exists():
        ids = set()
        empty = 0
        total = 0
        with open(combined, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                if not row.get("problem_id") or not row.get("question") or not row.get("answer"):
                    empty += 1
                ids.add(row.get("problem_id"))
        if len(ids) != total:
            fail(f"combined data has duplicate problem_id values: {total - len(ids)} duplicates", failures)
        if empty:
            fail(f"combined data has {empty} rows with empty required fields", failures)
        else:
            ok("combined data has unique ids and required fields")


def check_inference_outputs(out_dir: Path, cfg: dict, failures: list[str]) -> None:
    expected_rows = count_jsonl(Path(cfg["paths"]["data_dir"]) / "all_problems.jsonl")
    for budget_name, budget_cfg in cfg["inference"]["budgets"].items():
        path = out_dir / f"inference_{budget_name}.jsonl"
        if not path.exists():
            warn(f"missing inference output for budget '{budget_name}': {path}")
            continue
        n = count_jsonl(path)
        if n != expected_rows:
            fail(f"{path} has {n} rows, expected {expected_rows}", failures)
            continue
        bad = 0
        for row in load_jsonl_sample(path):
            if len(row.get("chains", [])) < budget_cfg["n_chains"]:
                bad += 1
        if bad:
            fail(f"{path} sample has too few chains", failures)
        else:
            ok(f"{budget_name} inference output present: {n} rows")


def check_gpu(require_gpu: bool, failures: list[str], package_ready: dict[str, bool]) -> None:
    torch_ok = check_import("torch", failures, required=False) if package_ready.get("torch") else False
    vllm_ok = check_import("vllm", failures, required=False) if package_ready.get("vllm") else False
    if require_gpu and package_ready.get("torch") and not torch_ok:
        fail("torch package is installed but not importable", failures)
    if require_gpu and package_ready.get("vllm") and not vllm_ok:
        fail("vllm package is installed but not importable", failures)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        result = None

    if result and result.returncode == 0:
        ok("nvidia-smi")
    elif require_gpu:
        fail("nvidia-smi failed or NVIDIA driver is unavailable", failures)
    else:
        warn("nvidia-smi failed or NVIDIA driver is unavailable")

    if torch_ok:
        try:
            import torch

            if torch.cuda.is_available():
                ok(f"torch CUDA devices: {torch.cuda.device_count()}")
            elif require_gpu:
                fail("torch.cuda.is_available() is False", failures)
            else:
                warn("torch.cuda.is_available() is False")
        except Exception as exc:
            if require_gpu:
                fail(f"torch CUDA check failed: {exc}", failures)
            else:
                warn(f"torch CUDA check failed: {exc}")
    elif vllm_ok:
        warn("vLLM import works but torch import did not; check GPU environment")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--check-inference", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    failures: list[str] = []
    base = Path(cfg["paths"].get("base_dir", "."))
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])

    print("== Config ==")
    ok(f"config loaded: {args.config}")
    ok(f"base_dir: {base}")

    print("\n== Python Packages ==")
    for name in [
        "yaml",
        "datasets",
        "pandas",
        "sklearn",
        "lightgbm",
        "sympy",
        "latex2sympy2",
        "jsonlines",
        "scipy",
        "transformers",
    ]:
        check_import(name, failures)
    for package in [
        "datasets",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "sympy",
        "latex2sympy2",
        "antlr4-python3-runtime",
        "jsonlines",
        "scipy",
        "transformers",
    ]:
        check_version(package, failures)

    print("\n== Data ==")
    check_data(data_dir, cfg, failures)

    print("\n== Models ==")
    check_model_dir("inference", Path(cfg["inference"]["model"]), failures)
    check_model_dir("prm", Path(cfg["prm"]["model"]), failures)

    print("\n== GPU ==")
    gpu_package_ready = {}
    if args.require_gpu:
        for package in ["torch", "vllm", "xformers"]:
            gpu_package_ready[package] = check_version(package, failures, required=True)
    else:
        for package in ["torch", "vllm", "xformers"]:
            gpu_package_ready[package] = check_version(package, failures, required=False)
    check_gpu(args.require_gpu, failures, gpu_package_ready)

    if args.check_inference:
        print("\n== Inference Outputs ==")
        check_inference_outputs(out_dir, cfg, failures)

    print("\n== Result ==")
    if failures:
        print(f"Preflight failed with {len(failures)} issue(s).")
        return 2
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
