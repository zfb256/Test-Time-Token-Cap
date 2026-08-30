#!/usr/bin/env python3
"""Structural validation for a clean inference rerun."""

import argparse
import json
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from reproducibility import REQUIRED_RUN_SIGNATURE, deterministic_chain_seed


def validate_file(
    path: Path,
    expected_rows: int,
    expected_records: dict | None = None,
) -> tuple[int, int]:
    seen: set[str] = set()
    chain_total = 0
    expected_budget = None

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            problem_id = row["problem_id"]
            if problem_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate problem_id {problem_id}")
            seen.add(problem_id)
            metadata = tuple(
                row.get(field, 0 if field == "math_level" else None)
                for field in ("dataset", "question", "answer", "math_level")
            )
            if expected_records is not None:
                if problem_id in expected_records:
                    if expected_records[problem_id] != metadata:
                        raise ValueError(
                            f"{path}:{line_no}: problem metadata differs "
                            f"across files for {problem_id}"
                        )
                else:
                    expected_records[problem_id] = metadata

            budget = row["budget"]
            seed_policy = budget.get("seed_policy", {})
            run_signature = budget.get("run_signature")
            if (
                not isinstance(run_signature, dict)
                or not REQUIRED_RUN_SIGNATURE.issubset(run_signature)
            ):
                raise ValueError(
                    f"{path}:{line_no}: missing required run signature fields"
                )
            signature = (
                budget["max_new_tokens"],
                budget["n_chains"],
                budget["temperature"],
                budget["top_p"],
                seed_policy.get("engine_seed"),
                seed_policy.get("base_seed"),
                seed_policy.get("mode"),
                seed_policy.get("request_seed"),
                budget.get("mode"),
                budget.get("deterministic_sampling"),
                json.dumps(
                    run_signature,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
            if expected_budget is None:
                expected_budget = signature
            elif signature != expected_budget:
                raise ValueError(
                    f"{path}:{line_no}: inconsistent budget {signature} != {expected_budget}"
                )

            chains = row["chains"]
            if len(chains) != budget["n_chains"]:
                raise ValueError(
                    f"{path}:{line_no}: {len(chains)} chains != {budget['n_chains']}"
                )
            chain_total += len(chains)
            chain_indexes = [chain.get("chain_index") for chain in chains]
            if chain_indexes != list(range(len(chains))):
                raise ValueError(
                    f"{path}:{line_no}: invalid chain indexes "
                    f"{chain_indexes}"
                )
            if seed_policy.get("mode") == "per_problem_per_chain_sha256":
                request_seeds = [chain.get("request_seed") for chain in chains]
                if None in request_seeds or len(set(request_seeds)) != len(chains):
                    raise ValueError(
                        f"{path}:{line_no}: deterministic chain seeds are "
                        "missing or not unique"
                    )
                base_seed = seed_policy.get("base_seed")
                if base_seed is None:
                    raise ValueError(
                        f"{path}:{line_no}: deterministic base seed is missing"
                    )
                expected_seeds = [
                    deterministic_chain_seed(base_seed, problem_id, index)
                    for index in range(len(chains))
                ]
                if request_seeds != expected_seeds:
                    raise ValueError(
                        f"{path}:{line_no}: request seeds do not match the "
                        "declared SHA256 policy"
                    )
            elif seed_policy.get("mode") == "per_request_single_chain":
                if any(
                    chain.get("request_seed")
                    != seed_policy.get("request_seed")
                    for chain in chains
                ):
                    raise ValueError(
                        f"{path}:{line_no}: request seed does not match "
                        "the declared single-chain policy"
                    )
            elif seed_policy.get("mode") == "engine_only_multi_sample":
                if any(
                    chain.get("request_seed") is not None for chain in chains
                ):
                    raise ValueError(
                        f"{path}:{line_no}: engine-only policy unexpectedly "
                        "contains per-chain seeds"
                    )
            else:
                raise ValueError(
                    f"{path}:{line_no}: unknown seed-policy mode "
                    f"{seed_policy.get('mode')!r}"
                )
            for chain_no, chain in enumerate(chains):
                for field in ("text", "token_count", "finish_reason", "stop_reason"):
                    if field not in chain:
                        raise ValueError(
                            f"{path}:{line_no}: chain {chain_no} missing {field}"
                        )
                if chain["finish_reason"] not in {"stop", "length"}:
                    raise ValueError(
                        f"{path}:{line_no}: unexpected finish_reason "
                        f"{chain['finish_reason']!r}"
                    )
                if not isinstance(chain["text"], str):
                    raise ValueError(
                        f"{path}:{line_no}: chain {chain_no} text is not a string"
                    )
                if (
                    not isinstance(chain["token_count"], int)
                    or isinstance(chain["token_count"], bool)
                    or chain["token_count"] < 0
                ):
                    raise ValueError(
                        f"{path}:{line_no}: invalid token_count "
                        f"{chain['token_count']!r}"
                    )
                if chain["token_count"] > budget["max_new_tokens"]:
                    raise ValueError(
                        f"{path}:{line_no}: token_count {chain['token_count']} exceeds cap "
                        f"{budget['max_new_tokens']}"
                    )
                if "token_ids" in chain:
                    token_ids = chain["token_ids"]
                    if (
                        not isinstance(token_ids, list)
                        or len(token_ids) != chain["token_count"]
                        or any(
                            not isinstance(token_id, int)
                            or isinstance(token_id, bool)
                            for token_id in token_ids
                        )
                    ):
                        raise ValueError(
                            f"{path}:{line_no}: chain {chain_no} has "
                            "inconsistent token_ids"
                        )
                elif budget.get("run_signature", {}).get("store_token_ids"):
                    raise ValueError(
                        f"{path}:{line_no}: chain {chain_no} lacks token_ids "
                        "despite its run signature"
                    )

    if len(seen) != expected_rows:
        raise ValueError(f"{path}: {len(seen)} rows != expected {expected_rows}")
    return len(seen), chain_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("../outputs")
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Override the inferred row count (useful for smoke-test roots).",
    )
    args = parser.parse_args()

    files = sorted(args.root.glob("*/inference_*.jsonl"))
    if not files:
        raise SystemExit(f"No inference JSONL files found below {args.root}")

    rows_total = 0
    chains_total = 0
    parent_records = {}
    for path in files:
        expected_rows = (
            args.expected_rows
            if args.expected_rows is not None
            else (60 if path.parent.name.startswith("aime_") else 800)
        )
        expected_records = parent_records.setdefault(path.parent, {})
        before = set(expected_records)
        rows, chains = validate_file(path, expected_rows, expected_records)
        if before and set(expected_records) != before:
            extra = sorted(set(expected_records) - before)
            raise ValueError(
                f"{path}: problem_id set differs from other files in "
                f"{path.parent} (extra examples: {extra[:5]})"
            )
        rows_total += rows
        chains_total += chains
        print(f"OK {path}: rows={rows}, chains={chains}")
    print(f"VALID files={len(files)} rows={rows_total} chains={chains_total}")


if __name__ == "__main__":
    main()
