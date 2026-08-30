"""Decompose resampling and token-budget effects with a three-arm design.

Arm A is an independently sampled base-cap run and defines eligibility.
Arms B and C are the same sampled trajectories observed at the base and
extended caps.  Therefore:

* A -> B measures recovery under independent same-cap resampling;
* A -> C measures recovery under resampling plus a larger token cap;
* C - B is the incremental contribution of extending the B trajectories.

The script operates on one model/seed.  Repeated-seed uncertainty should
cluster the emitted detail rows by problem ID.
"""

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from reproducibility import REQUIRED_RUN_SIGNATURE

SPEC = importlib.util.spec_from_file_location(
    "expanded_repair_eval", SCRIPT_DIR / "48_expanded_repair_eval.py"
)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)

DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "review_diagnostics", SCRIPT_DIR / "57_review_diagnostics.py"
)
DIAGNOSTICS = importlib.util.module_from_spec(DIAGNOSTICS_SPEC)
DIAGNOSTICS_SPEC.loader.exec_module(DIAGNOSTICS)


def _validate_maps(*maps):
    reference = set(maps[0])
    if any(set(mapping) != reference for mapping in maps[1:]):
        raise ValueError("three-arm problem_id sets differ")
    for pid in reference:
        first = maps[0][pid]
        for mapping in maps[1:]:
            for field in ("dataset", "question", "answer"):
                if mapping[pid].get(field) != first.get(field):
                    raise ValueError(f"{field} mismatch for problem_id={pid}")


def _decoding_signature(record):
    budget = record.get("budget", {})
    run_signature = budget.get("run_signature")
    if (
        not isinstance(run_signature, dict)
        or not REQUIRED_RUN_SIGNATURE.issubset(run_signature)
    ):
        raise ValueError("missing required run signature")
    return (
        budget.get("n_chains"),
        budget.get("temperature"),
        budget.get("top_p"),
        json.dumps(run_signature, sort_keys=True, ensure_ascii=False),
    )


def validate_protocol(
    base_map, samecap_map, long_map, base_budget, long_budget
):
    """Verify that B is an exact token prefix of C and A is independent."""
    _validate_maps(base_map, samecap_map, long_map)
    if base_budget < 1 or long_budget <= base_budget:
        raise ValueError("budgets must satisfy 0 < base_budget < long_budget")
    for pid in base_map:
        base_record = base_map[pid]
        samecap_record = samecap_map[pid]
        long_record = long_map[pid]
        records = (
            ("base-independent", base_record, base_budget),
            ("samecap-control", samecap_record, base_budget),
            ("long-control", long_record, long_budget),
        )
        for label, record, expected_cap in records:
            budget = record.get("budget", {})
            if budget.get("max_new_tokens") != expected_cap:
                raise ValueError(
                    f"{label} cap mismatch for problem_id={pid}: "
                    f"{budget.get('max_new_tokens')} != {expected_cap}"
                )
            if len(record.get("chains", [])) != budget.get("n_chains"):
                raise ValueError(
                    f"{label} chain-count mismatch for problem_id={pid}"
                )
        signatures = {
            _decoding_signature(record)
            for _, record, _ in records
        }
        if len(signatures) != 1:
            raise ValueError(f"decoding mismatch across arms for problem_id={pid}")

        samecap_budget = samecap_record["budget"]
        long_budget_metadata = long_record["budget"]
        reconstruction = samecap_budget.get("reconstructed_from", {})
        if (
            samecap_budget.get("mode") != "reconstructed_token_prefix"
            or reconstruction.get("source_cap") != long_budget
        ):
            raise ValueError(
                f"samecap arm is not declared as a {long_budget}-token "
                f"prefix reconstruction for problem_id={pid}"
            )
        if (
            samecap_budget.get("seed_policy")
            != long_budget_metadata.get("seed_policy")
        ):
            raise ValueError(f"B/C seed-policy mismatch for problem_id={pid}")

        base_policy = base_record["budget"].get("seed_policy", {})
        control_policy = long_budget_metadata.get("seed_policy", {})
        if (
            base_policy.get("base_seed") is None
            or control_policy.get("base_seed") is None
            or base_policy.get("base_seed") == control_policy.get("base_seed")
        ):
            raise ValueError(
                f"arm A is not demonstrably independent for problem_id={pid}"
            )

        base_request_seeds = {
            chain.get("request_seed") for chain in base_record["chains"]
        }
        control_request_seeds = {
            chain.get("request_seed") for chain in long_record["chains"]
        }
        if (
            None in base_request_seeds
            or None in control_request_seeds
            or base_request_seeds & control_request_seeds
        ):
            raise ValueError(
                f"A/C request-seed sets are missing values or overlap for "
                f"problem_id={pid}"
            )

        for chain_index, (base_chain, prefix_chain, long_chain) in enumerate(
            zip(
                base_record["chains"],
                samecap_record["chains"],
                long_record["chains"],
            )
        ):
            prefix_ids = prefix_chain.get("token_ids")
            long_ids = long_chain.get("token_ids")
            if prefix_ids is None or long_ids is None:
                raise ValueError(
                    f"B/C token_ids missing for problem_id={pid}, "
                    f"chain={chain_index}"
                )
            expected_prefix = long_ids[:base_budget]
            if prefix_ids != expected_prefix:
                raise ValueError(
                    f"B is not an exact C prefix for problem_id={pid}, "
                    f"chain={chain_index}"
                )
            if prefix_chain.get("token_count") != len(prefix_ids):
                raise ValueError(
                    f"B token_count mismatch for problem_id={pid}, "
                    f"chain={chain_index}"
                )
            if long_chain.get("token_count") != len(long_ids):
                raise ValueError(
                    f"C token_count mismatch for problem_id={pid}, "
                    f"chain={chain_index}"
                )
            for field in ("chain_index", "request_seed"):
                if prefix_chain.get(field) != long_chain.get(field):
                    raise ValueError(
                        f"B/C {field} mismatch for problem_id={pid}, "
                        f"chain={chain_index}"
                    )


def analyze(base_map, samecap_map, long_map, base_budget, long_budget):
    _validate_maps(base_map, samecap_map, long_map)
    rows = []
    for pid, base_record in base_map.items():
        base = REPAIR.chain_values(base_record, base_budget, REPAIR.VERIFIER)
        samecap = REPAIR.chain_values(
            samecap_map[pid], base_budget, REPAIR.VERIFIER
        )
        long = REPAIR.chain_values(long_map[pid], long_budget, REPAIR.VERIFIER)
        eligible = REPAIR.eligibility(base)
        base_vote = REPAIR.plurality(base)
        samecap_vote = REPAIR.plurality(samecap)
        long_vote = REPAIR.plurality(long)
        base_any = any(value["correct"] for value in base)
        samecap_any = any(value["correct"] for value in samecap)
        long_any = any(value["correct"] for value in long)

        for definition, is_eligible in eligible.items():
            if definition == "first_chain_wrong":
                samecap_repair_any = samecap_any
                long_repair_any = long_any
            else:
                samecap_repair_any = not base_any and samecap_any
                long_repair_any = not base_any and long_any
            rows.append(
                {
                    "problem_id": pid,
                    "dataset": base_record["dataset"],
                    "definition": definition,
                    "eligible": is_eligible,
                    "base_plurality_correct": base_vote["correct"],
                    "base_any_correct": base_any,
                    "samecap_plurality_correct": samecap_vote["correct"],
                    "long_plurality_correct": long_vote["correct"],
                    "samecap_repair_plurality": (
                        is_eligible and samecap_vote["correct"]
                    ),
                    "long_repair_plurality": (
                        is_eligible and long_vote["correct"]
                    ),
                    "samecap_repair_any": is_eligible and samecap_repair_any,
                    "long_repair_any": is_eligible and long_repair_any,
                    "incremental_plurality_helpful": (
                        is_eligible
                        and not samecap_vote["correct"]
                        and long_vote["correct"]
                    ),
                    "incremental_plurality_harmful": (
                        is_eligible
                        and samecap_vote["correct"]
                        and not long_vote["correct"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(detail):
    rows = []
    groups = [("all", detail)]
    groups.extend((name, group) for name, group in detail.groupby("dataset"))
    for dataset, dataset_frame in groups:
        for definition, group in dataset_frame.groupby("definition", sort=False):
            eligible = group[group["eligible"]]
            n = len(eligible)
            samecap = int(eligible["samecap_repair_plurality"].sum())
            long = int(eligible["long_repair_plurality"].sum())
            helpful = int(eligible["incremental_plurality_helpful"].sum())
            harmful = int(eligible["incremental_plurality_harmful"].sum())
            rows.append(
                {
                    "dataset": dataset,
                    "definition": definition,
                    "n_problems": len(group),
                    "n_eligible": n,
                    "samecap_repair_count": samecap,
                    "samecap_repair_rate": samecap / n if n else float("nan"),
                    "long_repair_count": long,
                    "long_repair_rate": long / n if n else float("nan"),
                    "incremental_token_effect": (
                        (long - samecap) / n if n else float("nan")
                    ),
                    "incremental_helpful_count": helpful,
                    "incremental_harmful_count": harmful,
                    "samecap_any_repair_count": int(
                        eligible["samecap_repair_any"].sum()
                    ),
                    "long_any_repair_count": int(
                        eligible["long_repair_any"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_harm(base_map, samecap_map, long_map, base_budget, long_budget):
    eligibility_values = {
        pid: REPAIR.chain_values(record, base_budget, REPAIR.VERIFIER)
        for pid, record in base_map.items()
    }
    contrasts = (
        ("A_to_B_resampling", base_map, samecap_map, base_budget, base_budget),
        ("A_to_C_combined", base_map, long_map, base_budget, long_budget),
        ("B_to_C_tokens", samecap_map, long_map, base_budget, long_budget),
    )
    frames = []
    for contrast, source, target, source_budget, target_budget in contrasts:
        cases = DIAGNOSTICS.build_cases(
            source, target, source_budget, target_budget
        )
        frame = DIAGNOSTICS.symmetric_harm(cases, eligibility_values)
        frame.insert(0, "contrast", contrast)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_harm(detail):
    frames = []
    for contrast, group in detail.groupby("contrast", sort=False):
        frame = DIAGNOSTICS.summarize_harm(group)
        frame.insert(0, "contrast", contrast)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _input_signature(paths, base_budget, long_budget):
    inputs = []
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        inputs.append((str(resolved), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        [inputs, base_budget, long_budget], sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aligned_batches(paths, batch_size=10):
    """Read aligned JSONL arms without retaining the full runs in memory."""
    sentinel = object()

    def records(handle):
        for line in handle:
            if line.strip():
                yield json.loads(line)

    seen = set()
    with ExitStack() as stack:
        handles = [
            stack.enter_context(path.open(encoding="utf-8")) for path in paths
        ]
        batch = ({}, {}, {})
        for rows in zip_longest(
            *(records(handle) for handle in handles), fillvalue=sentinel
        ):
            if sentinel in rows:
                raise ValueError("three-arm JSONL files have different lengths")
            problem_ids = [row["problem_id"] for row in rows]
            if len(set(problem_ids)) != 1:
                raise ValueError(
                    "three-arm JSONL files have different problem_id order"
                )
            problem_id = problem_ids[0]
            if problem_id in seen:
                raise ValueError(f"duplicate problem_id={problem_id}")
            seen.add(problem_id)
            for mapping, row in zip(batch, rows):
                mapping[problem_id] = row
            if len(batch[0]) == batch_size:
                yield batch
                batch = ({}, {}, {})
        if batch[0]:
            yield batch


def _analyze_batches_checkpointed(
    batches,
    base_budget,
    long_budget,
    checkpoint_path,
    signature,
):
    connection = sqlite3.connect(checkpoint_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS meta "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS progress "
        "(problem_id TEXT PRIMARY KEY, detail TEXT NOT NULL, harm TEXT NOT NULL)"
    )
    stored = connection.execute(
        "SELECT value FROM meta WHERE key = 'signature'"
    ).fetchone()
    if stored and stored[0] != signature:
        connection.close()
        raise ValueError("three-arm checkpoint input signature changed")
    connection.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('signature', ?)",
        (signature,),
    )
    connection.commit()

    done = {
        row[0] for row in connection.execute("SELECT problem_id FROM progress")
    }
    expected = []
    expected_set = set()
    for base_map, samecap_map, long_map in batches:
        overlap = expected_set.intersection(base_map)
        if overlap:
            raise ValueError(f"duplicate problem_id={next(iter(overlap))}")
        expected.extend(base_map)
        expected_set.update(base_map)
        problem_ids = [pid for pid in base_map if pid not in done]
        if not problem_ids:
            continue
        batch_base = {pid: base_map[pid] for pid in problem_ids}
        batch_samecap = {pid: samecap_map[pid] for pid in problem_ids}
        batch_long = {pid: long_map[pid] for pid in problem_ids}
        detail = analyze(
            batch_base,
            batch_samecap,
            batch_long,
            base_budget,
            long_budget,
        )
        harm = analyze_harm(
            batch_base,
            batch_samecap,
            batch_long,
            base_budget,
            long_budget,
        )
        rows = [
            (
                pid,
                detail[detail["problem_id"] == pid].to_json(
                    orient="records"
                ),
                harm[harm["problem_id"] == pid].to_json(orient="records"),
            )
            for pid in problem_ids
        ]
        with connection:
            connection.executemany(
                "INSERT INTO progress(problem_id, detail, harm) "
                "VALUES (?, ?, ?)",
                rows,
            )
        done.update(problem_ids)
        print(f"Checkpointed {len(done)} problems", flush=True)

    payloads = {
        problem_id: (detail, harm)
        for problem_id, detail, harm in connection.execute(
            "SELECT problem_id, detail, harm FROM progress"
        )
    }
    connection.close()
    if set(payloads) != expected_set:
        raise ValueError("three-arm checkpoint problem_id set is incomplete")
    detail_rows, harm_rows = [], []
    for problem_id in expected:
        detail, harm = payloads[problem_id]
        detail_rows.extend(json.loads(detail))
        harm_rows.extend(json.loads(harm))
    return pd.DataFrame(detail_rows), pd.DataFrame(harm_rows)


def analyze_checkpointed(
    base_map,
    samecap_map,
    long_map,
    base_budget,
    long_budget,
    checkpoint_path,
    signature,
):
    problem_ids = list(base_map)
    batches = (
        (
            {pid: base_map[pid] for pid in problem_ids[start : start + 10]},
            {pid: samecap_map[pid] for pid in problem_ids[start : start + 10]},
            {pid: long_map[pid] for pid in problem_ids[start : start + 10]},
        )
        for start in range(0, len(problem_ids), 10)
    )
    return _analyze_batches_checkpointed(
        batches, base_budget, long_budget, checkpoint_path, signature
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-independent", required=True)
    parser.add_argument("--samecap-control", required=True)
    parser.add_argument("--long-control", required=True)
    parser.add_argument("--base-budget", type=int, default=768)
    parser.add_argument("--long-budget", type=int, default=1536)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    paths = tuple(
        Path(value)
        for value in (
            args.base_independent,
            args.samecap_control,
            args.long_control,
        )
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(f"{prefix}_checkpoint.sqlite")

    def validated_batches():
        for batch in _aligned_batches(paths):
            validate_protocol(*batch, args.base_budget, args.long_budget)
            yield batch

    detail, harm_detail = _analyze_batches_checkpointed(
        validated_batches(),
        args.base_budget,
        args.long_budget,
        checkpoint,
        _input_signature(paths, args.base_budget, args.long_budget),
    )
    summary = summarize(detail)
    harm_summary = summarize_harm(harm_detail)
    detail.to_csv(f"{prefix}_detail.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    harm_detail.to_csv(f"{prefix}_harm_detail.csv", index=False)
    harm_summary.to_csv(f"{prefix}_harm_summary.csv", index=False)
    checkpoint.unlink()
    print(summary.to_string(index=False))
    print(harm_summary.to_string(index=False))


if __name__ == "__main__":
    main()
