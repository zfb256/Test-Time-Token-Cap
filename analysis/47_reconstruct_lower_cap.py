#!/usr/bin/env python3
"""Reconstruct an exact lower-token-cap run from stored endpoint token IDs.

Generate the longest sampled condition once, then use this script to recover
the same problem/chain/random-seed trajectories at a lower cap. Naturally
terminated chains are kept unchanged; longer chains are decoded from their
first ``cap`` generated token IDs and marked as length-stopped.
"""

import argparse
import copy
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from reproducibility import REQUIRED_RUN_SIGNATURE


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else SCRIPT_DIR / candidate


def decode_generated_tokens(tokenizer, token_ids):
    """Match vLLM's serialized output-text semantics for a token prefix.

    vLLM suppresses a trailing Unicode replacement character when generation
    stops in the middle of a byte-fallback sequence. Hugging Face's whole-list
    decode emits that marker, so remove it for exact cap reconstruction.
    """
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text[:-1] if text.endswith("\ufffd") else text


def reconstruct_chain(chain, cap, decode):
    token_ids = chain.get("token_ids")
    if token_ids is None:
        raise ValueError("chain lacks token_ids; exact cap reconstruction is impossible")
    if len(token_ids) != int(chain.get("token_count", -1)):
        raise ValueError("token_ids length does not match token_count")
    if len(token_ids) <= cap:
        return copy.deepcopy(chain)

    out = copy.deepcopy(chain)
    out["token_ids"] = token_ids[:cap]
    out["token_count"] = cap
    out["text"] = decode(token_ids[:cap])
    out["finish_reason"] = "length"
    out["stop_reason"] = None
    out["reconstructed_at_cap"] = cap
    return out


def load_records(path):
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record["problem_id"]
            if problem_id in records:
                raise ValueError(
                    f"{path}:{line_no}: duplicate problem_id={problem_id}"
                )
            records[problem_id] = record
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def validate_reconstruction(source_records, rebuilt_records, cap, decode=None):
    if set(source_records) != set(rebuilt_records):
        raise ValueError("source and reconstructed problem_id sets differ")
    for problem_id, source in source_records.items():
        rebuilt = rebuilt_records[problem_id]
        for field in ("dataset", "question", "answer", "math_level"):
            default = 0 if field == "math_level" else None
            if source.get(field, default) != rebuilt.get(field, default):
                raise ValueError(
                    f"{field} mismatch for problem_id={problem_id}"
                )
        source_budget = source.get("budget", {})
        rebuilt_budget = rebuilt.get("budget", {})
        source_signature = source_budget.get("run_signature")
        if (
            not isinstance(source_signature, dict)
            or not REQUIRED_RUN_SIGNATURE.issubset(source_signature)
            or rebuilt_budget.get("run_signature") != source_signature
        ):
            raise ValueError(
                f"run signature missing or changed for problem_id={problem_id}"
            )
        source_cap = source_budget.get("max_new_tokens")
        reconstruction = rebuilt_budget.get("reconstructed_from", {})
        if (
            not isinstance(source_cap, int)
            or source_cap <= cap
            or rebuilt_budget.get("max_new_tokens") != cap
            or rebuilt_budget.get("mode") != "reconstructed_token_prefix"
            or reconstruction.get("source_cap") != source_cap
        ):
            raise ValueError(
                f"invalid reconstruction metadata for problem_id={problem_id}"
            )
        if source_budget.get("seed_policy") != rebuilt_budget.get("seed_policy"):
            raise ValueError(
                f"seed policy changed during reconstruction for "
                f"problem_id={problem_id}"
            )
        source_chains = source.get("chains", [])
        rebuilt_chains = rebuilt.get("chains", [])
        if len(source_chains) != len(rebuilt_chains):
            raise ValueError(
                f"chain-count mismatch for problem_id={problem_id}"
            )
        for chain_index, (source_chain, rebuilt_chain) in enumerate(
            zip(source_chains, rebuilt_chains)
        ):
            source_ids = source_chain.get("token_ids")
            rebuilt_ids = rebuilt_chain.get("token_ids")
            if (
                source_ids is None
                or source_chain.get("token_count") != len(source_ids)
                or rebuilt_ids != source_ids[:cap]
            ):
                raise ValueError(
                    f"token-prefix mismatch for problem_id={problem_id}, "
                    f"chain={chain_index}"
                )
            if rebuilt_chain.get("token_count") != len(rebuilt_ids):
                raise ValueError(
                    f"token_count mismatch for problem_id={problem_id}, "
                    f"chain={chain_index}"
                )
            for field in ("chain_index", "request_seed"):
                if source_chain.get(field) != rebuilt_chain.get(field):
                    raise ValueError(
                        f"{field} changed for problem_id={problem_id}, "
                        f"chain={chain_index}"
                    )
            if len(source_ids) > cap:
                if (
                    rebuilt_chain.get("finish_reason") != "length"
                    or rebuilt_chain.get("stop_reason") is not None
                    or rebuilt_chain.get("reconstructed_at_cap") != cap
                ):
                    raise ValueError(
                        f"truncation metadata mismatch for "
                        f"problem_id={problem_id}, chain={chain_index}"
                    )
                if (
                    decode is not None
                    and rebuilt_chain.get("text") != decode(rebuilt_ids)
                ):
                    raise ValueError(
                        f"decoded prefix text mismatch for "
                        f"problem_id={problem_id}, chain={chain_index}"
                    )
            elif rebuilt_chain != source_chain:
                raise ValueError(
                    f"natural completion changed for problem_id={problem_id}, "
                    f"chain={chain_index}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--cap", required=True, type=int)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate and reuse an existing output instead of overwriting it.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Atomically rebuild an existing derived lower-cap output.",
    )
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("--cap must be positive")
    if args.validate_existing and args.rebuild_existing:
        parser.error(
            "--validate-existing and --rebuild-existing are mutually exclusive"
        )

    out_dir = resolve_path(args.results_dir)
    source_path = out_dir / f"inference_{args.source_name}.jsonl"
    output_path = out_dir / f"inference_{args.output_name}.jsonl"
    tokenizer = None
    if output_path.exists():
        if args.validate_existing:
            from transformers import AutoTokenizer

            tokenizer_path = resolve_path(args.tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True
            )
            validate_reconstruction(
                load_records(source_path),
                load_records(output_path),
                args.cap,
                decode=lambda ids: decode_generated_tokens(tokenizer, ids),
            )
            print(f"Validated existing exact lower-cap records -> {output_path}")
            return
        if not args.rebuild_existing:
            raise SystemExit(f"Refusing to overwrite existing {output_path}")

    from transformers import AutoTokenizer

    tokenizer_path = resolve_path(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )

    source_records = load_records(source_path)
    records = []
    for pid, record in source_records.items():
        source_cap = int(record.get("budget", {}).get("max_new_tokens", -1))
        if source_cap <= args.cap:
            raise ValueError(
                f"{source_path}: problem_id={pid}: source cap {source_cap} "
                f"must exceed reconstruction cap {args.cap}"
            )
        rebuilt = copy.deepcopy(record)
        rebuilt["chains"] = [
            reconstruct_chain(
                chain,
                args.cap,
                lambda ids: decode_generated_tokens(tokenizer, ids),
            )
            for chain in record.get("chains", [])
        ]
        rebuilt["budget"]["max_new_tokens"] = args.cap
        rebuilt["budget"]["mode"] = "reconstructed_token_prefix"
        rebuilt["budget"]["reconstructed_from"] = {
            "source_name": args.source_name,
            "source_cap": source_cap,
        }
        records.append(rebuilt)

    validate_reconstruction(
        source_records,
        {record["problem_id"]: record for record in records},
        args.cap,
        decode=lambda ids: decode_generated_tokens(tokenizer, ids),
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)
    print(f"Saved {len(records)} exact lower-cap records -> {output_path}")


if __name__ == "__main__":
    main()
