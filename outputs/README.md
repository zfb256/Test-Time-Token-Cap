# Canonical experiment artifacts

This GitHub snapshot contains compact summaries and reproducibility manifests.
The multi-gigabyte inference JSONL and per-problem detail files are
intentionally excluded from GitHub and can be regenerated with the scripts in
`pipeline/`.

## Three-arm trees (the primary result)

These back the attribution decomposition. Arm A is
`inference_long_sample8.jsonl`, arm C is `inference_xlong_sample8.jsonl`, and
arm B is `inference_long_reconstructed.jsonl`, rebuilt on CPU from arm C's
stored token identifiers by `analysis/47_reconstruct_lower_cap.py`.

| Directory | Contents |
| --- | --- |
| `three_arm_core/` | Qwen and Llama, 800-problem core split, 768/1536 tokens, seeds 42 / 314159 / 271828, plus `aggregate/` |
| `three_arm_expanded/` | Qwen and Llama, full GSM8K + MATH split (6,319 problems), seed 42 |
| `three_arm_r1/` | R1 at its own budget pair, 2048/4096 tokens, three seeds, plus `aggregate/` |

## Derived analyses

| Directory | Produced by | Backs |
| --- | --- | --- |
| `temperature_sensitivity/` | `analysis/62_temperature_multiseed.py` | temperature sweep, `t060/` and `t100/` (T=0.75 is `three_arm_core`) |
| `three_arm_routing/` | `analysis/61_three_arm_routing.py` | routing curve and summary |
| `equivalence/` | `analysis/60_equivalence_power.py`, `analysis/63_verifier_audit.py` | exact bounds, McNemar tests, verifier re-scoring |

## Earlier engine-seeded runs

These predate the request-seeded three-arm protocol and use engine-level
seeding, so they must not be pooled with the trees above. The paper reports
them as separate contrasts.

| Directory | Backs |
| --- | --- |
| `qwen_main/` | Qwen independent-budget comparison, greedy prefix audit, relaxed-verifier sensitivity, budget-scaling and cost accounting |
| `llama_validation/` | Llama independent-budget comparison and second-model sanity check |
| `r1_distill/` | R1 mixed-design contrasts at 1024 / 2048 / 4096 tokens |
| `aime_qwen/` | AIME 2024 and 2025, Qwen, 768 / 1536 / 3072 tokens, plus the greedy prefix audit |
| `aime_r1/` | AIME 2024 and 2025, R1, 2048 / 4096 / 8192 tokens |

## Validation

```
python analysis/45_validate_rerun.py outputs/three_arm_core
python analysis/45_validate_rerun.py outputs/three_arm_r1
```

The validator asserts the 800-problem core-split shape, so it applies to the
core and R1 trees, not to `three_arm_expanded/` (6,319 problems). See
`VALIDATION.txt` for the recorded result.

Protocol preconditions are enforced separately, and more strictly, by
`validate_protocol()` in `analysis/53_three_arm_repair_decomposition.py`, which
refuses to analyse a tree unless the three arms agree on problems, decoding
signature and budgets, arms A and C draw disjoint request seeds, and every
arm-B chain is an exact token prefix of its arm-C chain.

`reproducibility_manifest.json` records hashes for the canonical snapshot;
regenerate it with `analysis/28_reproducibility_manifest.py`.
