# Experiment summaries and provenance

This compact checkout includes aggregate results and historical experiment
manifests. Raw inference JSONL and per-problem detail tables are omitted.
Restore or regenerate those files before rerunning analyses that require them.
The root README explains figure reproduction from the included summaries.

## Primary three-arm experiments

| Directory | Population and settings |
| --- | --- |
| `three_arm_core/` | Qwen/Llama, 800 problems, 768/1536 tokens, eight chains, seeds 42 / 314159 / 271828 |
| `three_arm_expanded/` | Qwen/Llama, 6,319 problems, same budgets and chain count, seed 42 |
| `three_arm_r1/` | R1, 800 problems, 2048/4096 tokens, four chains, three seeds |

For Qwen/Llama, A is `inference_long_sample8.jsonl` and C is
`inference_xlong_sample8.jsonl`. R1 uses `inference_long_sample4.jsonl` and
`inference_xlong_sample4.jsonl`. In each tree, B is
`inference_long_reconstructed.jsonl`, reconstructed from C's token identifiers.
These filenames describe the full artifact, not files shipped here.

`three_arm_summary.csv` files back the primary correction table;
`three_arm_harm_summary.csv` files describe the separate A-selected reversal
population. The `aggregate/` directories contain repeated-seed summaries.

## Derived analyses

| Directory | Analysis | Manuscript use |
| --- | --- | --- |
| `temperature_sensitivity/` | `analysis/62_temperature_multiseed.py` | Three-seed temperature sensitivity; temperature 0.75 reuses `three_arm_core/` |
| `three_arm_routing/` | `analysis/61_three_arm_routing.py` | Routing curves and allocation table over all records |
| `equivalence/` | `analysis/60_equivalence_power.py`, `analysis/63_verifier_audit.py` | Event-rate bounds, paired tests, and C-only verifier sensitivity |

Core binomial intervals pool dependent problem–seed pairs and are descriptive.
Full-split bounds use one record per problem. Verifier sensitivity holds A
eligibility fixed; unchanged correct-plurality counts in C do not constitute
a full B/C audit with a new verifier.

## Earlier engine-seeded comparisons

These runs are separate comparisons and must not be pooled with the primary
request-seeded experiments.

| Directory | Contents |
| --- | --- |
| `qwen_main/` | Independent-budget and mixed-design comparisons, greedy prefix audit, verifier and budget sensitivity |
| `llama_validation/` | Llama independent-budget, mixed-design, and greedy checks |
| `r1_distill/` | R1 independent-budget comparisons at 1024 / 2048 / 4096 tokens |
| `aime_qwen/` | Sampled Qwen at 768 / 1536 / 3072 tokens and a separate greedy audit |
| `aime_r1/` | Sampled R1 at 2048 / 4096 / 8192 tokens |

The two `large_model_mechanism_summary.md` files retain historical greedy
summaries and their original scoring. They are not current three-arm result
tables. In particular, the Llama summary's 94 complete-but-wrong records must
not be substituted for the manuscript's mixed-design population of 98.
Historical `repair` field names may count correct endpoints, rather than
wrong-to-right transitions when relaxed eligibility admits a correct A vote.

## Validation and checksums

After restoring raw records, run from the repository root:

```bash
python analysis/45_validate_rerun.py outputs/three_arm_core
python analysis/45_validate_rerun.py outputs/three_arm_r1
python analysis/45_validate_rerun.py outputs/three_arm_expanded --expected-rows 6319
```

The validator expects 800 rows for these core trees unless overridden
(and infers 60 for AIME directories). `VALIDATION.txt` retains the original
validation record; it is not evidence that raw-data checks ran on this compact
checkout. `validate_protocol()` in `analysis/53_three_arm_repair_decomposition.py`
checks aligned problems and decoding settings, distinct A/C request seeds,
and exact B/C token-prefix identity.

The `reproducibility_manifest.*` files retain historical paths and hashes,
including omitted raw files and earlier source versions. Old manuscript paths
are provenance, not current build inputs. Current checkout hashes are in
`../SHA256SUMS`; use those to check this release. A newly generated experiment
can create its own manifest with `analysis/28_reproducibility_manifest.py`.
