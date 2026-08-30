# Attributing Test-Time Token-Cap Gains in Mathematical Reasoning: Continuation Versus Resampling

Research project: when does extra inference-time compute help mathematical
reasoning — by completing truncated traces, or by repairing complete but
incorrect ones? Includes a post-trace budget-allocation (routing) method
built on the answer.

This repository is the compact GitHub snapshot: source, tests, fixed dataset
snapshots, and result summaries are included. Raw inference
traces and per-problem detail tables are excluded because the full artifact
tree is about 3 GB; the reproducibility manifests retain their hashes.

## Experimental design

The primary protocol separates two effects that a two-budget comparison
confounds:

- arm A: an independent sample at the lower token cap;
- arm B: the exact lower-cap token prefix reconstructed from arm C;
- arm C: the paired sample at the higher token cap.

Thus A→B estimates same-budget resampling, while B→C isolates the effect of
additional tokens on the same trajectory. Protocol validation rejects runs
with mismatched decoding signatures, overlapping A/C request seeds, or a B arm
that is not an exact token prefix of C.

## Layout

- `pipeline/` — inference + scoring pipeline (steps 00–08) and all run
  configs. GPU steps run here; see `run_three_seed_core.sh` for the current
  experiment batch.
- `analysis/` — post-hoc analysis scripts. CPU only; defaults
  resolve to `../outputs/qwen_main` and `../pipeline/config.yaml`.
- `data/` — the 800-problem core set and `data/expanded/`, which contains all
  1,319 GSM8K and 5,000 MATH test problems. Reuse these snapshots so runs
  stay comparable.
- `models/` — model weights (not committed; download on the GPU server).
- `outputs/` — compact summaries for the three-arm, temperature, routing, and
  equivalence analyses. Full inference traces are not included.
## Install and test

The pinned environment targets Python 3.10 and CUDA 11.8:

```bash
python -m pip install -r pipeline/requirements.txt
python -m pytest -q
```

GPU inference uses vLLM. CPU-only tests and most post-hoc analyses do not
require model weights.

## Status (2026-08)

The current experiment uses three arms: A is an independent 768-token sample,
B is the exact 768-token prefix of C, and C uses the same request seed with a
1536-token cap. A→B measures same-budget resampling; B→C isolates additional
tokens.

Across three core seeds, arm A defines 52 Qwen and 115 Llama strict
complete-but-wrong pairs. Same-cap and long runs repair 0/0 Qwen pluralities
and the same 4/4 Llama pluralities, leaving no B→C plurality transition. The
one-seed expanded run gives the same equality on all 6,319 problems: 1/240
versus 1/240 for Qwen and 10/509 versus 10/509 for Llama. On arm-A-complete
correct cases, same-cap resampling harms 20/3,994 Qwen and 150/3,168 Llama
pluralities; extra tokens harm 0 and 1. Greedy prefix audits remain a protocol
check rather than evidence about independent resampling.

At temperatures 0.6, 0.75, and 1.0, the three-seed core sweep finds one strict
B→C repair across six model--temperature cells and no strict harmful
transition. The only event is Llama at temperature 0.6 (1 of 115). Eligibility
falls with temperature for Qwen, 60/52/38, and is flat before dropping for
Llama, 115/115/84. R1 runs the same design at 2048 against 4096 tokens and
contributes 24 pooled pairs with no repair from either arm.

## Running

The reproducible drivers support a small smoke test, the three-seed core run,
the one-seed full-split run, and the temperature sweep:

```bash
cd pipeline
bash run_three_seed_core.sh smoke
bash run_three_seed_core.sh full
bash run_three_seed_core.sh expanded
bash run_temperature_sensitivity.sh full
```

Each mode generates arms A and C, reconstructs B from stored token IDs, runs
the CPU analyses, validates row counts, and writes a reproducibility manifest.
Inference is resumable; completed records are reused rather than regenerated.

CPU analyses (from `analysis/`, after regenerating or restoring the full raw
artifact tree):

```bash
python 37_confidence_intervals.py     # CIs + same-fraction router comparison
python 38_trace_prefix_overlap.py     # greedy prefix audit
python 40_cost_accounting_v2.py       # rerun vs resume accounting
python 39_sampling_repair_eval.py ... # matched sampled repair
python 53_three_arm_repair_decomposition.py ... # A/B/C attribution
python 56_aggregate_three_arm.py ...   # clustered three-seed summary
python 58_temperature_sensitivity.py   # one-seed temperature table
python 60_equivalence_power.py         # equivalence bounds, detection limits
python 61_three_arm_routing.py         # allocation inside the three-arm design
python 62_temperature_multiseed.py     # pooled three-seed temperature table
python 63_verifier_audit.py            # can strict matching hide a repair?
```

Scripts 60--63 back the PeerJ manuscript. Script 62 pools whatever seed
directories exist; script 61 covers Qwen, Llama, and R1; script 63 re-scores
the eligible subset to bound what strict answer matching could be hiding. All
GPU runs these depend on have landed.

Do not move `pipeline/` and `analysis/` apart; analysis defaults are
relative to the script directory.
