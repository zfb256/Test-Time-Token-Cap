# Attributing Test-Time Token-Cap Gains in Mathematical Reasoning: Continuation Versus Resampling

Code, frozen dataset snapshots, and result aggregates for the PeerJ Computer
Science submission of the same name.

## Description

When a language model gets a larger output-token budget and its accuracy on a
mathematical benchmark rises, the gain is usually read as the model having
repaired a faulty derivation. This project shows that reading is not supported
by the measurement. A larger cap does two things at once: it lets a truncated
derivation finish, and — if the two budgets are sampled independently — it
replaces the trajectory outright. Both arrive in the accuracy curve as a
corrected answer.

The repository implements a three-arm protocol that separates them, applies it
to three released 7–8B models on GSM8K, MATH, and AIME, and reports the
attribution with exact bounds. It also contains the post-trace budget
allocation (routing) analysis that follows from the result.

This is the compact snapshot: source, tests, dataset snapshots, and result
aggregates are included. Raw per-chain inference traces and per-problem detail
tables are excluded because the full artifact tree is about 3 GB. Their
checksums are retained in `outputs/reproducibility_manifest.json`.

## Dataset Information

All evaluation data are public third-party benchmarks, used without any edit to
the problem text. Frozen snapshots live in `data/` so that every run and every
arm sees the same problems in the same order.

| Snapshot | Path | Contents | Source |
| --- | --- | --- | --- |
| Core split | `data/all_problems.jsonl` | 400 GSM8K + 400 MATH test problems | `openai/gsm8k` (config `main`, `test`); `SuperSecureHuman/competition_math_hf_dataset` (`test`) |
| Full split | `data/expanded/all_problems.jsonl` | all 1,319 GSM8K + 5,000 MATH test problems | same as above |
| AIME set | `data/aime/all_problems.jsonl` | all 60 AIME 2024 + 2025 problems | `Maxwell-Jia/AIME_2024`; `yentinglin/aime_2025` |

Source URLs:

- https://huggingface.co/datasets/openai/gsm8k (GSM8K, Cobbe et al., 2021)
- https://huggingface.co/datasets/SuperSecureHuman/competition_math_hf_dataset
  (MATH, Hendrycks et al., 2021)
- https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
- https://huggingface.co/datasets/yentinglin/aime_2025

Record schema (one JSON object per line): `problem_id`, `dataset`, `source`,
`source_split`, `question`, `answer`, `math_level`. GSM8K is assigned
`math_level` 0 and AIME 6 so the three sets order on one difficulty scale.

Preprocessing, implemented in `pipeline/01_download_data.py` and
`pipeline/09_download_aime.py`, is limited to these steps: sample the snapshot
(GSM8K uniformly, MATH stratified at 80 problems per difficulty level 1–5, both
under seed 42); rewrite each problem into the schema above; normalize AIME
answers to `\boxed{n}` and verify they are integers in 0–999, aborting the build
on any row that fails; wrap the question in the model's own zero-shot chat
template. No other filtering, deduplication, augmentation, or truncation is
applied. No language model is trained or fine-tuned anywhere in this project.

Regenerate the snapshots with:

```bash
cd pipeline
python 01_download_data.py --config config.yaml                       # core split
python 01_download_data.py --config config.yaml --data-dir ../data/expanded --full
python 09_download_aime.py
```

Redistribution of the problem text follows the source licences; see `LICENSE`.

## Code Information

| Directory | Contents |
| --- | --- |
| `pipeline/` | Inference and scoring steps `00`–`09`, the run drivers (`run_*.sh`), and every experiment config (`config*.yaml`). GPU steps run from here. |
| `analysis/` | Post-hoc analysis scripts, CPU only. Each writes CSV and TXT summaries into an `outputs/` tree. |
| `data/` | Frozen dataset snapshots described above. |
| `outputs/` | Problem-level aggregates and reproducibility manifests for every analysis the paper reports. `outputs/README.md` maps each directory to the tables and figures it backs. |
| `tests/` | CPU regression suite; no model weights required. |
| `models/` | Model weights. Not committed; download on the GPU server. |

Key entry points:

| Script | Produces |
| --- | --- |
| `pipeline/02_run_inference.py` | Sampled generation; stores token identifiers and termination reasons per chain. |
| `analysis/47_reconstruct_lower_cap.py` | Arm B, by truncating arm C's stored token identifiers at the base cap. |
| `analysis/53_three_arm_repair_decomposition.py` | The A/B/C attribution and its protocol preconditions. |
| `analysis/60_equivalence_power.py` | Clopper–Pearson bounds, exact McNemar tests, detection limits. |
| `analysis/61_three_arm_routing.py` | Post-trace allocation curves and cost accounting. |
| `analysis/62_temperature_multiseed.py` | Pooled three-seed temperature table. |
| `analysis/63_verifier_audit.py` | Re-scoring bound on what strict answer matching could hide. |
| `analysis/36_make_paper_figures.py` | Every figure in the paper. None is hand-edited. |

Do not move `pipeline/` and `analysis/` apart; analysis defaults are resolved
relative to the script directory.

## Usage Instructions

### Reproduce the tables and figures from the shipped aggregates

No GPU and no downloads are required; `outputs/` already holds the
problem-level aggregates.

```bash
python -m pip install -r pipeline/requirements.txt
python analysis/36_make_paper_figures.py
```

### Reproduce the experiments end to end

The drivers support a smoke test, the three-seed core run, the one-seed
full-split run, and the temperature sweep. Each mode generates arms A and C,
reconstructs arm B from the stored token identifiers, runs the CPU analyses,
validates row counts, and refreshes the reproducibility manifest. Inference is
resumable: completed records are reused rather than regenerated.

```bash
cd pipeline
bash run_three_seed_core.sh smoke        # small end-to-end check
bash run_three_seed_core.sh full         # three-seed, 800-problem core run
bash run_three_seed_core.sh expanded     # one-seed, 6,319-problem full split
bash run_temperature_sensitivity.sh full # T = 0.6 / 0.75 / 1.0 sweep
bash run_r1_three_arm.sh                 # R1 at 2048 against 4096 tokens
```

### Run a single analysis

From `analysis/`, after regenerating or restoring the raw artifact tree:

```bash
python 37_confidence_intervals.py     # CIs + same-fraction router comparison
python 38_trace_prefix_overlap.py     # greedy prefix audit
python 40_cost_accounting_v2.py       # rerun vs resume accounting
python 39_sampling_repair_eval.py ... # matched sampled repair
python 53_three_arm_repair_decomposition.py ... # A/B/C attribution
python 56_aggregate_three_arm.py ...  # clustered three-seed summary
python 60_equivalence_power.py        # equivalence bounds, detection limits
python 61_three_arm_routing.py        # allocation inside the three-arm design
python 62_temperature_multiseed.py    # pooled three-seed temperature table
python 63_verifier_audit.py           # can strict matching hide a repair?
```

Pass `--results-dir` to point a script at a tree other than its default.

### Validate a run

```bash
python analysis/45_validate_rerun.py outputs/three_arm_core
python analysis/45_validate_rerun.py outputs/three_arm_r1
```

## Requirements

- Python 3.10 (tested on 3.10.11)
- CUDA 11.8 or later, and one GPU with at least 40 GB of memory for inference.
  The reported runs used a single NVIDIA A800-SXM4-40GB in bfloat16 under
  Ubuntu 22.04 LTS with CUDA 12.4.
- No GPU is needed to reproduce the tables and figures from the shipped
  aggregates

All Python dependencies are pinned in `pipeline/requirements.txt`. The principal
ones are vLLM 0.6.6.post1, PyTorch 2.5.1, Transformers 4.47.1, scikit-learn
1.4.2, NumPy 1.26.4, pandas 2.2.2, SymPy 1.13.1, latex2sympy2 1.9.1, SciPy
1.12.0, and matplotlib 3.8.4.

```bash
python -m pip install -r pipeline/requirements.txt
python -m pytest -q
```

## Methodology

The protocol separates two effects that a two-budget comparison confounds:

- **arm A** — an independent sample at the base token cap;
- **arm B** — the exact base-cap token prefix, reconstructed from arm C at no
  additional inference cost;
- **arm C** — the paired sample at the extended token cap.

A→B therefore estimates same-budget resampling, and B→C isolates the effect of
additional tokens on the same trajectory. Eligibility is fixed on arm A, which
places both contrasts on one population that was chosen without reference to
the paired arms.

Pipeline order:

1. Build the frozen dataset snapshot (`01`, `09`).
2. Generate arms A and C with per-request deterministic seeds derived by
   SHA-256 from the engine seed, the problem identifier, and the chain index
   (`02`).
3. Reconstruct arm B on CPU from arm C's stored token identifiers
   (`analysis/47`).
4. Score every chain with one verifier, aggregate by plurality vote, and
   classify each record as complete-but-wrong or not.
5. Run the attribution, bounds, temperature, and routing analyses
   (`analysis/53`, `60`–`63`).
6. Validate structure and refresh the manifest (`analysis/45`, `28`).

Protocol validation refuses to analyse a tree unless the three arms agree on
problems, decoding signature, and budgets; arms A and C draw disjoint request
seeds; and every arm-B chain is an exact token prefix of its arm-C chain.

Evaluation is a paired within-problem comparison. Assessment metrics are
plurality accuracy, any-chain accuracy, repair and reversal counts reported
separately by direction, cap-hit fraction, and token-cap cost ratios, with exact
one-sided Clopper–Pearson upper bounds, exact McNemar tests, and a detection
scale marker for the zero counts. Precision, recall, and F1 are not reported:
each record has one reference answer and one aggregated prediction, so scoring
returns a single correct-or-incorrect verdict with no confusion matrix to trade
off. Robustness comes from three-seed replication, full-split replication,
per-dataset reporting, three models, and four one-factor ablations (temperature,
chain count, eligibility predicate, verifier strictness). The router is scored
out of fold under five-fold cross-validation grouped by problem identifier.

## Results Snapshot

Across three core seeds, arm A defines 52 Qwen and 115 Llama strict
complete-but-wrong pairs. Same-cap and extended runs repair 0/0 Qwen
pluralities and the same 4/4 Llama pluralities, leaving no B→C plurality
transition. The one-seed full split gives the same equality on all 6,319
problems: 1/240 versus 1/240 for Qwen and 10/509 versus 10/509 for Llama. On
arm-A-complete correct cases, same-cap resampling harms 20/3,994 Qwen and
150/3,168 Llama pluralities, while extra tokens harm 0 and 1. Greedy prefix
audits are the measured evidence for prefix preservation; under sampling the
property holds by construction rather than by measurement.

At temperatures 0.6, 0.75, and 1.0, the three-seed core sweep finds one strict
B→C repair across six model–temperature cells and no strict harmful transition.
The only event is Llama at temperature 0.6 (1 of 115). Eligibility falls with
temperature for Qwen, 60/52/38, and is flat before dropping for Llama,
115/115/84. R1 runs the same design at 2048 against 4096 tokens and contributes
24 pooled pairs with no repair from either arm.

## Citations

If you use this code or these snapshots, please cite the archived release. The
machine-readable metadata is in `CITATION.cff`, and the Zenodo concept DOI
always resolves to the latest version:

> Zhuo, F. Attributing test-time token-cap gains in mathematical reasoning:
> continuation versus resampling. https://doi.org/10.5281/zenodo.22172646

Please also cite the benchmarks this project evaluates on:

- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems.
  arXiv:2110.14168.
- Hendrycks, D. et al. (2021). Measuring Mathematical Problem Solving With the
  MATH Dataset. NeurIPS Datasets and Benchmarks.

## License & Contribution Guidelines

The code in this repository is released under the MIT License; see `LICENSE`.
The dataset snapshots under `data/` remain subject to their own upstream
licences, which `LICENSE` records.

This repository is the frozen artifact for a specific paper, so it is not an
actively developed project and pull requests that change the analysis are not
accepted — an exact copy of what produced the published results has to remain
available. Bug reports, reproduction failures, and questions are welcome as
GitHub issues, and corrections that affect the published results will be
handled as a new archived version with its own DOI.
