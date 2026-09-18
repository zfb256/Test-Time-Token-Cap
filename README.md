# Attributing test-time token-cap gains in mathematical reasoning: continuation versus resampling

Code, frozen benchmark snapshots, and aggregate results accompanying the
manuscript prepared for the International Journal of Machine Learning and
Cybernetics. The manuscript has not been published in that journal.

## Study and main findings

Comparing independently sampled token budgets combines trajectory replacement
with continuation. The three-arm protocol separates these effects: A is an
independent base-budget sample, C is an extended-budget sample, and B is C's
exact base-cap token prefix. Eligibility is selected on A alone. A→B measures
same-budget resampling, while B→C measures continuation of a shared trajectory.

Strict complete-but-wrong eligibility requires every A chain to terminate
naturally with a parseable, incorrect answer. Across five primary comparisons,
continuation adds no plurality correction on this population:

| Model | Split | Eligible records | A→B corrections | A→C corrections |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-Math-7B-Instruct | Core, three seeds | 52 | 0 | 0 |
| Llama-3.1-8B-Instruct | Core, three seeds | 115 | 4 | 4 |
| DeepSeek-R1-Distill-Qwen-7B | Core, three seeds | 24 | 0 | 0 |
| Qwen2.5-Math-7B-Instruct | Full, one seed | 240 | 1 | 1 |
| Llama-3.1-8B-Instruct | Full, one seed | 509 | 10 | 10 |

Full-split one-sided 95% upper bounds on incremental correction are 1.24% for
Qwen and 0.59% for Llama. Core records pool repeated problems across seeds;
their binomial bounds are descriptive, and the core and full splits overlap.
Temperature and eligibility changes yield exceptions: Llama has one strict
incremental correction at temperature 0.6, and first-chain eligibility admits
1 Qwen and 6 Llama incremental corrections. The zero counts do not establish
a general inability to self-correct.

Offline routing uses B features and archived B/C outcomes over all records.
The learned router is evaluated out of fold, grouping seeds of each problem.
Routing all records with capped B chains reproduces C accuracy by construction.
Rerun and resume savings are token-cap projections, not measured execution
savings; online execution requires separate validation.

## Included files

| Directory | Contents |
| --- | --- |
| `pipeline/` | Generation, scoring, configurations, and experiment drivers |
| `analysis/` | Reconstruction, aggregation, statistical analysis, and plotting |
| `tests/` | CPU regression tests |
| `data/` | Frozen core, full-split, and AIME benchmark snapshots |
| `outputs/` | Aggregate CSV/TXT reports and historical experiment manifests |

Raw per-chain generation records and per-problem detail tables are excluded.
They can be regenerated and are available from the corresponding author on
request. Their historical checksums remain in the experiment manifests.
Model weights and the `paper/` directory are not included.
See [outputs/README.md](outputs/README.md) for the result-directory map.

## Reproduce the figures

From the repository root, install the two plotting dependencies and run:

```bash
python -m pip install pandas==2.2.2 matplotlib==3.8.4
python analysis/36_make_paper_figures.py
```

This uses the included routing aggregates and writes `paper/Fig1.pdf` and
`paper/Fig2.pdf` locally. The generated `paper/` directory is ignored by Git
and is not part of the release. No inference or benchmark download is needed. Other historical
plot functions remain in the script but are not called by its entry point.
The included summaries allow inspection of reported table values; regenerating
analyses that need per-problem records requires restoring or regenerating those
records first.

## Reproduce the experiments

The reported inference environment used Python 3.10.11, an NVIDIA
A800-SXM4-40GB, bfloat16, CUDA 12.4, vLLM 0.6.6.post1, PyTorch 2.5.1, and
Transformers 4.47.1. Full dependencies are pinned in
`pipeline/requirements.txt`. Download the three model checkpoints into
`models/` using the directory names in the configurations. Llama access may
require accepting its upstream license and authenticating.

```bash
python -m pip install -r pipeline/requirements.txt
python -m pytest -q
cd pipeline
bash run_three_seed_core.sh smoke
bash run_three_seed_core.sh full
bash run_three_seed_core.sh expanded
TEMPERATURE_SEEDS="42 314159 271828" bash run_temperature_sensitivity.sh full
bash run_r1_three_arm.sh full
```

Qwen/Llama primary runs use 768/1536-token caps, eight chains, and temperature
0.75. R1 uses 2048/4096 tokens, four chains, and temperature 0.6. Sampled runs
use top-p 0.95. Greedy audits use one chain, temperature 0, and top-p 1.
Core seeds are 42, 314159, and 271828; full-split Qwen/Llama runs use seed 42.
Each primary driver generates independent A/C endpoints, reconstructs B,
and runs its associated analyses. Additional analyses, including routing,
are separate entry points described in [pipeline/README.md](pipeline/README.md).

Completed records are reused on resume. Request seeds combine a
budget-specific seed, problem identifier, and chain index; A and C use distinct
budget offsets. Stored token identifiers establish B/C prefix identity.
Independent greedy audits check decoded text only. Deterministic request seeds
do not guarantee bitwise reproduction across hardware or software versions.

## Data

| Snapshot | Path | Contents |
| --- | --- | --- |
| Core | `data/all_problems.jsonl` | 400 GSM8K + 400 MATH test problems |
| Full | `data/expanded/all_problems.jsonl` | 1,319 GSM8K + 5,000 MATH test problems |
| AIME | `data/aime/all_problems.jsonl` | 60 problems from AIME 2024 and 2025 |

Sources are [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[MATH](https://huggingface.co/datasets/SuperSecureHuman/competition_math_hf_dataset),
[AIME 2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024), and
[AIME 2025](https://huggingface.co/datasets/yentinglin/aime_2025).
Records contain `problem_id`, `dataset`, `source`, `source_split`, `question`,
`answer`, and `math_level`. Values 0 for GSM8K and 6 for AIME are dataset
placeholders, not calibrated difficulty levels comparable to MATH levels 1–5.

Core sampling uses seed 42, uniformly for GSM8K and with 80 MATH problems per
difficulty level. AIME answers are normalized to boxed integers from 0 to 999.
The generation stage applies the model's zero-shot chat template. No language
model is trained or fine-tuned; the offline routing classifiers are fitted.
To recreate the snapshots, run from `pipeline/`:

```bash
python 01_download_data.py --config config.yaml
python 01_download_data.py --config config.yaml --data-dir ../data/expanded --full
python 09_download_aime.py
```

## Checksums and citation

`SHA256SUMS` covers the files in this GitHub snapshot, excluding itself and
Git metadata. On systems with GNU coreutils, verify it from the repository root:

```bash
sha256sum -c SHA256SUMS
```

The older `outputs/**/reproducibility_manifest.*` files describe historical
experiment artifacts, including raw files omitted here and source files that
have since been edited. They are retained as provenance, not as current
checkout checksums. Historical paths can refer to the previous manuscript
layout. Do not overwrite these manifests merely to match this compact release.

The frozen experiment release is archived at
[Zenodo, version DOI 10.5281/zenodo.22319436](https://doi.org/10.5281/zenodo.22319436).
The [concept DOI](https://doi.org/10.5281/zenodo.22172646) points to the latest
archived version. This GitHub documentation and figure update does not modify
that deposit or assign a new release DOI. Citation metadata is in `CITATION.cff`.
Please also cite the original benchmarks when using their data.

Code is released under the MIT License. Benchmark snapshots remain subject to
their upstream licenses; see [LICENSE](LICENSE). Reproduction issues and bug
reports are welcome. Corrections affecting reported results should be documented
and archived as a new version, preserving the original experiment evidence.
