# Pipeline: Inference and Scoring Workspace

Numbered steps 00–08 plus run configs. GPU steps run from this directory;
paths in the configs point at the repo-root `../data`, `../models`, and
`../outputs/*` directories. Analysis lives in `../analysis/` and resolves
its defaults relative to this layout — do not move the two directories
apart.

## Current focused experiment

```bash
cd pipeline
bash run_three_seed_core.sh smoke
bash run_three_seed_core.sh full
python 01_download_data.py --config config.yaml \
  --data-dir ../data/expanded --full
bash run_three_seed_core.sh expanded
```

This runs the Qwen/Llama three-seed exact-prefix study, validates every
artifact, aggregates problem-clustered uncertainty, and refreshes the
reproducibility manifest.

The focused run executes, resumably and in order:

1. twelve sampled endpoints: two models × three seeds at both 768 and 1536
   tokens, temperature 0.75, eight chains;
2. exact 768-token reconstruction from stored token IDs;
3. endpoint/repair analyses and structural validation;
4. repeated-seed aggregation with problem-clustered uncertainty.

The optional `expanded` mode uses the complete configured GSM8K and MATH
test splits for one seed. It complements, rather than replaces, the
three-seed 800-problem run.

Sampled inference uses `deterministic_sampling: true`: each problem and
chain is issued as a separate request with a stable SHA-256-derived seed.
Artifacts record both the derivation policy and each chain's request seed.
This removes request-order and batching from the random-number assignment;
GPU kernels can still prevent a promise of bitwise identity across different
hardware or library versions.

For a token-only budget comparison, generate the longest endpoint once and
reconstruct the lower cap from stored token IDs:

```bash
python ../analysis/47_reconstruct_lower_cap.py \
  --results-dir ../outputs/new_run --source-name xlong_sample8 \
  --output-name long_sample8_paired --cap 768 \
  --tokenizer ../models/Qwen2.5-Math-7B-Instruct
```

This is both cheaper and methodologically cleaner than independently
resampling the two token caps. Independent same-cap reruns should be kept as
a separate resampling analysis.

For the focused three-seed Qwen/Llama experiment, first run the eight-problem
GPU smoke test and inspect its validation result, then start the full run:

```bash
cd pipeline
bash run_three_seed_core.sh smoke
bash run_three_seed_core.sh full
```

The seeds are 42, 314159, and 271828. For each model and seed, the script
generates an independent 768-token arm A and a 1536-token arm C, then
reconstructs arm B as the exact 768-token prefix of C. A→B measures
same-budget resampling, while the B→C contrast isolates
the incremental token-cap effect on the same trajectories. CPU repair and
problem-clustered aggregate reports are produced automatically under
`outputs/three_arm_core`.

Existing traces imply about 29.6 million generated tokens for all twelve
generated endpoints. On the previously used GPU, reserve 3.5--4 hours for
generation, smoke tests, analysis, and contingencies.

## Configs

| File | Purpose |
|------|---------|
| `config.yaml` | Main Qwen2.5-Math-7B run |
| `config_llama_validation.yaml` | Llama-3.1-8B second-model validation |
| `config_r1_distill.yaml` | Reasoning-tuned model contrast |
| `config_aime_qwen.yaml` / `config_aime_r1.yaml` | AIME hard-task validation |

## Server setup

```bash
pip install -r requirements.txt
huggingface-cli login        # needed for gated meta-llama weights
```

Download model weights into `../models/` (names must match the config
paths): `Qwen2.5-Math-7B-Instruct`, `Llama-3.1-8B-Instruct`,
`DeepSeek-R1-Distill-Qwen-7B`. The PRM model is only needed to reproduce
the PRM scoring steps.

Hardware: single 40GB GPU (A100/A800) recommended; 24GB works with lower
throughput. The focused batch includes twelve 800-example, eight-chain
endpoints: six at 768 tokens and six at 1536; budget a full rental session
including setup.

## Troubleshooting

- **OOM**: lower `batch_size` / `gpu_memory_utilization` in the config.
- **MATH dataset download**: `pip install datasets==2.18.0`; the config
  lists three HF fallback dataset ids. Normally unnecessary — the sampled
  problems are committed under `../data/`.
- **Windows console encoding** (local runs): `set PYTHONIOENCODING=utf-8`.
