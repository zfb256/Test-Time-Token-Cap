# Inference and scoring pipeline

Numbered steps 00–09 and experiment drivers run from this directory.
Configurations resolve data, model, and output paths relative to `pipeline/`.
Keep `pipeline/` and `analysis/` alongside each other.

## Setup and primary runs

From the repository root:

```bash
python -m pip install -r pipeline/requirements.txt
cd pipeline
```

Download `Qwen2.5-Math-7B-Instruct`, `Llama-3.1-8B-Instruct`, and
`DeepSeek-R1-Distill-Qwen-7B` into `../models/`. Authenticate with Hugging Face
if required for Llama. The reported runs used a 40 GB A800 GPU.

Then run from `pipeline/`:

```bash
bash run_three_seed_core.sh smoke
bash run_three_seed_core.sh full
bash run_three_seed_core.sh expanded
TEMPERATURE_SEEDS="42 314159 271828" bash run_temperature_sensitivity.sh full
bash run_r1_three_arm.sh full
```

| Driver/mode | Population and settings |
| --- | --- |
| `run_three_seed_core.sh smoke` | Eight problems, three seeds, Qwen and Llama; separate smoke output tree |
| `run_three_seed_core.sh full` | 800 core problems, three seeds, Qwen and Llama, 768/1536 tokens, eight chains, temperature 0.75 |
| `run_three_seed_core.sh expanded` | 6,319 full-split problems, seed 42, Qwen and Llama |
| Temperature driver with the three seeds above | Qwen/Llama temperature 0.6 and 1.0; the analysis reuses the primary 0.75 runs |
| `run_r1_three_arm.sh full` | 800 core problems, three seeds, R1, 2048/4096 tokens, four chains, temperature 0.6 |

Without a mode argument, the core and R1 drivers default to `smoke`.
The temperature driver defaults to seed 42; the command above explicitly
selects all three seeds required for the reported pooled analysis.
The primary drivers generate independent arms A and C, reconstruct B from C's
stored token identifiers, score outcomes, validate row counts, and write
manifests. Repeated runs reuse completed records. Request-specific seeds make
random-number assignment independent of request ordering; they do not guarantee
identical outputs across different kernels or hardware.

Frozen data are included. If the full snapshot is missing, recreate it with:

```bash
python 01_download_data.py --config config.yaml --data-dir ../data/expanded --full
```

## Additional CPU analyses

After restoring or regenerating the raw inference records and per-problem
detail tables, run from the repository root:

```bash
python analysis/60_equivalence_power.py
python analysis/61_three_arm_routing.py
python analysis/62_temperature_multiseed.py
python analysis/63_verifier_audit.py
```

Use each script's `--help` for its own input options; not all scripts accept
`--results-dir`. Script 63 re-scores C on fixed A-eligible core records. Its
numeric fallback does not reselect A or establish a complete relaxed B/C
comparison. The routing script uses all records and grouped out-of-fold scores.

Figures alone can be regenerated from the compact checkout; see the root
README. Full protocol validation needs the omitted raw files.

## Earlier experiments

`config.yaml`, `config_llama_validation.yaml`, `config_r1_distill.yaml`, and
the AIME configurations also support earlier independent-budget and greedy
experiments. These outputs remain separate from the request-seeded primary
runs. `config_r1_three_arm.yaml` supplies the primary R1 comparison.

The legacy PRM scoring and classifier pipeline remains available. Its PRM
checkpoint and LightGBM dependency are not used in the manuscript's primary
three-arm or current routing comparisons. Legacy file and column names can
contain `repair`; consult the eligibility definition before interpreting them
as wrong-to-right transitions.

## Troubleshooting

- For GPU out-of-memory errors, reduce `batch_size` or adjust
  `gpu_memory_utilization` in the configuration.
- Reuse the frozen data for comparison with reported results; a fresh upstream
  download can differ if the source changes.
- On Windows, set `PYTHONIOENCODING=utf-8` if console encoding fails.
