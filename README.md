# Sim-to-Real Accuracy Validation

Compare 7 inference-serving simulators against ground-truth latency data collected from vLLM, producing MAPE/MPE error tables, CSV exports, and publication figures.

## Simulators

| Adapter | Type | Description |
|---------|------|-------------|
| `blis-blackbox` | Subprocess (Go) | BLIS with trained alpha/beta regression coefficients per model |
| `blis-roofline` | Subprocess (Go) | BLIS with hardware roofline latency model |
| `blis-crossmodel` | Subprocess (Go) | BLIS with globally-fitted cross-model coefficients |
| `blis-trained-roofline` | Subprocess (Go) | BLIS with trained roofline coefficients |
| `vidur` | Subprocess (Python) | Discrete-event simulator with vLLM scheduler emulation |
| `llm-optimizer-estimate` | In-process (Python) | Analytical roofline estimator from llm-optimizer |
| `aiconfigurator-estimate` | In-process (Python) | Analytical estimator from AIConfigurator SDK |

## Project Structure

```
sim-to-real-accuracy-validation/
├── experiment/                 # Core Python package
│   ├── data_model.py           # Dataclasses (Experiment, StageMetrics, SimulatorResult, etc.)
│   ├── ground_truth.py         # Discover and parse ground-truth experiment directories
│   ├── kv_cache_extractor.py   # Extract KV cache block counts from vllm.log / kv_events.jsonl
│   ├── trace_converter.py      # Convert per-request JSON → BLIS trace (header YAML + CSV)
│   ├── vidur_trace_converter.py# Convert per-request JSON → Vidur trace CSV
│   ├── metrics.py              # MAPE, MPE, absolute error computation
│   ├── report.py               # Formatted tables and CSV export
│   ├── run.py                  # Pipeline orchestrator (CLI entry point)
│   ├── figures.py              # Publication figures (independent CLI entry point)
│   └── adapters/
│       ├── base.py             # SimulatorAdapter ABC + shared BLIS logic
│       ├── blis_blackbox.py
│       ├── blis_roofline.py
│       ├── blis_crossmodel.py
│       ├── blis_trained_roofline.py
│       ├── vidur.py
│       ├── llm_optimizer_est.py
│       └── aiconfigurator_est.py
├── tests/                      # Unit + integration tests (pytest)
├── vllm_data/ground_truth/     # 16 ground-truth experiment directories (not tracked, see below)
├── inference-sim -> ../inference-sim   # Symlink to BLIS simulator repo
├── vidur -> ../vidur                   # Symlink to Vidur simulator repo
└── llm-optimizer -> ../llm-optimizer   # Symlink to LLM optimizer repo
```

## Prerequisites

- **Python** >= 3.10
- **Go** >= 1.21 (for building BLIS)
- **Internet access** (llm-optimizer downloads model configs from HuggingFace Hub)
- Local clones of: [inference-sim](https://github.com/inference-sim/inference-sim), [vidur](https://github.com/microsoft/vidur), [llm-optimizer](https://github.com/inference-sim/llm-optimizer)

## Setup

### 1. Clone dependency repos and create symlinks

This repo expects `inference-sim`, `llm-optimizer`, and `vidur` to be symlinks to their respective local clones. The experiments in this repo were run against the following commits:

| Repo | Commit | Description |
|------|--------|-------------|
| [inference-sim](https://github.com/inference-sim/inference-sim) | `b05154c` | hypothesis(H30-H32): BLIS replay vs real vLLM — three-way crossmodel validation |
| [llm-optimizer](https://github.com/bentoml/llm-optimizer) | `bb82d22` | feat: add support for max workers |
| [vidur](https://github.com/microsoft/vidur) | `8383d29` | [Bugfix]: Revert scheduler regression and introduce canary branch |

Clone at the pinned versions and create symlinks (assuming repos live under the same parent directory):

```bash
# Clone at pinned commits
git clone git@github.com:inference-sim/inference-sim.git ../inference-sim && git -C ../inference-sim checkout b05154c
git clone git@github.com:bentoml/llm-optimizer.git ../llm-optimizer && git -C ../llm-optimizer checkout bb82d22
git clone git@github.com:microsoft/vidur.git ../vidur && git -C ../vidur checkout 8383d29

# Symlink into this repo
ln -s ../inference-sim inference-sim
ln -s ../llm-optimizer llm-optimizer
ln -s ../vidur vidur
```

### 2. Ground truth data

Ground truth data collected from vLLM must be placed under `vllm_data/ground_truth/`. This directory is not tracked by git due to its size. Each experiment directory should follow the naming convention `YYYYMMDD-HHMMSS-*-tp<N>-<workload>` and contain the files described in the [Ground-Truth Data](#ground-truth-data) section below.

```bash
mkdir -p vllm_data/ground_truth
# Copy or symlink your experiment directories here
```

### 3. Build the BLIS binary

```bash
cd inference-sim
go build -o blis main.go
cd ..
```

### 4. Install Python dependencies

```bash
pip install numpy pyyaml pandas matplotlib   # experiment package deps (pandas/matplotlib for figures)
pip install -e vidur/                         # Vidur simulator
pip install -e llm-optimizer/                 # LLM optimizer estimator
pip install aiconfigurator                    # AIConfigurator SDK
```

### 5. (Optional) HuggingFace authentication

The `llm-optimizer-estimate` adapter uses `huggingface_hub` to download model `config.json` files. If models are gated:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
```

## Usage

### Run all simulators

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --blis-binary inference-sim/blis \
  --vidur-dir vidur \
  --output-dir results
```

### Run only the non-BLIS adapters

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters vidur llm-optimizer-estimate aiconfigurator-estimate
```

### Run a subset of adapters

```bash
python -m experiment.run --adapters blis-roofline vidur
```

### Generate publication figures

Figures are generated independently from the pipeline, reading the CSVs it produces:

```bash
# Basic — reads error_records.csv and runtime.csv from results/
python -m experiment.figures --results-dir results

# With metadata enrichment (adds hardware/config breakdowns)
python -m experiment.figures --results-dir results --metadata experiment_metadata.csv

# Custom output directory
python -m experiment.figures --results-dir results --output-dir results/figures
```

This produces 5 PDF figures and 1 LaTeX table under `results/figures/`.

### Pipeline CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `vllm_data/ground_truth` | Directory containing ground-truth experiment folders |
| `--blis-binary` | `inference-sim/blis` | Path to compiled BLIS binary |
| `--vidur-dir` | `vidur` | Path to cloned Vidur repository |
| `--output-dir` | `results` | Where reports and CSV are saved |
| `--adapters` | all 7 | Space-separated list of adapters to run |

### Figures CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir` | `results` | Directory containing `error_records.csv` and `runtime.csv` |
| `--output-dir` | `results/figures` | Where figures are saved |
| `--metadata` | *(none)* | Path to `experiment_metadata.csv` for hardware/config enrichment |

Valid adapter names: `blis-blackbox`, `blis-roofline`, `blis-crossmodel`, `blis-trained-roofline`, `vidur`, `llm-optimizer-estimate`, `aiconfigurator-estimate`.

## Pipeline

The orchestrator (`experiment.run`) executes this sequence:

1. **Discover** — load `experiments.json` from `--data-dir` and resolve each entry to its directory
2. **Parse** — load each experiment's configs, metrics, and KV cache data into `Experiment` dataclasses
3. **Run** — for each (experiment, adapter) pair, check `adapter.can_run()`, then `adapter.run()` to produce a `SimulatorResult`
4. **Compare** — compute MAPE, MPE, and absolute error across 9 latency metrics (e2e/ttft/itl × mean/p90/p99)
5. **Report** — print formatted tables to stdout and save `error_records.csv`

Failures at any step are logged and skipped — the pipeline does not abort on individual errors.

## Adapter Compatibility

Not every adapter can run every experiment. The `can_run()` method filters incompatible pairs, and the pipeline skips them automatically. See [docs/simulator-limitations.md](docs/simulator-limitations.md) for full details.

| Adapter | Key filters | Coverage (49 experiments) |
|---------|-------------|--------------------------|
| `blis-blackbox` | Model must have coefficients in `inference-sim/defaults.yaml` | Varies |
| `blis-roofline` | Always runs | All 49 |
| `blis-crossmodel` | Always runs | All 49 |
| `blis-trained-roofline` | Model must have trained coefficients | Varies |
| `vidur` | 3 pre-profiled models, H100/A100 only, no FP8 | ~9 |
| `llm-optimizer-estimate` | H100/A100, `shared_prefix` workloads, no Llama-4-Scout | ~40 |
| `aiconfigurator-estimate` | H100 only, dense models, `shared_prefix` workloads | ~20 |

## Output

### Pipeline output (`results/`)

- **`error_records.csv`** — one row per (simulator, experiment, stage, metric) with columns: `simulator`, `experiment_folder`, `model`, `workload`, `stage_index`, `metric_name`, `predicted`, `actual`, `mape`, `mpe`, `absolute_error`, plus metadata (`exp_id`, `hardware`, `dp`, `cpu_offload`, `gpu_mem_util`, `precision`, `config_tag`)
- **`runtime.csv`** — one row per (simulator, experiment) with wall-clock time and metadata
- **Stdout tables** — MAPE by simulator, MAPE by model, MAPE by workload, MPE by simulator (signed), runtime summary

### Figures output (`results/figures/`)

Generated separately via `python -m experiment.figures`:

- **`fig1_model_sensitivity.pdf`** — MAPE by model across simulators
- **`fig2_hardware_portability.pdf`** — MAPE by hardware platform
- **`fig3_workload_sensitivity.pdf`** — MAPE by workload type
- **`fig4a_config_dense.pdf`** / **`fig4b_config_moe.pdf`** — Config sensitivity (mbt, cpu-offload, gpu-mem, dp)
- **`fig5_pareto.pdf`** — Accuracy vs. runtime Pareto frontier
- **`table1_runtime.tex`** — LaTeX runtime comparison table

## Ground-Truth Data

Experiments are discovered via `experiments.json` (a manifest file in `vllm_data/ground_truth/`). Each entry maps an experiment ID to its metadata (hardware, precision, dp, etc.). Directories are named `<id>-<slug>` and resolved by prefix matching.

Each experiment directory contains:

| File | Purpose |
|------|---------|
| `exp-config.yaml` | Model name, TP degree, scheduler limits |
| `profile.yaml` | Load stages (rate, duration), data type config |
| `vllm.log` | GPU KV cache block count |
| `kv_events.jsonl` | CPU KV cache offloading events |
| `results/summary_lifecycle_metrics.json` | Aggregate latency and throughput |
| `results/stage_N_lifecycle_metrics.json` | Per-stage latency and throughput |
| `results/per_request_lifecycle_metrics.json` | Per-request timings (used for trace replay) |

The perf data directory is auto-detected: `results/` is preferred, with `inference-perf-data/` as a legacy fallback.

## Tests

```bash
# Run all tests (unit + integration)
python -m pytest tests/ -v

# Run only unit tests (no ground-truth data needed)
python -m pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires vllm_data/ground_truth/)
python -m pytest tests/test_integration.py -v
```

Integration tests are automatically skipped if `vllm_data/ground_truth/` is not present.
