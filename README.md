# Sim-to-Real Accuracy Validation

Compare 5 inference-serving simulators against ground-truth latency data collected from vLLM, producing MAPE/MPE error tables and a flat CSV for further analysis.

## Simulators

| Adapter | Type | Description |
|---------|------|-------------|
| `blis-blackbox` | Subprocess (Go) | BLIS with trained alpha/beta regression coefficients per model |
| `blis-roofline` | Subprocess (Go) | BLIS with hardware roofline latency model |
| `blis-crossmodel` | Subprocess (Go) | BLIS with globally-fitted cross-model coefficients |
| `vidur` | Subprocess (Python) | Discrete-event simulator with vLLM scheduler emulation |
| `llm-optimizer-estimate` | In-process (Python) | Analytical roofline estimator from llm-optimizer |

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
│   ├── run.py                  # Orchestrator (CLI entry point)
│   └── adapters/
│       ├── base.py             # SimulatorAdapter ABC + shared BLIS logic
│       ├── blis_blackbox.py
│       ├── blis_roofline.py
│       ├── blis_crossmodel.py
│       ├── vidur.py
│       └── llm_optimizer_est.py
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

This repo expects `inference-sim`, `llm-optimizer`, and `vidur` to be symlinks to their respective local clones. For example, if all repos live under the same parent directory:

```bash
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
pip install numpy pyyaml            # experiment package deps
pip install -e vidur/                # Vidur simulator
pip install -e llm-optimizer/        # LLM optimizer estimator
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

### Run a subset of adapters

```bash
python -m experiment.run --adapters blis-roofline vidur
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `vllm_data/ground_truth` | Directory containing ground-truth experiment folders |
| `--blis-binary` | `inference-sim/blis` | Path to compiled BLIS binary |
| `--vidur-dir` | `vidur` | Path to cloned Vidur repository |
| `--output-dir` | `results` | Where reports and CSV are saved |
| `--adapters` | all 5 | Space-separated list of adapters to run |

Valid adapter names: `blis-blackbox`, `blis-roofline`, `blis-crossmodel`, `vidur`, `llm-optimizer-estimate`.

## Pipeline

The orchestrator (`experiment.run`) executes this sequence:

1. **Discover** — scan `--data-dir` for experiment directories matching `YYYYMMDD-HHMMSS-*-tp<N>-<workload>`
2. **Parse** — load each experiment's configs, metrics, and KV cache data into `Experiment` dataclasses
3. **Run** — for each (experiment, adapter) pair, check `adapter.can_run()`, then `adapter.run()` to produce a `SimulatorResult`
4. **Compare** — compute MAPE, MPE, and absolute error across 9 latency metrics (e2e/ttft/itl × mean/p90/p99)
5. **Report** — print formatted tables to stdout and save `error_records.csv`

Failures at any step are logged and skipped — the pipeline does not abort on individual errors.

## Adapter Compatibility

Not every adapter can run every experiment. The `can_run()` method filters:

| Adapter | Filter | Expected coverage (16 experiments) |
|---------|--------|------------------------------------|
| `blis-blackbox` | Model must have coefficients in `inference-sim/defaults.yaml` | Low — current defaults target newer models (Llama-3.x) |
| `blis-roofline` | Always runs | All 16 |
| `blis-crossmodel` | Always runs | All 16 |
| `vidur` | Model must be Llama-2-7b, Llama-2-70b, or CodeLlama-34b | 12 of 16 (excludes Mixtral) |
| `llm-optimizer-estimate` | Workload must be `shared_prefix` type with `question_len`, `system_prompt_len`, `output_len` | All 16 |

## Output

Results are written to `--output-dir` (default: `results/`):

- **`error_records.csv`** — one row per (simulator, experiment, stage, metric) with columns: `simulator`, `experiment_folder`, `model`, `workload`, `stage_index`, `metric_name`, `predicted`, `actual`, `mape`, `mpe`, `absolute_error`
- **Stdout tables** — MAPE by simulator, MAPE by model, MAPE by workload, MPE by simulator (signed)

## Ground-Truth Data

Each experiment directory under `vllm_data/ground_truth/` contains:

| File | Purpose |
|------|---------|
| `exp-config.yaml` | Model name, TP degree, scheduler limits |
| `profile.yaml` | Load stages (rate, duration), data type config |
| `vllm.log` | GPU KV cache block count |
| `kv_events.jsonl` | CPU KV cache offloading events |
| `inference-perf-data/summary_lifecycle_metrics.json` | Aggregate latency and throughput |
| `inference-perf-data/stage_N_lifecycle_metrics.json` | Per-stage latency and throughput |
| `inference-perf-data/per_request_lifecycle_metrics.json` | Per-request timings (used for trace replay) |

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
