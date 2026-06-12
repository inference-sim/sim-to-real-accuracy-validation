# Experiment Adapters

This directory contains simulator adapters that wrap different LLM serving simulators behind a common interface (`SimulatorAdapter`). Each adapter translates ground-truth experiment configurations into simulator-specific inputs, runs the simulator, and parses its output into standardised `SimulatorResult` objects.

## LLMServingSim Adapter

### Overview

The LLMServingSim adapter (v1.1.0) validates LLMServingSim's prediction accuracy against real vLLM experiments using cycle-accurate simulation.

### Supported Configurations

- **Hardware**: H100 only
- **Models**:
  - Llama-3.1-8B (tp1, tp2)
  - Mixtral-8x7B-v0.1 (tp1, tp2, tp4)
- **Precision**: FP16 only
- **Features**: CPU offloading, multi-instance (dp>1)

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters llmservingsim \
  --llmservingsim-dir LLMServingSim
```

### Requirements

- LLMServingSim v1.1.0 installed at `LLMServingSim/` (or custom path via `--llmservingsim-dir`)
- Profiler data at `profiler/perf/H100/{model}/bf16/tp{N}/dense.csv` + `attention.csv`
- Ground-truth data with `per_request_lifecycle_metrics.json`

### How It Works

1. **Eligibility**: Checks H100 hardware, model support, profiler data existence, FP16 precision
2. **Config Generation**: Creates cluster config YAML with `tp_size`, `num_npus`, memory settings
3. **Workload Generation**: Reads ground-truth token counts, generates constant-rate arrivals
4. **Execution**: Runs `python3 -m serving` via subprocess with temp configs
5. **Parsing**: Converts CSV output (ns) to SimulatorResult (ms) with nested LatencyDistribution/ThroughputMetrics

## BLIS Trained-Physics Adapter

### Overview

The BLIS Trained-Physics adapter (`blis-trained-physics`) uses `--latency-model trained-physics` with globally-fitted roofline basis functions and architecture-aware corrections. The trained-physics model loads its 13 coefficients (10 beta + 3 alpha) from `defaults.yaml` automatically via the BLIS binary — no command-line coefficient injection needed.

This adapter generalizes across model architectures, workloads, and TP configurations without per-model calibration. It is the recommended adapter for new models.

### Architecture

The trained-physics model uses **13 coefficients** (10 beta + 3 alpha):

**Beta coefficients (10 values):**
- Prefill compute/memory split
- Decode compute/memory split
- Weight loading correction
- TP communication correction
- Per-layer overhead
- Per-request batch overhead
- Per-step constant overhead
- MoE-layer overhead (architecture-aware)

**Alpha coefficients (3 values):**
- API queueing overhead
- Post-decode fixed overhead
- Per-token output processing overhead

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S (any hardware supported by BLIS)
- **Architectures**: MoE (e.g., Mixtral, Scout), GQA (e.g., Llama, Qwen), dense
- **Models**: Any model -- cross-model coefficients require no per-model profiling
- **Precision**: FP16, FP8
- **Features**: KV offloading, multi-stage workloads, shared-prefix workloads

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-trained-physics \
  --blis-binary /path/to/blis
```

### Requirements

- BLIS binary compiled with the `trained-physics` latency backend
- The binary must support the `--latency-model trained-physics` CLI flag
- Coefficients are loaded from `defaults.yaml` by the BLIS binary (no command-line coefficient injection needed)

### How It Works

1. **Eligibility**: `can_run()` returns `True` for all experiments (cross-model, no per-model profiling)
2. **Workload Generation**: Converts experiment profile config into a BLIS `WorkloadSpec` YAML
3. **Execution**: Runs the BLIS binary via subprocess with `--latency-model trained-physics` and all common flags (model, tp, hardware, KV offloading parameters, seed)
4. **Parsing**: Reads the JSON results file and constructs a `SimulatorResult` with per-stage `StageMetrics`

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `BLIS trained-physics failed (rc=1)` | Binary does not support `--latency-model trained-physics` | Rebuild BLIS with trained-physics backend enabled |
| Missing `defaults.yaml` coefficients | BLIS cannot find the trained-physics coefficients file | Ensure `defaults.yaml` is present in the BLIS directory with trained-physics coefficients |

## AIConfigurator Adapter

### Overview

The AIConfigurator adapter (`aiconfigurator-estimate`) uses the AIConfigurator SDK (v0.9.0) to produce analytical latency predictions. It calls `TaskConfig` + `TaskRunner` once per experiment, which returns a Pareto DataFrame with predictions across multiple concurrency levels. For each stage, it matches the arrival rate to predicted throughput to find the equilibrium concurrency.

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S
- **Architectures**: Dense and MoE (MoE uses HYBRID database mode)
- **Precision**: FP16 (bfloat16 profile), FP8 (fp8 profile)
- **Workloads**: `shared_prefix` type only (requires question_len, system_prompt_len, output_len)

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters aiconfigurator-estimate
```

### Requirements

- `aiconfigurator==0.9.0` installed via pip
- No external binary or data files needed (SDK includes profiling data)

### How It Works

1. **Eligibility**: Checks hardware in supported set, precision FP16/FP8, shared_prefix data type
2. **Config**: Maps hardware to system_name (`h100_sxm`, `a100_sxm`, `l40s`), detects MoE for HYBRID mode
3. **Execution**: Runs `TaskRunner().run(task_config)` to get Pareto DataFrame
4. **Throughput Matching**: For each stage, finds the row where predicted `seq/s` ≈ stage arrival rate
5. **Latency Derivation**: E2E = TTFT + TPOT × (output_length - 1); only mean predictions (no P90/P99)
