# Experiment Adapters

This directory contains simulator adapters that wrap different LLM serving simulators behind a common interface (`SimulatorAdapter`). Each adapter translates ground-truth experiment configurations into simulator-specific inputs, runs the simulator, and parses its output into standardised `SimulatorResult` objects.

## LLMServingSim Adapter

### Overview

The LLMServingSim adapter validates LLMServingSim's prediction accuracy against real vLLM experiments.

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

- LLMServingSim installed at `LLMServingSim/` (or custom path via `--llmservingsim-dir`)
- Performance models at `llm_profile/perf_models/H100/{model}/tp{N}/`
- Ground-truth data with `per_request_lifecycle_metrics.json`

### How It Works

1. **Eligibility**: Checks H100 hardware, model support, perf model existence, FP16 precision
2. **Config Generation**: Creates cluster_config JSON with TP, memory, instance settings
3. **Workload Generation**: Reads ground-truth token counts, generates constant-rate arrivals
4. **Execution**: Runs LLMServingSim via subprocess with temp configs
5. **Parsing**: Converts CSV output (ns) to SimulatorResult (ms) with nested LatencyDistribution/ThroughputMetrics

## BLIS Evolved Adapter

### Overview

The BLIS Evolved adapter (`blis-evolved`) uses the `evolved` latency backend, which combines roofline basis functions with learned correction terms. Coefficients were optimised during iter16 training, achieving **60.19% MAPE** across 15 experiments on H100/FP16.

Unlike the blackbox adapter, the evolved adapter does not require per-model profiled coefficients in `defaults.yaml`. Instead, it passes static cross-model alpha and beta coefficients on the command line, making it applicable to any model.

### Architecture

The evolved backend uses a **7-term physics-informed formula** with two sets of coefficients:

**Alpha coefficients (3 values):**
| Index | Name | Semantics |
|-------|------|-----------|
| `alpha_0` | QueueingTime scale | Scales the queueing delay component |
| `alpha_1` | Prefill attention scale | Scales prefill attention latency |
| `alpha_2` | Decode attention scale | Scales decode attention latency |

**Beta coefficients (7 values):**
| Index | Name | Semantics |
|-------|------|-----------|
| `beta_0` | Prefill roofline correction | Multiplicative correction to prefill roofline estimate |
| `beta_1` | Prefill correction term 1 | Additive correction for prefill compute |
| `beta_2` | Prefill correction term 2 | Additive correction for prefill memory |
| `beta_3` | Decode roofline correction | Multiplicative correction to decode roofline estimate |
| `beta_4` | Decode correction term 1 | Additive correction for decode compute |
| `beta_5` | Decode correction term 2 | Additive correction for decode memory |
| `beta_6` | Scheduling overhead | Fixed per-iteration scheduling cost |

The coefficients are optimised via differential evolution (inner loop) during BLIS training iterations.

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S (any hardware supported by BLIS)
- **Models**: Any model (cross-model coefficients; no per-model profiling needed)
- **Precision**: FP16
- **Features**: KV offloading, multi-stage workloads, shared-prefix workloads

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary /path/to/blis
```

### Requirements

- BLIS binary compiled with the `evolved` latency backend
- The binary must support the following CLI flags:
  - `--latency-model evolved`
  - `--alpha-coeffs <comma-separated>`
  - `--beta-coeffs <comma-separated>`

### How It Works

1. **Eligibility**: `can_run()` returns `True` for all experiments (cross-model coefficients)
2. **Workload Generation**: Converts experiment profile config into a BLIS `WorkloadSpec` YAML (inference_perf format with stages, token distributions, and shared-prefix settings)
3. **Coefficient Injection**: Formats the 3 alpha and 7 beta iter16 coefficients as comma-separated strings with 6 decimal places and passes them via `--alpha-coeffs` and `--beta-coeffs`
4. **Execution**: Runs the BLIS binary via subprocess with `--latency-model evolved` and all common flags (model, tp, hardware, KV offloading parameters, seed)
5. **Parsing**: Reads the JSON results file and constructs a `SimulatorResult` with per-stage `StageMetrics` (E2E, TTFT, ITL latency distributions and throughput)

### Expected Accuracy

Based on iter16 training results across 15 H100/FP16 experiments:

- **Overall MAPE**: 60.19%
- **Best case**: Low-contention single-stage workloads
- **Worst case**: High-rate multi-stage workloads with KV pressure
- **Typical**: E2E latency predictions within 40-80% MAPE depending on load

> **Note**: The evolved backend is under active development. Accuracy is expected to improve in future training iterations as the coefficient search space expands.

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `BLIS evolved failed (rc=1)` | Binary does not support `--latency-model evolved` | Rebuild BLIS with evolved backend enabled |
| `RuntimeError: BLIS evolved failed ... model not found` | BLIS binary cannot locate hardware config for the specified model/hardware combination | Verify `hardware_config.json` exists for the target hardware |
| `--alpha-coeffs: unrecognized argument` | BLIS binary version too old | Update to a BLIS version that supports the evolved latency model CLI flags |
