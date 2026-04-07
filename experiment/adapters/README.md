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

The BLIS Evolved adapter (`blis-evolved`) uses the `evolved` latency backend, which combines roofline basis functions with learned correction terms. The latest iteration (iter26) achieves **37.42% overall MAPE** (TTFT: 24.34%, E2E: 13.09%) across 15 experiments on H100/FP16, with physics-based TP All-Reduce modeling.

Unlike the blackbox adapter, the evolved adapter does not require per-model profiled coefficients in `defaults.yaml`. Instead, it passes static cross-model alpha and beta coefficients on the command line, making it applicable to any model.

### Architecture

The evolved backend uses a **10-coefficient physics-informed formula** with prefill/decode compute/memory split:

**Alpha coefficients (3 values):**
| Index | Name | Semantics |
|-------|------|-----------|
| `α₀` | QueueingTime | Fixed API overhead (~15.6ms per request) |
| `α₁` | PostDecodeFixedOverhead | Per-request completion overhead (~0.8ms) |
| `α₂` | OutputTokenProcessingTime | Per-output-token streaming cost (µs/token) |

**Beta coefficients (10 values):**
| Index | Name | iter24 | iter26 | Semantics |
|-------|------|--------|--------|-----------|
| `β₁ₐ` | Prefill compute correction | 0.139 | 0.139 | FlashAttention reduces effective FLOPs by 7.2× |
| `β₂ₐ` | Decode compute correction | **0.0** | **0.0** | **Dropped — decode is memory-bound** |
| `β₃` | Weight loading correction | 1.363 | 1.363 | 36% overhead above roofline weight bandwidth |
| `β₄` | TP communication correction | 0.396 | **0.410** | **TP All-Reduce activated (iter26)** |
| `β₅` | Per-layer overhead | 62.3 µs | **49.6 µs** | **Decreased after TP term activation** |
| `β₆` | Per-request scheduling | 2.8 µs/req | 2.8 µs/req | Per-request scheduling in batch |
| `β₇` | Per-step constant | 169.4 µs/step | 169.4 µs/step | Fixed per-step dispatch overhead |
| `β₈` | Per-MoE-layer overhead | 427.3 µs | 427.3 µs | Router + permutation + EP communication |
| `β₁ᵦ` | Prefill memory correction | **0.0** | **0.0** | **Dropped — prefill is compute-bound** |
| `β₂ᵦ` | Decode memory correction | 1.263 | 1.263 | 26% overhead above roofline memory bandwidth |

**Key insights:**
- **Iter24**: Clean physical split — prefill uses only compute (β₁ₐ), decode uses only memory (β₂ᵦ)
- **Iter26**: TP All-Reduce activated (β₄: 0.396 → 0.410), per-layer overhead decreased (β₅: 62.3 → 49.6 µs)

The coefficients are optimised via 2D grid search + golden section polish during BLIS training iterations.

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S (any hardware supported by BLIS)
- **Architectures**: MoE (e.g., Mixtral, Scout), GQA (e.g., Llama, Qwen), dense
- **Models**: Any model -- cross-model coefficients require no per-model profiling. Validated model families include Scout, Llama, Qwen, Yi, and Mistral.
- **Precision**: FP16, FP8
- **Features**: KV offloading, multi-stage workloads, shared-prefix workloads

### Usage

```bash
# Using iter26 coefficients (default, requires 10-beta BLIS with TP All-Reduce)
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary /path/to/blis

# Using iter24 coefficients (10-beta BLIS)
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary /path/to/blis \
  --blis-evolved-iteration 24

# Using iter16 coefficients (7-beta BLIS)
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary /path/to/blis-iter16 \
  --blis-evolved-iteration 16
```

### Requirements

- BLIS binary compiled with the `evolved` latency backend
- The binary must support the following CLI flags:
  - `--latency-model evolved`
  - `--alpha-coeffs <comma-separated>` (3 values)
  - `--beta-coeffs <comma-separated>` (7 or 10 values)

**Iteration requirements:**
- **Iter16** (7 betas): Requires BLIS that supports 7-beta mode (60.19% MAPE)
- **Iter24** (10 betas): Requires BLIS with decode-split support (10-beta mode, 39.18% MAPE)
- **Iter26** (10 betas): Requires BLIS with TP All-Reduce support (10-beta mode, 37.42% MAPE)

> **Note**: Use `--blis-evolved-iteration` to select which coefficient set to use (default: 26)

### How It Works

1. **Eligibility**: `can_run()` returns `True` for all experiments (cross-model coefficients)
2. **Workload Generation**: Converts experiment profile config into a BLIS `WorkloadSpec` YAML (inference_perf format with stages, token distributions, and shared-prefix settings)
3. **Coefficient Injection**: Formats the 3 alpha and 10 beta iter24 coefficients as comma-separated strings with 6 decimal places and passes them via `--alpha-coeffs` and `--beta-coeffs`
4. **Execution**: Runs the BLIS binary via subprocess with `--latency-model evolved` and all common flags (model, tp, hardware, KV offloading parameters, seed)
5. **Parsing**: Reads the JSON results file and constructs a `SimulatorResult` with per-stage `StageMetrics` (E2E, TTFT, ITL latency distributions and throughput)

### Expected Accuracy

Based on iter26 training results across 15 H100/FP16 experiments:

- **Overall MAPE**: 37.42% (TTFT: 24.34%, E2E: 13.09%)
- **Improvement over iter24**: -1.76 points (-4.5% relative)
- **Best case**: Yi general-lite (TTFT: 2.7%, E2E: ~14%)
- **Worst case**: Scout reasoning-lite (TTFT: ~58%, E2E: ~11% — long 934-token prefill)
- **Typical**: 12 of 15 experiments below 30% TTFT MAPE

**Training journey** (iter16 → iter26):
- Iter16: 60.19% MAPE (trained-roofline architecture)
- Iter20: 40.58% MAPE (β₈·nMoELayers breakthrough — 19.5pt improvement)
- Iter21: 39.86% MAPE (prefill compute-only split)
- Iter24: 39.18% MAPE (decode memory-only split)
- Iter26: 37.42% MAPE (TP All-Reduce activation)

Total improvement: **37.8% relative reduction** (60.19% → 37.42%)

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `BLIS evolved failed (rc=1)` | Binary does not support `--latency-model evolved` | Rebuild BLIS with evolved backend enabled |
| `Error: invalid beta-coeffs format` / `expected 10 beta coefficients, got N` | Wrong number of beta coefficients (iter24 requires 10, not 7 or 9) | Update BLIS to version with decode-split support (10-beta mode) |
| `Error: invalid alpha-coeffs format` / `expected 3 alpha coefficients, got N` | Wrong number of alpha coefficients or malformed comma-separated values | Verify alpha has exactly 3 comma-separated float values |
| `--alpha-coeffs: unrecognized argument` | BLIS binary version too old | Update to a BLIS version that supports the evolved latency model CLI flags |

## BLIS Trained-Physics Adapter

### Overview

The BLIS Trained-Physics adapter (`blis-trained-physics`) uses `--latency-model trained-physics` with globally-fitted roofline basis functions and architecture-aware corrections. Unlike the evolved adapter which passes coefficients on the command line, the trained-physics model loads its 13 coefficients (10 beta + 3 alpha) from `defaults.yaml` automatically via the BLIS binary.

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
