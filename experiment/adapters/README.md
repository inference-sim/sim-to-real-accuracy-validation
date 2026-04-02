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

The BLIS Evolved adapter (`blis-evolved`) uses the `evolved` latency backend, which combines roofline basis functions with learned correction terms. Coefficients were optimised during iter24 training, achieving **39.18% overall MAPE** (TTFT: 24.13%, E2E: 15.05%) across 15 experiments on H100/FP16.

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
| Index | Name | Value (iter24) | Semantics |
|-------|------|----------------|-----------|
| `β₁ₐ` | Prefill compute correction | 0.139 | FlashAttention reduces effective FLOPs by 7.2× |
| `β₂ₐ` | Decode compute correction | **0.0** | **Dropped — decode is memory-bound** |
| `β₃` | Weight loading correction | 1.363 | 36% overhead above roofline weight bandwidth |
| `β₄` | TP communication correction | 0.396 | TP cost partially absorbed into β₅·L |
| `β₅` | Per-layer overhead | 62.3 µs/layer | Kernel launch + layer norm per layer |
| `β₆` | Per-request scheduling | 2.8 µs/req | Per-request scheduling in batch |
| `β₇` | Per-step constant | 169.4 µs/step | Fixed per-step dispatch overhead |
| `β₈` | Per-MoE-layer overhead | 427.3 µs/MoE-layer | Router + permutation + EP communication |
| `β₁ᵦ` | Prefill memory correction | **0.0** | **Dropped — prefill is compute-bound** |
| `β₂ᵦ` | Decode memory correction | 1.263 | 26% overhead above roofline memory bandwidth |

**Key insight (iter24)**: Clean physical split discovered — prefill uses only compute (β₁ₐ), decode uses only memory (β₂ᵦ). The non-binding constraints (prefill memory, decode compute) are physically meaningful zeros.

The coefficients are optimised via 2D grid search + golden section polish during BLIS training iterations.

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S (any hardware supported by BLIS)
- **Architectures**: MoE (e.g., Mixtral, Scout), GQA (e.g., Llama, Qwen), dense
- **Models**: Any model -- cross-model coefficients require no per-model profiling. Validated model families include Scout, Llama, Qwen, Yi, and Mistral.
- **Precision**: FP16, FP8
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

- BLIS binary compiled with the `evolved` latency backend supporting **10-beta mode (decode split)**
- The binary must support the following CLI flags:
  - `--latency-model evolved`
  - `--alpha-coeffs <comma-separated>` (3 values)
  - `--beta-coeffs <comma-separated>` (10 values)

> **Note**: Iter24 requires BLIS with decode-split support. Earlier BLIS versions support only 7-9 betas and cannot use iter24 coefficients.

### How It Works

1. **Eligibility**: `can_run()` returns `True` for all experiments (cross-model coefficients)
2. **Workload Generation**: Converts experiment profile config into a BLIS `WorkloadSpec` YAML (inference_perf format with stages, token distributions, and shared-prefix settings)
3. **Coefficient Injection**: Formats the 3 alpha and 10 beta iter24 coefficients as comma-separated strings with 6 decimal places and passes them via `--alpha-coeffs` and `--beta-coeffs`
4. **Execution**: Runs the BLIS binary via subprocess with `--latency-model evolved` and all common flags (model, tp, hardware, KV offloading parameters, seed)
5. **Parsing**: Reads the JSON results file and constructs a `SimulatorResult` with per-stage `StageMetrics` (E2E, TTFT, ITL latency distributions and throughput)

### Expected Accuracy

Based on iter24 training results across 15 H100/FP16 experiments:

- **Overall MAPE**: 39.18% (TTFT: 24.13%, E2E: 15.05%)
- **Best case**: Yi general-lite (TTFT: 2.7%, E2E: 17.0%)
- **Worst case**: Scout reasoning-lite (TTFT: 60.3%, E2E: 11.8% — long 934-token prefill)
- **Typical**: 12 of 15 experiments below 30% TTFT MAPE

**Training journey** (iter16 → iter24):
- Iter16: 60.19% MAPE (trained-roofline architecture)
- Iter20: 40.58% MAPE (β₈·nMoELayers breakthrough — 19.5pt improvement)
- Iter21: 39.86% MAPE (prefill compute-only split)
- Iter24: 39.18% MAPE (decode memory-only split)

Total improvement: **34.9% relative reduction** (60.19% → 39.18%)

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `BLIS evolved failed (rc=1)` | Binary does not support `--latency-model evolved` | Rebuild BLIS with evolved backend enabled |
| `Error: invalid beta-coeffs format` / `expected 10 beta coefficients, got N` | Wrong number of beta coefficients (iter24 requires 10, not 7 or 9) | Update BLIS to version with decode-split support (10-beta mode) |
| `Error: invalid alpha-coeffs format` / `expected 3 alpha coefficients, got N` | Wrong number of alpha coefficients or malformed comma-separated values | Verify alpha has exactly 3 comma-separated float values |
| `--alpha-coeffs: unrecognized argument` | BLIS binary version too old | Update to a BLIS version that supports the evolved latency model CLI flags |
