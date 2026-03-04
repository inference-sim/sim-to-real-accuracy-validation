# Sim-to-Real Accuracy Validation: Design Document

**Date**: 2026-03-04
**Status**: Approved
**Goal**: Evaluate BLIS (blackbox, roofline, cross-model), Vidur, and llm-optimizer estimate mode against ground-truth vLLM data, reporting mean/P90/P99 E2E, TTFT, and ITL errors across workloads and LLMs.

---

## 1. Ground-Truth Dataset

16 experiments covering a 4×4 matrix:

| Model | HuggingFace ID | TP |
|-------|----------------|----|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | 1 |
| Llama-2-70B | `meta-llama/Llama-2-70b-hf` | 4 |
| Mixtral-8x7B | `mistralai/Mixtral-8x7B-v0.1` | 2 |
| CodeLlama-34B | `codellama/CodeLlama-34b-Instruct-hf` | 2 |

| Workload | Input Tokens | Output Tokens | Load Pattern |
|----------|-------------|---------------|--------------|
| general | ~547 | ~248 | 8→20 req/s (2 stages, 600s each) |
| codegen | ~566 | ~247 | 5→10 req/s (2 stages, 600s each) |
| roleplay | ~750 | ~251 | 6 req/s (1 stage, 1200s) |
| reasoning | ~1034 | ~1448 | 4 req/s (1 stage, 1200s) |

All experiments: vLLM v0.15.1, H100 GPU, max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128, prefix caching enabled, chunked prefill enabled, fp16. **KV cache CPU offloading enabled** (8 GiB via `OffloadingConnector`, `recompute` failure policy).

### Available Metrics
- Per-stage and summary: E2E, TTFT, ITL, TPOT at full percentile distributions (mean, P90, P99, etc.)
- Throughput: input_tokens/s, output_tokens/s, requests/s
- Per-request: individual request timings with per-token timestamps
- KV cache capacity: GPU KV cache size in tokens from `vllm.log` (divided by block_size=16 → `total_kv_blocks`)
- KV cache CPU offloading events: `kv_events.jsonl` with block-level GPU↔CPU transfer logs (used to derive `cpu_kv_blocks`)

---

## 2. Simulators Under Test

| Simulator | Mode | Coverage | Notes |
|-----------|------|----------|-------|
| BLIS blackbox | Pre-trained α/β coefficients | CodeLlama-34b, Mixtral-8x7B only (no Llama-2 coefficients) | Highest expected accuracy where available |
| BLIS roofline | Analytical FLOPs/bandwidth | All 4 models | Requires HF config.json + hardware specs |
| BLIS crossmodel | Physics-informed global coefficients | All 4 models | MoE-aware; 7 global coefficients |
| Vidur | Discrete-event simulation with profiled execution times | Llama-2-7B, Llama-2-70B, CodeLlama-34b (no Mixtral profiling data) | vLLM scheduler mode; trace replay; MLSys'24 |
| llm-optimizer estimate | Roofline analysis (no batching sim) | All 4 models | Single-point estimates per concurrency; heuristic percentiles |

---

## 3. Architecture: Adapter Pattern

### 3.1 SimulatorAdapter ABC

```python
class SimulatorAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """e.g. 'blis-blackbox', 'blis-roofline', 'vidur', 'llm-optimizer-estimate'"""

    def can_run(self, experiment: Experiment) -> bool:
        """Returns True if this adapter can simulate the given experiment."""
        return True

    @abstractmethod
    def run(self, experiment: Experiment) -> SimulatorResult:
        """Run simulation, return standardized results."""
```

### 3.2 Data Model

```python
@dataclass
class LatencyDistribution:
    mean: float   # milliseconds
    p90: float
    p99: float

@dataclass
class ThroughputMetrics:
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    requests_per_sec: float

@dataclass
class StageMetrics:
    stage_index: int
    rate: float              # req/s
    duration: float          # seconds
    num_requests: int
    e2e: LatencyDistribution
    ttft: LatencyDistribution
    itl: LatencyDistribution
    throughput: ThroughputMetrics

@dataclass
class Experiment:
    folder: str
    model: str
    tp: int
    workload: str
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    total_kv_blocks: int     # GPU blocks: "GPU KV cache size" tokens / 16
    cpu_kv_blocks: int       # CPU blocks: peak from kv_events.jsonl
    stages: list[StageMetrics]
    summary: StageMetrics
    profile_config: dict

@dataclass
class SimulatorResult:
    adapter_name: str
    experiment_folder: str
    stages: list[StageMetrics]
    summary: StageMetrics
```

---

## 4. BLIS Adapter Details

### 4.1 Common Base (BaseBLISAdapter)

All BLIS adapters share:

1. **Trace replay workload**: Convert `per_request_lifecycle_metrics.json` → BLIS trace v2 format (arrival_time_us, input_tokens, output_tokens, streaming=true). This gives the most faithful comparison — BLIS simulates the exact request sequence the real server processed.

2. **CLI flags mapped from ground-truth**:
   ```
   --model <model> --tp <tp> --hardware H100
   --max-running <max_num_seqs>
   --max-tokens <max_num_batched_tokens>
   --total-kv-blocks <total_kv_blocks>
   --seed 0
   --workload-spec <trace_v2_spec.yaml>
   --results-path <output.json>
   ```

3. **KV cache CPU offloading flags** (all experiments use CPU offloading):
   ```
   --kv-cpu-blocks <cpu_kv_blocks>
   --kv-offload-threshold 0.9
   --kv-transfer-bandwidth 100.0
   ```
   BLIS's `TieredKVCache` is activated when `--kv-cpu-blocks > 0`. It offloads GPU free blocks with cached prefix content to CPU when GPU utilization exceeds the offload threshold, and reloads them on cache hits with transfer latency modeled as `base_latency + ceil(block_size / bandwidth)` ticks per block.

4. **Result parsing**: JSON stdout → StageMetrics mapping.

### 4.2 KV Cache Extraction

**GPU blocks (`total_kv_blocks`)**: Parsed from `vllm.log` line matching `GPU KV cache size: N tokens`. Divide by block_size (16) to get blocks:

| Model | GPU KV cache (tokens) | total_kv_blocks |
|-------|----------------------|-----------------|
| Llama-2-7B | 119,408 | 7,463 |
| Llama-2-70B | 501,712 | 31,357 |
| Mixtral-8x7B | 440,176 | 27,511 |
| CodeLlama-34B | 425,648 | 26,603 |

**CPU blocks (`cpu_kv_blocks`)**: Parsed from `kv_events.jsonl`. Each line logs a block-level event (offload, reload, evict). The extractor computes the peak concurrent CPU-resident block count across the entire experiment run.

### 4.3 Subclass-Specific

- **BLISBlackboxAdapter**: No extra flags. `can_run()` checks `defaults.yaml` for model coefficients.
- **BLISRooflineAdapter**: Adds `--latency-model roofline`.
- **BLISCrossModelAdapter**: Adds `--latency-model crossmodel`.

---

## 5. Vidur Adapter Details

### 5.1 Overview

Vidur is a discrete-event LLM inference simulator (MSR-India, MLSys'24) that predicts batch execution times using profiled GPU kernel timings (MLP, attention, communication) rather than analytical models. It simulates PagedAttention-style block management, preemption, and scheduling at batch granularity.

### 5.2 Invocation

Invoked via CLI subprocess:

```bash
python -m vidur.main \
    --replica_config_model_name <model> \
    --replica_config_device h100 \
    --replica_config_tensor_parallel_size <tp> \
    --replica_config_num_pipeline_stages 1 \
    --cluster_config_num_replicas 1 \
    --replica_scheduler_config_type vllm \
    --vllm_scheduler_config_batch_size_cap <max_num_seqs> \
    --vllm_scheduler_config_chunk_size <max_num_batched_tokens> \
    --request_generator_config_type trace_replay \
    --trace_request_generator_config_trace_file <trace.csv> \
    --metrics_config_output_dir <output_dir>
```

### 5.3 Trace Conversion

A separate `vidur_trace_converter.py` converts `per_request_lifecycle_metrics.json` → Vidur CSV format:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens
0.0,547,248
0.125,566,247
...
```

Fields:
- `arrived_at`: request arrival time in seconds (relative to first request)
- `num_prefill_tokens`: `question_len + system_prompt_len` from the request
- `num_decode_tokens`: actual output token count from the request

### 5.4 Model Coverage

`can_run()` returns False for Mixtral-8x7B — Vidur lacks H100 profiling data for MoE architectures. Supported models:

| Model | Profiling Data | Vidur Config Class |
|-------|---------------|--------------------|
| Llama-2-7B | `data/profiling/compute/h100/meta-llama/Llama-2-7b-hf/` | `Llama2_7BModelConfig` |
| Llama-2-70B | `data/profiling/compute/h100/meta-llama/Llama-2-70b-hf/` | `Llama2_70BModelConfig` |
| CodeLlama-34B | `data/profiling/compute/h100/codellama/CodeLlama-34b-Instruct-hf/` | `CodeLlama34BModelConfig` |

### 5.5 Result Parsing

Vidur writes per-request metrics to `simulator_output/<timestamp>/`. The adapter parses the request-level data to compute mean/P90/P99 for E2E, TTFT, and ITL — consistent with how ground-truth percentiles are computed. Key metrics extracted:

- `request_e2e_time` → E2E latency
- `prefill_e2e_time` → TTFT
- `decode_time_execution_plus_preemption_normalized` → ITL (per-request average inter-token latency)

### 5.6 Limitations

- No CPU KV offloading modeling — Vidur's `MemoryPlanner` computes KV blocks from GPU memory only. This may cause divergence from real behavior under high KV pressure.
- Execution time predictions depend on profiling data quality — batch sizes or KV cache sizes outside the profiled range are extrapolated.
- No prefix caching modeling — all requests compute full prefill regardless of shared prefixes.

---

## 6. LLM-Optimizer Adapter Details

Calls Python API directly:

```python
from llm_optimizer.performance import estimate_llm_performance
from llm_optimizer.common import get_model_config_from_hf
```

**Per-stage estimation**: For each stage, derives concurrency via Little's Law:
```
concurrency = rate × mean_e2e_seconds
```

Input/output lengths from profile config (question_len + system_prompt_len, output_len).

**Percentile mapping**: Heuristic multipliers (p90≈1.2x, p95≈1.3x, p99≈1.6x of mean). Flagged in report as heuristic-based.

---

## 7. Error Metrics

For each (experiment, simulator, stage) triple, computed on E2E/TTFT/ITL at mean/P90/P99:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MAPE** | `mean(\|pred - actual\| / actual × 100)` | Error magnitude |
| **MPE** (signed) | `mean((pred - actual) / actual × 100)` | Error direction (+ = over-predict) |
| **Absolute Error** | `\|pred - actual\|` in ms or tokens/s | Raw magnitude |

---

## 8. Report Structure

### 8.1 Aggregate Summary Table (main deliverable)

Rows: simulators. Columns: MAPE for each metric variant.

```
| Simulator           | E2E Mean | E2E P90 | E2E P99 | TTFT Mean | TTFT P90 | TTFT P99 | ITL Mean | ITL P90 | ITL P99 |
|---------------------|----------|---------|---------|-----------|----------|----------|----------|---------|---------|
| blis-blackbox       | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| blis-roofline       | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| blis-crossmodel     | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| vidur               | ...‡     | ...‡    | ...‡    | ...‡      | ...‡     | ...‡     | ...‡     | ...‡    | ...‡    |
| llm-optimizer-est   | ...      | ...†    | ...†    | ...       | ...†     | ...†     | ...      | ...†    | ...†    |
```

`†` = heuristic percentile estimate
`‡` = excludes Mixtral-8x7B (no profiling data); no CPU KV offloading or prefix caching modeled

### 8.2 Additional Views

- **Per-experiment detail**: One row per (experiment, simulator) pair
- **Per-model breakdown**: MAPE grouped by model
- **Per-workload breakdown**: MAPE grouped by workload type
- **Signed error (MPE)**: Direction of bias per simulator

### 8.3 Output

- Printed to stdout as formatted tables
- Saved as CSV for further analysis

---

## 9. File Layout

```
sim-to-real-accuracy-validation/
  experiment/
    __init__.py
    adapters/
      __init__.py
      base.py              # SimulatorAdapter ABC, SimulatorResult, BaseBLISAdapter
      blis_blackbox.py
      blis_roofline.py
      blis_crossmodel.py
      vidur.py             # VidurAdapter
      llm_optimizer_est.py
    ground_truth.py        # Experiment dataclass, discover_experiments()
    trace_converter.py     # per_request_lifecycle_metrics.json → BLIS trace v2
    vidur_trace_converter.py  # per_request_lifecycle_metrics.json → Vidur CSV
    kv_cache_extractor.py  # Extract total_kv_blocks + cpu_kv_blocks from logs
    metrics.py             # compute_errors(), MAPE, MPE
    report.py              # generate_report(), format tables, CSV
    run.py                 # Entry point: python -m experiment.run
  vllm_data/               # (existing ground-truth)
  inference-sim/            # (existing, cloned BLIS v0.6.7)
  llm-optimizer/            # (existing, cloned)
  vidur/                    # (existing, cloned)
  docs/plans/               # This design doc
```

---

## 10. Prerequisites

```bash
# Build BLIS
cd inference-sim && go build -o blis main.go

# Install llm-optimizer
cd llm-optimizer && pip install -e .

# Install Vidur
cd vidur && pip install -e .

# Run experiment
python -m experiment.run
```

---

## 11. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Trace replay over synthetic workloads | Isolates latency model accuracy from workload generation variance |
| Skip blackbox for Llama-2 models | No pre-trained coefficients; proxy coefficients would add confounding error |
| Skip Vidur for Mixtral-8x7B | No H100 profiling data for MoE models; profiling requires GPU access |
| vLLM scheduler for Vidur | Matches the real deployment's scheduler; Sarathi would add confounding variable |
| Separate Vidur trace converter | Vidur CSV format (arrived_at, prefill_tokens, decode_tokens) differs from BLIS v2 YAML; decoupling avoids format coupling |
| Parse Vidur request-level metrics | Computing percentiles from per-request data matches ground-truth methodology; CDF bins may differ |
| Enable BLIS CPU KV offloading | All 16 experiments use vLLM OffloadingConnector (8 GiB CPU); without it BLIS would over-predict preemption under KV pressure |
| Extract cpu_kv_blocks from kv_events.jsonl | Most faithful to actual CPU tier usage; analytical derivation from 8 GiB budget may overcount |
| Extract total_kv_blocks from vllm.log | Parse "GPU KV cache size: N tokens" and divide by block_size=16; critical for BLIS to match real GPU KV capacity |
| Per-stage estimation for llm-optimizer | Multi-stage workloads have different load characteristics per stage |
| Little's Law for concurrency derivation | llm-optimizer doesn't model arrivals; concurrency is the natural input |
| Include heuristic percentiles (flagged) | User requested; clearly marked as heuristic in report |
| Signed error (MPE) alongside MAPE | Shows whether simulators systematically over- or under-predict |
| Flag Vidur limitations in report | No prefix caching or CPU offloading modeled; readers need context for interpreting errors |
