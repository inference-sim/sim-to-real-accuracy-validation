# Sim-to-Real Accuracy Validation: Design Document

**Date**: 2026-03-04
**Status**: Approved
**Goal**: Evaluate BLIS (blackbox, roofline, cross-model) and llm-optimizer estimate mode against ground-truth vLLM data, reporting mean/P90/P99 E2E, TTFT, and ITL errors across workloads and LLMs.

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

All experiments: vLLM v0.15.1, H100 GPU, max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128, prefix caching enabled, chunked prefill enabled, fp16.

### Available Metrics
- Per-stage and summary: E2E, TTFT, ITL, TPOT at full percentile distributions (mean, P90, P99, etc.)
- Throughput: input_tokens/s, output_tokens/s, requests/s
- Per-request: individual request timings with per-token timestamps
- KV cache capacity: `total_kv_blocks` extracted from `vllm.log`

---

## 2. Simulators Under Test

| Simulator | Mode | Coverage | Notes |
|-----------|------|----------|-------|
| BLIS blackbox | Pre-trained α/β coefficients | CodeLlama-34b, Mixtral-8x7B only (no Llama-2 coefficients) | Highest expected accuracy where available |
| BLIS roofline | Analytical FLOPs/bandwidth | All 4 models | Requires HF config.json + hardware specs |
| BLIS crossmodel | Physics-informed global coefficients | All 4 models | MoE-aware; 7 global coefficients |
| llm-optimizer estimate | Roofline analysis (no batching sim) | All 4 models | Single-point estimates per concurrency; heuristic percentiles |

---

## 3. Architecture: Adapter Pattern

### 3.1 SimulatorAdapter ABC

```python
class SimulatorAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """e.g. 'blis-blackbox', 'blis-roofline', 'llm-optimizer-estimate'"""

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
    total_kv_blocks: int     # from vllm.log
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

3. **Result parsing**: JSON stdout → StageMetrics mapping.

### 4.2 Subclass-Specific

- **BLISBlackboxAdapter**: No extra flags. `can_run()` checks `defaults.yaml` for model coefficients.
- **BLISRooflineAdapter**: Adds `--latency-model roofline`.
- **BLISCrossModelAdapter**: Adds `--latency-model crossmodel`.

---

## 5. LLM-Optimizer Adapter Details

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

## 6. Error Metrics

For each (experiment, simulator, stage) triple, computed on E2E/TTFT/ITL at mean/P90/P99:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MAPE** | `mean(\|pred - actual\| / actual × 100)` | Error magnitude |
| **MPE** (signed) | `mean((pred - actual) / actual × 100)` | Error direction (+ = over-predict) |
| **Absolute Error** | `\|pred - actual\|` in ms or tokens/s | Raw magnitude |

---

## 7. Report Structure

### 7.1 Aggregate Summary Table (main deliverable)

Rows: simulators. Columns: MAPE for each metric variant.

```
| Simulator           | E2E Mean | E2E P90 | E2E P99 | TTFT Mean | TTFT P90 | TTFT P99 | ITL Mean | ITL P90 | ITL P99 |
|---------------------|----------|---------|---------|-----------|----------|----------|----------|---------|---------|
| blis-blackbox       | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| blis-roofline       | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| blis-crossmodel     | ...      | ...     | ...     | ...       | ...      | ...      | ...      | ...     | ...     |
| llm-optimizer-est   | ...      | ...†    | ...†    | ...       | ...†     | ...†     | ...      | ...†    | ...†    |
```

`†` = heuristic percentile estimate

### 7.2 Additional Views

- **Per-experiment detail**: One row per (experiment, simulator) pair
- **Per-model breakdown**: MAPE grouped by model
- **Per-workload breakdown**: MAPE grouped by workload type
- **Signed error (MPE)**: Direction of bias per simulator

### 7.3 Output

- Printed to stdout as formatted tables
- Saved as CSV for further analysis

---

## 8. File Layout

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
      llm_optimizer_est.py
    ground_truth.py        # Experiment dataclass, discover_experiments()
    trace_converter.py     # per_request_lifecycle_metrics.json → BLIS trace v2
    metrics.py             # compute_errors(), MAPE, MPE
    report.py              # generate_report(), format tables, CSV
    run.py                 # Entry point: python -m experiment.run
  vllm_data/               # (existing ground-truth)
  inference-sim/            # (existing, cloned BLIS v0.6.7)
  llm-optimizer/            # (existing, cloned)
  docs/plans/               # This design doc
```

---

## 9. Prerequisites

```bash
# Build BLIS
cd inference-sim && go build -o blis main.go

# Install llm-optimizer
cd llm-optimizer && pip install -e .

# Run experiment
python -m experiment.run
```

---

## 10. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Trace replay over synthetic workloads | Isolates latency model accuracy from workload generation variance |
| Skip blackbox for Llama-2 models | No pre-trained coefficients; proxy coefficients would add confounding error |
| Per-stage estimation for llm-optimizer | Multi-stage workloads have different load characteristics per stage |
| Little's Law for concurrency derivation | llm-optimizer doesn't model arrivals; concurrency is the natural input |
| Include heuristic percentiles (flagged) | User requested; clearly marked as heuristic in report |
| Signed error (MPE) alongside MAPE | Shows whether simulators systematically over- or under-predict |
| Extract total_kv_blocks from vllm.log | Critical for BLIS to match real KV cache capacity |
