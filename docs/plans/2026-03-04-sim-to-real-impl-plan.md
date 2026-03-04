# Sim-to-Real Accuracy Validation: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a validation harness that runs 5 simulators (BLIS blackbox/roofline/crossmodel, Vidur, llm-optimizer) against 16 ground-truth vLLM experiments and reports MAPE/MPE error tables for E2E, TTFT, and ITL at mean/P90/P99.

**Architecture:** Adapter pattern — each simulator is wrapped in a `SimulatorAdapter` subclass with `can_run()` and `run()`. A shared `Experiment` dataclass loads ground-truth data. The orchestrator iterates over all (experiment, adapter) pairs, collects `SimulatorResult`s, computes error metrics, and generates tabular reports. All trace conversion and KV cache extraction is handled by dedicated modules.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, numpy (percentiles), subprocess (BLIS/Vidur CLI), tabulate (report formatting), csv (output).

**Design doc:** `docs/plans/2026-03-04-sim-to-real-accuracy-validation-design.md`

---

## Dependency Graph

```
Task 1: Data Model
  ↓
Task 2: KV Cache Extractor ──────────────────────────────┐
  ↓                                                       │
Task 3: Ground Truth Loader ──────────────────────────────┤
  ↓                                                       │
Task 4: Adapter Base Classes ─────────────────────────────┤
  ↓                                                       │
Task 5: BLIS Trace Converter ─┬─→ Task 8: BLIS Adapters  │
Task 6: Vidur Trace Converter ┼─→ Task 9: Vidur Adapter  │
                              │                           │
Task 7: LLM-Optimizer Adapter ┘                          │
  ↓                                                       │
Task 10: Error Metrics ←──────────────────────────────────┘
  ↓
Task 11: Report Generator
  ↓
Task 12: Orchestrator (run.py)
  ↓
Task 13: Integration Test with Real Data
```

---

## Task 1: Data Model

**Files:**
- Create: `experiment/__init__.py`
- Create: `experiment/data_model.py`
- Test: `tests/test_data_model.py`

**What to build:** All shared dataclasses used across the codebase.

```python
# experiment/data_model.py
from dataclasses import dataclass

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
    model: str               # HuggingFace ID, e.g. "meta-llama/Llama-2-7b-hf"
    tp: int
    workload: str            # "general", "codegen", "roleplay", "reasoning"
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    total_kv_blocks: int     # GPU blocks: "GPU KV cache size" tokens / 16
    cpu_kv_blocks: int       # CPU blocks: peak from kv_events.jsonl
    stages: list[StageMetrics]
    summary: StageMetrics
    profile_config: dict     # Raw parsed profile.yaml

@dataclass
class SimulatorResult:
    adapter_name: str
    experiment_folder: str
    stages: list[StageMetrics]
    summary: StageMetrics
```

**Step 1:** Write test that constructs each dataclass and verifies field access.

**Step 2:** Create the file with the dataclasses above.

**Step 3:** Run `pytest tests/test_data_model.py -v`, verify PASS.

**Step 4:** Commit: `feat: add core data model dataclasses`

---

## Task 2: KV Cache Extractor

**Files:**
- Create: `experiment/kv_cache_extractor.py`
- Test: `tests/test_kv_cache_extractor.py`

**What to build:** Two functions that parse ground-truth logs:

### 2a: `extract_total_kv_blocks(vllm_log_path) -> int`

Parse `vllm.log` for the line:
```
INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 119,408 tokens
```
Regex: `r"GPU KV cache size:\s+([\d,]+)\s+tokens"`
Strip commas, parse int, divide by 16 (block_size), return.

Expected values per model:
| Model | Tokens | Blocks |
|-------|--------|--------|
| Llama-2-7B | 119,408 | 7,463 |
| Llama-2-70B | 501,712 | 31,357 |
| Mixtral-8x7B | 440,176 | 27,511 |
| CodeLlama-34B | 425,648 | 26,603 |

### 2b: `extract_cpu_kv_blocks(kv_events_path) -> int`

Parse `kv_events.jsonl` line-by-line. Each line is:
```json
[timestamp, [event1, event2, ...], flags, metadata]
```

Events are positional arrays. Track CPU-resident blocks:
- `["BlockStored", ..., "CPU", ...]` at position [6] — increment CPU block count
- `["TransferCompleted", transfer_id, req_id, "GPU", "CPU", block_count, success, seq]` — blocks moved GPU→CPU, add block_count
- `["TransferCompleted", transfer_id, req_id, "CPU", "GPU", block_count, success, seq]` — blocks moved CPU→GPU, subtract block_count

Track peak concurrent CPU blocks across the full file. Return peak.

**Note:** The kv_events.jsonl files are 100-270 MB. Process line-by-line, never load entire file into memory.

**Step 1:** Write test with small synthetic JSONL fixtures. Test both functions.

**Step 2:** Implement both functions.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add KV cache block extraction from vllm.log and kv_events.jsonl`

---

## Task 3: Ground Truth Loader

**Files:**
- Create: `experiment/ground_truth.py`
- Test: `tests/test_ground_truth.py`

**What to build:** Functions to discover and parse the 16 experiment directories.

### 3a: `discover_experiments(base_dir) -> list[str]`

Scan `vllm_data/ground_truth/` for directories matching pattern `YYYYMMDD-HHMMSS-*-tp*-*`. Return sorted list of absolute paths. Exclude `SCHEMA.md` and any non-directory entries.

### 3b: `parse_experiment(folder_path) -> Experiment`

For a single experiment directory:

1. **Parse `exp-config.yaml`** → `yaml.safe_load()`:
   - `model`, `tensor_parallelism` (→ `tp`), `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`

2. **Extract workload** from folder name: last segment after the final `-tp<N>-` prefix.

3. **Parse `profile.yaml`** → `yaml.safe_load()` (it's a single-line JSON, YAML parses it):
   - Store as `profile_config` dict
   - Extract `load.stages` for stage rate/duration

4. **Parse each `inference-perf-data/stage_N_lifecycle_metrics.json`**:
   - Discover stage files by globbing `stage_*_lifecycle_metrics.json`
   - For each stage, extract:
     - `load_summary.count` → `num_requests`
     - `load_summary.requested_rate` → `rate`
     - `load_summary.send_duration` → `duration`
     - `successes.latency.request_latency` → E2E: `{"mean", "p90", "p99"}` (values are in **seconds**, convert to **ms** by × 1000)
     - `successes.latency.time_to_first_token` → TTFT: same conversion
     - `successes.latency.inter_token_latency` → ITL: same conversion
     - `successes.throughput` → ThroughputMetrics directly

   **CRITICAL:** Percentile JSON keys use dots: `"p0.1"`, `"p99.9"`. Use bracket access: `d["p90"]`, `d["p99"]`.

   **CRITICAL:** The stage lifecycle metric latency values are in **seconds**. Convert to **milliseconds** for the `LatencyDistribution` dataclass.

5. **Parse `inference-perf-data/summary_lifecycle_metrics.json`** → same extraction as stage, but `load_summary` lacks `send_duration`/`requested_rate`/`achieved_rate`. Use `rate=0.0` and `duration=0.0` for summary, `stage_index=-1`.

6. **Extract KV blocks**: Call `extract_total_kv_blocks(folder/vllm.log)` and `extract_cpu_kv_blocks(folder/kv_events.jsonl)`.

7. **Return `Experiment`** with all fields populated.

**Step 1:** Write tests using a fixture directory with minimal synthetic files that mirrors the real structure. Test `discover_experiments` and `parse_experiment`.

**Step 2:** Implement both functions.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add ground truth experiment discovery and parsing`

---

## Task 4: Adapter Base Classes

**Files:**
- Create: `experiment/adapters/__init__.py`
- Create: `experiment/adapters/base.py`
- Test: `tests/test_adapter_base.py`

**What to build:**

### 4a: `SimulatorAdapter` ABC

```python
from abc import ABC, abstractmethod

class SimulatorAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def can_run(self, experiment: Experiment) -> bool:
        return True

    @abstractmethod
    def run(self, experiment: Experiment) -> SimulatorResult: ...
```

### 4b: `BaseBLISAdapter` (shared BLIS logic)

Encapsulates the common logic all 3 BLIS adapters share:

```python
class BaseBLISAdapter(SimulatorAdapter):
    def __init__(self, blis_binary: str):
        self.blis_binary = blis_binary  # path to compiled BLIS binary

    def _build_common_args(self, experiment: Experiment, trace_spec: str, results_path: str) -> list[str]:
        """Build CLI args shared by all BLIS modes."""
        return [
            self.blis_binary, "run",
            "--model", experiment.model,
            "--tp", str(experiment.tp),
            "--hardware", "H100",
            "--max-running", str(experiment.max_num_seqs),
            "--max-tokens", str(experiment.max_num_batched_tokens),
            "--total-kv-blocks", str(experiment.total_kv_blocks),
            "--kv-cpu-blocks", str(experiment.cpu_kv_blocks),
            "--kv-offload-threshold", "0.9",
            "--kv-transfer-bandwidth", "100.0",
            "--seed", "0",
            "--workload-spec", trace_spec,
            "--results-path", results_path,
        ]

    def _parse_blis_results(self, results_path: str, experiment: Experiment) -> SimulatorResult:
        """Parse BLIS JSON output into SimulatorResult with per-stage breakdown."""
        # 1. Load JSON from results_path
        # 2. Parse aggregate metrics from top-level fields:
        #    e2e_mean_ms, e2e_p90_ms, e2e_p99_ms, ttft_mean_ms, etc.
        # 3. Parse per-request data from "requests" array
        # 4. Split requests into stages by matching arrival times to stage boundaries
        #    (stage boundaries from experiment.profile_config["load"]["stages"])
        # 5. Compute per-stage percentiles from per-request E2E/TTFT/ITL
        # 6. Return SimulatorResult with stages + summary
```

**Per-stage splitting logic:**

BLIS outputs a flat `requests` array with per-request `arrived_at` (seconds), `ttft_ms`, `itl_ms`, `e2e_ms`. The ground-truth `profile_config["load"]["stages"]` defines stage boundaries as cumulative durations. Split requests by arrival time into stage buckets, then compute numpy percentiles within each bucket.

```python
def _split_requests_by_stage(self, requests: list[dict], stages_config: list[dict]) -> list[list[dict]]:
    boundaries = []
    cumulative = 0.0
    for s in stages_config:
        cumulative += s["duration"]
        boundaries.append(cumulative)
    # Assign each request to a stage based on arrived_at vs boundaries
    stage_buckets = [[] for _ in stages_config]
    for req in requests:
        for i, boundary in enumerate(boundaries):
            if req["arrived_at"] <= boundary or i == len(boundaries) - 1:
                stage_buckets[i].append(req)
                break
    return stage_buckets
```

**BLIS JSON field reference (exact keys):**
- Top-level: `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p99_ms`, `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p99_ms`, `itl_mean_ms`, `itl_p90_ms`, `itl_p99_ms`, `responses_per_sec`, `tokens_per_sec`, `total_input_tokens`, `total_output_tokens`, `completed_requests`
- Per-request (in `requests` array): `arrived_at`, `ttft_ms`, `itl_ms`, `e2e_ms`, `num_prefill_tokens`, `num_decode_tokens`
- All latencies in **milliseconds** (matches our LatencyDistribution)

**Step 1:** Write test for `_build_common_args` verifying correct flag construction including KV offloading flags.

**Step 2:** Write test for `_parse_blis_results` using a synthetic BLIS JSON fixture.

**Step 3:** Write test for `_split_requests_by_stage` with known stage boundaries.

**Step 4:** Implement `SimulatorAdapter` ABC and `BaseBLISAdapter`.

**Step 5:** Run all tests, verify PASS.

**Step 6:** Commit: `feat: add SimulatorAdapter ABC and BaseBLISAdapter with KV offloading support`

---

## Task 5: BLIS Trace Converter

**Files:**
- Create: `experiment/trace_converter.py`
- Test: `tests/test_trace_converter.py`

**What to build:** Convert `per_request_lifecycle_metrics.json` → BLIS trace v2 format (YAML header + CSV data).

### Input format (`per_request_lifecycle_metrics.json`)

JSON array of objects:
```json
{
  "start_time": 1393.251,    // monotonic seconds
  "end_time": 1395.195,
  "request": "{\"model\": \"...\", \"prompt\": \"...\", \"max_tokens\": 247, ...}",
  "info": {
    "input_tokens": 591,
    "output_tokens": 140,
    "output_token_times": [1393.381, ...]
  }
}
```

**Note:** `request` field is a **JSON-encoded string** — needs double-parse.

### Output format (BLIS trace v2)

**Header YAML file** (`trace_header.yaml`):
```yaml
trace_version: 2
time_unit: "microseconds"
mode: "real"
warm_up_requests: 0
```

**Data CSV file** (`trace_data.csv`) — 22 columns:
```
request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message
```

For each request:
- `request_id`: sequential integer (0, 1, 2, ...)
- `client_id`, `tenant_id`, `slo_class`, `session_id`, `prefix_group`: empty string `""`
- `round_index`: `0`
- `streaming`: `true`
- `input_tokens`: from `info.input_tokens`
- `output_tokens`: from `info.output_tokens`
- `text_tokens`: same as `input_tokens`
- `image_tokens`, `audio_tokens`, `video_tokens`: `0`
- `reason_ratio`: `0.0`
- `arrival_time_us`: `int(start_time * 1_000_000)` (convert seconds → microseconds, relative to first request)
- `send_time_us`: same as `arrival_time_us`
- `first_chunk_time_us`: `0` (unknown to generator)
- `last_chunk_time_us`: `0`
- `num_chunks`: `0`
- `status`: `"ok"`
- `error_message`: `""`

**Function signature:**
```python
def convert_to_blis_trace(
    per_request_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """Returns (header_yaml_path, data_csv_path)."""
```

Additionally, create a **workload spec YAML** that BLIS's `--workload-spec` flag expects. This wraps the trace:
```yaml
workloads:
  - name: "trace-replay"
    trace_header: "trace_header.yaml"
    trace_data: "trace_data.csv"
```

**Step 1:** Write test with a small synthetic `per_request_lifecycle_metrics.json` (3-5 requests). Verify output CSV column count, arrival time conversion, and streaming=true.

**Step 2:** Implement `convert_to_blis_trace`.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add BLIS trace v2 converter`

---

## Task 6: Vidur Trace Converter

**Files:**
- Create: `experiment/vidur_trace_converter.py`
- Test: `tests/test_vidur_trace_converter.py`

**What to build:** Convert `per_request_lifecycle_metrics.json` → Vidur CSV format.

### Output format

Simple CSV with 3 columns:
```csv
arrived_at,num_prefill_tokens,num_decode_tokens
0.0,591,140
0.125,547,248
```

- `arrived_at`: `start_time - first_start_time` (seconds, relative to first request)
- `num_prefill_tokens`: from `info.input_tokens`
- `num_decode_tokens`: from `info.output_tokens`

**Function signature:**
```python
def convert_to_vidur_trace(
    per_request_path: str,
    output_csv_path: str,
) -> str:
    """Converts per-request metrics to Vidur CSV. Returns output path."""
```

**Step 1:** Write test with synthetic input. Verify 3-column CSV output, relative timestamps.

**Step 2:** Implement `convert_to_vidur_trace`.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add Vidur trace converter`

---

## Task 7: LLM-Optimizer Adapter

**Files:**
- Create: `experiment/adapters/llm_optimizer_est.py`
- Test: `tests/test_llm_optimizer_adapter.py`

**What to build:** Adapter that calls `estimate_llm_performance` per-stage.

### Key logic

For each stage in the experiment:

1. **Derive concurrency** via Little's Law:
   ```python
   concurrency = max(1, round(stage.rate * stage.e2e.mean / 1000))
   # stage.e2e.mean is in ms, convert to seconds for Little's Law
   ```

2. **Extract input/output lengths** from profile config:
   ```python
   data_cfg = experiment.profile_config["data"]["shared_prefix"]
   input_length = data_cfg["question_len"] + data_cfg["system_prompt_len"]
   output_length = data_cfg["output_len"]
   ```

3. **Call llm-optimizer**:
   ```python
   from llm_optimizer.performance import estimate_llm_performance
   from llm_optimizer.common import get_model_config_from_hf

   model_config = get_model_config_from_hf(experiment.model)
   result = estimate_llm_performance(
       num_gpus=experiment.tp,
       gpu_name="H100",
       model_config=model_config,
       precision="fp16",
       concurrency=concurrency,
       input_length=input_length,
       output_length=output_length,
   )
   ```

4. **Map result → StageMetrics**:
   - `e2e.mean = result.e2e_latency_s * 1000` (seconds → ms)
   - `ttft.mean = result.ttft_ms` (already in ms)
   - `itl.mean = result.itl_ms` (already in ms)
   - **Heuristic percentiles**: `p90 = mean * 1.2`, `p99 = mean * 1.6`
   - Throughput: `result.output_throughput_tps`, `result.input_throughput_tps`, `result.requests_per_sec`

5. **Summary**: Weighted average across stages (by num_requests) or call with overall concurrency.

**Step 1:** Write test with mocked `estimate_llm_performance` return value. Verify concurrency derivation, unit conversion, heuristic percentile multipliers.

**Step 2:** Implement `LLMOptimizerEstimateAdapter`.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add llm-optimizer estimate adapter`

---

## Task 8: BLIS Adapters (Blackbox, Roofline, CrossModel)

**Files:**
- Create: `experiment/adapters/blis_blackbox.py`
- Create: `experiment/adapters/blis_roofline.py`
- Create: `experiment/adapters/blis_crossmodel.py`
- Test: `tests/test_blis_adapters.py`

**What to build:** Three thin subclasses of `BaseBLISAdapter`.

### 8a: BLISBlackboxAdapter

```python
class BLISBlackboxAdapter(BaseBLISAdapter):
    @property
    def name(self) -> str:
        return "blis-blackbox"

    def can_run(self, experiment: Experiment) -> bool:
        """Check defaults.yaml for matching model/TP/GPU coefficients."""
        # Parse defaults.yaml, look for entry matching:
        #   id == experiment.model (case-insensitive)
        #   tensor_parallelism == experiment.tp
        #   GPU == "H100" (or similar)
        # Return True only if alpha_coeffs and beta_coeffs are non-null
        ...

    def run(self, experiment: Experiment) -> SimulatorResult:
        # 1. Convert trace: convert_to_blis_trace(...)
        # 2. Build args: self._build_common_args(...) — no extra flags
        # 3. subprocess.run(args, capture_output=True, check=True)
        # 4. Parse: self._parse_blis_results(...)
        ...
```

**Important:** The model IDs in `defaults.yaml` are **lowercase** (e.g., `codellama/codellama-34b-instruct-hf`, `mistralai/mixtral-8x7b-instruct-v0.1`). The ground-truth uses mixed case (`codellama/CodeLlama-34b-Instruct-hf`). The `can_run()` check must do **case-insensitive** comparison.

Also note: `defaults.yaml` has `mixtral-8x7b-instruct-v0.1` but our ground truth model is `Mixtral-8x7B-v0.1` (base model, not instruct). Check if BLIS matches on model prefix or exact ID.

### 8b: BLISRooflineAdapter

```python
class BLISRooflineAdapter(BaseBLISAdapter):
    @property
    def name(self) -> str:
        return "blis-roofline"

    def run(self, experiment: Experiment) -> SimulatorResult:
        # Same as blackbox but add: "--latency-model", "roofline"
        ...
```

`can_run()` → always True (roofline works for all models analytically).

### 8c: BLISCrossModelAdapter

```python
class BLISCrossModelAdapter(BaseBLISAdapter):
    @property
    def name(self) -> str:
        return "blis-crossmodel"

    def run(self, experiment: Experiment) -> SimulatorResult:
        # Same as blackbox but add: "--latency-model", "crossmodel"
        ...
```

`can_run()` → always True.

**Step 1:** Write tests that mock `subprocess.run` and verify correct CLI args for each adapter, including `--latency-model` flag differences and KV offloading flags.

**Step 2:** Write test for `BLISBlackboxAdapter.can_run()` with a mock defaults.yaml fixture.

**Step 3:** Implement all three adapters.

**Step 4:** Run tests, verify PASS.

**Step 5:** Commit: `feat: add BLIS blackbox, roofline, and crossmodel adapters`

---

## Task 9: Vidur Adapter

**Files:**
- Create: `experiment/adapters/vidur.py`
- Test: `tests/test_vidur_adapter.py`

**What to build:**

```python
class VidurAdapter(SimulatorAdapter):
    SUPPORTED_MODELS = {
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-hf",
        "codellama/CodeLlama-34b-Instruct-hf",
    }

    def __init__(self, vidur_dir: str):
        self.vidur_dir = vidur_dir  # path to cloned vidur/

    @property
    def name(self) -> str:
        return "vidur"

    def can_run(self, experiment: Experiment) -> bool:
        return experiment.model in self.SUPPORTED_MODELS

    def run(self, experiment: Experiment) -> SimulatorResult:
        # 1. Convert trace: convert_to_vidur_trace(...)
        # 2. Build CLI args:
        #    python -m vidur.main
        #      --replica_config_model_name <model>
        #      --replica_config_device h100
        #      --replica_config_tensor_parallel_size <tp>
        #      --replica_config_num_pipeline_stages 1
        #      --cluster_config_num_replicas 1
        #      --replica_scheduler_config_type vllm
        #      --vllm_scheduler_config_batch_size_cap <max_num_seqs>
        #      --vllm_scheduler_config_chunk_size <max_num_batched_tokens>
        #      --request_generator_config_type trace_replay
        #      --trace_request_generator_config_trace_file <trace.csv>
        #      --metrics_config_output_dir <output_dir>
        # 3. subprocess.run(..., cwd=self.vidur_dir)
        # 4. Parse request_metrics.csv from output dir
        ...
```

### Result parsing from `request_metrics.csv`

Vidur writes a CSV with per-request metrics. Key columns (all in **seconds**):
- `request_e2e_time` → E2E
- `prefill_e2e_time` → TTFT
- `decode_time_execution_plus_preemption_normalized` → mean ITL per request

**Parse logic:**
1. Read CSV with pandas or csv.DictReader
2. Convert all values from **seconds → milliseconds** (× 1000)
3. Split rows by stage using the same arrival-time boundary logic as BLIS (the original `arrived_at` from the trace maps to request order)
4. Compute numpy percentiles (mean, P90, P99) per stage
5. Compute summary across all requests
6. Return `SimulatorResult`

**Important:** The Vidur output directory has a timestamp subdirectory. After running Vidur, find the latest `simulator_output/<timestamp>/request_metrics.csv`.

**Step 1:** Write test with mock subprocess and a synthetic `request_metrics.csv` fixture. Verify seconds→ms conversion and per-stage splitting.

**Step 2:** Implement `VidurAdapter`.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add Vidur adapter with trace replay and request-level parsing`

---

## Task 10: Error Metrics

**Files:**
- Create: `experiment/metrics.py`
- Test: `tests/test_metrics.py`

**What to build:** Error computation functions.

```python
def compute_mape(predicted: list[float], actual: list[float]) -> float:
    """Mean Absolute Percentage Error."""
    # mean(|pred - actual| / actual * 100) for each pair
    # Guard against actual == 0 (skip or return inf)

def compute_mpe(predicted: list[float], actual: list[float]) -> float:
    """Mean Percentage Error (signed)."""
    # mean((pred - actual) / actual * 100)

def compute_absolute_error(predicted: float, actual: float) -> float:
    """Absolute error in original units (ms or tokens/s)."""
    return abs(predicted - actual)
```

Higher-level function:

```python
@dataclass
class ErrorRecord:
    simulator: str
    experiment_folder: str
    model: str
    workload: str
    stage_index: int
    metric_name: str     # "e2e_mean", "e2e_p90", "e2e_p99", "ttft_mean", etc.
    predicted: float
    actual: float
    mape: float          # percentage
    mpe: float           # signed percentage
    absolute_error: float

def compute_errors(
    experiment: Experiment,
    result: SimulatorResult,
) -> list[ErrorRecord]:
    """Compute all error records for one (experiment, simulator) pair."""
    records = []
    for gt_stage, pred_stage in zip(experiment.stages, result.stages):
        for metric_type in ["e2e", "ttft", "itl"]:
            for percentile in ["mean", "p90", "p99"]:
                gt_val = getattr(getattr(gt_stage, metric_type), percentile)
                pred_val = getattr(getattr(pred_stage, metric_type), percentile)
                records.append(ErrorRecord(
                    simulator=result.adapter_name,
                    experiment_folder=experiment.folder,
                    model=experiment.model,
                    workload=experiment.workload,
                    stage_index=gt_stage.stage_index,
                    metric_name=f"{metric_type}_{percentile}",
                    predicted=pred_val,
                    actual=gt_val,
                    mape=compute_mape([pred_val], [gt_val]),
                    mpe=compute_mpe([pred_val], [gt_val]),
                    absolute_error=compute_absolute_error(pred_val, gt_val),
                ))
    # Also compute for summary stage
    ...
    return records
```

**Step 1:** Write tests for `compute_mape`, `compute_mpe` with known values. Include edge case: actual=0.

**Step 2:** Write test for `compute_errors` with synthetic Experiment and SimulatorResult.

**Step 3:** Implement all functions.

**Step 4:** Run tests, verify PASS.

**Step 5:** Commit: `feat: add error metrics computation (MAPE, MPE, absolute error)`

---

## Task 11: Report Generator

**Files:**
- Create: `experiment/report.py`
- Test: `tests/test_report.py`

**What to build:** Generate formatted tables from error records.

### 11a: Aggregate summary table

Input: `list[ErrorRecord]` from all (experiment, simulator) pairs.

Group by simulator → for each metric_name, compute mean MAPE across all experiments where that simulator ran.

Output (using `tabulate` library):
```
| Simulator         | E2E Mean | E2E P90 | E2E P99 | TTFT Mean | ... |
|-------------------|----------|---------|---------|-----------|-----|
| blis-blackbox     | 12.3%    | 15.1%   | 18.2%   | 8.4%      | ... |
| blis-roofline     | 22.1%    | ...     | ...     | ...       | ... |
| blis-crossmodel   | 18.5%    | ...     | ...     | ...       | ... |
| vidur             | 14.2%‡  | ...     | ...     | ...       | ... |
| llm-optimizer-est | 35.0%   | 42.0%†  | 56.0%†  | ...       | ... |
```

### 11b: Additional views

- **Per-experiment detail**: one row per (experiment, simulator), all 9 metric MAPEs
- **Per-model breakdown**: group by model, mean MAPE per simulator
- **Per-workload breakdown**: group by workload, mean MAPE per simulator
- **Signed error (MPE)**: same structure as summary but using MPE instead of MAPE

### 11c: CSV output

Write all `ErrorRecord`s as a flat CSV for further analysis:
```
simulator,experiment_folder,model,workload,stage_index,metric_name,predicted,actual,mape,mpe,absolute_error
```

**Function signatures:**
```python
def generate_report(records: list[ErrorRecord], output_dir: str) -> None:
    """Print all tables to stdout and save CSVs."""

def format_aggregate_table(records: list[ErrorRecord]) -> str:
    """Format the main MAPE summary table."""

def format_per_model_table(records: list[ErrorRecord]) -> str:
def format_per_workload_table(records: list[ErrorRecord]) -> str:
def format_signed_error_table(records: list[ErrorRecord]) -> str:
```

**Step 1:** Write test with synthetic ErrorRecords. Verify table output contains expected simulators and metrics.

**Step 2:** Implement all formatting functions.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add report generator with aggregate, per-model, per-workload tables`

---

## Task 12: Orchestrator (run.py)

**Files:**
- Create: `experiment/run.py`
- Test: `tests/test_run.py`

**What to build:** The entry point that ties everything together.

```python
# experiment/run.py
"""Entry point: python -m experiment.run"""

import argparse
import os
from experiment.ground_truth import discover_experiments, parse_experiment
from experiment.adapters.blis_blackbox import BLISBlackboxAdapter
from experiment.adapters.blis_roofline import BLISRooflineAdapter
from experiment.adapters.blis_crossmodel import BLISCrossModelAdapter
from experiment.adapters.vidur import VidurAdapter
from experiment.adapters.llm_optimizer_est import LLMOptimizerEstimateAdapter
from experiment.metrics import compute_errors
from experiment.report import generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="vllm_data/ground_truth")
    parser.add_argument("--blis-binary", default="inference-sim/blis")
    parser.add_argument("--vidur-dir", default="vidur")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--adapters", nargs="+",
                        default=["blis-blackbox", "blis-roofline", "blis-crossmodel",
                                 "vidur", "llm-optimizer-estimate"],
                        help="Which adapters to run (default: all)")
    args = parser.parse_args()

    # 1. Discover and parse experiments
    folders = discover_experiments(args.data_dir)
    experiments = [parse_experiment(f) for f in folders]

    # 2. Build adapter registry
    all_adapters = {
        "blis-blackbox": BLISBlackboxAdapter(args.blis_binary),
        "blis-roofline": BLISRooflineAdapter(args.blis_binary),
        "blis-crossmodel": BLISCrossModelAdapter(args.blis_binary),
        "vidur": VidurAdapter(args.vidur_dir),
        "llm-optimizer-estimate": LLMOptimizerEstimateAdapter(),
    }
    adapters = [all_adapters[name] for name in args.adapters]

    # 3. Run all (experiment, adapter) pairs
    all_records = []
    for exp in experiments:
        for adapter in adapters:
            if not adapter.can_run(exp):
                print(f"  SKIP {adapter.name} for {exp.folder} (unsupported)")
                continue
            print(f"  RUN  {adapter.name} for {exp.folder}...")
            try:
                result = adapter.run(exp)
                records = compute_errors(exp, result)
                all_records.extend(records)
            except Exception as e:
                print(f"  FAIL {adapter.name} for {exp.folder}: {e}")

    # 4. Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    generate_report(all_records, args.output_dir)

if __name__ == "__main__":
    main()
```

**Step 1:** Write test with mocked adapters and experiments. Verify the orchestration loop calls `can_run()` then `run()`, and skips when `can_run()` returns False.

**Step 2:** Implement `run.py`.

**Step 3:** Run tests, verify PASS.

**Step 4:** Commit: `feat: add orchestrator entry point (python -m experiment.run)`

---

## Task 13: Integration Test with Real Data

**Files:**
- Test: `tests/test_integration.py`

**What to build:** A test that loads one real experiment and runs a lightweight validation.

### 13a: Ground truth parsing smoke test

Load one real experiment (e.g., `20260217-155451-llama-2-7b-tp1-codegen`), verify:
- `total_kv_blocks == 7463`
- `model == "meta-llama/Llama-2-7b-hf"`
- `tp == 1`
- `workload == "codegen"`
- `len(stages) == 2` (codegen has 2 stages: 5 RPS, 10 RPS)
- `stages[0].rate == 5.0`
- `stages[0].e2e.mean > 0` (sanity check)

### 13b: Trace converter smoke test

Convert the real `per_request_lifecycle_metrics.json` and verify:
- BLIS trace CSV has the correct number of rows
- Vidur trace CSV has 3 columns, correct row count
- Arrival times are relative (first = 0)

### 13c: KV cache extractor smoke test

- `extract_total_kv_blocks` returns 7463 for Llama-2-7b
- `extract_cpu_kv_blocks` returns a positive integer

**Step 1:** Implement and run integration tests.

**Step 2:** Commit: `test: add integration tests with real ground-truth data`

---

## Summary: Task Order and Estimated Scope

| Task | Description | Depends On | Files Created |
|------|-------------|------------|---------------|
| 1 | Data Model | — | 2 files |
| 2 | KV Cache Extractor | 1 | 2 files |
| 3 | Ground Truth Loader | 1, 2 | 2 files |
| 4 | Adapter Base Classes | 1 | 2 files |
| 5 | BLIS Trace Converter | 1 | 2 files |
| 6 | Vidur Trace Converter | 1 | 2 files |
| 7 | LLM-Optimizer Adapter | 1, 4 | 2 files |
| 8 | BLIS Adapters (×3) | 4, 5 | 4 files |
| 9 | Vidur Adapter | 4, 6 | 2 files |
| 10 | Error Metrics | 1 | 2 files |
| 11 | Report Generator | 10 | 2 files |
| 12 | Orchestrator | all | 2 files |
| 13 | Integration Test | all | 1 file |

**Parallelizable pairs:** Tasks 5 & 6 (trace converters). Tasks 7, 8, 9 (adapters, after 4 is done). Tasks 10 & 11 can start once data model is stable.
