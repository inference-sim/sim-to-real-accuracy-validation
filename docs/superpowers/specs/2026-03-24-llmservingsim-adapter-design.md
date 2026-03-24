# LLMServingSim Adapter Design

**Date**: 2026-03-24
**Purpose**: Design an adapter for LLMServingSim to enable validation against vLLM ground-truth experiments
**Status**: Approved

## Overview

The LLMServingSim adapter enables accuracy validation of LLMServingSim predictions against real vLLM ground-truth experiments. It translates experiment configurations into LLMServingSim-compatible formats, generates matching workloads, executes simulations, and parses results into standardized metrics.

## Architecture & Scope

### Purpose

Validate LLMServingSim's accuracy by running it on the same configurations as ground-truth vLLM experiments and comparing predicted vs actual latency metrics (E2E, TTFT, ITL).

### Supported Experiments

- **Hardware**: H100 only (LLMServingSim has performance models for H100)
- **Models**:
  - Llama-3.1-8B (tp1, tp2)
  - Mixtral-8x7B-v0.1 (tp1, tp2, tp4)
- **Precision**: FP16 (LLMServingSim default)
- **CPU Offloading**: Supported (6 out of 20 runnable experiments use this)
- **Multi-replica**: Supported (dp>1 with round-robin routing)
- **Coverage**: Approximately 20 out of 49 ground-truth experiments

### Eligibility Criteria

The adapter's `can_run()` method returns `True` only when:

1. `experiment.hardware == "H100"`
2. `experiment.model` maps to a supported LLMServingSim model
3. Performance model exists at `llm_profile/perf_models/H100/{model}/tp{N}/`
4. `experiment.precision == "FP16"`

### Integration

Follows the existing `SimulatorAdapter` interface and integrates into `experiment/run.py` alongside BLIS, Vidur, and AIConfigurator adapters.

## Components

### 2.1 LLMServingSimAdapter Class

**Responsibilities**:
- Implements `SimulatorAdapter` interface
- Orchestrates workload generation, config creation, simulation execution, and result parsing
- Manages temporary directories for per-experiment isolation

**Key Methods**:
```python
@property
def name(self) -> str:
    return "llmservingsim"

def can_run(self, experiment: Experiment) -> bool:
    # Check hardware, model, tp combination, precision

def run(self, experiment: Experiment) -> SimulatorResult:
    # Main execution flow with temp directory management
```

### 2.2 Model Name Mapper

**Purpose**: Translate ground-truth model names to LLMServingSim format

**Mapping**:
```python
MODEL_MAP = {
    "Llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
}
```

### 2.3 Cluster Config Generator

**Purpose**: Create experiment-specific cluster_config JSON files

**Strategy**:
- Load base template: `LLMServingSim/cluster_config/single_node_single_instance.json`
- Modify fields:
  - Hardware type: "H100"
  - TP degree: from `experiment.tp`
  - Instance count: 1 for dp≤1, `experiment.dp` for dp>1
  - GPU memory: compute from `experiment.total_kv_blocks`
  - CPU memory: compute from `experiment.cpu_kv_blocks` (if offloading enabled)
  - Routing policy: "RR" via CLI for dp>1
- Write to temp file

**Memory Calculation**:
```python
# GPU memory (from total_kv_blocks)
gpu_mem_gb = (experiment.total_kv_blocks * 16 * 2) / (1024**3)  # FP16 = 2 bytes/value

# CPU memory (from cpu_kv_blocks, if offloading enabled)
if experiment.cpu_offload and experiment.cpu_kv_blocks > 0:
    cpu_mem_gb = (experiment.cpu_kv_blocks * 16 * 2) / (1024**3)
```

**Note on Terminology**: LLMServingSim uses `npu_mem`, `npu_num` as generic **accelerator** configuration fields. For H100 experiments, these configure **GPU** memory (80GB HBM3) and GPU device count.

### 2.4 Workload Generator

**Purpose**: Generate LLMServingSim-compatible `.jsonl` workload files

**Data Source**: Read ground-truth `per_request_lifecycle_metrics.json` to extract:
- `info.input_tokens` → `input_toks`
- `info.output_tokens` → `output_toks`

**Arrival Time Generation**:
Uses constant-rate spacing (matching inference-perf's `type: constant`):
- Stage 1 (rate=R1, duration=D1): arrivals at 0, 1/R1, 2/R1, ...
- Stage 2 (rate=R2, duration=D2): arrivals at D1, D1+1/R2, D1+2/R2, ...

**Token IDs**: Generate random token IDs (LLMServingSim only needs token counts for performance modeling)

**Output Format**:
```json
{"input_toks": 574, "output_toks": 251, "arrival_time_ns": 0, "input_tok_ids": [1, 2, 3, ...]}
{"input_toks": 571, "output_toks": 252, "arrival_time_ns": 125000000, "input_tok_ids": [4, 5, 6, ...]}
```

### 2.5 CLI Arguments Builder

**Purpose**: Construct LLMServingSim command-line arguments

**Key Parameters**:
```python
args = [
    "python", "main.py",
    "--cluster-config", cluster_config_path,
    "--dataset", workload_path,
    "--output", output_csv_path,
    "--fp", "16",
    "--block-size", "16",
    "--max-batch", str(experiment.max_num_seqs),
    "--max-num-batched-tokens", str(experiment.max_num_batched_tokens),
    "--num-req", str(total_requests),
    "--log-level", "WARNING",
]

# For multi-instance (dp > 1)
if experiment.dp and experiment.dp > 1:
    args += ["--request-routing-policy", "RR"]
```

### 2.6 Result Parser

**Purpose**: Parse LLMServingSim's CSV output into `SimulatorResult`

**Steps**:
1. Read CSV file
2. Convert nanoseconds → milliseconds
3. Split requests by stage using arrival times
4. Compute per-stage metrics (mean/p90/p99 for e2e, ttft, itl)
5. Compute summary as weighted average across stages

**Column Mapping**:
- `latency` (ns) → e2e_ms (verified: latency = end_time - arrival)
- `TTFT` (ns) → ttft_ms
- `TPOT` (ns) → itl_ms (inter-token latency)
- `input`, `output` → token counts for throughput calculation

## Data Flow

### Overall Execution Flow

```
Experiment → can_run() → run() → SimulatorResult
                ↓
    [temp directory created]
                ↓
    ┌───────────────────────────┐
    │ 1. Generate Cluster Config│
    │    - Load template        │
    │    - Modify: model, tp,   │
    │      memory, instances    │
    └───────────┬───────────────┘
                ↓
    ┌───────────────────────────┐
    │ 2. Generate Workload      │
    │    - Read ground-truth    │
    │      per_request metrics  │
    │    - Extract token counts │
    │    - Generate arrivals    │
    │    - Write .jsonl         │
    └───────────┬───────────────┘
                ↓
    ┌───────────────────────────┐
    │ 3. Execute LLMServingSim  │
    │    - Build CLI args       │
    │    - subprocess.run()     │
    │    - Capture output CSV   │
    └───────────┬───────────────┘
                ↓
    ┌───────────────────────────┐
    │ 4. Parse Results          │
    │    - Read CSV             │
    │    - Split by stages      │
    │    - Compute metrics      │
    │    - Return SimulatorResult│
    └───────────┬───────────────┘
                ↓
    [temp directory cleaned up]
```

### Cluster Config Modification

**Template Modifications**:
```python
config = load_json(template_path)

# 1. Model and hardware
config["nodes"][0]["instances"][0]["model_name"] = llmservingsim_model
config["nodes"][0]["instances"][0]["hardware"] = "H100"

# 2. Tensor parallelism
config["nodes"][0]["instances"][0]["npu_num"] = experiment.tp  # Number of GPUs
config["nodes"][0]["instances"][0]["npu_group"] = experiment.tp  # TP group size

# 3. GPU memory (from total_kv_blocks)
gpu_mem_gb = (experiment.total_kv_blocks * 16 * 2) / (1024**3)  # FP16 = 2 bytes
config["nodes"][0]["instances"][0]["npu_mem"]["mem_size"] = gpu_mem_gb

# 4. CPU memory (from cpu_kv_blocks, if offloading enabled)
if experiment.cpu_offload and experiment.cpu_kv_blocks > 0:
    cpu_mem_gb = (experiment.cpu_kv_blocks * 16 * 2) / (1024**3)
    config["nodes"][0]["cpu_mem"]["mem_size"] = cpu_mem_gb

# 5. Multi-instance (for dp > 1)
if experiment.dp and experiment.dp > 1:
    config["nodes"][0]["num_instances"] = experiment.dp
    # Duplicate instance config dp times
    instances = config["nodes"][0]["instances"][0]
    config["nodes"][0]["instances"] = [instances.copy() for _ in range(experiment.dp)]
```

### Workload Generation

**Step 1: Read ground-truth metrics**
```python
perf_dir = resolve_perf_dir(experiment.folder)
metrics_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")
with open(metrics_path) as f:
    requests = json.load(f)
```

**Step 2: Extract token counts**
```python
token_pairs = [
    (req["info"]["input_tokens"], req["info"]["output_tokens"])
    for req in requests
]
```

**Step 3: Generate arrival times (constant rate)**
```python
arrivals = []
t = 0.0  # seconds
for stage in experiment.profile_config["load"]["stages"]:
    rate = stage["rate"]  # requests/sec
    duration = stage["duration"]  # seconds
    num_requests = round(rate * duration)

    inter_arrival = 1.0 / rate  # seconds between requests
    for _ in range(num_requests):
        arrivals.append(t)
        t += inter_arrival
```

**Step 4: Generate .jsonl**
```python
for i, ((input_toks, output_toks), arrival_sec) in enumerate(zip(token_pairs, arrivals)):
    input_tok_ids = list(range(1, input_toks + 1))  # dummy token IDs
    record = {
        "input_toks": input_toks,
        "output_toks": output_toks,
        "arrival_time_ns": int(arrival_sec * 1e9),
        "input_tok_ids": input_tok_ids,
    }
    json.dump(record, file)
    file.write("\n")
```

### Result Parsing

**CSV Column Mapping**:
- `latency` (ns) → e2e_ms = latency / 1e6
- `TTFT` (ns) → ttft_ms = TTFT / 1e6
- `TPOT` (ns) → itl_ms = TPOT / 1e6
- `input`, `output` → token counts for throughput
- `arrival` (ns) → used for stage splitting

**Stage Splitting Logic**:
```python
def split_by_stage(rows, stages_config):
    boundaries = []
    cumulative_time = 0.0
    for stage in stages_config:
        cumulative_time += stage["duration"]
        boundaries.append(cumulative_time * 1e9)  # convert to nanoseconds

    buckets = [[] for _ in stages_config]
    for row in rows:
        arrival_ns = float(row["arrival"])
        for i, boundary_ns in enumerate(boundaries):
            if arrival_ns <= boundary_ns:
                buckets[i].append(row)
                break
            if i == len(boundaries) - 1:
                buckets[i].append(row)  # fallback to last stage
    return buckets
```

## Error Handling

### Eligibility Validation

**`can_run()` checks** (fail gracefully by returning `False`):
- Hardware must be "H100"
- Model must exist in `MODEL_MAP` and have H100 perf models
- TP combination must exist in `llm_profile/perf_models/H100/{model}/tp{N}/`
- Precision must be FP16

```python
def can_run(self, experiment: Experiment) -> bool:
    if experiment.hardware != "H100":
        return False

    model_sim = MODEL_MAP.get(experiment.model)
    if not model_sim:
        return False

    perf_model_path = os.path.join(
        self.llmservingsim_dir,
        f"llm_profile/perf_models/H100/{model_sim}/tp{experiment.tp}"
    )
    if not os.path.exists(perf_model_path):
        return False

    if experiment.precision not in ("FP16",):
        return False

    return True
```

### Runtime Error Handling

**LLMServingSim execution failures**:
```python
try:
    result = subprocess.run(
        args,
        capture_output=True,
        check=True,
        cwd=self.llmservingsim_dir,
        timeout=3600,  # 1 hour timeout
    )
except subprocess.TimeoutExpired:
    raise RuntimeError(
        f"LLMServingSim timed out after 1 hour for {experiment.folder}"
    )
except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.decode("utf-8", errors="replace")
    raise RuntimeError(
        f"LLMServingSim failed (rc={exc.returncode}) for {experiment.folder}: {stderr}"
    )
```

**Missing ground-truth data**:
```python
metrics_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")
if not os.path.exists(metrics_path):
    raise FileNotFoundError(
        f"Ground-truth metrics not found: {metrics_path}. "
        f"Cannot generate workload without token counts."
    )
```

**CSV parsing failures**:
```python
if not os.path.exists(output_csv_path):
    raise FileNotFoundError(
        f"LLMServingSim output CSV not found: {output_csv_path}"
    )

with open(output_csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    if not rows:
        raise ValueError(f"LLMServingSim output CSV is empty: {output_csv_path}")
```

### Data Validation

**Stage splitting validation**:
```python
# Warn if any stage has no requests
for i, bucket in enumerate(buckets):
    if not bucket:
        logger.warning(
            f"Stage {i} has no requests. "
            f"This may indicate a mismatch between workload and stage config."
        )
```

**Token count mismatch detection**:
```python
# Check if generated workload matches ground-truth request count
expected_requests = sum(
    round(s["rate"] * s["duration"])
    for s in experiment.profile_config["load"]["stages"]
)
actual_requests = len(token_pairs)

if actual_requests != expected_requests:
    logger.warning(
        f"Token count mismatch: expected {expected_requests} requests, "
        f"got {actual_requests} from ground-truth metrics"
    )
```

### Cleanup

**Guaranteed cleanup using context manager**:
```python
def run(self, experiment: Experiment) -> SimulatorResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Generate configs, run simulator, parse results
            return result
        except Exception:
            # Temp directory automatically cleaned up even on error
            raise
```

## Testing Strategy

### Unit Tests

**Test eligibility checking** (`test_llmservingsim_adapter.py`):
- Verify `can_run()` for supported configurations (H100 + Llama/Mixtral + tp + FP16)
- Verify `can_run()` returns `False` for unsupported hardware (A100, L40S)
- Verify `can_run()` returns `False` for unsupported models (Codellama-34b, Qwen3-14B)
- Verify `can_run()` returns `False` for unsupported precision (FP8)

**Test workload generation**:
- Verify constant-rate arrival times are uniformly spaced
- Verify multi-stage workloads have correct stage boundaries
- Verify token counts match ground-truth metrics
- Verify .jsonl format is valid

**Test cluster config modification**:
- Verify single-instance config (dp=1)
- Verify multi-instance config (dp>1)
- Verify TP settings
- Verify memory calculations
- Verify CPU offloading configuration

**Test result parsing**:
- Verify nanosecond to millisecond conversion
- Verify stage splitting by arrival time
- Verify percentile calculations (mean, p90, p99)
- Verify weighted summary computation

### Integration Tests

**End-to-end test with real LLMServingSim** (requires LLMServingSim installed):
- Run on a small ground-truth experiment (e.g., Llama-3.1-8B tp1)
- Verify output CSV is generated
- Verify result structure is valid
- Verify metrics are reasonable (e.g., ITL > 0, TTFT > ITL)

### Validation Tests

**Compare against existing adapters** (sanity checks):
- Verify stage splitting produces same request counts as VidurAdapter
- Verify summary aggregation follows same pattern as other adapters

### Manual Validation

**Smoke test checklist**:
1. Run adapter on 1-2 experiments from each supported configuration:
   - Llama-3.1-8B tp1 (no cpu_offload)
   - Llama-3.1-8B tp1 (with cpu_offload)
   - Mixtral-8x7B tp2 (with cpu_offload)
   - Multi-instance dp=2 experiment
2. Verify output CSV contains expected columns
3. Check that parsed metrics are reasonable
4. Compare against ground-truth: should see similar order-of-magnitude

## Additional Considerations

### Compatibility with Orchestration Layer

**Integration with `experiment/run.py`**:
- Follows the same interface as existing adapters (BLIS, Vidur, AIConfigurator)
- Respects `--no-dp-scaling` flag (orchestrator filters dp>1 experiments before calling adapter)
- Returns standardized `SimulatorResult` for metrics computation
- Wall-clock time tracked via `time.perf_counter()` like other adapters

**Adapter registry addition**:
```python
# In experiment/run.py
def build_adapter_registry(...):
    factories = {
        ...
        "llmservingsim": lambda: LLMServingSimAdapter(llmservingsim_dir),
    }
```

### Configuration Management

**LLMServingSim directory location**:
- Default: `LLMServingSim/` in project root
- Configurable via constructor: `LLMServingSimAdapter(llmservingsim_dir="/path/to/sim")`
- Must contain: `main.py`, `llm_profile/perf_models/`, `cluster_config/`

**Template selection**:
- Base template: `LLMServingSim/cluster_config/single_node_single_instance.json`
- Always modify dynamically per experiment (no separate templates needed)
- Each experiment gets its own modified config in a temp directory

### Limitations and Scope

**Out of scope for initial implementation**:
- FP8 precision (only FP16 supported)
- Non-H100 hardware (A100, L40S, TPU)
- Models without H100 perf models (CodeLlama-34b, Llama-2-70b, Qwen3-14B)
- Prefill/Decode disaggregation (P/D instances)
- MoE expert routing customization (uses defaults)
- Prefix caching (ground-truth doesn't specify this)

**Expected coverage**: Approximately 20 out of 49 ground-truth experiments

### Future Extensions

**If additional hardware support is needed**:
1. Profile new hardware using `llm_profile/` tools
2. Add hardware to availability check in `can_run()`
3. Update memory bandwidth/latency values in cluster_config generation

**If additional models are needed**:
1. Check if model architecture is Llama-based
2. Profile using `llm_profile/` tools
3. Add to `MODEL_MAP`

**If FP8 support is needed**:
1. Verify LLMServingSim has FP8 perf models
2. Adjust `--fp` CLI flag from 16 → 8
3. Update memory calculations (1 byte per value instead of 2)
4. Add FP8 to precision check in `can_run()`

## Summary

The LLMServingSimAdapter enables validation of LLMServingSim against approximately 20 H100 ground-truth experiments by:

1. **Eligibility**: Filtering to H100 + (Llama-3.1-8B, Mixtral-8x7B) + FP16
2. **Config Generation**: Dynamically modifying cluster_config JSON with TP, memory, instance settings
3. **Workload Generation**: Reading ground-truth token counts, generating constant-rate arrivals
4. **Execution**: Running LLMServingSim via subprocess with temp configs
5. **Parsing**: Converting CSV output (nanoseconds) to SimulatorResult (milliseconds)
6. **Integration**: Following existing adapter patterns for seamless orchestration

**Key Features**:
- ✓ CPU offloading support (6/20 experiments)
- ✓ Multi-instance support (dp>1)
- ✓ Exact workload matching (reads ground-truth token counts)
- ✓ Per-stage metrics with weighted summary
- ✓ Comprehensive error handling and validation
- ✓ Clean temporary file management

**File Structure**:
```
experiment/
  adapters/
    llmservingsim.py          # Main adapter implementation
    __init__.py               # Add LLMServingSimAdapter export
tests/
  test_llmservingsim_adapter.py  # Unit and integration tests
```
