# Vidur Evaluation Methodology

**Date**: 2026-03-04
**Simulator**: Vidur (discrete-event simulation with profiled execution times)
**Adapter**: `experiment/adapters/vidur.py`
**Trace converter**: `experiment/vidur_trace_converter.py`
**Reference**: Agrawal et al., "Vidur: A Large-Scale Simulation Framework For LLM Inference", MLSys 2024

---

## 1. What Vidur Simulates

Vidur is a **discrete-event LLM inference simulator** that predicts request-level latencies by combining:

- **Profiled GPU kernel execution times** — MLP, attention, and communication kernels profiled on real hardware (H100) across a sweep of batch sizes and KV cache lengths. Stored as lookup tables under `data/profiling/compute/h100/<model>/`.
- **PagedAttention-style block management** — models GPU memory allocation, block-level KV cache tracking, and preemption when memory is exhausted.
- **vLLM scheduling** — models continuous batching with the vLLM scheduler (batch size cap, chunked prefill, waiting queue management).

Unlike llm-optimizer's analytical roofline, Vidur **simulates every scheduling step** and produces **per-request output metrics**, from which true percentiles (mean, P90, P99) are computed without heuristic multipliers.

---

## 2. Model Coverage

Vidur requires pre-profiled GPU kernel timing data. Only models with H100 profiling data are supported:

| Model | HuggingFace ID | TP | Profiling Path |
|-------|----------------|----|----------------|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | 1 | `data/profiling/compute/h100/meta-llama/Llama-2-7b-hf/` |
| Llama-2-70B | `meta-llama/Llama-2-70b-hf` | 4 | `data/profiling/compute/h100/meta-llama/Llama-2-70b-hf/` |
| CodeLlama-34B | `codellama/CodeLlama-34b-Instruct-hf` | 2 | `data/profiling/compute/h100/codellama/CodeLlama-34b-Instruct-hf/` |

**Excluded**: Mixtral-8x7B (`mistralai/Mixtral-8x7B-v0.1`) — Vidur has no H100 profiling data for Mixture-of-Experts architectures. Profiling MoE models requires GPU access for kernel measurement.

This means Vidur is evaluated on **12 of 16 experiments** (3 models × 4 workloads). All Vidur results in the aggregate report are annotated with `‡` to indicate partial model coverage.

---

## 3. Ground-Truth Dataset

Same 16 vLLM experiments as all other simulators. See the [design doc](2026-03-04-sim-to-real-accuracy-validation-design.md) §1 for the full 4×4 matrix. Key parameters matched:

| Parameter | Ground Truth | Mapped to Vidur |
|-----------|-------------|-----------------|
| GPU | H100 | `--replica_config_device h100` |
| Tensor parallelism | per-model | `--replica_config_tensor_parallel_size` |
| `max_num_seqs` | 128 | `--vllm_scheduler_config_batch_size_cap` |
| `max_num_batched_tokens` | 2048 | `--vllm_scheduler_config_max_tokens_in_batch` |
| Scheduler | vLLM continuous batching | `--replica_scheduler_config_type vllm` |
| Pipeline stages | 1 | `--replica_config_num_pipeline_stages 1` |
| Replicas | 1 | `--cluster_config_num_replicas 1` |

---

## 4. Trace Conversion Pipeline

Vidur is evaluated using **trace replay** — the exact request sequence from the ground-truth experiment is replayed through the simulator. This isolates Vidur's latency prediction accuracy from any workload generation variance.

### 4.1 Input Format

The ground-truth `per_request_lifecycle_metrics.json` contains per-request records:

```json
{
  "start_time": 1393.251,
  "end_time": 1395.195,
  "request": "{\"model\": \"...\", \"max_tokens\": 247, ...}",
  "info": {
    "input_tokens": 591,
    "output_tokens": 140,
    "output_token_times": [1393.381, ...]
  }
}
```

### 4.2 Output Format

Vidur's `trace_replay` request generator expects a 3-column CSV:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens
0.0,591,140
0.125,547,248
```

### 4.3 Conversion Logic

Implemented in `experiment/vidur_trace_converter.py`:

| Output field | Source | Transformation |
|-------------|--------|----------------|
| `arrived_at` | `start_time` | Relative to first request: `start_time - first_start_time` (seconds) |
| `num_prefill_tokens` | `info.input_tokens` | Direct copy |
| `num_decode_tokens` | `info.output_tokens` | Direct copy |

The arrival times preserve the exact inter-request timing from the real experiment. The first request always has `arrived_at = 0.0`.

---

## 5. Simulator Invocation

Vidur is invoked as a subprocess:

```bash
python -m vidur.main \
    --replica_config_model_name <model> \
    --replica_config_device h100 \
    --replica_config_tensor_parallel_size <tp> \
    --replica_config_num_pipeline_stages 1 \
    --cluster_config_num_replicas 1 \
    --replica_scheduler_config_type vllm \
    --vllm_scheduler_config_batch_size_cap <max_num_seqs> \
    --vllm_scheduler_config_max_tokens_in_batch <max_num_batched_tokens> \
    --request_generator_config_type trace_replay \
    --trace_request_generator_config_trace_file <trace.csv> \
    --metrics_config_output_dir <output_dir>
```

The subprocess runs with `cwd` set to the cloned Vidur repository directory so that relative profiling data paths resolve correctly.

On failure, the adapter wraps the `CalledProcessError` into a `RuntimeError` with the stderr output for diagnostics.

---

## 6. Result Parsing

### 6.1 Output Location

Vidur writes results to a timestamped subdirectory: `<output_dir>/<timestamp>/request_metrics.csv`. The adapter discovers this file by globbing `<output_dir>/*/request_metrics.csv`.

### 6.2 Per-Request Metrics CSV

The CSV contains one row per simulated request. Key columns:

| Column | Meaning | Unit |
|--------|---------|------|
| `request_e2e_time` | End-to-end latency | seconds |
| `prefill_e2e_time` | Time to first token (prefill latency) | seconds |
| `decode_time_execution_plus_preemption_normalized` | Average inter-token latency (includes preemption time) | seconds |
| `request_num_prefill_tokens` | Input token count | tokens |
| `request_num_decode_tokens` | Output token count | tokens |

### 6.3 Unit Conversion

All Vidur latency values are in **seconds**. They are converted to **milliseconds** (× 1000) to match the `LatencyDistribution` dataclass used throughout the evaluation harness.

### 6.4 Data Validation

Rows missing any of the three required latency columns (`request_e2e_time`, `prefill_e2e_time`, `decode_time_execution_plus_preemption_normalized`) or containing non-numeric values are silently dropped. Token count columns that appear as floats (e.g., `"512.0"`) are parsed via `int(float(...))`.

---

## 7. Per-Stage Splitting

Ground-truth experiments use multi-stage load profiles (e.g., codegen: 5 req/s for 600s, then 10 req/s for 600s). To compute per-stage error metrics, Vidur's flat request output must be split into stage buckets.

### Splitting Strategy: Request-Count Boundaries

Unlike BLIS (which splits by arrival timestamp), Vidur's adapter splits by **request index** because:
1. Vidur's output CSV preserves request-ID order (which matches arrival order from the trace)
2. The output does not include an explicit arrival timestamp column

Stage boundaries are computed as cumulative expected request counts:

```
boundary[i] = Σ(rate[j] × duration[j]) for j = 0..i
```

For example, with stages `[{rate: 5, duration: 600}, {rate: 10, duration: 600}]`:
- Stage 0 boundary: `5 × 600 = 3000` (requests 0–2999)
- Stage 1 boundary: `3000 + 10 × 600 = 9000` (requests 3000–8999)

Requests past all boundaries are assigned to the last stage.

### Percentile Computation

Within each stage bucket, numpy percentiles are computed directly from per-request latency values:

```python
e2e_vals = [float(row["request_e2e_time"]) * 1000 for row in bucket]
mean = np.mean(e2e_vals)
p90  = np.percentile(e2e_vals, 90)
p99  = np.percentile(e2e_vals, 99)
```

This is a **true percentile computation** from simulated per-request data — no heuristic multipliers are needed (unlike llm-optimizer).

### Throughput Computation

Per-stage throughput is derived from token counts and stage duration:

```
input_tokens_per_sec  = sum(prefill_tokens in bucket) / stage_duration
output_tokens_per_sec = sum(decode_tokens in bucket) / stage_duration
requests_per_sec      = len(bucket) / stage_duration
```

---

## 8. Summary Computation

The summary aggregates all rows across all stages. It uses the same `_compute_stage` method with:
- `stage_index = -1`
- `duration = total experiment duration` (sum of all stage durations)

This means summary percentiles are computed across all requests in the experiment, matching how the ground truth computes its `summary_lifecycle_metrics.json`.

---

## 9. Error Metrics

For each (experiment, stage) pair, errors are computed on 9 metric variants (E2E/TTFT/ITL × mean/P90/P99):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAPE** | `\|predicted − actual\| / actual × 100` | Unsigned error magnitude (%) |
| **MPE** | `(predicted − actual) / actual × 100` | Signed error direction (+ = over-predict) |
| **Absolute Error** | `\|predicted − actual\|` | Raw magnitude in ms or tokens/s |

Aggregation:
- **Per-simulator MAPE**: Mean across the 12 experiments where Vidur can run (excludes Mixtral-8x7B)
- **Per-model breakdown**: MAPE averaged across workloads for each of the 3 supported models
- **Per-workload breakdown**: MAPE averaged across the 3 supported models for each workload

---

## 10. Known Limitations and Caveats

| Limitation | Impact on Evaluation |
|------------|---------------------|
| **No MoE support** | Vidur has zero MoE references. All MoE models (Mixtral-8x7B, Mixtral-8x22B, Llama-4-Scout) are excluded. Aggregate MAPE reflects only dense-architecture models. |
| **No CPU KV offloading** | All ground-truth experiments use vLLM's `OffloadingConnector` (8 GiB CPU). Vidur's `MemoryPlanner` computes KV blocks from GPU memory only. Under high KV pressure, real vLLM offloads blocks to CPU (avoiding preemption), but Vidur may preempt or reject requests instead. Expect Vidur to **over-predict** E2E latency under memory pressure. |
| **No prefix caching** | All experiments enable prefix caching. Vidur computes full prefill for every request regardless of shared prefixes. This causes Vidur to **over-predict TTFT** for workloads with high prefix reuse (e.g., `shared_prefix` data type where all requests share `system_prompt_len` tokens). |
| **Profiling data extrapolation** | Execution time predictions for batch sizes or KV cache lengths outside the profiled range are extrapolated from the nearest profiled data points. Accuracy degrades at the extremes. |
| **Scheduler model fidelity** | Vidur models vLLM's scheduler but may not capture every scheduling heuristic or edge case in vLLM v0.15.1 (e.g., exact chunked prefill budget allocation, priority queue ordering). |
| **No network/host overhead** | Vidur models GPU execution time only. Real-world overhead from tokenization, detokenization, HTTP transport, and host-side scheduling is not captured. |
| **Stage splitting approximation** | Splitting by `rate × duration` request counts assumes the exact expected number of requests per stage was achieved. In practice, request counts may differ slightly from `rate × duration` due to timing jitter and ramp-up effects. |
| **Multi-instance (dp) approximation** | For dp>1 experiments, Vidur uses `--cluster_config_num_replicas` with `round_robin` global scheduler. This assumes: (1) the trace represents the **aggregate** workload across all dp instances, (2) round-robin distribution approximates the ground truth's actual load balancing, (3) `num_blocks` passed is the per-instance value. Vidur models independent per-replica queuing/batching correctly, but routing policy mismatch may affect tail latency accuracy. |
| **No chunked prefill in vLLM scheduler** | Vidur's vLLM scheduler processes entire prefill in one shot; real vLLM v0.15.1 chunks prefill when `max_num_batched_tokens` < prefill length. This is a TTFT fidelity gap (not just an edge case). Only the Sarathi scheduler has chunked prefill, but our adapter uses the vLLM scheduler. |
| **gpu_mem not modeled** | The adapter passes explicit `num_blocks` from ground truth, bypassing Vidur's `memory_margin_fraction`. Experiments with `gpu_mem=0.95` (IDs 25, 30) are simulated with the same KV block budget as `gpu_mem=0.9` — the difference in available memory is not reflected. |

---

## 11. Expected Error Profile

Based on Vidur's simulation methodology:

- **TTFT**: Likely over-predicted due to missing prefix caching. The `shared_prefix` workload type means many requests share a common system prompt — real vLLM caches these prefill blocks, but Vidur recomputes them. Over-prediction magnitude depends on system prompt length relative to total input length.
- **E2E mean**: Moderate accuracy expected. Vidur models batch execution, scheduling, and preemption but misses CPU offloading benefits. Under light load (roleplay at 6 req/s), accuracy should be good. Under high load with KV pressure (general at 20 req/s), expect divergence.
- **ITL**: Should be the most accurate metric — Vidur's profiled kernel timings directly model per-token decode execution. The `decode_time_execution_plus_preemption_normalized` metric includes preemption delays, matching real behavior.
- **P90/P99**: Unlike llm-optimizer, these are true percentiles from simulated per-request data. Expect better tail accuracy than heuristic multipliers, though KV-pressure-driven tail spikes may be under- or over-predicted depending on whether Vidur preempts where real vLLM offloads.
- **Per-model variation**: Accuracy likely correlates with how well the profiling data covers the experiment's operating regime. Llama-2-7B (TP=1, moderate batches) may be more accurate than Llama-2-70B (TP=4, larger batches, more communication overhead).

---

## 12. Comparison with LLM-Optimizer Methodology

| Dimension | Vidur | LLM-Optimizer |
|-----------|-------|---------------|
| **Approach** | Discrete-event simulation | Analytical roofline |
| **Percentiles** | True (from per-request data) | Heuristic (mean × multiplier) |
| **Scheduling** | Modeled (vLLM continuous batching) | Not modeled |
| **Memory management** | Modeled (PagedAttention blocks) | Not modeled |
| **Model coverage** | 3/4 models (no Mixtral) | 4/4 models |
| **Input dependency on ground truth** | Trace replay only (arrival times + token counts) | Oracle concurrency (uses ground-truth E2E) |
| **Execution** | Subprocess (seconds to minutes per experiment) | In-process Python call (milliseconds) |
| **KV offloading** | Not modeled | Not modeled |
| **Prefix caching** | Not modeled | Not modeled |
