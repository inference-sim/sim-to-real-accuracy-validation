# LLM-Optimizer Evaluation Methodology

**Date**: 2026-03-04
**Simulator**: `llm-optimizer` (estimate mode — roofline analysis)
**Adapter**: `experiment/adapters/llm_optimizer_est.py`

---

## 1. What LLM-Optimizer Estimates

LLM-optimizer's `estimate_llm_performance` is an **analytical roofline model** that predicts LLM inference latency from first principles — GPU FLOPs, memory bandwidth, model architecture parameters, and concurrency level. It does **not** simulate batching, scheduling, queuing, or memory management. It produces a **single-point mean estimate** per invocation.

Key outputs per call:
- `e2e_latency_s` — end-to-end latency in **seconds**
- `ttft_ms` — time to first token in **milliseconds**
- `itl_ms` — inter-token latency in **milliseconds**
- `output_throughput_tps`, `input_throughput_tps`, `requests_per_sec` — throughput metrics

**Model coverage**: All 4 ground-truth models (Llama-2-7B, Llama-2-70B, Mixtral-8x7B, CodeLlama-34B). Model architecture is fetched at runtime via `get_model_config_from_hf()`.

---

## 2. Ground-Truth Dataset

The evaluation uses 16 vLLM experiments on H100 GPUs across a 4-model x 4-workload matrix. All experiments share a fixed configuration:

| Parameter | Value |
|-----------|-------|
| vLLM version | v0.15.1 |
| GPU | H100 |
| `max_model_len` | 4096 |
| `max_num_batched_tokens` | 2048 |
| `max_num_seqs` | 128 |
| Precision | fp16 |
| Prefix caching | Enabled |
| Chunked prefill | Enabled |
| KV CPU offloading | Enabled (8 GiB) |

Ground-truth latency percentiles (mean, P90, P99) for E2E, TTFT, and ITL are parsed from `stage_*_lifecycle_metrics.json` and `summary_lifecycle_metrics.json`. Values in those files are in **seconds** and are converted to **milliseconds** during parsing.

---

## 3. Per-Stage Evaluation Pipeline

LLM-optimizer is invoked **once per stage** within each experiment, not once per experiment. Multi-stage workloads (e.g., codegen at 5 req/s then 10 req/s) produce separate predictions per load level.

### 3.1 Input Derivation

For each stage, three inputs are derived from the ground-truth experiment:

#### Concurrency (via Little's Law)

```
concurrency = max(1, round(stage.rate × stage.e2e.mean / 1000))
```

- `stage.rate` — request arrival rate (req/s) from the ground-truth load profile
- `stage.e2e.mean` — ground-truth mean E2E latency in milliseconds
- Division by 1000 converts ms → seconds for dimensional correctness
- Result is clamped to `[1, max_num_seqs]` to respect the scheduler's batch cap

**Important**: This uses the ground-truth E2E latency as an *input*, which means the concurrency estimate is oracle-informed. In a real deployment scenario, concurrency would need to be estimated or measured independently. This methodology choice isolates the roofline model's latency prediction accuracy from concurrency estimation error.

#### Input/Output Token Lengths

Extracted from the experiment's `profile.yaml` configuration:

```
input_length  = profile_config["data"]["shared_prefix"]["question_len"]
              + profile_config["data"]["shared_prefix"]["system_prompt_len"]
output_length = profile_config["data"]["shared_prefix"]["output_len"]
```

These are the *configured* lengths, not per-request actual lengths. All requests in a given workload share the same distribution parameters.

#### Hardware and Model Configuration

| Parameter | Source |
|-----------|--------|
| `num_gpus` | `experiment.tp` (tensor parallelism from `exp-config.yaml`) |
| `gpu_name` | `"H100"` (hardcoded — all experiments use H100) |
| `model_config` | Fetched from HuggingFace via `get_model_config_from_hf(experiment.model)` |
| `precision` | `model_config.inferred_precision`, falling back to `"fp16"` |

### 3.2 LLM-Optimizer Invocation

```python
result = estimate_llm_performance(
    num_gpus=experiment.tp,
    gpu_name="H100",
    model_config=model_config,
    precision=precision,
    concurrency=concurrency,
    input_length=input_length,
    output_length=output_length,
)
```

The function returns a single `PerformanceResult` object with mean latency and throughput values.

### 3.3 Unit Conversion

| LLM-optimizer field | Unit | Conversion | Target unit |
|---------------------|------|------------|-------------|
| `e2e_latency_s` | seconds | × 1000 | milliseconds |
| `ttft_ms` | milliseconds | none | milliseconds |
| `itl_ms` | milliseconds | none | milliseconds |
| Throughput fields | tokens/s or req/s | none | tokens/s or req/s |

### 3.4 Heuristic Percentile Estimation

LLM-optimizer produces only mean estimates. P90 and P99 are derived via fixed multipliers:

| Percentile | Multiplier | Rationale |
|------------|------------|-----------|
| P90 | mean × 1.2 | Conservative heuristic for moderate tail |
| P99 | mean × 1.6 | Conservative heuristic for heavy tail |

These multipliers are applied identically to E2E, TTFT, and ITL. They are **not** calibrated to any empirical tail distribution — they serve as rough approximations. In the final report, all P90/P99 values from llm-optimizer are flagged with `†` to indicate heuristic origin.

---

## 4. Summary Computation

After all stages are evaluated, a weighted-average summary is computed:

- **Latency metrics** (E2E, TTFT, ITL mean/P90/P99): **request-weighted** average across stages. This is correct because latency is a per-request quantity — stages with more requests contribute proportionally more.

  ```
  summary.e2e.mean = Σ(stage.e2e.mean × stage.num_requests) / Σ(stage.num_requests)
  ```

- **Throughput metrics** (tokens/s, req/s): **duration-weighted** average across stages. This is correct because throughput is a rate quantity — stages with longer durations contribute proportionally more.

  ```
  summary.throughput.rps = Σ(stage.rps × stage.duration) / Σ(stage.duration)
  ```

Summary percentiles (P90, P99) also use the heuristic multipliers applied to the weighted-average mean, rather than aggregating per-stage percentiles. This is a further approximation.

---

## 5. Eligibility Check (`can_run`)

The adapter only runs for experiments whose `profile.yaml` uses the `shared_prefix` data type with all three required keys: `question_len`, `system_prompt_len`, `output_len`. All 16 ground-truth experiments satisfy this condition.

Experiments with non-`shared_prefix` data types (e.g., `random`) are skipped because llm-optimizer requires fixed input/output lengths — it cannot model variable-length request distributions.

---

## 6. Error Metrics

For each (experiment, stage) pair, the following errors are computed on each of the 9 metric variants (E2E/TTFT/ITL × mean/P90/P99):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAPE** | `\|predicted − actual\| / actual × 100` | Unsigned error magnitude (%) |
| **MPE** | `(predicted − actual) / actual × 100` | Signed error direction (+  = over-predict) |
| **Absolute Error** | `\|predicted − actual\|` | Raw magnitude in ms or tokens/s |

Aggregation across experiments:
- **Per-simulator MAPE**: Mean of per-experiment MAPEs for each metric variant
- **Per-model breakdown**: MAPE grouped by model (averages across workloads)
- **Per-workload breakdown**: MAPE grouped by workload type (averages across models)

---

## 7. Known Limitations and Caveats

| Limitation | Impact on Evaluation |
|------------|---------------------|
| **No batching simulation** | LLM-optimizer models a single concurrency level, not dynamic batch formation. Real vLLM batching decisions (continuous batching, chunked prefill) are not captured. Expect over-prediction of TTFT under high load. |
| **Oracle concurrency** | Concurrency is derived from ground-truth E2E latency. This gives llm-optimizer an advantage it would not have in practice. |
| **Heuristic percentiles** | P90 and P99 are not modeled — they are fixed multiples of the mean. Workloads with heavy tails (e.g., reasoning with 1448 output tokens) will likely show large P99 errors. |
| **No KV cache pressure modeling** | LLM-optimizer does not model GPU memory, KV block exhaustion, preemption, or CPU offloading. Under high KV pressure, real latencies spike but llm-optimizer predictions remain flat. |
| **No prefix caching modeling** | All ground-truth experiments have prefix caching enabled. LLM-optimizer models full prefill for every request, likely over-predicting TTFT for workloads with high prefix reuse. |
| **Fixed input/output lengths** | The adapter uses configured distribution parameters, not actual per-request token counts. Variance in real request sizes is not captured. |
| **Single precision assumption** | Uses `model_config.inferred_precision` (typically fp16). If the real deployment uses mixed precision or quantization, predictions may diverge. |

---

## 8. Expected Error Profile

Based on the methodology's constraints, the expected error pattern is:

- **TTFT**: Likely the most accurate metric — roofline prefill latency estimation is well-grounded in compute/bandwidth analysis.
- **E2E mean**: Moderate accuracy — captures the roofline decode throughput but misses queuing, preemption, and batching effects.
- **ITL**: Should be close for compute-bound decoding but may diverge under memory-bandwidth-bound regimes or high batch sizes.
- **P90/P99 (all metrics)**: Expected to be the least accurate due to heuristic multipliers. Real tail latencies are driven by scheduling jitter, preemption, and KV cache pressure — none of which are modeled.
