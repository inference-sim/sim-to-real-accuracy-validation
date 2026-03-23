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

LLM-optimizer is invoked via a **concurrency sweep** once per experiment, then per-stage predictions are extracted by matching the stage's arrival rate to the predicted throughput. This approach avoids oracle data leakage by deriving concurrency from the simulator's own predictions rather than ground-truth measurements.

### 3.1 Input Derivation

For each experiment, the following inputs are extracted:

#### Concurrency (via Throughput Matching)

Instead of using ground-truth E2E latency (oracle-informed), concurrency is now determined by **throughput matching**:

1. **Sweep concurrency levels**: `[1, 2, 4, 8, 16, 32, 64, 128, ...]` up to `max_num_seqs`
2. **For each concurrency N**, call `estimate_llm_performance(concurrency=N)` to get `predicted_requests_per_sec = N / predicted_E2E_latency`
3. **For each stage with arrival rate λ**, find the concurrency N where `predicted_requests_per_sec ≈ λ`
4. **Extract predictions** (TTFT, ITL, E2E) from the matched concurrency level

This is **self-consistent via Little's Law**: at steady state, `arrival_rate = concurrency / E2E_latency`. By finding the concurrency where the roofline model's predicted throughput matches the stage's arrival rate, we establish an equilibrium point without using ground-truth latency as input.

**Example**:
```
Stage rate = 10 req/s
Sweep results:
  N=1  → predicted_rate=6.1 req/s   (too low)
  N=2  → predicted_rate=10.8 req/s  (MATCH! ✓)
  N=4  → predicted_rate=18.6 req/s  (too high)
→ Use predictions from N=2
```

**Key advantage**: No oracle data leakage. The only ground-truth input is the arrival rate (λ), which is part of the workload specification, not a measurement. All latency predictions come entirely from the roofline model.

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

The adapter performs a **concurrency sweep** once per experiment:

```python
# Sweep concurrency levels [1, 2, 4, 8, 16, ...]
sweep_results = []
for concurrency in [1, 2, 4, 8, 16, 32, 64, 128, ...]:
    result = estimate_llm_performance(
        num_gpus=experiment.tp,
        gpu_name="H100",
        model_config=model_config,
        precision=precision,
        concurrency=concurrency,
        input_length=input_length,
        output_length=output_length,
    )
    sweep_results.append((concurrency, result))

    # Stop if VRAM exhausted
    if result.ttft_ms == float("inf"):
        break

# Then for each stage, match throughput
for stage in experiment.stages:
    concurrency, perf = match_throughput(
        stage_rate=stage.rate,
        sweep_results=sweep_results,
    )
    # Use perf.ttft_ms, perf.itl_ms, perf.e2e_latency_s
```

Each sweep call returns a `PerformanceResult` object with mean latency and throughput values. The sweep stops when VRAM capacity is exceeded (indicated by `ttft_ms == inf`).

### 3.3 Unit Conversion

| LLM-optimizer field | Unit | Conversion | Target unit |
|---------------------|------|------------|-------------|
| `e2e_latency_s` | seconds | × 1000 | milliseconds |
| `ttft_ms` | milliseconds | none | milliseconds |
| `itl_ms` | milliseconds | none | milliseconds |
| Throughput fields | tokens/s or req/s | none | tokens/s or req/s |

### 3.4 Percentile Handling

LLM-optimizer produces only **mean** estimates — it does not model latency distributions. P90 and P99 are left as `None` (not reported). The metrics layer skips comparisons where the simulator does not provide a value, and the report tables display **N/A** for those columns.

---

## 4. Summary Computation

After all stages are evaluated, a weighted-average summary is computed:

- **Latency metrics** (E2E, TTFT, ITL mean): **request-weighted** average across stages. This is correct because latency is a per-request quantity — stages with more requests contribute proportionally more.

  ```
  summary.e2e.mean = Σ(stage.e2e.mean × stage.num_requests) / Σ(stage.num_requests)
  ```

- **Throughput metrics** (tokens/s, req/s): **duration-weighted** average across stages. This is correct because throughput is a rate quantity — stages with longer durations contribute proportionally more.

  ```
  summary.throughput.rps = Σ(stage.rps × stage.duration) / Σ(stage.duration)
  ```

Summary P90 and P99 are `None` (not reported), consistent with per-stage handling.

---

## 5. Eligibility Check (`can_run`)

The adapter only runs for experiments whose `profile.yaml` uses the `shared_prefix` data type with all three required keys: `question_len`, `system_prompt_len`, `output_len`. All 16 ground-truth experiments satisfy this condition.

Experiments with non-`shared_prefix` data types (e.g., `random`) are skipped because llm-optimizer requires fixed input/output lengths — it cannot model variable-length request distributions.

---

## 6. Error Metrics

For each (experiment, stage) pair, the following errors are computed on each metric variant where the simulator provides a value. Since llm-optimizer only produces mean estimates, errors are computed for 3 metrics (E2E/TTFT/ITL × mean). P90 and P99 columns show **N/A** in the report.

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
| **Throughput-matching concurrency** | Concurrency is now derived via throughput matching (no oracle data). This is realistic but assumes the roofline model's throughput predictions are accurate. Errors in predicted throughput propagate to concurrency selection, which then affects latency predictions. |
| **No percentile modeling** | P90 and P99 are not produced — only mean estimates are available. The report shows N/A for these columns. |
| **No KV cache pressure modeling** | LLM-optimizer does not model GPU memory, KV block exhaustion, preemption, or CPU offloading. Under high KV pressure, real latencies spike but llm-optimizer predictions remain flat. |
| **No prefix caching modeling** | All ground-truth experiments have prefix caching enabled. LLM-optimizer models full prefill for every request, likely over-predicting TTFT for workloads with high prefix reuse. |
| **Fixed input/output lengths** | The adapter uses configured distribution parameters, not actual per-request token counts. Variance in real request sizes is not captured. |
| **Precision now from experiment** | Previously used `model_config.inferred_precision` (HF model default). Now passes `experiment.precision.lower()` directly. This correctly uses FP8 TFLOPS (1978 on H100, 2x FP16) for FP8 experiments. A100+FP8 is excluded via `can_run()` since `A100.FP8_TFLOPS=None`. |
| **MoE treated as dense** | Roofline hardcodes `d_ff = 4 * d_model` with no expert routing. MoE experiments (Mixtral-8x7B, Mixtral-8x22B, Llama-4-Scout) run but produce inaccurate results — the model treats all activated parameters as if they fire on every token. |
| **Config knobs not modeled** | `max_num_batched_tokens`, `cpu_offload`, `gpu_mem_util` are not roofline inputs. Experiments sweeping these (IDs 22–25, 27–30) produce identical estimates regardless of the sweep value. |
| **dp not in roofline** | Multi-instance experiments (IDs 32–35) are simulated as single-instance. DP exists in llm-optimizer's tuning/command-gen layer but not in the roofline estimator used by the adapter. |

---

## 8. Computational Cost

The throughput-matching approach changes the cost profile compared to the previous oracle-informed method:

**Previous (oracle-informed)**:
- Per stage: 1 call to `estimate_llm_performance()`
- Total for N-stage experiment: N calls

**Current (throughput-matching)**:
- One-time sweep: ~10 calls to `estimate_llm_performance()` (stops at VRAM limit)
- Per stage: O(log N) lookup/interpolation to match throughput
- Total for N-stage experiment: ~10 calls (amortized across all stages)

**Result**: For multi-stage experiments, throughput-matching can be **more efficient** than oracle-informed, since the sweep cost is amortized across all stages. For single-stage experiments, throughput-matching is ~10× slower (10 calls vs 1).

---

## 9. Expected Error Profile

Based on the methodology's constraints, the expected error pattern is:

- **TTFT**: Likely the most accurate metric — roofline prefill latency estimation is well-grounded in compute/bandwidth analysis.
- **E2E mean**: Moderate accuracy — captures the roofline decode throughput but misses queuing, preemption, and batching effects.
- **ITL**: Should be close for compute-bound decoding but may diverge under memory-bandwidth-bound regimes or high batch sizes.
- **P90/P99 (all metrics)**: Not reported — llm-optimizer does not model latency distributions. The report shows N/A for these columns.
