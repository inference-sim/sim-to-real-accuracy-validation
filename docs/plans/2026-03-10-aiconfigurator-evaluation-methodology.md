# AIConfigurator Evaluation Methodology

**Date**: 2026-03-10
**Simulator**: AIConfigurator v0.4.0 (NVIDIA analytical configuration optimizer)
**Adapter**: `experiment/adapters/aiconfigurator_est.py`
**Reference**: [ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)

---

## 1. What AIConfigurator Estimates

AIConfigurator is NVIDIA's **analytical performance model** for LLM inference serving. Given a model, GPU system, backend, and sequence lengths, it sweeps across parallelism strategies (TP, PP, DP) and concurrency levels to produce a **Pareto DataFrame** of achievable configurations. Each row contains predicted latency and throughput for one (parallelism, concurrency) combination.

Key outputs per row in the Pareto DataFrame:
- `ttft` — time to first token in **milliseconds**
- `tpot` — time per output token in **milliseconds**
- `concurrency` — number of concurrent in-flight requests
- `tp`, `pp`, `dp` — parallelism decomposition
- `seq/s` — request throughput
- `tokens/s` — output token throughput
- `tokens/s/gpu` — per-GPU throughput efficiency
- `memory` — estimated GPU memory consumption (GiB)

AIConfigurator does **not** simulate batching, scheduling, queuing, or memory management. It models TTFT and TPOT independently from profiled GPU kernel performance data (GEMM, attention, NCCL communication). Like llm-optimizer, it produces **point estimates** — one value per metric per configuration row.

**Difference from llm-optimizer**: llm-optimizer is called once per concurrency level and returns a single result. AIConfigurator is called once per experiment and returns an entire DataFrame of results across a concurrency sweep, from which we look up the row matching each stage's concurrency.

---

## 2. Model Coverage

AIConfigurator has a built-in `SupportedModels` dictionary mapping model names to architecture parameters (layers, heads, hidden size, etc.). Models not in this dictionary are resolved at runtime via HuggingFace model configs.

| Ground-Truth Model | HuggingFace ID | AIConfigurator Name | Supported |
|---|---|---|---|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | `LLAMA2_7B` | Yes |
| Llama-2-70B | `meta-llama/Llama-2-70b-hf` | `LLAMA2_70B` | Yes |
| CodeLlama-34B | `codellama/CodeLlama-34b-Instruct-hf` | (HF ID passthrough) | Yes |
| Mixtral-8x7B | `mistralai/Mixtral-8x7B-v0.1` | `MOE_Mixtral8x7B` | **No** — vLLM backend unsupported |

**Excluded**: Mixtral-8x7B is a Mixture-of-Experts model. AIConfigurator raises `NotImplementedError("AIConfigurator does not yet support MOE models for VLLM backend.")` during `TaskConfig.validate()`. Since all ground-truth data was collected on vLLM, we must use the `vllm` backend, which means MoE models cannot be evaluated.

This means AIConfigurator is evaluated on **12 of 16 experiments** (3 models × 4 workloads).

### Model Name Mapping

The adapter maintains an explicit mapping for models with known AIConfigurator entries:

```python
_MODEL_MAP = {
    "meta-llama/Llama-2-7b-hf": "LLAMA2_7B",
    "meta-llama/Llama-2-70b-hf": "LLAMA2_70B",
}
```

Models not in `_MODEL_MAP` (e.g., CodeLlama-34B) are passed through as raw HuggingFace IDs. AIConfigurator resolves these via `get_model_config_from_hf_id()`, which downloads the model config from HuggingFace and infers architecture parameters. CodeLlama uses `LlamaForCausalLM`, so this works transparently.

---

## 3. Ground-Truth Dataset

The evaluation uses 16 vLLM experiments on H100 GPUs across a 4-model × 4-workload matrix. All experiments share a fixed configuration:

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

## 4. Argument Mapping (Ground Truth → AIConfigurator)

The adapter translates ground-truth experiment parameters into AIConfigurator's `TaskConfig` constructor:

| Ground Truth Source | AIConfigurator `TaskConfig` Parameter | Value / Derivation |
|---|---|---|
| `experiment.model` | `model_name` | Via `_MODEL_MAP` or HF ID passthrough |
| `experiment.tp` | `total_gpus` | Direct (constrains parallelism search) |
| `question_len + system_prompt_len` | `isl` | From `profile_config["data"]["shared_prefix"]` |
| `output_len` | `osl` | From `profile_config["data"]["shared_prefix"]` |
| H100 (all experiments) | `system_name` | `"h100_sxm"` (hardcoded) |
| vLLM (all experiments) | `backend_name` | `"vllm"` (hardcoded) |
| — | `serving_mode` | `"agg"` (standard non-disaggregated serving) |
| — | `ttft` | `5000.0` ms (relaxed constraint for full sweep) |
| — | `tpot` | `200.0` ms (relaxed constraint for full sweep) |

The `ttft` and `tpot` parameters are intentionally set to relaxed values (5000 ms and 200 ms respectively). These act as **upper-bound constraints** in AIConfigurator's Pareto analysis — configurations exceeding these thresholds are excluded. By setting them high, we ensure the full concurrency sweep is returned, allowing us to look up any concurrency level.

All other configuration (quantization modes, kernel profiling data, etc.) is left to AIConfigurator's auto-detection.

---

## 5. Per-Stage Evaluation Pipeline

Unlike llm-optimizer (called once per stage), AIConfigurator is called **once per experiment**. The resulting Pareto DataFrame contains rows for multiple concurrency levels, from which per-stage values are looked up.

### 5.1 AIConfigurator Invocation

```python
task_config = TaskConfig(
    serving_mode="agg",
    model_name=resolved_model_name,
    system_name="h100_sxm",
    backend_name="vllm",
    total_gpus=experiment.tp,
    isl=input_length,
    osl=output_length,
    ttft=5000.0,
    tpot=200.0,
    yaml_config=_H100_VLLM_QUANT_CONFIG,
)
result = TaskRunner().run(task_config)
pareto_df = result["pareto_df"]
```

`TaskRunner.run()` dispatches to the vLLM backend's `run_agg()` method, which iterates over parallelism configurations (TP, PP, DP combinations that multiply to `total_gpus`) and concurrency levels, computing TTFT and TPOT for each combination using profiled kernel timing data.

The returned `pareto_df` typically contains dozens of rows — one per feasible (parallelism, concurrency) point.

### 5.2 TP Filtering

The Pareto DataFrame may contain rows with different TP/PP/DP decompositions. We filter to rows matching the experiment's exact tensor parallelism:

```python
tp_df = pareto_df[pareto_df["tp"] == experiment.tp]
```

With `total_gpus=experiment.tp`, the only valid decomposition is `tp=total_gpus, pp=1, dp=1`, so this filter typically retains all rows. It exists as a safety check.

### 5.3 Per-Stage Concurrency Derivation (Little's Law)

For each ground-truth stage, concurrency is estimated identically to the llm-optimizer adapter:

```
concurrency = max(1, round(stage.rate × stage.e2e.mean / 1000))
```

- `stage.rate` — request arrival rate (req/s) from the ground-truth load profile
- `stage.e2e.mean` — ground-truth mean E2E latency in milliseconds
- Division by 1000 converts ms → seconds for dimensional correctness
- Result is clamped to `[1, max_num_seqs]` to respect the scheduler's batch cap

**Important**: This uses the ground-truth E2E latency as an *input*, making the concurrency estimate oracle-informed. See §7.

### 5.4 Nearest-Concurrency Lookup

The derived concurrency may not exactly match any row in the Pareto DataFrame. The adapter finds the **nearest available concurrency**:

```python
idx = (df["concurrency"] - target).abs().idxmin()
row = df.loc[idx]
```

For example, if the derived concurrency is 9 but the DataFrame has rows for concurrency 5, 10, 20, the adapter selects the row with concurrency 10 (distance 1 < distance 4).

### 5.5 Metric Extraction and E2E Derivation

From the selected row:

| Metric | Source | Unit |
|--------|--------|------|
| TTFT | `row["ttft"]` | milliseconds (direct) |
| ITL | `row["tpot"]` | milliseconds (TPOT ≈ inter-token latency) |
| E2E | `ttft + tpot × output_length` | milliseconds (derived) |

**E2E derivation**: AIConfigurator does not directly output an end-to-end latency. E2E is computed as the sum of prefill time (TTFT) plus total decode time (TPOT × number of output tokens). This assumes sequential prefill-then-decode with no overlap, which matches the standard autoregressive inference pipeline.

### 5.6 Throughput Extraction

| Metric | Source | Derivation |
|--------|--------|------------|
| `requests_per_sec` | `row["seq/s"]` | Direct from DataFrame |
| `output_tokens_per_sec` | `row["tokens/s"]` | Direct from DataFrame |
| `input_tokens_per_sec` | `row["seq/s"] × isl` | Derived (requests × input length) |

### 5.7 Heuristic Percentile Estimation

AIConfigurator produces only mean estimates per concurrency level. P90 and P99 are derived via fixed multipliers (identical to llm-optimizer):

| Percentile | Multiplier | Rationale |
|------------|------------|-----------|
| P90 | mean × 1.2 | Conservative heuristic for moderate tail |
| P99 | mean × 1.6 | Conservative heuristic for heavy tail |

These multipliers are applied to E2E, TTFT, and ITL. They are **not** calibrated to any empirical tail distribution.

---

## 6. Summary Computation

After all stages are evaluated, a weighted-average summary is computed:

- **Latency metrics** (E2E, TTFT, ITL mean/P90/P99): **request-weighted** average across stages. This is correct because latency is a per-request quantity — stages with more requests contribute proportionally more.

  ```
  summary.e2e.mean = Σ(stage.e2e.mean × stage.num_requests) / Σ(stage.num_requests)
  ```

- **Throughput metrics** (tokens/s, req/s): **duration-weighted** average across stages. This is correct because throughput is a rate quantity — stages with longer durations contribute proportionally more.

  ```
  summary.throughput.rps = Σ(stage.rps × stage.duration) / Σ(stage.duration)
  ```

Summary percentiles (P90, P99) also use the heuristic multipliers applied to the weighted-average mean, rather than aggregating per-stage percentiles.

---

## 7. Eligibility Check (`can_run`)

The adapter requires two conditions:

1. **`shared_prefix` data type** — The experiment's `profile.yaml` must use `shared_prefix` with all three keys: `question_len`, `system_prompt_len`, `output_len`. This is needed because AIConfigurator requires fixed ISL/OSL values.

2. **Non-MoE model** — The experiment's model must not be in the `_MOE_MODELS` set (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x22B-v0.1`). AIConfigurator's vLLM backend does not support Mixture-of-Experts architectures.

All 12 non-Mixtral experiments satisfy both conditions.

---

## 8. Error Metrics

For each (experiment, stage) pair, the following errors are computed on each of the 9 metric variants (E2E/TTFT/ITL × mean/P90/P99):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAPE** | `\|predicted − actual\| / actual × 100` | Unsigned error magnitude (%) |
| **MPE** | `(predicted − actual) / actual × 100` | Signed error direction (+ = over-predict) |
| **Absolute Error** | `\|predicted − actual\|` | Raw magnitude in ms or tokens/s |

Aggregation:
- **Per-simulator MAPE**: Mean across the 12 experiments where AIConfigurator can run (excludes Mixtral-8x7B)
- **Per-model breakdown**: MAPE averaged across workloads for each of the 3 supported models
- **Per-workload breakdown**: MAPE averaged across the 3 supported models for each workload

---

## 9. Installation Note: Git LFS Data Files

AIConfigurator ships GPU kernel profiling data as CSV files under `systems/data/` (e.g., `gemm_perf.txt`, `context_attention_perf.txt`). These files are tracked by **Git LFS** in the source repository. When installing from a local clone via `pip install -e .` or `pip install .`, the LFS pointer files (3-line text stubs) may be copied instead of the actual data.

**Symptom**: `KeyError: 'gemm_dtype'` during `PerfDatabase` initialization — the CSV reader parses the LFS pointer text instead of tabular data.

**Fix**: Ensure Git LFS data is fetched before installing:

```bash
cd /path/to/aiconfigurator
git lfs pull
pip install -e .
```

Or manually copy the resolved data files from the source repo into site-packages:

```bash
cp -r src/aiconfigurator/systems/data/ \
  $(python -c "import aiconfigurator; print(aiconfigurator.__path__[0])")/systems/data/
```

---

## 10. Known Limitations and Caveats

| Limitation | Impact on Evaluation |
|------------|---------------------|
| **No batching simulation** | AIConfigurator models steady-state TTFT and TPOT at a given concurrency level. It does not simulate dynamic batch formation, continuous batching, or chunked prefill. Real vLLM batching decisions affect latency under load. |
| **Oracle concurrency** | Concurrency is derived from ground-truth E2E latency. This gives AIConfigurator an advantage it would not have in practice. |
| **Heuristic percentiles** | P90 and P99 are fixed multiples of the mean (×1.2, ×1.6). Workloads with heavy tails (e.g., reasoning with 1448 output tokens) will likely show large P99 errors. |
| **No KV cache pressure modeling** | AIConfigurator does not model GPU memory exhaustion, preemption, or CPU offloading. Under high KV pressure, real latencies spike but predictions remain flat. |
| **No prefix caching modeling** | All ground-truth experiments have prefix caching enabled. AIConfigurator models full prefill for every request, likely over-predicting TTFT for workloads with high prefix reuse. |
| **Discrete concurrency grid** | The Pareto DataFrame contains rows at specific concurrency levels. Derived concurrency values between grid points are snapped to the nearest available row, introducing quantization error. |
| **E2E is derived, not measured** | E2E = TTFT + TPOT × OSL assumes pure sequential decode with no overlap or scheduling delay. Real E2E includes queuing time, preemption delays, and decode scheduling overhead. |
| **MoE exclusion** | Mixtral-8x7B cannot be evaluated. Aggregate MAPE reflects only dense-architecture models. |
| **Fixed ISL/OSL** | The adapter uses configured distribution parameters from `profile.yaml`, not actual per-request token counts. Variance in real request sizes is not captured. |

---

## 11. Expected Error Profile

Based on the methodology's constraints:

- **TTFT**: Should be reasonably accurate for compute-bound prefill. AIConfigurator uses profiled GEMM and attention kernel timings on H100, which should closely match real prefill latency. However, prefix caching (not modeled) causes over-prediction for workloads with high system prompt reuse.
- **E2E mean**: The derived E2E (TTFT + TPOT × OSL) misses queuing, preemption, and scheduling overhead. Expect under-prediction under high load where real requests queue, and over-prediction under low load where real prefix caching reduces prefill time.
- **ITL (TPOT)**: Should be the most accurate metric — TPOT directly models per-token decode latency from profiled kernel data. However, it represents steady-state decode throughput and does not capture per-token variance.
- **P90/P99 (all metrics)**: Expected to be the least accurate due to heuristic multipliers. Real tail latencies are driven by scheduling jitter, preemption, and KV cache pressure — none of which are modeled.

---

## 12. Comparison with Other Analytical Adapters

| Dimension | AIConfigurator | LLM-Optimizer |
|-----------|----------------|---------------|
| **Approach** | Profiled kernel data + analytical model | Roofline analysis (compute/bandwidth bound) |
| **SDK interaction** | One call per experiment → DataFrame sweep | One call per stage → single result |
| **Concurrency handling** | Sweep all levels, look up nearest | Single concurrency per invocation |
| **Parallelism** | Explores TP/PP/DP decompositions | Fixed TP only |
| **E2E latency** | Derived: `TTFT + TPOT × OSL` | Direct: `e2e_latency_s` |
| **Percentiles** | Heuristic (mean × multiplier) | Heuristic (mean × multiplier) |
| **Model coverage** | 3/4 models (no Mixtral via vLLM) | 4/4 models |
| **GPU data source** | Profiled kernel CSVs (GEMM, attention, NCCL) | Theoretical peak FLOPs/bandwidth |
| **Quantization** | Explicit quant mode per kernel type | Uses `inferred_precision` from model config |
| **Execution** | In-process Python (seconds per experiment) | In-process Python (milliseconds per stage) |
