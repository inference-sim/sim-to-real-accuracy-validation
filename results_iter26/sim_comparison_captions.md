# Simulator Comparison Figures

## Overview

These figures provide pairwise head-to-head comparisons between **both BLIS variants** (BLIS-Roofline and BLIS-Evolved iter26) and each other simulator (Vidur, LLM-Optimizer, AIConfigurator, LLMServingSim). Each figure uses a **2×3 grid layout** combining aggregate and model-wise breakdowns across three latency metrics (E2E Mean, TTFT Mean, ITL Mean). All comparisons show both BLIS variants side-by-side against the comparison simulator.

**Figure Layout:**
- **Top row (Aggregate):** 3 panels showing median MAPE aggregated across all experiments, models, configs, and workloads for E2E, TTFT, and ITL
- **Bottom row (Model Breakdown):** 3 panels showing median MAPE per model, aggregated across configs and workloads for E2E, TTFT, and ITL

**Data Filtering:** Each comparison includes only experiments where BOTH simulators have data (intersection of coverage). All configurations and workloads are included without filtering — no restrictions on default configs, safe flags, or workload types.

**Aggregation Method:** For aggregate panels, compute median MAPE across all data points. For model breakdown panels, compute median MAPE per model across all experiments/configs/workloads for that model.

**Iter26 Improvements:** BLIS-Evolved (iter26) achieves **37.42% overall MAPE** (TTFT: 24.34%, E2E: 13.09%) through activated TP All-Reduce physics-based modeling (β₄=0.410) and optimized per-layer overhead (β₅=49.6 µs/layer). This represents a **1.76-point improvement** over iter24 (39.18% MAPE) and **37.8% relative reduction** from iter16 baseline (60.19% MAPE).

---

## Comparison Figures

### BLIS-Roofline & BLIS-Evolved vs. Vidur

![BLIS vs Vidur](figures/sim_comparisons/blis_vs_vidur.png)

**Shared experiments:** 4 experiments with both BLIS variants
**Models:** CodeLlama-34b-Instruct-hf, Llama-2-70b-hf
**Workload:** general-lite only (Vidur only ran on this workload)
**Hardware:** H100 (Vidur lacks A100/L40S profiles in this dataset)

Vidur requires pre-built model profiles and currently only supports 3 models in the dataset. This comparison reflects Vidur's coverage limitations — it represents head-to-head accuracy on the small subset of experiments where all three simulators have data. The limited model diversity (2 large dense models, both 70B/34B class) and single workload type mean this comparison does not generalize to the full workload/model space.

**Key observations:**
- Vidur's discrete-event simulation approach with vLLM scheduler emulation
- Limited to pre-profiled models (requires separate profiling run per architecture)
- Does not support MoE models
- Requires trace replay infrastructure
- BLIS-Evolved (iter26) provides cross-model generalization without per-model profiling

---

### BLIS-Roofline & BLIS-Evolved vs. LLM-Optimizer

![BLIS vs LLM-Optimizer](figures/sim_comparisons/blis_vs_llm_optimizer.png)

**Shared experiments:** 38 experiments with both BLIS variants
**Models:** Qwen3-14B, CodeLlama-34b-Instruct-hf, Llama-2-70b-hf, Llama-3.1-8B-Instruct, Mixtral-8x22B-Instruct-v0.1, Mixtral-8x7B-v0.1
**Workload:** general, general-lite, codegen, roleplay (shared\_prefix workloads)
**Hardware:** H100, A100-80GB

LLM-Optimizer is an analytical roofline estimator that queries model configs from HuggingFace Hub and estimates latency using hardware compute/memory roofline models. It supports the broadest model coverage among non-BLIS simulators and includes MoE models (approximated as dense with 4×hidden\_size FFN dimension). This comparison represents head-to-head accuracy across a diverse set of dense and MoE models at various scales, showing both the baseline BLIS-Roofline and the improved BLIS-Evolved (iter26) with learned corrections including TP All-Reduce modeling.

**Key observations:**
- Analytical estimator (no trace replay, no scheduling simulation)
- Supports both dense and MoE models (MoE approximated as dense)
- Requires only model config from HuggingFace Hub
- Limited to shared\_prefix workloads (cannot model multi-turn conversations)
- Does not model serving parameters beyond TP (no chunk size, CPU offload, GPU mem util, DP)
- BLIS-Evolved (iter26) captures queueing delays and communication overhead missing from pure roofline

---

### BLIS-Roofline & BLIS-Evolved vs. AIConfigurator

![BLIS vs AIConfigurator](figures/sim_comparisons/blis_vs_aiconfigurator.png)

**Shared experiments:** 19 experiments with both BLIS variants
**Models:** Qwen3-14B, CodeLlama-34b-Instruct-hf, Llama-2-70b-hf, Llama-3.1-8B-Instruct
**Workload:** general, general-lite, codegen, roleplay (shared\_prefix workloads)
**Hardware:** H100 only

AIConfigurator is an analytical estimator from the AIConfigurator SDK that focuses on H100 hardware and dense models. It excludes MoE architectures entirely and is limited to H100 (no A100/L40S support). This comparison represents head-to-head accuracy on dense models at various scales on H100 hardware, showing both the baseline BLIS-Roofline and the improved BLIS-Evolved (iter26) with learned corrections including TP All-Reduce modeling and optimized per-layer overhead.

**Key observations:**
- Analytical estimator (no trace replay, no scheduling simulation)
- Dense models only (no MoE support)
- H100 only (no multi-GPU-type portability)
- Limited to shared\_prefix workloads (cannot model multi-turn conversations)
- Does not model serving parameters beyond TP (no chunk size, CPU offload, GPU mem util, DP)
- BLIS-Evolved (iter26) provides broader coverage (MoE, A100, L40S) with learned corrections

---

### BLIS-Roofline & BLIS-Evolved vs. LLMServingSim

![BLIS vs LLMServingSim](figures/sim_comparisons/blis_vs_llmservingsim.png)

**Shared experiments:** 1 experiment with all three simulators
**Models:** Mixtral-8x7B-v0.1
**Workload:** general workload with 2000 requests (cluster dataset)
**Hardware:** H100
**Configuration:** TP=4, no CPU offload, 90% GPU memory utilization

**Important:** This comparison uses the cluster_2000req dataset where all three simulators were tested on the **exact same 2000-request sample** from a real cluster workload. This controlled comparison eliminates workload variance as a confounding factor. LLMServingSim has extremely sparse coverage in the main dataset (~1 experiment) due to its prohibitive runtime (hours per experiment, 700× slower than BLIS). The cluster dataset captures this single high-quality apples-to-apples comparison on a high-TP MoE configuration.

**Key results on shared experiment (Mixtral-8x7B TP4, E2E Mean MAPE):**
- BLIS-Evolved (iter26): **15.84%** (6.2× more accurate than roofline, 5.8× better than LLMServingSim)
- BLIS-Roofline: 97.57% (severe underestimation)
- LLMServingSim: 91.42% (overestimation)

BLIS-Evolved (iter26)'s learned correction terms—including activated TP All-Reduce modeling (β₄=0.410) and optimized per-layer overhead (β₅=49.6 µs/layer)—capture queueing delays, communication overhead, and weight loading that both the pure roofline model and LLMServingSim's trace-driven simulation miss on this high-parallelism MoE workload. The TP All-Reduce activation in iter26 specifically addresses communication bottlenecks in multi-GPU scenarios, contributing to the dramatic accuracy improvement on this TP=4 configuration.

**Iter26 TP All-Reduce Impact:**
The physics-based TP All-Reduce term (activated in iter26) models cross-GPU synchronization as:
```
β₄ × (2·numDenseLayers + numMoELayers) × totalTokens × d × 2B × 2phases × (TP-1)/TP / bwHBM
```
This term captures the NVLink-mediated gradient synchronization overhead that becomes significant at higher TP values (TP≥2), explaining BLIS-Evolved's superior performance on this TP=4 configuration compared to simulators lacking explicit communication modeling.

---

## Methodology Notes

**Concurrency for analytical estimators:** LLM-Optimizer and AIConfigurator are analytical estimators that require a concurrency input. For each ground-truth load stage, concurrency is derived via Little's Law (L = λ × W) using the stage's request rate (λ) and the ground-truth mean E2E latency (W). This means the concurrency input is informed by observed performance, not purely predicted — see main figure\_captions.md for full methodology discussion.

**MAPE calculation:** All error metrics are computed as absolute percentage error: `|predicted - actual| / actual × 100`. Median MAPE is used for aggregation to handle outliers robustly.

**Stage filtering:** All comparisons use summary-level predictions (`stage_index = -1`) aggregated across all load stages, consistent with the main publication figures.

**Iter26 coefficient optimization:** The iter26 coefficients were optimized via golden section search on β₄ (TP All-Reduce) and β₅ (per-layer overhead) starting from iter24's decode-split architecture. The optimization achieved a 1.76-point MAPE improvement (39.18% → 37.42%) by activating physics-based TP communication modeling while reducing per-layer overhead from 62.3 to 49.6 µs/layer.
