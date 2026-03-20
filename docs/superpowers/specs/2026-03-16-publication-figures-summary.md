# Publication Figures — Executive Summary

**Paper thesis:** BLIS predicts LLM inference latency accurately and fast, across models, hardware, workloads, and serving configurations — using only datasheet-level hardware specs.

**Benchmark:** 55 experiments across 8 models (7B–141B, dense + MoE + FP8), 3 GPUs (H100, A100, L40S), 4 workloads, and 5 vLLM config knobs. Five simulators compared: BLIS-Trained, BLIS-Roofline, Vidur, LLM-Optimizer, AIConfigurator. Metric: MAPE (Mean Absolute Percentage Error) for E2E latency, TTFT, and ITL — both mean and P99 tail.

---

## Figures

### Figure 1 — Model Sensitivity (2x3 grouped bar grid)

**What it shows:** MAPE of all 5 simulators across 7 model architectures on H100, default config.

**Why it matters:** Demonstrates BLIS generalizes across dense models (7B→70B) and MoE architectures (Mixtral, Scout FP8) without model-specific tuning. Reviewers need to see that accuracy doesn't degrade as model complexity grows.

**Caption:** "Prediction accuracy across model architectures. MAPE of five simulators across seven LLM models spanning dense (7B–70B) and MoE (47B–141B) architectures on H100 (default vLLM config). Top row: mean latency; bottom row: P99 tail latency. BLIS-Trained maintains low MAPE across all architectures. LLM-Optimizer and AIConfigurator produce only mean estimates (tail-latency bars absent)."

### Figure 2 — Hardware Portability (2x3 grouped bar grid)

**What it shows:** MAPE across H100, A100-80GB, and L40S, aggregated over 7 models per GPU.

**Why it matters:** Most simulators are profiled/calibrated on one GPU and don't transfer. BLIS uses only datasheet specs (FLOPS, bandwidth), so it should generalize across GPU generations without re-profiling. This is the portability claim.

**Caption:** "Hardware portability. MAPE across three GPU types (default config). Each bar aggregates across all viable models for that GPU. BLIS variants generalize across GPU generations using only datasheet specifications."

### Figure 3 — Workload Sensitivity (2x3 grouped bar grid)

**What it shows:** MAPE across 4 workload types (General-Purpose, Code Generation, Roleplay, Reasoning), aggregated over 4 models on H100.

**Why it matters:** Real deployments serve diverse workloads with different token-length distributions. If a simulator only works for one workload shape, it's not useful. This shows BLIS accuracy is stable across workload diversity.

**Caption:** "Workload sensitivity. MAPE across four workload types, aggregated over four models spanning dense and MoE architectures (H100, default config). BLIS-Trained shows the smallest degradation across workload diversity."

### Figure 4 — Configuration Generalization (2x3 grouped bar grid, two panels)

**What it shows:** MAPE as individual vLLM knobs are varied from defaults. Panel (a): Llama-3.1-8B (dense, 6 configs). Panel (b): Mixtral-8x7B (MoE, 7 configs including expert parallelism).

**Why it matters:** Practitioners need to explore config spaces (batch size, parallelism, memory tuning) before deploying. If the simulator breaks when you change one knob, you can't trust it for capacity planning. This shows BLIS stays accurate across config perturbations — including MoE-specific expert parallelism, which no other simulator models.

**Caption (a):** "Configuration sensitivity for a dense model (Llama-3.1-8B, H100). Each group varies one vLLM knob from the default. BLIS prediction error remains stable across configuration changes."

**Caption (b):** "Configuration sensitivity for an MoE model (Mixtral-8x7B, H100). Includes expert parallelism (EP=4 via DP=2). BLIS handles both standard knobs and MoE-specific parallelism without accuracy degradation."

### Figure 5 — Accuracy-Speed Pareto (scatter plot)

**What it shows:** Each simulator as one point: median MAPE (x) vs. median simulation runtime (y, log scale). Error bars show IQR.

**Why it matters:** This is the money shot. It answers the question every reviewer will ask: "Is BLIS just more accurate, or is it also fast?" BLIS-Trained should sit in the bottom-left (accurate + fast), dominating all other simulators. The Pareto framing makes the contribution visually unambiguous.

**Caption:** "Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE (x-axis) and median simulation runtime (y-axis, log scale) across all available variations. Error bars show interquartile range. BLIS-Trained achieves the best accuracy-speed tradeoff."

---

## Table

### Table 1 — Runtime Comparison

**What it shows:** Median wall-clock simulation time per simulator, and speedup vs. running the actual vLLM experiment.

**Why it matters:** Quantifies the practical value proposition. If BLIS completes in 1–2 seconds what takes 20+ minutes on real hardware, that's 1000× speedup — enabling design-space exploration that's infeasible with real experiments.

**Caption:** "Simulator runtime and speedup. Median wall-clock time per variation and speedup relative to running the actual vLLM experiment. BLIS variants complete in seconds, enabling rapid exploration of the model-hardware-config design space."

---

## Paper Narrative Arc

| Figure | Reviewer question answered |
|--------|---------------------------|
| Fig 1 | "Does it work across different models?" |
| Fig 2 | "Does it transfer to new hardware without re-profiling?" |
| Fig 3 | "Does it handle diverse real-world workloads?" |
| Fig 4 | "Can I trust it when tuning serving configs?" |
| Fig 5 | "Is there a catch — is it slow?" |
| Table 1 | "How much faster, exactly?" |

Figures 1–4 systematically eliminate concerns about generalization. Figure 5 delivers the punchline: BLIS is both the most accurate and the fastest. Table 1 quantifies the speedup for the abstract.
