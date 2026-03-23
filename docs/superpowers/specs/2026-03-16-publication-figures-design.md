# Publication Figures Design Spec

**Date:** 2026-03-16
**Goal:** Generate best-paper-quality figures from `experiment.run` output (`results/error_records.csv`, `results/runtime.csv`) for a systems paper at NSDI/EuroSys.
**Protagonist:** BLIS simulator (2 variants: Trained, Roofline). Baselines: Vidur, LLM-Optimizer, AIConfigurator.
**Data collection plan:** [inference-sim/inference-sim#598](https://github.com/inference-sim/inference-sim/discussions/598)

---

### Simulator Name Mapping

CSV identifiers map to display names in figures:

| CSV Identifier | Display Name |
|----------------|-------------|
| `blis-trained-roofline` | BLIS-Trained |
| `blis-roofline` | BLIS-Roofline |
| `vidur` | Vidur |
| `llm-optimizer-estimate` | LLM-Optimizer |
| `aiconfigurator-estimate` | AIConfigurator |

`blis-blackbox` and `blis-crossmodel` are excluded from all figures.

### Model Short Labels

| CSV Model ID | Short Label | Type |
|-------------|------------|------|
| `meta-llama/Llama-2-7b-hf` | Ll-2-7B | Dense |
| `meta-llama/Llama-3.1-8B-Instruct` | Ll-3.1-8B | Dense |
| `Qwen/Qwen3-14B` | Qw-14B | Dense |
| `codellama/CodeLlama-34b-Instruct-hf` | CL-34B | Dense |
| `meta-llama/Llama-2-70b-hf` | Ll-2-70B | Dense |
| `mistralai/Mixtral-8x7B-v0.1` | Mx-8x7B | MoE |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | Mx-8x22B | MoE |
| `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic` | Scout-17B | MoE, FP8 |

### Workload Display Names

| CSV Value | Display Name |
|-----------|-------------|
| `general` | General-Purpose |
| `codegen` | Code Generation |
| `roleplay` | Roleplay |
| `reasoning` | Reasoning |
| `general-lite` | General-Lite |

### Known Simulator Limitations

- **AIConfigurator:** Excludes MoE architectures (Mixtral-8x7B, Mixtral-8x22B, Scout-17B-16E). Bars absent per the missing-data rule. E2E latency computed analytically as `ttft + itl × output_length`.
- **Vidur:** Requires pre-profiled GPU kernel timings. Model and hardware coverage may be incomplete; absent bars where profiles are unavailable.
- **LLM-Optimizer and AIConfigurator:** Produce only mean estimates (E2E, TTFT, ITL). P99 (tail) bars are absent for these simulators in all figures.

---

## Experiment Matrix

Based on the [55-experiment benchmark plan](https://github.com/inference-sim/inference-sim/discussions/598).

### Dimensions

- **Models (8):** Llama-2-7B, Llama-3.1-8B, Qwen3-14B, CodeLlama-34B, Llama-2-70B, Mixtral-8x7B, Mixtral-8x22B, Llama-4-Scout-17B-16E (FP8)
- **Hardware (3):** H100-80GB SXM, A100-80GB SXM, L40S
- **Workloads (4+1):** General-Purpose, Code Generation, Roleplay, Reasoning, General-Lite (same token distributions as General at 6 rps peak; used for models whose safe rate is below General's 20 rps peak)
- **Config knobs (5):** max_num_batched_tokens, cpu_offloading, gpu_memory_utilization, TP, DP/EP

### Collection Status (55 experiments)

| Phase | Scope | Experiments | Status |
|-------|-------|-------------|--------|
| 0 | Existing H100 data (4 models x 3 workloads) | 12 | Done |
| 0.5 | Default-config baselines + workload consistency (rows 56-59) | 4 | Pending |
| 1-3 | H100 new model baselines + workloads | 9 | Done |
| 4-5 | H100 config sweeps (Llama-3.1-8b + Mixtral-8x7B) | 11 | Done |
| 6-7 | H100 EP + DP experiments | 3 | Partial (2/3 done) |
| 8 | Hardware variation (A100, L40S) | 9 | Pending |
| 9 | H100 Reasoning workloads | 4 | Partial |

The figure code renders whatever data is present in the CSVs. As new experiments complete, re-running `experiment.run` and the figure generator will incorporate them automatically.

### Figure Derivation from Experiment Matrix

Each figure applies a filter to the full matrix, then groups by one dimension:

| Figure | Filter | Group By |
|--------|--------|----------|
| Fig 1 | HW=H100, Workload=General/General-Lite, config=defaults | Model (7 models) |
| Fig 2 | Workload=General/General-Lite (per-model consistent across HW), config=defaults, TP=default, DP<=1 | Hardware (3 GPUs) |
| Fig 3 | HW=H100, config=defaults, TP=default, DP<=1, 4 models only | Workload (4 workloads) |
| Fig 4a | HW=H100, Workload=General, Model=Llama-3.1-8b | Config variant |
| Fig 4b | HW=H100, Workload=General, Model=Mixtral-8x7B | Config variant |
| Fig 5 | All available variations | Simulator (aggregate) |

---

## Deliverables

**5 figures (4a/4b count as one) + 1 table.**

| # | Title | Type | Purpose |
|---|-------|------|---------|
| Fig 1 | Model Sensitivity | 2x3 grouped bar grid | Accuracy across 7 model architectures |
| Fig 2 | Hardware Portability | 2x3 grouped bar grid | Accuracy across 3 GPU types |
| Fig 3 | Workload Sensitivity | 2x3 grouped bar grid | Accuracy across 4 workload types |
| Fig 4(a) | Config Sensitivity — Dense | 2x3 grouped bar grid | Llama-3.1-8b config sweeps (5 knobs) |
| Fig 4(b) | Config Sensitivity — MoE | 2x3 grouped bar grid | Mixtral-8x7B config sweeps (6 knobs incl. EP) |
| Fig 5 | Accuracy-Speed Pareto | Scatter plot | Tradeoff frontier (the money shot) |
| Table 1 | Runtime Comparison | LaTeX/text table | Simulator runtime + speedup vs real |

---

## Global Design Language

### Color Palette

BLIS variants use a blue/teal gradient (dark = most sophisticated). Baselines use gray with hatching for B&W distinguishability.

| Simulator | Hex | Fill | Visual Role |
|-----------|-----|------|-------------|
| BLIS-Trained | `#0077B6` | Solid | Hero (darkest, most prominent) |
| BLIS-Roofline | `#90E0EF` | Solid | BLIS family |
| Vidur | `#6C757D` | Hatched `//` | Baseline |
| LLM-Optimizer | `#ADB5BD` | Hatched `\\` | Baseline |
| AIConfigurator | `#DEE2E6` | Hatched `xx` | Baseline |

### Typography

- `font.family: serif` with `text.usetex: True` for LaTeX-rendered labels
- Axis labels: 8pt
- Subplot titles: 9pt
- Matches `\small` font in two-column ACM/USENIX templates

### Shared Elements

- **20% MAPE threshold line:** Thin horizontal dashed line at 20% MAPE in all accuracy subplots as a visual reference for quick scanning. Exact threshold may be adjusted based on data distribution.
- **Shared legend:** Single horizontal legend below the bottom row of each figure, spanning all 3 columns. Saves vertical space.
- **Missing data:** When a simulator does not produce a metric (e.g., LLM-Optimizer has no P99), the bar is absent. A small "N/A" text annotation at the baseline marks the gap.
- **Y-axis:** MAPE (%), shared scale within each row for easy cross-panel comparison.

---

## Figures 1-4: Accuracy Bar Chart Grid

### Layout

Each figure (including 4a and 4b) is a **2-row x 3-column** subplot grid:

```
              E2E Latency       TTFT              ITL
           ┌───────────────┬───────────────┬───────────────┐
     Mean  │  grouped bar  │  grouped bar  │  grouped bar  │
           ├───────────────┼───────────────┼───────────────┤
     Tail  │  grouped bar  │  grouped bar  │  grouped bar  │
    (P99)  │               │               │               │
           └───────────────┴───────────────┴───────────────┘
```

- **Top row:** MAPE for `e2e_mean`, `ttft_mean`, `itl_mean`
- **Bottom row:** MAPE for `e2e_p99`, `ttft_p99`, `itl_p99`
- **Metric reported:** MAPE (Mean Absolute Percentage Error)
- **Why P99 over P90 for tail:** P99 is the stricter test of simulator fidelity and the standard tail metric at systems venues. P90 results remain available in supplementary material.

### Bar Layout

- 5 bars per x-tick group (one per simulator), computed dynamically from the number of simulators present in the data
- Bar widths and group spacing are proportional to figure width and x-tick count (the `_bar_chart_grid()` helper computes `bar_width` and offsets from `n_simulators` and `n_groups`)
- BLIS bars always appear first (left) in each group, baselines to the right
- Simulator ordering is fixed: BLIS-Trained, BLIS-Roofline, Vidur, LLM-Optimizer, AIConfigurator

### Aggregation

When a figure's x-axis dimension has multiple underlying variations (e.g., Figure 2's H100 group spans many models, Figure 3 aggregates across models per workload), the bar height is the **median MAPE** across those variations, with **IQR error bars**. When n <= 3 (e.g., L40S with 2 anchors in Figure 2), overlay **individual data points** as semi-transparent dots instead of error bars for statistical honesty.

### Per-Figure Specifics

#### Figure 1 — Model Sensitivity

- **X-axis:** 7 models: Llama-3.1-8B, Qwen3-14B, CodeLlama-34B, Llama-2-70B, Mixtral-8x7B, Mixtral-8x22B, Scout-17B-16E. Ordered by parameter count within type (dense first, then MoE).
- **Filter:** HW=H100, Workload=General or General-Lite, config=defaults (from discussion: rows 13, 16, 17, 49, 56, 57, 58; rows 57-58 use General-Lite for safe rate)
- **Aggregation:** 1 variation per (model, simulator) — no aggregation, no error bars
- **Note:** Llama-2-7B excluded because its default config uses DP=2 (router datapoint), making it non-comparable to single-replica baselines.
- **Caption:** "Prediction accuracy across model architectures. MAPE of five simulators across seven LLM models spanning dense (7B-70B) and MoE (47B-141B) architectures on H100 (General-Purpose workload, default vLLM config). Top row: mean latency (E2E, TTFT, ITL); bottom row: P99 tail latency. BLIS-Trained (dark blue) maintains low MAPE across all architectures. LLM-Optimizer and AIConfigurator report mean estimates for all three metrics but lack tail-latency predictions (P99 bars absent)."

#### Figure 2 — Hardware Portability

- **X-axis:** 3 GPU types: H100, A100-80GB, L40S
- **Filter:** Workload=General or General-Lite (per-model consistent across all HW), config=defaults, TP=default, DP<=1. General-Lite models: Qwen3-14B (rows 59/36/54), Codellama-34b (rows 57/40), Llama-2-70b (rows 58/41). General models: Llama-3.1-8b, Mixtral-8x7B, Scout, Mixtral-8x22B. Llama-3.1-8b L40S (row 55) uses General-Lite by necessity — the only cross-hardware workload mismatch.
- **Aggregation:** H100 aggregates across 7 models (median + IQR). A100 aggregates across up to 7 models. L40S has 2 models (overlay dots). Not all models run on all hardware — L40S excludes CodeLlama-34B, Llama-2-70B, Mixtral-8x22B due to VRAM constraints.
- **Caption:** "Hardware portability. MAPE across three GPU types (default config). Each bar aggregates across all viable models for that GPU. BLIS variants generalize across GPU generations using only datasheet specifications."

#### Figure 3 — Workload Sensitivity

- **X-axis:** 4 workloads: General-Purpose, Code Generation, Roleplay, Reasoning
- **Filter:** HW=H100, config=defaults, TP=default, DP<=1, Model IN (Llama-3.1-8b, Qwen3-14B, Llama-4-Scout-17B-16E, Mixtral-8x22B) (from discussion: rows 13–21, 46–51, 53; up to 15 experiments)
- **Models:** 4 models with multi-workload data at defaults. Phase 0 models (Mixtral-8x7B, Codellama-34b, Llama-2-70b) excluded — their workload data uses cpu_offload=Enabled.
- **Aggregation:** Each workload aggregates across the 4 models (median + IQR error bars; overlay dots when n <= 3). Reasoning has 3 models (Qwen3-14B excluded — row 47 is ⚠️ unsafe).
- **Caption:** "Workload sensitivity. MAPE across four workload types, aggregated over four models spanning dense and MoE architectures (H100, default config). BLIS-Trained shows the smallest degradation across workload diversity."

#### Figure 4(a) — Config Sensitivity: Dense Model (Llama-3.1-8B)

- **X-axis:** 6 configs: `Default`, `mbt=1024`, `mbt=8192`, `CPU-Offload`, `GPU-0.95`, `TP=2`
- **Filter:** HW=H100, Workload=General, Model=Llama-3.1-8B (from discussion: rows 16, 22-26)
- **Aggregation:** 1 variation per (config, simulator) — no aggregation
- **Caption:** "Configuration sensitivity for a dense model (Llama-3.1-8B-Instruct, H100, General-Purpose). Each group varies one vLLM knob from the default. BLIS prediction error remains stable across configuration changes."

#### Figure 4(b) — Config Sensitivity: MoE Model (Mixtral-8x7B)

- **X-axis:** 7 configs: `Default`, `mbt=1024`, `mbt=8192`, `CPU-Offload`, `GPU-0.95`, `TP=4`, `EP=4 (DP=2)`
- **Filter:** HW=H100, Workload=General, Model=Mixtral-8x7B (from discussion: rows 56, 27-32; row 56 is the Disabled-default baseline)
- **Aggregation:** 1 variation per (config, simulator) — no aggregation
- **Caption:** "Configuration sensitivity for an MoE model (Mixtral-8x7B, H100, General-Purpose). Includes expert parallelism (EP=4 via DP=2). BLIS handles both standard knobs and MoE-specific parallelism without accuracy degradation."

---

## Figure 5 — Accuracy-Speed Pareto

### Layout

Single-panel scatter plot.

- **X-axis:** Median MAPE across all available variations (lower = more accurate). Caption states n per simulator.
- **Y-axis:** Median wall-clock simulation time, **log scale** (lower = faster)
- **Points:** One per simulator, using the color palette above. Marker size larger for BLIS-Trained (hero).
- **Error bars:** Horizontal = IQR of MAPE; Vertical = IQR of runtime
- **Pareto shading:** Light blue shaded region from BLIS-Trained toward the origin, marking the dominated region
- **Annotations:** Each point labeled with simulator name + "(n=X)" showing the number of variations

### Caption

"Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE (x-axis) and median simulation runtime (y-axis, log scale) across all available variations. Error bars show interquartile range. BLIS-Trained achieves the best accuracy-speed tradeoff."

---

## Table 1 — Runtime Comparison

| Simulator | Median Runtime (s) | Speedup vs. Real |
|-----------|-------------------:|------------------:|
| BLIS-Trained | — | — |
| BLIS-Roofline | — | — |
| Vidur | — | — |
| AIConfigurator | — | — |
| LLM-Optimizer | — | — |

- Values computed from `runtime.csv`; dashes above are placeholders showing table structure
- Median is computed over all variations each simulator ran; sample sizes differ because some simulators exclude certain models (see Known Simulator Limitations)
- Sorted by BLIS family first, then baselines by descending runtime
- "Speedup vs. Real" = median real vLLM experiment duration / median simulator runtime. Real experiment duration is derived from the ground-truth experiment's total wall-clock time (sum of stage durations from the experiment metadata, or estimated from the trace as `max(request_end_time) - min(request_start_time)` per experiment).
- **Caption:** "Simulator runtime and speedup. Median wall-clock time per variation and speedup relative to running the actual vLLM experiment. BLIS variants complete in seconds, enabling rapid exploration of the model-hardware-config design space."

---

## Implementation Notes

### Technology

- **matplotlib** for all figures (the standard for systems papers; reviewers expect it)
- **pandas** for data loading and aggregation
- Output formats: PDF (vector, for LaTeX `\includegraphics`) and PNG (300 DPI, for review)

### Code Structure

A single module `experiment/figures.py` that:

1. Reads `results/error_records.csv` and `results/runtime.csv` via pandas
2. Exposes one function per figure: `plot_model_sensitivity()`, `plot_hardware_portability()`, `plot_workload_sensitivity()`, `plot_config_sensitivity_dense()`, `plot_config_sensitivity_moe()`, `plot_pareto()`, `format_runtime_table()`. Each function filters out excluded simulators (`blis-blackbox`, `blis-crossmodel`).
3. Shares a common `_bar_chart_grid()` helper for the 2x3 grouped bar layout (Figures 1-4)
4. Uses a shared style configuration dict for colors, hatching, fonts, and the 20% threshold line
5. Outputs to `results/figures/` directory

### File Naming

Output files follow a consistent convention for LaTeX `\includegraphics` references:

- `results/figures/fig1_model_sensitivity.pdf` (and `.png`)
- `results/figures/fig2_hardware_portability.pdf`
- `results/figures/fig3_workload_sensitivity.pdf`
- `results/figures/fig4a_config_dense.pdf`
- `results/figures/fig4b_config_moe.pdf`
- `results/figures/fig5_pareto.pdf`
- `results/figures/table1_runtime.tex` (LaTeX tabular for direct inclusion)

### CLI Integration

Add a `--figures` flag to `experiment/run.py` (or a separate `python -m experiment.figures` CLI entry point) so figures can be regenerated from existing CSVs without re-running simulations.

### Data Filtering

The CSV schema currently encodes model and workload but not hardware, TP, DP, or config knobs. To support Figures 2 and 4, either:

1. **Extend the CSV schema** to include `hardware`, `tp`, `dp`, `max_num_batched_tokens`, `cpu_offloading`, `gpu_memory_utilization`, `precision` columns in both `error_records.csv` and `runtime.csv`, OR
2. **Infer from experiment folder names** using a naming convention (e.g., `YYYYMMDD-HHMMSS-<model>-tp<N>-<workload>-<hw>-<config_tag>`)

Option 1 is more robust and recommended. The `experiment.run` pipeline should propagate these dimensions through to the CSV output.

### Data Quality Policy

- **Exclude unsafe experiments:** Any experiment marked ⚠️ unsafe in the collection matrix (safe rate below workload peak RPS) is excluded from all figures. Only ✅ safe experiments contribute data.

### Robustness

- Gracefully handle missing simulators or metrics (absent bars, not crashes)
- Handle varying numbers of models, workloads, hardware, and configs in the CSV
- Stage index filtering: use `stage_index == -1` (aggregate) rows for the figures unless per-stage breakdown is explicitly needed
- Skip figure generation (with a warning) when required data dimensions are entirely absent (e.g., Figure 2 when no A100/L40 data exists)

### Figure Sizing

- Figures 1-3: full-page width in two-column format (~7.0" x 3.5")
- Figure 4(a), 4(b): full-page width each (~7.0" x 3.5"); presented as sub-figures in the paper
- Figure 5: single-column width (~3.5" x 3.0")
- All saved at 300 DPI for PNG, vector for PDF
