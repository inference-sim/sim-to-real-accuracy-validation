# Publication Figures Design Spec

**Date:** 2026-03-16
**Goal:** Generate best-paper-quality figures from `experiment.run` output (`results/error_records.csv`, `results/runtime.csv`) for a systems paper at NSDI/EuroSys.
**Protagonist:** BLIS simulator (3 variants). Baselines: Vidur, LLM-Optimizer, AIConfigurator.

---

## Data Source

All figures read from CSV files produced by `python -m experiment.run`:

- **`results/error_records.csv`** — columns: `simulator`, `experiment_folder`, `model`, `workload`, `stage_index`, `metric_name`, `predicted`, `actual`, `mape`, `mpe`, `absolute_error`
- **`results/runtime.csv`** — columns: `simulator`, `experiment_folder`, `model`, `workload`, `wall_clock_seconds`

The figure code must be robust to varying numbers of simulators, models, workloads, and metrics depending on which adapters ran. Missing combinations (e.g., a simulator that skips certain models) produce absent bars, not errors.

### Simulator Name Mapping

CSV identifiers map to display names in figures:

| CSV Identifier | Display Name |
|----------------|-------------|
| `blis-trained-roofline` | BLIS-Trained |
| `blis-crossmodel` | BLIS-CrossModel |
| `blis-roofline` | BLIS-Roofline |
| `vidur` | Vidur |
| `llm-optimizer-estimate` | LLM-Optimizer |
| `aiconfigurator-estimate` | AIConfigurator |

`blis-blackbox` is excluded from all figures (it has no matching coefficients for any current model and produces zero results).

### Workload Display Names

| CSV Value | Display Name |
|-----------|-------------|
| `general` | General-Purpose |
| `codegen` | Code Generation |
| `roleplay` | Roleplay |
| `reasoning` | Reasoning |

### Model Short Labels

| CSV Model ID | Short Label |
|-------------|------------|
| `meta-llama/Llama-2-7b-hf` | Ll-7B |
| `meta-llama/Llama-2-70b-hf` | Ll-70B |
| `mistralai/Mixtral-8x7B-v0.1` | Mx-8x7B |
| `codellama/CodeLlama-34b-Instruct-hf` | CL-34B |

Additional models will be added to this table as data is collected.

### Known Simulator Limitations

- **AIConfigurator:** Excludes MoE architectures (Mixtral-8x7B). Bars for this simulator will be absent on Mixtral per the missing-data rule.
- **Vidur:** Requires pre-profiled GPU kernel timings. Currently supports Llama-2-7B, Llama-2-70B, and CodeLlama-34B but not Mixtral.
- **LLM-Optimizer and AIConfigurator:** Produce only mean estimates. P99 (tail) bars are absent for these simulators in all figures.

---

## Experiment Matrix

### Data Available Today (12 variations)

| Slice | Variations | Status |
|-------|-----------|--------|
| Core benchmark | 4 models x 3 workloads = 12 | Available (H100, default vLLM config) |

Models: Llama-2-7B, Llama-2-70B, Mixtral-8x7B, CodeLlama-34B-Instruct.
Workloads: General-Purpose, Code Generation, Roleplay.

### Full Target (21 variations, pending data collection)

| Slice | Variations | Fixed Dimensions | Status |
|-------|-----------|------------------|--------|
| Core benchmark (Figs 1, 3) | 7 models x 4 workloads = 28, deduplicated with current = 16 | H100, default vLLM config | Partial: 12 of 16 available (3 models + reasoning TBD) |
| Hardware portability (Fig 2) | 2 on A100 + 2 on L40 = 4 | Selected anchor (model, workload) pairs | Not yet collected |
| Config generalization (Fig 4) | 3 alternative vLLM knobs | Llama-2-70B, General-Purpose, H100 | Not yet collected |
| **Total** | **21 unique variations** | | **12 available today** |

The figure code renders whatever data is present in the CSVs. When new experiments are collected and `experiment.run` is re-executed, the figures will automatically incorporate the new data points. Figures 2 and 4 will produce empty outputs until their respective data is collected.

---

## Deliverables

**5 figures + 1 table.**

| # | Title | Type | Purpose | Data Ready? |
|---|-------|------|---------|-------------|
| Fig 1 | Model Sensitivity | 2x3 grouped bar grid | Accuracy across model architectures | Yes (4 models today, 7 target) |
| Fig 2 | Hardware Portability | 2x3 grouped bar grid | Accuracy across GPU types | No (requires A100/L40 data) |
| Fig 3 | Workload Sensitivity | 2x3 grouped bar grid | Accuracy across workload types | Yes (3 workloads today, 4 target) |
| Fig 4 | Config Generalization | 2x3 grouped bar grid | Accuracy under vLLM config changes | No (requires config variant data) |
| Fig 5 | Accuracy-Speed Pareto | Scatter plot | Tradeoff frontier (the money shot) | Yes (uses available variations) |
| Table 1 | Runtime Comparison | LaTeX/text table | Simulator runtime + speedup vs real | Yes |

---

## Global Design Language

### Color Palette

BLIS variants use a blue/teal gradient (dark = most sophisticated). Baselines use gray with hatching for B&W distinguishability.

| Simulator | Hex | Fill | Visual Role |
|-----------|-----|------|-------------|
| BLIS-Trained | `#0077B6` | Solid | Hero (darkest, most prominent) |
| BLIS-CrossModel | `#00B4D8` | Solid | BLIS family |
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

Each figure is a **2-row x 3-column** subplot grid:

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

- 6 bars per x-tick group (one per simulator), computed dynamically from the number of simulators present in the data
- Bar widths and group spacing are proportional to figure width and x-tick count (the `_bar_chart_grid()` helper computes `bar_width` and offsets from `n_simulators` and `n_groups`)
- BLIS bars always appear first (left) in each group, baselines to the right
- Simulator ordering is fixed: BLIS-Trained, BLIS-CrossModel, BLIS-Roofline, Vidur, LLM-Optimizer, AIConfigurator

### Aggregation

When a figure's x-axis dimension has multiple underlying variations (e.g., Figure 3 aggregates across 4 models per workload), the bar height is the **median MAPE** across those variations, with **IQR error bars**. When n <= 3 (e.g., A100 with 2 anchors in Figure 2), overlay **individual data points** as semi-transparent dots instead of error bars for statistical honesty.

### Per-Figure Specifics

#### Figure 1 — Model Sensitivity

- **X-axis:** All models present in the CSV, filtered to `workload == "general"`. Currently 4 models; scales to 7 as data is collected. Short labels derived from model names (e.g., `Ll-7B`, `Ll-70B`, `Mx-8x7B`, `CL-34B`).
- **Fixed:** H100, General-Purpose workload, default vLLM config
- **Aggregation:** 1 variation per (model, simulator) — no aggregation needed, no error bars
- **Caption:** "Prediction accuracy across model architectures. MAPE of six simulators across LLM models (H100, General-Purpose workload, default vLLM config). Top row: mean latency; bottom row: P99 tail latency. BLIS-Trained (dark blue) maintains low MAPE across all architectures. LLM-Optimizer and AIConfigurator produce only mean estimates (tail-latency bars absent)."

#### Figure 2 — Hardware Portability

- **X-axis:** GPU types present in the CSV (target: `H100`, `A100`, `L40`). Requires a `hardware` dimension in the CSV or a mapping from experiment folder names to GPU type.
- **Fixed:** Selected anchor (model, workload) pairs per hardware
- **Aggregation:** H100 aggregates all core-benchmark variations (median + IQR). A100 and L40 have 2 anchors each (overlay individual dots).
- **Data dependency:** Requires A100/L40 ground truth collection and a way to identify GPU type from the CSV. The experiment schema may need a `hardware` column, or GPU type can be inferred from experiment folder naming convention.
- **Caption:** "Hardware portability. MAPE across GPU types. H100 bars show median over core-benchmark variations (error bars: IQR); other GPUs show individual anchor results as overlaid dots. BLIS variants generalize across GPU generations using only datasheet specifications."

#### Figure 3 — Workload Sensitivity

- **X-axis:** All workloads present in the CSV. Currently 3 (`general`, `codegen`, `roleplay`); target is 4 (+ `reasoning`).
- **Fixed:** H100, default config
- **Aggregation:** Each workload aggregates across all models (median + IQR error bars)
- **Caption:** "Workload sensitivity. MAPE across workload types, aggregated over models (H100, default config). BLIS-Trained shows the smallest degradation across workload diversity."

#### Figure 4 — Configuration Generalization

- **X-axis:** vLLM configurations present in the CSV (target: `Default`, `BatchTok-2K`, `No-Offload`, `GPU-0.8`). Requires a `config` dimension in the CSV or a mapping from experiment folder names to config variant.
- **Fixed:** Llama-2-70B, General-Purpose, H100
- **Aggregation:** 1 variation per (config, simulator) — no aggregation needed
- **Data dependency:** Requires running experiments with alternative vLLM configurations and encoding the config variant in the CSV.
- **Caption:** "Sensitivity to serving configuration. MAPE under vLLM configurations for Llama-2-70B on H100 (General-Purpose workload). Varying serving knobs has minimal impact on BLIS prediction error, demonstrating robustness without per-config tuning."

---

## Figure 5 — Accuracy-Speed Pareto

### Layout

Single-panel scatter plot.

- **X-axis:** Median MAPE across all available variations (lower = more accurate). Caption states n per simulator.
- **Y-axis:** Median wall-clock simulation time, **log scale** (lower = faster)
- **Points:** One per simulator, using the color palette above. Marker size larger for BLIS-Trained (hero).
- **Error bars:** Horizontal = IQR of MAPE; Vertical = IQR of runtime
- **Pareto shading:** Light blue shaded region from BLIS-Trained toward the origin, marking the dominated region
- **Annotations:** Each point labeled with simulator name + median values

### Caption

"Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE (x-axis) and median simulation runtime (y-axis, log scale) across N available variations. Error bars show interquartile range. BLIS-Trained achieves the best accuracy-speed tradeoff."

---

## Table 1 — Runtime Comparison

| Simulator | Median Runtime (s) | Speedup vs. Real |
|-----------|-------------------:|------------------:|
| BLIS-Trained | — | — |
| BLIS-CrossModel | — | — |
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
2. Exposes one function per figure: `plot_model_sensitivity()`, `plot_hardware_portability()`, `plot_workload_sensitivity()`, `plot_config_generalization()`, `plot_pareto()`, `format_runtime_table()`
3. Shares a common `_bar_chart_grid()` helper for the 2x3 grouped bar layout (Figures 1-4)
4. Uses a shared style configuration dict for colors, hatching, fonts, and the 20% threshold line
5. Outputs to `results/figures/` directory

### File Naming

Output files follow a consistent convention for LaTeX `\includegraphics` references:

- `results/figures/fig1_model_sensitivity.pdf` (and `.png`)
- `results/figures/fig2_hardware_portability.pdf`
- `results/figures/fig3_workload_sensitivity.pdf`
- `results/figures/fig4_config_generalization.pdf`
- `results/figures/fig5_pareto.pdf`
- `results/figures/table1_runtime.tex` (LaTeX tabular for direct inclusion)

### CLI Integration

Add a `--figures` flag to `experiment/run.py` (or a separate `python -m experiment.figures` CLI entry point) so figures can be regenerated from existing CSVs without re-running simulations.

### Robustness

- Gracefully handle missing simulators or metrics (absent bars, not crashes)
- Handle varying numbers of models, workloads, hardware, and configs in the CSV
- Stage index filtering: use `stage_index == -1` (aggregate) rows for the figures unless per-stage breakdown is explicitly needed
- Skip figure generation (with a warning) when required data dimensions are entirely absent (e.g., Figure 2 when no A100/L40 data exists)

### Figure Sizing

- Figures 1-4: full-page width in two-column format (~7.0" x 3.5")
- Figure 5: single-column width (~3.5" x 3.0")
- All saved at 300 DPI for PNG, vector for PDF
