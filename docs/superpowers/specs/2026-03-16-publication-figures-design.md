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

---

## Experiment Matrix (21 Variations)

| Slice | Variations | Fixed Dimensions |
|-------|-----------|------------------|
| Core benchmark (Figs 1, 3) | 4 models x 4 workloads = 16 | H100, default vLLM config |
| Hardware portability (Fig 2) | 2 on A100 + 2 on L40 = 4 | Selected anchor (model, workload) pairs |
| Config generalization (Fig 4) | 3 alternative vLLM knobs | Llama-2-70B, General-Purpose, H100 |
| **Total** | **21 unique variations** (with overlap counted once) | |

Figure 1 uses a 7-model superset of the core benchmark (same conditions: H100, General-Purpose, default config).

---

## Deliverables

**5 figures + 1 table.**

| # | Title | Type | Purpose |
|---|-------|------|---------|
| Fig 1 | Model Sensitivity | 2x3 grouped bar grid | Accuracy across 7 model architectures |
| Fig 2 | Hardware Portability | 2x3 grouped bar grid | Accuracy across 3 GPU types |
| Fig 3 | Workload Sensitivity | 2x3 grouped bar grid | Accuracy across 4 workload types |
| Fig 4 | Config Generalization | 2x3 grouped bar grid | Accuracy under vLLM config changes |
| Fig 5 | Accuracy-Speed Pareto | Scatter plot | Tradeoff frontier (the money shot) |
| Table 1 | Runtime Comparison | LaTeX/text table | Simulator runtime + speedup vs real |

---

## Global Design Language

### Color Palette

BLIS variants use a blue/teal gradient (dark = most sophisticated). Baselines use gray with hatching for B&W distinguishability.

| Simulator | Hex | Fill | Visual Role |
|-----------|-----|------|-------------|
| BLIS-trained-roofline | `#0077B6` | Solid | Hero (darkest, most prominent) |
| BLIS-crossmodel | `#00B4D8` | Solid | BLIS family |
| BLIS-roofline | `#90E0EF` | Solid | BLIS family |
| Vidur | `#6C757D` | Hatched `//` | Baseline |
| LLM-Optimizer | `#ADB5BD` | Hatched `\\` | Baseline |
| AIConfigurator | `#DEE2E6` | Hatched `xx` | Baseline |

### Typography

- `font.family: serif` with `text.usetex: True` for LaTeX-rendered labels
- Axis labels: 8pt
- Subplot titles: 9pt
- Matches `\small` font in two-column ACM/USENIX templates

### Shared Elements

- **20% MAPE threshold line:** Thin horizontal dashed line at 20% MAPE in all accuracy subplots as a visual "acceptable accuracy" marker.
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

### Bar Layout

- 6 bars per x-tick group (one per simulator)
- Bar width: ~3pt, inter-bar gap: 1pt
- Group width: ~20pt, inter-group gap: ~8pt
- BLIS bars always appear first (left) in each group, baselines to the right

### Aggregation

When a figure's x-axis dimension has multiple underlying variations (e.g., Figure 2's H100 group spans 16 core variations), the bar height is the **median MAPE** across those variations, with **IQR error bars**. When n <= 4 (e.g., A100 with 2 anchors), overlay **individual data points** as semi-transparent dots instead of error bars for honesty.

### Per-Figure Specifics

#### Figure 1 — Model Sensitivity

- **X-axis:** 7 models (short labels, e.g., `Ll-7B`, `Ll-70B`, `Mx-8x7B`, `CL-34B`, + 3 TBD)
- **Fixed:** H100, General-Purpose workload, default vLLM config
- **Aggregation:** 1 variation per (model, simulator) — no aggregation needed, no error bars
- **Caption:** "Prediction accuracy across model architectures. MAPE of six simulators on seven LLM models (H100, General-Purpose workload, default vLLM config). Top row: mean latency; bottom row: P99 tail latency. BLIS-trained-roofline (dark blue) maintains low MAPE across all architectures, while analytical estimators degrade at P99. LLM-Optimizer and AIConfigurator produce only mean estimates (P99 bars absent)."

#### Figure 2 — Hardware Portability

- **X-axis:** 3 GPU types (`H100`, `A100`, `L40`)
- **Fixed:** Selected anchor (model, workload) pairs per hardware
- **Aggregation:** H100 aggregates 16 core variations (median + IQR error bars). A100 and L40 have 2 anchors each (overlay individual dots).
- **Caption:** "Hardware portability. MAPE across three GPU types. H100 bars show median over 16 variations (error bars: IQR); A100 and L40 show individual anchor results as overlaid dots. BLIS variants generalize across GPU generations using only datasheet specifications."

#### Figure 3 — Workload Sensitivity

- **X-axis:** 4 workloads (`General`, `Codegen`, `Roleplay`, `Reasoning`)
- **Fixed:** H100, default config
- **Aggregation:** Each workload aggregates across 4 models (median + IQR error bars)
- **Caption:** "Workload sensitivity. MAPE across four workload types, aggregated over models (H100, default config). Workloads with more extreme token-length distributions (Code Generation, Reasoning) increase prediction error across all simulators, but BLIS-trained-roofline shows the smallest degradation."

#### Figure 4 — Configuration Generalization

- **X-axis:** 4 configs (`Default`, `BatchTok-2K`, `No-Offload`, `GPU-0.8`)
- **Fixed:** Llama-2-70B, General-Purpose, H100
- **Aggregation:** 1 variation per (config, simulator) — no aggregation needed
- **Caption:** "Sensitivity to serving configuration. MAPE under four vLLM configurations for Llama-2-70B on H100 (General-Purpose workload). Varying `max_num_batched_tokens`, `cpu_offloading`, and `gpu_memory_utilization` has minimal impact on BLIS prediction error, demonstrating robustness to configuration without per-config tuning."

---

## Figure 5 — Accuracy-Speed Pareto

### Layout

Single-panel scatter plot.

- **X-axis:** Median MAPE across all 21 variations (lower = more accurate)
- **Y-axis:** Median wall-clock simulation time, **log scale** (lower = faster)
- **Points:** One per simulator, using the color palette above
- **Error bars:** Horizontal = IQR of MAPE; Vertical = IQR of runtime
- **Pareto shading:** Light blue shaded region from BLIS-trained-roofline toward the origin, marking the dominated region
- **Annotations:** Each point labeled with simulator name + median values

### Caption

"Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE (x-axis) and median simulation runtime (y-axis, log scale) across all 21 variations. Error bars show interquartile range. BLIS-trained-roofline achieves the best accuracy-speed tradeoff, matching Vidur's prediction quality at 20x lower computational cost."

---

## Table 1 — Runtime Comparison

| Simulator | Median Runtime (s) | Speedup vs. Real |
|-----------|-------------------:|------------------:|
| BLIS-trained-roofline | 0.6 | 500x |
| BLIS-crossmodel | 0.6 | 500x |
| BLIS-roofline | 1.3 | 250x |
| Vidur | 16.8 | 20x |
| AIConfigurator | 3.4 | 100x |
| LLM-Optimizer | 0.05 | 6,000x |

- Sorted by BLIS family first, then baselines by descending runtime
- "Speedup vs. Real" = median real vLLM experiment duration / median simulator runtime
- **Caption:** "Simulator runtime and speedup. Median wall-clock time per variation and speedup relative to running the actual vLLM experiment. BLIS variants complete in under 1.5 seconds, enabling rapid exploration of the model-hardware-config design space."

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

### CLI Integration

Add a `--figures` flag to `experiment/run.py` (or a separate `experiment/figures.py` CLI entry point) so figures can be regenerated from existing CSVs without re-running simulations.

### Robustness

- Gracefully handle missing simulators or metrics (absent bars, not crashes)
- Handle varying numbers of models, workloads, hardware, and configs in the CSV
- Stage index filtering: use `stage_index == -1` (aggregate) rows for the figures unless per-stage breakdown is explicitly needed

### Figure Sizing

- Figures 1-4: full-page width in two-column format (~7.0" x 3.5")
- Figure 5: single-column width (~3.5" x 3.0")
- All saved at 300 DPI for PNG, vector for PDF
