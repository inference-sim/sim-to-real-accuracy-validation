# Simulator Comparison Figures Design

**Date:** 2026-03-27
**Goal:** Add pairwise BLIS-roofline vs other simulator comparison figures

## Overview

Generate 4 new figures showing head-to-head comparisons between BLIS-roofline and each other simulator (Vidur, LLM-Optimizer, AIConfigurator, LLMServingSim). Each figure uses a 2×3 grid layout combining aggregate and model-wise breakdowns.

## Output

**Location:** `results/figures/sim_comparisons/`

**Files:**
- `blis_vs_vidur.pdf` (+ PNG)
- `blis_vs_llm_optimizer.pdf` (+ PNG)
- `blis_vs_aiconfigurator.pdf` (+ PNG)
- `blis_vs_llmservingsim.pdf` (+ PNG)

## Figure Layout

Each figure has a **2×3 grid**:

**Top row (Aggregate):** 3 panels showing E2E, TTFT, ITL
- 2 bars per panel (BLIS vs other simulator)
- Median MAPE aggregated across all experiments, models, configs, workloads

**Bottom row (Model Breakdown):** 3 panels showing E2E, TTFT, ITL
- Models on x-axis (MODEL_ORDER)
- 2 bars per model (BLIS vs other simulator)
- Median MAPE aggregated across configs and workloads for each model

**Styling:**
- Figsize: `(10, 6.5)`
- Reuse existing COLOR_PALETTE, HATCH_PATTERNS, SIMULATOR_DISPLAY_NAMES
- Independent y-axis per column with 20% headroom
- Single legend at bottom (2 entries)
- Title: `"BLIS-Roofline vs {Other} Simulator Comparison (n={count}) ↓"`

## Data Filtering

**Per comparison:**
- Filter to experiments where BOTH simulators have data (intersection)
- Include ALL configs and workloads (no filtering)
- stage_index == -1 (summary rows)
- All hardware types (H100, A100, L40S)

## Implementation

**Three new functions in `figures.py`:**

1. **`_plot_aggregate_panel(ax, df, sim1, sim2, metric_key, metric_label)`**
   - Draw 2-bar aggregate comparison on a single axis
   - Compute median MAPE per simulator across all experiments
   - Return max height for y-axis scaling

2. **`_plot_model_breakdown_panel(ax, df, sim1, sim2, metric_key, metric_label)`**
   - Draw grouped bars by model on a single axis
   - Compute median MAPE per (model, simulator)
   - Reuse bar positioning logic from `_grouped_bar()`
   - Return max height for y-axis scaling

3. **`plot_simulator_comparison(df, sim1, sim2, output_path)`**
   - Create 2×3 subplot grid
   - Call helpers for each panel
   - Set y-axes, legend, title
   - Save via `_save_figure()`

**Integration in `main()`:**
- Add after existing figure generation
- Loop through 4 simulator pairs
- Use `error_df_full` (unfiltered data)
- Save to `sim_comparisons/` subdirectory

## Edge Cases

- If no shared experiments: return None, print "SKIP"
- If model has no data: skip that model in breakdown
- LLMServingSim has sparse data (~1-2 experiments): show what's available
- Empty metric panels: draw but will be empty (consistent with existing figures)

## Testing

**Validation:**
1. Run `python experiment/figures.py`
2. Check 4 new PDFs/PNGs in `sim_comparisons/`
3. Verify 2×3 layout and experiment counts
4. Confirm existing figures unchanged

**Expected counts:**
- BLIS vs LLM-Optimizer/AIConfigurator: ~40+ experiments
- BLIS vs Vidur: ~10-15 experiments (3 models only)
- BLIS vs LLMServingSim: ~1-2 experiments (sparse)
