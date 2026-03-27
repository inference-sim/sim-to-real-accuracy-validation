# Simulator Comparison Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pairwise BLIS-roofline vs other simulator comparison figures with 2×3 grid layout (aggregate + model breakdown).

**Architecture:** Add three new functions to `experiment/figures.py` that generate 2×3 subplot grids comparing two simulators. Top row shows aggregate MAPE across all experiments, bottom row shows per-model MAPE breakdown.

**Tech Stack:** matplotlib, pandas, numpy (existing dependencies)

---

## File Structure

**Modify:**
- `experiment/figures.py` - Add 3 new functions, integrate into main()

**Output:**
- `results/figures/sim_comparisons/blis_vs_vidur.pdf` (+ PNG)
- `results/figures/sim_comparisons/blis_vs_llm_optimizer.pdf` (+ PNG)
- `results/figures/sim_comparisons/blis_vs_aiconfigurator.pdf` (+ PNG)
- `results/figures/sim_comparisons/blis_vs_llmservingsim.pdf` (+ PNG)

---

### Task 1: Add aggregate panel helper function

**Files:**
- Modify: `experiment/figures.py` (after line 867, after `plot_aggregate_comparison_llmservingsim()`)

- [ ] **Step 1: Add `_plot_aggregate_panel()` function**

Add this function after `plot_aggregate_comparison_llmservingsim()`:

```python
def _plot_aggregate_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sim1: str,
    sim2: str,
    metric_key: str,
    metric_label: str,
) -> float:
    """Plot aggregate comparison bars for a single metric on the given axis.

    Returns the maximum bar height for y-axis scaling.
    """
    # Filter to just the two simulators and this metric
    df_filtered = df[
        (df["simulator"].isin([sim1, sim2])) &
        (df["metric_name"] == metric_key)
    ]

    if df_filtered.empty:
        return 0.0

    # Compute median MAPE per simulator
    agg_data = df_filtered.groupby("simulator")["mape"].median()

    # Order simulators according to SIMULATOR_ORDER
    sims_ordered = [s for s in [sim1, sim2] if s in agg_data.index]
    if not sims_ordered:
        return 0.0

    x = np.arange(len(sims_ordered))
    heights = [agg_data[sim] for sim in sims_ordered]
    colors = [COLOR_PALETTE[sim] for sim in sims_ordered]
    hatches = [HATCH_PATTERNS.get(sim, "") for sim in sims_ordered]

    bar_width = 0.6

    for i, (pos, height, color, hatch, sim) in enumerate(
        zip(x, heights, colors, hatches, sims_ordered)
    ):
        ax.bar(
            pos, height, bar_width,
            color=color, hatch=hatch,
            edgecolor="black", linewidth=0.5,
            label=SIMULATOR_DISPLAY_NAMES[sim],
        )

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(sims_ordered) - 0.5)
    ax.set_title(metric_label, fontsize=10, fontweight="bold")

    return max(heights) if heights else 0.0
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiment/figures.py`
Expected: No output (successful compilation)

- [ ] **Step 3: Commit**

```bash
git add experiment/figures.py
git commit -m "feat: add aggregate panel helper for simulator comparisons

Add _plot_aggregate_panel() to draw 2-bar aggregate comparison
for a single metric on a given matplotlib axis.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Add model breakdown panel helper function

**Files:**
- Modify: `experiment/figures.py` (after `_plot_aggregate_panel()`)

- [ ] **Step 1: Add `_plot_model_breakdown_panel()` function**

Add this function after `_plot_aggregate_panel()`:

```python
def _plot_model_breakdown_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sim1: str,
    sim2: str,
    metric_key: str,
    metric_label: str,
) -> float:
    """Plot model-wise grouped bars for a single metric on the given axis.

    Returns the maximum bar height for y-axis scaling.
    """
    # Filter to just the two simulators and this metric
    df_filtered = df[
        (df["simulator"].isin([sim1, sim2])) &
        (df["metric_name"] == metric_key)
    ]

    if df_filtered.empty:
        return 0.0

    # Only show models that have data for at least one simulator
    present_models = [m for m in MODEL_ORDER if m in df_filtered["model"].values]
    if not present_models:
        return 0.0

    n_models = len(present_models)
    n_sims = 2
    bar_width = 0.8 / n_sims
    x = np.arange(n_models)
    global_max = 0.0

    for sim_idx, sim in enumerate([sim1, sim2]):
        offset = (sim_idx - n_sims / 2 + 0.5) * bar_width
        positions = []
        heights = []

        for m_idx, model in enumerate(present_models):
            vals = df_filtered[
                (df_filtered["model"] == model) &
                (df_filtered["simulator"] == sim)
            ]["mape"]
            if vals.empty:
                continue
            mape = vals.median()
            positions.append(x[m_idx] + offset)
            heights.append(mape)

        if not positions:
            continue

        global_max = max(global_max, max(heights))

        ax.bar(
            positions, heights, bar_width,
            color=COLOR_PALETTE[sim],
            hatch=HATCH_PATTERNS.get(sim, ""),
            edgecolor="black", linewidth=0.5,
            label=SIMULATOR_DISPLAY_NAMES[sim],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_SHORT_LABELS.get(m, m) for m in present_models],
        rotation=35, ha="right",
    )
    ax.set_title(metric_label, fontsize=10, fontweight="bold")

    return global_max
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiment/figures.py`
Expected: No output (successful compilation)

- [ ] **Step 3: Commit**

```bash
git add experiment/figures.py
git commit -m "feat: add model breakdown panel helper for simulator comparisons

Add _plot_model_breakdown_panel() to draw grouped bars by model
for a single metric on a given matplotlib axis.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Add main simulator comparison function

**Files:**
- Modify: `experiment/figures.py` (after `_plot_model_breakdown_panel()`)

- [ ] **Step 1: Add `plot_simulator_comparison()` function**

Add this function after `_plot_model_breakdown_panel()`:

```python
def plot_simulator_comparison(
    df: pd.DataFrame,
    sim1: str,
    sim2: str,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Compare two simulators with 2×3 grid (aggregate + model breakdown).

    Top row: aggregate MAPE for E2E, TTFT, ITL
    Bottom row: per-model MAPE for E2E, TTFT, ITL

    Only includes experiments where both simulators have data.
    Aggregates across all configs and workloads.
    """
    _apply_rc_params()

    # Find experiments with data from both simulators
    exp_sims = df.groupby("experiment_folder")["simulator"].apply(set)
    common_exps = exp_sims[exp_sims.apply(lambda s: {sim1, sim2}.issubset(s))].index

    if len(common_exps) == 0:
        warnings.warn(f"No experiments with data from both {sim1} and {sim2}")
        return None

    df_filtered = df[df["experiment_folder"].isin(common_exps)]

    # Create 2×3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

    metrics = [("e2e_mean", "E2E Mean"), ("ttft_mean", "TTFT Mean"), ("itl_mean", "ITL Mean")]

    # Top row: aggregate panels
    col_maxes_top = []
    for col_idx, (metric_key, metric_label) in enumerate(metrics):
        max_height = _plot_aggregate_panel(
            axes[0, col_idx], df_filtered, sim1, sim2, metric_key, metric_label
        )
        col_maxes_top.append(max_height)

    # Bottom row: model breakdown panels
    col_maxes_bottom = []
    for col_idx, (metric_key, metric_label) in enumerate(metrics):
        max_height = _plot_model_breakdown_panel(
            axes[1, col_idx], df_filtered, sim1, sim2, metric_key, metric_label
        )
        col_maxes_bottom.append(max_height)

    # Set y-axes with 20% headroom per row
    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    for col_idx in range(3):
        # Top row
        y_top = col_maxes_top[col_idx] * 1.20 if col_maxes_top[col_idx] > 0 else 1.0
        axes[0, col_idx].set_ylim(bottom=0, top=y_top)
        axes[0, col_idx].set_ylabel(f"MAPE ({pct})")

        # Bottom row
        y_top = col_maxes_bottom[col_idx] * 1.20 if col_maxes_bottom[col_idx] > 0 else 1.0
        axes[1, col_idx].set_ylim(bottom=0, top=y_top)
        axes[1, col_idx].set_ylabel(f"MAPE ({pct})")

    # Title
    sim2_display = SIMULATOR_DISPLAY_NAMES.get(sim2, sim2)
    fig.suptitle(
        f"BLIS-Roofline vs {sim2_display} Simulator Comparison (n={len(common_exps)}) ↓",
        fontsize=11, fontweight="bold"
    )

    # Collect legend handles/labels from all axes (deduplicate)
    all_handles, all_labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label and label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)

    if all_handles:
        fig.legend(
            all_handles, all_labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=2,
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.08)

    if output_path:
        _save_figure(fig, output_path)
    return fig
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiment/figures.py`
Expected: No output (successful compilation)

- [ ] **Step 3: Commit**

```bash
git add experiment/figures.py
git commit -m "feat: add main simulator comparison function

Add plot_simulator_comparison() to generate 2×3 grid comparing
two simulators (aggregate + model breakdown for E2E/TTFT/ITL).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Integrate into main() to generate comparison figures

**Files:**
- Modify: `experiment/figures.py:1502-1512` (after table generation, before final print)

- [ ] **Step 1: Add comparison figure generation to main()**

Find this section in `main()` (around line 1502):

```python
    except Exception as e:
        print(f"  FAIL: table1_runtime.tex ({e})")
        logger.exception("Failed to generate table1_runtime.tex")

    print(f"\nFigures saved to {out}")
```

Replace it with:

```python
    except Exception as e:
        print(f"  FAIL: table1_runtime.tex ({e})")
        logger.exception("Failed to generate table1_runtime.tex")

    # Simulator comparison figures
    sim_comparison_dir = os.path.join(out, "sim_comparisons")
    os.makedirs(sim_comparison_dir, exist_ok=True)

    comparison_pairs = [
        ("blis-roofline", "vidur", "blis_vs_vidur.pdf"),
        ("blis-roofline", "llm-optimizer-estimate", "blis_vs_llm_optimizer.pdf"),
        ("blis-roofline", "aiconfigurator-estimate", "blis_vs_aiconfigurator.pdf"),
        ("blis-roofline", "llmservingsim", "blis_vs_llmservingsim.pdf"),
    ]

    for sim1, sim2, filename in comparison_pairs:
        try:
            # Use error_df_full (unfiltered) so we always include all simulators
            fig = plot_simulator_comparison(
                error_df_full, sim1, sim2,
                os.path.join(sim_comparison_dir, filename)
            )
            if fig is not None:
                plt.close(fig)
                print(f"  OK: sim_comparisons/{filename}")
            else:
                print(f"  SKIP: sim_comparisons/{filename} (no shared experiments)")
        except Exception as e:
            print(f"  FAIL: sim_comparisons/{filename} ({e})")
            logger.exception("Failed to generate %s", filename)

    print(f"\nFigures saved to {out}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiment/figures.py`
Expected: No output (successful compilation)

- [ ] **Step 3: Commit**

```bash
git add experiment/figures.py
git commit -m "feat: integrate simulator comparisons into main figure generation

Generate 4 pairwise BLIS comparison figures in sim_comparisons/
subdirectory alongside existing figures.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Test the full pipeline

**Files:**
- Test: `experiment/figures.py` (full run)
- Verify: `results/figures/sim_comparisons/*.pdf`

- [ ] **Step 1: Run figure generation script**

Run: `python experiment/figures.py --results-dir results --output-dir results/figures`

Expected output should include:
```
  OK: fig0a_aggregate_analytical.pdf
  ...
  OK: table1_runtime.tex
  OK: sim_comparisons/blis_vs_vidur.pdf
  OK: sim_comparisons/blis_vs_llm_optimizer.pdf
  OK: sim_comparisons/blis_vs_aiconfigurator.pdf
  SKIP: sim_comparisons/blis_vs_llmservingsim.pdf (no shared experiments)

Figures saved to results/figures
```

Note: LLMServingSim may show SKIP or OK depending on available data.

- [ ] **Step 2: Verify output files exist**

Run: `ls -lh results/figures/sim_comparisons/`

Expected: 6-8 files (PDFs and PNGs for 3-4 simulator pairs)

- [ ] **Step 3: Verify figure structure**

Open one PDF (e.g., `results/figures/sim_comparisons/blis_vs_llm_optimizer.pdf`)

Expected:
- 2×3 grid of subplots
- Top row: 3 panels with 2 bars each (aggregate)
- Bottom row: 3 panels with multiple models on x-axis, 2 bars per model
- Legend at bottom with 2 entries (BLIS-Roofline, other simulator)
- Title includes experiment count

- [ ] **Step 4: Verify existing figures unchanged**

Run: `git diff results/figures/fig*.pdf`

Expected: No differences (or only timestamp-related metadata changes)

- [ ] **Step 5: Commit generated figures**

```bash
git add results/figures/sim_comparisons/
git commit -m "test: add generated simulator comparison figures

Add initial set of BLIS vs other simulator comparison figures
showing 2×3 grid (aggregate + model breakdown).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Validation Checklist

After completing all tasks:

- [ ] All 4 comparison figures generated (or 3 if LLMServingSim skipped)
- [ ] Each figure has 2×3 grid layout
- [ ] Top row shows aggregate bars (2 per metric)
- [ ] Bottom row shows model breakdown (2 bars per model)
- [ ] Experiment counts in titles are reasonable (10-40+ depending on simulator)
- [ ] Existing figures (fig0-fig5, table1) are unchanged
- [ ] No syntax errors in `figures.py`
- [ ] Figures saved to `results/figures/sim_comparisons/` subdirectory

## Success Criteria

1. Script runs without errors: `python experiment/figures.py`
2. All existing figures still generate correctly
3. New `sim_comparisons/` directory created with 3-4 figure sets (PDF + PNG)
4. Each comparison figure shows clear visual distinction between simulators
5. No modifications to existing figure generation logic
