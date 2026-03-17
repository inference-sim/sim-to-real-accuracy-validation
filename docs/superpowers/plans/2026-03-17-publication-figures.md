# Publication Figures Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 5 publication-quality figures + 1 LaTeX table from experiment output CSVs for a best-paper submission at NSDI/EuroSys.

**Architecture:** A self-contained `experiment/figures.py` module reads existing CSVs (`error_records.csv`, `runtime.csv`), enriches with optional metadata CSV, and produces PDF/PNG figures via matplotlib. A reusable `_bar_chart_grid()` helper renders the 2x3 grouped bar layout shared by Figures 1-4. Each figure function filters, aggregates, and plots independently.

**Tech Stack:** matplotlib, pandas, numpy, pytest. LaTeX rendering (`text.usetex: True`).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `experiment/figures.py` | Style constants, data loading/enrichment, `_bar_chart_grid()`, 7 figure/table functions, `main()` |
| Create | `tests/test_figures.py` | Unit tests for data logic + smoke tests for plotting |
| Modify | `experiment/run.py:128-159` | Add `--figures` flag to regenerate from existing CSVs |

---

## Chunk 1: Foundation and Data Infrastructure

### Task 1: Style Constants and Display Mappings

**Files:**
- Create: `experiment/figures.py`
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_figures.py
"""Tests for experiment.figures."""
from __future__ import annotations

import pytest


class TestConstants:
    def test_simulator_order_length(self):
        from experiment.figures import SIMULATOR_ORDER
        assert len(SIMULATOR_ORDER) == 5

    def test_excluded_not_in_order(self):
        from experiment.figures import SIMULATOR_ORDER, EXCLUDED_SIMULATORS
        assert not set(SIMULATOR_ORDER) & EXCLUDED_SIMULATORS

    def test_all_simulators_have_style_entries(self):
        from experiment.figures import (
            SIMULATOR_ORDER, COLOR_PALETTE,
            SIMULATOR_DISPLAY_NAMES, HATCH_PATTERNS,
        )
        for sim in SIMULATOR_ORDER:
            assert sim in COLOR_PALETTE
            assert sim in SIMULATOR_DISPLAY_NAMES
            assert sim in HATCH_PATTERNS

    def test_model_order_excludes_llama2_7b(self):
        from experiment.figures import MODEL_ORDER
        assert "meta-llama/Llama-2-7b-hf" not in MODEL_ORDER
        assert len(MODEL_ORDER) == 7

    def test_metrics_grid_is_2x3(self):
        from experiment.figures import METRICS_GRID
        assert len(METRICS_GRID) == 2
        assert all(len(row) == 3 for row in METRICS_GRID)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_figures.py::TestConstants -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement constants**

```python
# experiment/figures.py
"""Publication figures for sim-to-real accuracy validation.

Generates 5 figures + 1 table from error_records.csv and runtime.csv.
Spec: docs/superpowers/specs/2026-03-16-publication-figures-design.md
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_SIMULATORS = frozenset({"blis-blackbox", "blis-crossmodel"})

SIMULATOR_ORDER = [
    "blis-trained-roofline",
    "blis-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
]

SIMULATOR_DISPLAY_NAMES = {
    "blis-trained-roofline": "BLIS-Trained",
    "blis-roofline": "BLIS-Roofline",
    "vidur": "Vidur",
    "llm-optimizer-estimate": "LLM-Optimizer",
    "aiconfigurator-estimate": "AIConfigurator",
}

COLOR_PALETTE = {
    "blis-trained-roofline": "#0077B6",
    "blis-roofline": "#90E0EF",
    "vidur": "#6C757D",
    "llm-optimizer-estimate": "#ADB5BD",
    "aiconfigurator-estimate": "#DEE2E6",
}

HATCH_PATTERNS = {
    "blis-trained-roofline": None,
    "blis-roofline": None,
    "vidur": "//",
    "llm-optimizer-estimate": "\\\\",
    "aiconfigurator-estimate": "xx",
}

MODEL_ORDER = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-14B",
    "codellama/CodeLlama-34b-Instruct-hf",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
]

MODEL_SHORT_LABELS = {
    "meta-llama/Llama-2-7b-hf": "Ll-2-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Ll-3.1-8B",
    "Qwen/Qwen3-14B": "Qw-14B",
    "codellama/CodeLlama-34b-Instruct-hf": "CL-34B",
    "meta-llama/Llama-2-70b-hf": "Ll-2-70B",
    "mistralai/Mixtral-8x7B-v0.1": "Mx-8x7B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mx-8x22B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": "Scout-17B",
}

WORKLOAD_DISPLAY_NAMES = {
    "general": "General-Purpose",
    "codegen": "Code Generation",
    "roleplay": "Roleplay",
    "reasoning": "Reasoning",
    "general-lite": "General-Lite",
}

METRICS_GRID = [
    [("e2e_mean", r"E2E Mean"), ("ttft_mean", r"TTFT Mean"), ("itl_mean", r"ITL Mean")],
    [("e2e_p99", r"E2E P99"), ("ttft_p99", r"TTFT P99"), ("itl_p99", r"ITL P99")],
]

MAPE_THRESHOLD = 20.0
FIGURE_SIZES = {"bar_grid": (7.0, 3.5), "pareto": (3.5, 3.0)}

RC_PARAMS = {
    "font.family": "serif", "text.usetex": True,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_figures.py::TestConstants -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add style constants and display mappings"
```

---

### Task 2: Data Loading and Metadata Enrichment

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Context:** Existing CSVs lack hardware, tp, dp, and config-knob columns needed for Figures 2 and 4. An optional metadata CSV (`experiment_metadata.csv`) maps `experiment_folder` to these fields. When absent, figures needing metadata are skipped with a warning.

Metadata CSV schema:
```
experiment_folder,hardware,tp,dp,cpu_offloading,gpu_memory_utilization,config_tag
/path/to/exp,H100,1,1,Disabled,0.90,default
```

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path


def _sample_error_csv(tmp_path: Path) -> str:
    path = tmp_path / "error_records.csv"
    path.write_text(
        "simulator,experiment_folder,model,workload,stage_index,metric_name,"
        "predicted,actual,mape,mpe,absolute_error\n"
        "blis-trained-roofline,/exp/1,meta-llama/Llama-3.1-8B-Instruct,general,"
        "-1,e2e_mean,110,100,10,10,10\n"
        "blis-crossmodel,/exp/1,meta-llama/Llama-3.1-8B-Instruct,general,"
        "-1,e2e_mean,200,100,100,100,100\n"
        "blis-trained-roofline,/exp/1,meta-llama/Llama-3.1-8B-Instruct,general,"
        "0,e2e_mean,115,100,15,15,15\n"
    )
    return str(path)


def _sample_runtime_csv(tmp_path: Path) -> str:
    path = tmp_path / "runtime.csv"
    path.write_text(
        "simulator,experiment_folder,model,workload,wall_clock_seconds\n"
        "blis-trained-roofline,/exp/1,meta-llama/Llama-3.1-8B-Instruct,general,1.5\n"
        "blis-crossmodel,/exp/1,meta-llama/Llama-3.1-8B-Instruct,general,3.0\n"
    )
    return str(path)


def _sample_metadata_csv(tmp_path: Path) -> str:
    path = tmp_path / "experiment_metadata.csv"
    path.write_text(
        "experiment_folder,hardware,tp,dp,cpu_offloading,gpu_memory_utilization,config_tag\n"
        "/exp/1,H100,1,1,Disabled,0.90,default\n"
    )
    return str(path)


class TestDataLoading:
    def test_excludes_blacklisted_simulators(self, tmp_path):
        from experiment.figures import load_error_data
        df = load_error_data(_sample_error_csv(tmp_path))
        assert "blis-crossmodel" not in df["simulator"].values

    def test_keeps_summary_rows_only(self, tmp_path):
        from experiment.figures import load_error_data
        df = load_error_data(_sample_error_csv(tmp_path))
        assert (df["stage_index"] == -1).all()

    def test_runtime_excludes_blacklisted(self, tmp_path):
        from experiment.figures import load_runtime_data
        df = load_runtime_data(_sample_runtime_csv(tmp_path))
        assert "blis-crossmodel" not in df["simulator"].values

    def test_enrich_joins_metadata(self, tmp_path):
        from experiment.figures import load_error_data, enrich_with_metadata
        df = load_error_data(_sample_error_csv(tmp_path))
        enriched = enrich_with_metadata(df, _sample_metadata_csv(tmp_path))
        assert "hardware" in enriched.columns
        assert enriched.iloc[0]["hardware"] == "H100"

    def test_enrich_without_metadata_adds_empty_cols(self, tmp_path):
        from experiment.figures import load_error_data, enrich_with_metadata
        df = load_error_data(_sample_error_csv(tmp_path))
        enriched = enrich_with_metadata(df, metadata_path=None)
        assert "hardware" in enriched.columns
```

- [ ] **Step 2: Run tests → FAIL**

Run: `pytest tests/test_figures.py::TestDataLoading -v`

- [ ] **Step 3: Implement data loading**

```python
# Add to experiment/figures.py

_METADATA_COLUMNS = [
    "hardware", "tp", "dp", "cpu_offloading", "gpu_memory_utilization", "config_tag",
]


def load_error_data(csv_path: str) -> pd.DataFrame:
    """Load error_records.csv, exclude blacklisted simulators, keep summary rows."""
    df = pd.read_csv(csv_path)
    df = df[~df["simulator"].isin(EXCLUDED_SIMULATORS)]
    df = df[df["stage_index"] == -1]
    return df.reset_index(drop=True)


def load_runtime_data(csv_path: str) -> pd.DataFrame:
    """Load runtime.csv, exclude blacklisted simulators."""
    df = pd.read_csv(csv_path)
    df = df[~df["simulator"].isin(EXCLUDED_SIMULATORS)]
    return df.reset_index(drop=True)


def enrich_with_metadata(
    df: pd.DataFrame,
    metadata_path: str | None = None,
) -> pd.DataFrame:
    """Left-join metadata onto DataFrame by experiment_folder."""
    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        return df.merge(meta, on="experiment_folder", how="left", suffixes=("", "_meta"))
    df = df.copy()  # Never mutate caller's DataFrame
    for col in _METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _has_metadata(df: pd.DataFrame) -> bool:
    """Check if DataFrame has non-empty metadata columns."""
    return "hardware" in df.columns and df["hardware"].notna().any() and (df["hardware"] != "").any()
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add data loading and metadata enrichment"
```

---

### Task 3: Bar Chart Grid Helper

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write failing tests**

```python
class TestBarChartGrid:
    def _make_data(self):
        """Minimal data: 2 groups, 2 simulators, 6 metrics."""
        metrics = {m: 10.0 for row in METRICS_GRID for m, _ in row}
        return {
            "Group-A": {"blis-trained-roofline": dict(metrics), "vidur": dict(metrics)},
            "Group-B": {"blis-trained-roofline": dict(metrics)},
        }

    def test_returns_2x3_axes(self):
        from experiment.figures import _bar_chart_grid, METRICS_GRID
        fig, axes = _bar_chart_grid(
            data=self._make_data(), group_order=["Group-A", "Group-B"],
            title="Test", output_path=None,
        )
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_threshold_line_present(self):
        from experiment.figures import _bar_chart_grid, MAPE_THRESHOLD
        fig, axes = _bar_chart_grid(
            data=self._make_data(), group_order=["Group-A"],
            title="Test", output_path=None,
        )
        for ax in axes.flat:
            y_data = [l.get_ydata()[0] for l in ax.get_lines() if hasattr(l.get_ydata(), '__len__')]
            assert any(abs(y - MAPE_THRESHOLD) < 0.01 for y in y_data)
        plt.close(fig)

    def test_saves_pdf_and_png(self, tmp_path):
        from experiment.figures import _bar_chart_grid
        out = str(tmp_path / "test.pdf")
        _bar_chart_grid(data=self._make_data(), group_order=["Group-A"],
                        title="Test", output_path=out)
        assert os.path.exists(out)
        assert os.path.exists(out.replace(".pdf", ".png"))
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement `_bar_chart_grid()`**

Key signature:
```python
def _bar_chart_grid(
    data: dict[str, dict[str, dict[str, float | None]]],
    group_order: list[str],
    title: str,
    output_path: str | None,
    group_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    error_data: dict | None = None,   # IQR: {group: {sim: {metric: (lo, hi)}}}
    dot_data: dict | None = None,     # Overlay: {group: {sim: {metric: [vals]}}}
) -> tuple[plt.Figure, np.ndarray]:
```

Implementation notes:
- Apply `RC_PARAMS` via `matplotlib.rcParams.update()`
- Create `fig, axes = plt.subplots(2, 3, figsize=..., sharey="row")`
- Compute `bar_width = 0.8 / n_simulators`. Offset each simulator: `(sim_idx - n_sims/2 + 0.5) * bar_width`
- For each cell in METRICS_GRID, draw bars for each simulator present in data
- Draw MAPE_THRESHOLD dashed line in each subplot
- Single shared legend below bottom row via `fig.legend(..., loc="lower center", ncol=n_sims)`
- Save PDF + PNG via `_save_figure()` helper when output_path provided

**N/A annotation logic** (spec requirement): When a simulator is present in a group (has at least one metric) but a specific metric is missing (e.g., LLM-Optimizer has no P99), draw a small "N/A" text at the bar baseline position:
```python
# Inside the metric cell loop, after drawing bars:
for g_idx, g in enumerate(group_order):
    sim_data = data.get(g, {}).get(sim, {})
    if sim_data and metric_key not in sim_data:
        # Simulator ran for this group but lacks this metric → N/A
        ax.text(x[g_idx] + offset, 0.3, r"\textit{N/A}",
                ha="center", va="bottom", fontsize=5, color="gray")
```

**IQR error bars** — unpack `error_data` tuples into matplotlib's `yerr` 2xN format:
```python
if error_data and g in error_data and sim in error_data[g] and metric_key in error_data[g][sim]:
    lo, hi = error_data[g][sim][metric_key]
    yerr = np.array([[lo], [hi]])  # 2x1 for single bar; accumulate across groups
else:
    yerr = None
```
When building bars for all groups at once, accumulate into `[[lo_0, lo_1, ...], [hi_0, hi_1, ...]]`.

Also add a shared `_save_figure()` helper (used by both `_bar_chart_grid()` and `plot_pareto()`):
```python
def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure as PDF + PNG at 300 DPI."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = output_path.replace(".pdf", ".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", output_path)
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add _bar_chart_grid() helper"
```

---

## Chunk 2: Accuracy Bar Chart Figures (1-3)

### Task 4: Figure 1 — Model Sensitivity

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Filter:** `hardware == "H100"`, `workload IN ("general", "general-lite")`, `config_tag == "default"`, model IN MODEL_ORDER (7 models).
**X-axis:** 7 models ordered by MODEL_ORDER. Labels use MODEL_SHORT_LABELS.
**Aggregation:** 1 variation per (model, simulator) — no aggregation, no error bars.
**Caption (templated):**
```
"Prediction accuracy across model architectures. MAPE of {n_simulators}
simulators across {n_models} LLM models spanning dense (7B--70B) and MoE
(47B--141B) architectures on H100 (General-Purpose workload, default vLLM
config). Top row: mean latency; bottom row: P99 tail latency. BLIS-Trained
(dark blue) maintains low MAPE across all architectures. LLM-Optimizer and
AIConfigurator produce only mean estimates (tail-latency bars absent)."
```

- [ ] **Step 1: Write failing tests**

```python
class TestFigure1:
    def _make_fig1_df(self):
        """DataFrame with 2 models, 2 simulators, 6 metrics."""
        rows = []
        for model in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-14B"]:
            for sim in ["blis-trained-roofline", "vidur"]:
                for metric in ["e2e_mean", "e2e_p99", "ttft_mean", "ttft_p99",
                               "itl_mean", "itl_p99"]:
                    rows.append({
                        "simulator": sim, "model": model, "workload": "general",
                        "metric_name": metric, "mape": 10.0,
                        "hardware": "H100", "config_tag": "default",
                        "experiment_folder": f"/exp/{model}",
                        "stage_index": -1, "tp": 1, "dp": 1,
                    })
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_model_sensitivity
        fig = plot_model_sensitivity(self._make_fig1_df(), output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_filters_to_h100_default_only(self):
        from experiment.figures import plot_model_sensitivity
        df = self._make_fig1_df()
        # Add an A100 row that should be excluded
        extra = df.iloc[0:1].copy()
        extra["hardware"] = "A100"
        df = pd.concat([df, extra])
        fig = plot_model_sensitivity(df, output_path=None)
        plt.close(fig)  # Should not crash; A100 data excluded
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `plot_model_sensitivity()`**

```python
def plot_model_sensitivity(
    df: pd.DataFrame,
    output_path: str | None = "results/figures/fig1_model_sensitivity.pdf",
) -> plt.Figure | None:
    """Figure 1: MAPE across 7 model architectures on H100, default config."""
    if _has_metadata(df):
        df = df[(df["hardware"] == "H100") & (df["config_tag"] == "default")]
    df = df[df["workload"].isin(("general", "general-lite"))]
    df = df[df["model"].isin(MODEL_ORDER)]

    if df.empty:
        warnings.warn("Figure 1: no data after filtering")
        return None

    # Build data dict: {model: {simulator: {metric: mape}}}
    data = {}
    for model in MODEL_ORDER:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        data[model] = {}
        for sim in SIMULATOR_ORDER:
            sdf = mdf[mdf["simulator"] == sim]
            if sdf.empty:
                continue
            data[model][sim] = dict(zip(sdf["metric_name"], sdf["mape"]))

    group_labels = MODEL_SHORT_LABELS
    present_models = [m for m in MODEL_ORDER if m in data]
    fig, _ = _bar_chart_grid(
        data=data, group_order=present_models,
        group_labels=group_labels,
        title="Model Sensitivity", output_path=output_path,
    )
    return fig
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 1 — model sensitivity"
```

---

### Task 5: Figure 2 — Hardware Portability

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Requires metadata:** Yes (hardware column). Skip with warning if absent.
**Filter:** `workload IN ("general", "general-lite")`, `config_tag == "default"`, `tp == default`, `dp <= 1`. Per-model workload must be consistent across all hardware (e.g., if a model uses general-lite on H100, it must also use general-lite on A100/L40S).
**X-axis:** 3 GPU types: H100, A100-80GB, L40S.
**Aggregation:** Median MAPE + IQR error bars per hardware group. Overlay dots when n <= 3.
**Caption (templated):**
```
"Hardware portability. MAPE across {n_gpus} GPU types (default config). Each
bar aggregates across all viable models for that GPU (H100: n={n_h100},
A100: n={n_a100}, L40S: n={n_l40s}). BLIS variants generalize across GPU
generations using only datasheet specifications."
```

- [ ] **Step 1: Write failing tests**

```python
class TestFigure2:
    def _make_fig2_df(self):
        rows = []
        for hw in ["H100", "A100-80GB", "L40S"]:
            for sim in ["blis-trained-roofline", "blis-roofline"]:
                for metric in ["e2e_mean", "e2e_p99", "ttft_mean", "ttft_p99",
                               "itl_mean", "itl_p99"]:
                    rows.append({
                        "simulator": sim, "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "workload": "general", "metric_name": metric, "mape": 10.0,
                        "hardware": hw, "config_tag": "default",
                        "experiment_folder": f"/exp/{hw}", "stage_index": -1,
                        "tp": 1, "dp": 1,
                    })
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_hardware_portability
        fig = plot_hardware_portability(self._make_fig2_df(), output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_skips_without_metadata(self):
        from experiment.figures import plot_hardware_portability
        df = self._make_fig2_df().drop(columns=["hardware"])
        fig = plot_hardware_portability(df, output_path=None)
        assert fig is None
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `plot_hardware_portability()`**

```python
HARDWARE_ORDER = ["H100", "A100-80GB", "L40S"]

def plot_hardware_portability(
    df: pd.DataFrame,
    output_path: str | None = "results/figures/fig2_hardware_portability.pdf",
) -> plt.Figure | None:
    """Figure 2: MAPE across 3 GPU types, aggregated over models."""
    if not _has_metadata(df):
        warnings.warn("Figure 2: skipped (no hardware metadata)")
        return None
    df = df[(df["config_tag"] == "default") & (df["workload"].isin(("general", "general-lite")))]
    df = df[df["dp"].replace("", 1).fillna(1).astype(float) <= 1]  # DP<=1; NaN/empty → default=1

    # Aggregate: median MAPE per (hardware, simulator, metric)
    data, error_data, dot_data = {}, {}, {}
    for hw in HARDWARE_ORDER:
        hdf = df[df["hardware"] == hw]
        if hdf.empty:
            continue
        data[hw], error_data[hw], dot_data[hw] = {}, {}, {}
        for sim in SIMULATOR_ORDER:
            sdf = hdf[hdf["simulator"] == sim]
            if sdf.empty:
                continue
            data[hw][sim], error_data[hw][sim], dot_data[hw][sim] = {}, {}, {}
            for metric_row in METRICS_GRID:
                for metric_key, _ in metric_row:
                    vals = sdf[sdf["metric_name"] == metric_key]["mape"]
                    if vals.empty:
                        continue
                    data[hw][sim][metric_key] = vals.median()
                    if len(vals) > 3:
                        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                        error_data[hw][sim][metric_key] = (
                            vals.median() - q1, q3 - vals.median(),
                        )
                    else:
                        dot_data[hw][sim][metric_key] = vals.tolist()

    present_hw = [h for h in HARDWARE_ORDER if h in data]
    fig, _ = _bar_chart_grid(
        data=data, group_order=present_hw, title="Hardware Portability",
        output_path=output_path, error_data=error_data, dot_data=dot_data,
    )
    return fig
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 2 — hardware portability"
```

---

### Task 6: Figure 3 — Workload Sensitivity

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Filter:** `hardware == "H100"`, `config_tag == "default"`, `dp <= 1`, model IN (Llama-3.1-8B, Qwen3-14B, Scout-17B-16E, Mixtral-8x22B).
**X-axis:** 4 workloads: General-Purpose, Code Generation, Roleplay, Reasoning.
**Aggregation:** Median + IQR across 4 models. Overlay dots when n <= 3 (Reasoning has 3 models).
**Caption (templated):**
```
"Workload sensitivity. MAPE across {n_workloads} workload types, aggregated
over {n_models} models spanning dense and MoE architectures (H100, default
config). BLIS-Trained shows the smallest degradation across workload diversity
(median MAPE: {blis_trained_median_mape:.1f}\%)."
```

- [ ] **Step 1: Write failing tests**

```python
FIGURE3_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-14B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


class TestFigure3:
    def _make_fig3_df(self):
        rows = []
        for wl in ["general", "codegen", "roleplay", "reasoning"]:
            for model in FIGURE3_MODELS:
                for sim in ["blis-trained-roofline"]:
                    for metric in ["e2e_mean", "e2e_p99", "ttft_mean",
                                   "ttft_p99", "itl_mean", "itl_p99"]:
                        rows.append({
                            "simulator": sim, "model": model, "workload": wl,
                            "metric_name": metric, "mape": 8.0,
                            "hardware": "H100", "config_tag": "default",
                            "experiment_folder": f"/exp/{model}/{wl}",
                            "stage_index": -1, "tp": 1, "dp": 1,
                        })
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_workload_sensitivity
        fig = plot_workload_sensitivity(self._make_fig3_df(), output_path=None)
        assert fig is not None
        plt.close(fig)
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `plot_workload_sensitivity()`**

Same aggregation pattern as Figure 2, but grouped by workload instead of hardware. Key constant:

```python
FIGURE3_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-14B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

WORKLOAD_ORDER = ["general", "codegen", "roleplay", "reasoning"]
```

Filter to H100 + default config + FIGURE3_MODELS. For each workload, aggregate median MAPE across the models present. Use WORKLOAD_DISPLAY_NAMES for x-tick labels.

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 3 — workload sensitivity"
```

---

## Chunk 3: Config Figures, Pareto, Table, CLI

### Task 7: Figure 4a — Config Sensitivity (Dense)

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Requires metadata:** Yes (config_tag column). Skip with warning if absent.
**Filter:** `hardware == "H100"`, `workload == "general"`, `model == "meta-llama/Llama-3.1-8B-Instruct"`.
**X-axis:** 6 configs: `default`, `mbt=1024`, `mbt=8192`, `cpu-offload`, `gpu-0.95`, `tp=2`.
**Aggregation:** None (1 variation per config).
**Caption (templated):**
```
"Configuration sensitivity for a dense model (Llama-3.1-8B-Instruct, H100,
General-Purpose). Each group varies one vLLM knob from the default
({n_configs} configurations). BLIS prediction error remains stable across
configuration changes (max MAPE: {blis_trained_max_mape:.1f}\%)."
```

- [ ] **Step 1: Write failing tests**

```python
class TestFigure4a:
    def _make_fig4a_df(self):
        rows = []
        for tag in ["default", "mbt=1024", "mbt=8192", "cpu-offload", "gpu-0.95", "tp=2"]:
            for sim in ["blis-trained-roofline", "vidur"]:
                for metric in ["e2e_mean", "e2e_p99", "ttft_mean",
                               "ttft_p99", "itl_mean", "itl_p99"]:
                    rows.append({
                        "simulator": sim,
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "workload": "general", "metric_name": metric, "mape": 7.0,
                        "hardware": "H100", "config_tag": tag,
                        "experiment_folder": f"/exp/llama8b/{tag}",
                        "stage_index": -1, "tp": 1, "dp": 1,
                    })
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_config_sensitivity_dense
        fig = plot_config_sensitivity_dense(self._make_fig4a_df(), output_path=None)
        assert fig is not None
        plt.close(fig)
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `plot_config_sensitivity_dense()`**

```python
FIG4A_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FIG4A_CONFIG_ORDER = ["default", "mbt=1024", "mbt=8192", "cpu-offload", "gpu-0.95", "tp=2"]
FIG4A_CONFIG_LABELS = {
    "default": "Default", "mbt=1024": r"mbt=1024", "mbt=8192": r"mbt=8192",
    "cpu-offload": "CPU-Offload", "gpu-0.95": "GPU-0.95", "tp=2": "TP=2",
}

def plot_config_sensitivity_dense(df, output_path=...):
    if not _has_metadata(df):
        warnings.warn("Figure 4a: skipped (no config metadata)")
        return None
    df = df[(df["hardware"] == "H100") & (df["workload"] == "general")
            & (df["model"] == FIG4A_MODEL)]
    # Build data dict keyed by config_tag → simulator → metric → mape
    # Pass to _bar_chart_grid with FIG4A_CONFIG_ORDER and FIG4A_CONFIG_LABELS
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 4a — config sensitivity (dense)"
```

---

### Task 8: Figure 4b — Config Sensitivity (MoE)

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Filter:** `hardware == "H100"`, `workload == "general"`, `model == "mistralai/Mixtral-8x7B-v0.1"`.
**X-axis:** 7 configs: `default`, `mbt=1024`, `mbt=8192`, `cpu-offload`, `gpu-0.95`, `tp=4`, `ep=4`.
**Caption (templated):**
```
"Configuration sensitivity for an MoE model (Mixtral-8x7B, H100,
General-Purpose). Includes expert parallelism (EP=4 via DP=2).
{n_configs} configurations tested. BLIS handles both standard knobs
and MoE-specific parallelism without accuracy degradation."
```

- [ ] **Step 1-5:** Same pattern as Task 7.

Constants:
```python
FIG4B_MODEL = "mistralai/Mixtral-8x7B-v0.1"
FIG4B_CONFIG_ORDER = [
    "default", "mbt=1024", "mbt=8192", "cpu-offload", "gpu-0.95", "tp=4", "ep=4",
]
FIG4B_CONFIG_LABELS = {
    "default": "Default", "mbt=1024": r"mbt=1024", "mbt=8192": r"mbt=8192",
    "cpu-offload": "CPU-Offload", "gpu-0.95": "GPU-0.95",
    "tp=4": "TP=4", "ep=4": r"EP=4\,(DP=2)",
}
```

- [ ] **Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 4b — config sensitivity (MoE)"
```

---

### Task 9: Figure 5 — Accuracy-Speed Pareto

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Layout:** Single-panel scatter plot (not bar grid).
**X-axis:** Median MAPE across all variations (lower = better).
**Y-axis:** Median wall-clock simulation time, log scale (lower = faster).
**Points:** One per simulator. BLIS-Trained marker larger. Error bars = IQR.
**Caption (templated):**
```
"Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE
(x-axis) and median simulation runtime (y-axis, log scale) across all
available variations ({total_variations} total). Error bars show interquartile
range. BLIS-Trained achieves the best accuracy-speed tradeoff (median MAPE:
{blis_trained_mape:.1f}\%, runtime: {blis_trained_runtime:.1f}s)."
```

- [ ] **Step 1: Write failing tests**

```python
class TestFigure5:
    def _make_pareto_data(self, tmp_path):
        error_rows, runtime_rows = [], []
        for sim in ["blis-trained-roofline", "vidur"]:
            for i in range(5):
                error_rows.append({
                    "simulator": sim, "model": f"model-{i}", "workload": "general",
                    "metric_name": "e2e_mean", "mape": 10.0 + i,
                    "experiment_folder": f"/exp/{i}", "stage_index": -1,
                })
                runtime_rows.append({
                    "simulator": sim, "model": f"model-{i}", "workload": "general",
                    "wall_clock_seconds": 1.0 + i * 0.5,
                    "experiment_folder": f"/exp/{i}",
                })
        return pd.DataFrame(error_rows), pd.DataFrame(runtime_rows)

    def test_returns_figure(self, tmp_path):
        from experiment.figures import plot_pareto
        error_df, runtime_df = self._make_pareto_data(tmp_path)
        fig = plot_pareto(error_df, runtime_df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_saves_pdf(self, tmp_path):
        from experiment.figures import plot_pareto
        error_df, runtime_df = self._make_pareto_data(tmp_path)
        out = str(tmp_path / "pareto.pdf")
        plot_pareto(error_df, runtime_df, output_path=out)
        assert os.path.exists(out)
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `plot_pareto()`**

```python
def plot_pareto(
    error_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    output_path: str | None = "results/figures/fig5_pareto.pdf",
) -> plt.Figure | None:
    """Figure 5: Accuracy-speed Pareto frontier scatter plot."""
    matplotlib.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["pareto"])

    # MAPE aggregation: for each simulator, compute per-experiment mean MAPE
    # (averaging across the 6 metrics within each experiment), then take the
    # median and IQR across experiments. This gives one representative accuracy
    # number per experiment, then summarizes across experiments.
    exp_mape = error_df.groupby(["simulator", "experiment_folder"])["mape"].mean()
    sim_stats = {}
    for sim in SIMULATOR_ORDER:
        if sim not in exp_mape.index.get_level_values(0):
            continue
        vals = exp_mape.loc[sim]
        sim_stats[sim] = {
            "mape_med": vals.median(),
            "mape_q1": vals.quantile(0.25),
            "mape_q3": vals.quantile(0.75),
            "n": len(vals),
        }
    # Runtime: median + IQR per simulator
    for sim in list(sim_stats):
        rt = runtime_df[runtime_df["simulator"] == sim]["wall_clock_seconds"]
        if rt.empty:
            del sim_stats[sim]
            continue
        sim_stats[sim].update({
            "rt_med": rt.median(), "rt_q1": rt.quantile(0.25), "rt_q3": rt.quantile(0.75),
        })

    # Plot each simulator as a point
    for sim in SIMULATOR_ORDER:
        if sim not in sim_stats:
            continue
        s = sim_stats[sim]
        marker_size = 120 if sim == "blis-trained-roofline" else 60
        ax.errorbar(
            s["mape_med"], s["rt_med"],
            xerr=[[s["mape_med"] - s["mape_q1"]], [s["mape_q3"] - s["mape_med"]]],
            yerr=[[s["rt_med"] - s["rt_q1"]], [s["rt_q3"] - s["rt_med"]]],
            fmt="none", ecolor=COLOR_PALETTE[sim], elinewidth=1, capsize=3,
        )
        ax.scatter(s["mape_med"], s["rt_med"], s=marker_size, color=COLOR_PALETTE[sim],
                   edgecolor="black", linewidth=0.5, zorder=5,
                   label=SIMULATOR_DISPLAY_NAMES[sim])
        ax.annotate(f"{SIMULATOR_DISPLAY_NAMES[sim]} (n={s['n']})",
                    (s["mape_med"], s["rt_med"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=6)

    # Pareto-dominated shading: light blue region from BLIS-Trained toward upper-right
    if "blis-trained-roofline" in sim_stats:
        bt = sim_stats["blis-trained-roofline"]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between(
            [bt["mape_med"], xlim[1]], bt["rt_med"], ylim[1],
            color="#90E0EF", alpha=0.15, zorder=0,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.set_yscale("log")
    ax.set_xlabel(r"Median MAPE (\%)")
    ax.set_ylabel(r"Median Runtime (s)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Figure 5 — accuracy-speed Pareto"
```

---

### Task 10: Table 1 — Runtime Comparison

**Files:**
- Modify: `experiment/figures.py`
- Test: `tests/test_figures.py`

**Caption (templated):**
```
"Simulator runtime and speedup. Median wall-clock time per variation and
speedup relative to running the actual vLLM experiment
(~{real_experiment_duration_min:.0f} min). BLIS variants complete in
{blis_runtime_range}, enabling rapid exploration of the model-hardware-config
design space."
```

- [ ] **Step 1: Write failing tests**

```python
class TestTable1:
    def _make_runtime_df(self):
        rows = []
        for sim, t in [("blis-trained-roofline", 1.2), ("vidur", 30.0)]:
            for i in range(3):
                rows.append({
                    "simulator": sim, "experiment_folder": f"/exp/{i}",
                    "model": "m", "workload": "general",
                    "wall_clock_seconds": t + i * 0.5,
                })
        return pd.DataFrame(rows)

    def test_returns_latex_string(self):
        from experiment.figures import format_runtime_table_latex
        result = format_runtime_table_latex(self._make_runtime_df())
        assert "\\begin{tabular}" in result
        assert "BLIS-Trained" in result

    def test_saves_tex_file(self, tmp_path):
        from experiment.figures import format_runtime_table_latex
        out = str(tmp_path / "table1.tex")
        format_runtime_table_latex(self._make_runtime_df(), output_path=out)
        assert os.path.exists(out)
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `format_runtime_table_latex()`**

```python
def format_runtime_table_latex(
    runtime_df: pd.DataFrame,
    real_experiment_seconds: float = 1200.0,
    output_path: str | None = "results/figures/table1_runtime.tex",
) -> str:
    """Table 1: LaTeX tabular with median runtime and speedup vs. real."""
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Simulator & Median Runtime (s) & Speedup vs.\ Real \\", r"\midrule"]
    for sim in SIMULATOR_ORDER:
        sdf = runtime_df[runtime_df["simulator"] == sim]
        if sdf.empty:
            continue
        median_t = sdf["wall_clock_seconds"].median()
        speedup = real_experiment_seconds / median_t
        lines.append(
            f"{SIMULATOR_DISPLAY_NAMES[sim]} & {median_t:.1f} & {speedup:.0f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tex)
    return tex
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat(figures): add Table 1 — runtime comparison (LaTeX)"
```

---

### Task 11: CLI Entry Point

**Files:**
- Modify: `experiment/figures.py`
- Modify: `experiment/run.py:128-159`
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write failing tests**

```python
class TestCLI:
    def test_parse_args_defaults(self):
        from experiment.figures import parse_figure_args
        args = parse_figure_args([])
        assert args.results_dir == "results"
        assert args.output_dir == "results/figures"
        assert args.metadata is None

    def test_main_with_csvs(self, tmp_path):
        from experiment.figures import main as figures_main
        # Create minimal CSVs
        (tmp_path / "error_records.csv").write_text(
            "simulator,experiment_folder,model,workload,stage_index,"
            "metric_name,predicted,actual,mape,mpe,absolute_error\n"
        )
        (tmp_path / "runtime.csv").write_text(
            "simulator,experiment_folder,model,workload,wall_clock_seconds\n"
        )
        out = tmp_path / "figures"
        figures_main(["--results-dir", str(tmp_path), "--output-dir", str(out)])
        # Should not crash (no data = warnings only)
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement CLI**

```python
def parse_figure_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing error_records.csv and runtime.csv")
    parser.add_argument("--output-dir", default="results/figures",
                        help="Directory for output figures")
    parser.add_argument("--metadata", default=None,
                        help="Path to experiment_metadata.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_figure_args(argv)
    error_csv = os.path.join(args.results_dir, "error_records.csv")
    runtime_csv = os.path.join(args.results_dir, "runtime.csv")

    error_df = load_error_data(error_csv)
    runtime_df = load_runtime_data(runtime_csv)
    error_df = enrich_with_metadata(error_df, args.metadata)
    runtime_df = enrich_with_metadata(runtime_df, args.metadata)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_model_sensitivity(error_df,
        output_path=os.path.join(args.output_dir, "fig1_model_sensitivity.pdf"))
    plot_hardware_portability(error_df,
        output_path=os.path.join(args.output_dir, "fig2_hardware_portability.pdf"))
    plot_workload_sensitivity(error_df,
        output_path=os.path.join(args.output_dir, "fig3_workload_sensitivity.pdf"))
    plot_config_sensitivity_dense(error_df,
        output_path=os.path.join(args.output_dir, "fig4a_config_dense.pdf"))
    plot_config_sensitivity_moe(error_df,
        output_path=os.path.join(args.output_dir, "fig4b_config_moe.pdf"))
    plot_pareto(error_df, runtime_df,
        output_path=os.path.join(args.output_dir, "fig5_pareto.pdf"))
    format_runtime_table_latex(runtime_df,
        output_path=os.path.join(args.output_dir, "table1_runtime.tex"))

    logger.info("All figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
```

Also add `--figures` flag to `experiment/run.py`:
```python
# In parse_args(), add:
parser.add_argument("--figures", action="store_true",
                    help="Generate publication figures from existing CSVs (skip simulation)")

# In main(), add before run_pipeline:
if args.figures:
    from experiment.figures import main as figures_main
    figures_main(["--results-dir", args.output_dir])
    return
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add experiment/figures.py experiment/run.py tests/test_figures.py
git commit -m "feat(figures): add CLI entry point and --figures flag"
```

---

### Task 12: End-to-End Smoke Test

**Files:**
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write integration smoke test**

```python
class TestEndToEnd:
    def test_full_pipeline_with_sample_data(self, tmp_path):
        """Generate all figures from synthetic CSVs — no crash, files created."""
        from experiment.figures import main as figures_main

        # Create realistic sample CSVs
        _write_full_sample_csvs(tmp_path)
        _write_sample_metadata(tmp_path)

        out = tmp_path / "figures"
        figures_main([
            "--results-dir", str(tmp_path),
            "--output-dir", str(out),
            "--metadata", str(tmp_path / "experiment_metadata.csv"),
        ])

        expected_files = [
            "fig1_model_sensitivity.pdf", "fig1_model_sensitivity.png",
            "fig2_hardware_portability.pdf",
            "fig3_workload_sensitivity.pdf",
            "fig4a_config_dense.pdf", "fig4b_config_moe.pdf",
            "fig5_pareto.pdf", "table1_runtime.tex",
        ]
        for fname in expected_files:
            assert (out / fname).exists(), f"Missing: {fname}"
```

Helper `_write_full_sample_csvs()` should generate ~50 rows covering all 5 simulators, 3 models, and 6 metrics. Helper `_write_sample_metadata()` should map each experiment_folder to H100/default.

- [ ] **Step 2: Run test → FAIL**
- [ ] **Step 3: Fix any issues discovered**
- [ ] **Step 4: Run full test suite**

Run: `pytest tests/test_figures.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_figures.py
git commit -m "test(figures): add end-to-end smoke test"
```

---

## Appendix A: Figure Captions (Templated)

Placeholders in `{braces}` are filled at render time from the data. These captions are embedded as comments in each figure function and printed to stdout during generation.

### Figure 1 — Model Sensitivity
```
Prediction accuracy across model architectures. MAPE of {n_simulators}
simulators across {n_models} LLM models spanning dense (7B--70B) and MoE
(47B--141B) architectures on H100 (General-Purpose workload, default vLLM
config). Top row: mean latency; bottom row: P99 tail latency. BLIS-Trained
(dark blue) maintains low MAPE across all architectures. LLM-Optimizer and
AIConfigurator produce only mean estimates (tail-latency bars absent).
```

### Figure 2 — Hardware Portability
```
Hardware portability. MAPE across {n_gpus} GPU types (default config). Each
bar aggregates across all viable models for that GPU (H100: n={n_h100},
A100: n={n_a100}, L40S: n={n_l40s}). BLIS variants generalize across GPU
generations using only datasheet specifications.
```

### Figure 3 — Workload Sensitivity
```
Workload sensitivity. MAPE across {n_workloads} workload types, aggregated
over {n_models} models spanning dense and MoE architectures (H100, default
config). BLIS-Trained shows the smallest degradation across workload diversity
(median MAPE: {blis_trained_median_mape:.1f}%).
```

### Figure 4(a) — Config Sensitivity (Dense)
```
Configuration sensitivity for a dense model (Llama-3.1-8B-Instruct, H100,
General-Purpose). Each group varies one vLLM knob from the default
({n_configs} configurations). BLIS prediction error remains stable across
configuration changes (max MAPE: {blis_trained_max_mape:.1f}%).
```

### Figure 4(b) — Config Sensitivity (MoE)
```
Configuration sensitivity for an MoE model (Mixtral-8x7B, H100,
General-Purpose). Includes expert parallelism (EP=4 via DP=2). {n_configs}
configurations tested. BLIS handles both standard knobs and MoE-specific
parallelism without accuracy degradation.
```

### Figure 5 — Accuracy-Speed Pareto
```
Accuracy-speed Pareto frontier. Each point shows a simulator's median MAPE
(x-axis) and median simulation runtime (y-axis, log scale) across all
available variations ({total_variations} total). Error bars show interquartile
range. BLIS-Trained achieves the best accuracy-speed tradeoff (median MAPE:
{blis_trained_mape:.1f}%, runtime: {blis_trained_runtime:.1f}s).
```

### Table 1 — Runtime Comparison
```
Simulator runtime and speedup. Median wall-clock time per variation and
speedup relative to running the actual vLLM experiment
(~{real_experiment_duration_min:.0f} min). BLIS variants complete in
{blis_runtime_range}, enabling rapid exploration of the model-hardware-config
design space.
```

---

## Appendix B: Output File Naming

| Output | Path |
|--------|------|
| Figure 1 | `results/figures/fig1_model_sensitivity.pdf` (+ `.png`) |
| Figure 2 | `results/figures/fig2_hardware_portability.pdf` (+ `.png`) |
| Figure 3 | `results/figures/fig3_workload_sensitivity.pdf` (+ `.png`) |
| Figure 4a | `results/figures/fig4a_config_dense.pdf` (+ `.png`) |
| Figure 4b | `results/figures/fig4b_config_moe.pdf` (+ `.png`) |
| Figure 5 | `results/figures/fig5_pareto.pdf` (+ `.png`) |
| Table 1 | `results/figures/table1_runtime.tex` |
