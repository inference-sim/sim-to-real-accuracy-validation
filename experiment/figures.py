"""Publication figures for sim-to-real accuracy validation.

Generates 5 figures + 1 LaTeX table from error_records.csv and runtime.csv.
Spec: docs/superpowers/specs/2026-03-16-publication-figures-design.md
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
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
    [("e2e_mean", "E2E Mean"), ("ttft_mean", "TTFT Mean"), ("itl_mean", "ITL Mean")],
]

MAPE_THRESHOLD = 20.0
FIGURE_SIZES = {"bar_grid": (7.0, 2.2), "pareto": (5.5, 4.5)}

RC_PARAMS = {
    "font.family": "serif",
    "text.usetex": True,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}

HARDWARE_ORDER = ["H100", "A100-80GB", "L40S"]
WORKLOAD_ORDER = ["general", "codegen", "roleplay", "reasoning"]

FIGURE3_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-14B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

FIG4A_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FIG4A_CONFIG_ORDER = [
    "default", "mbt=1024", "mbt=8192", "cpu-offload", "gpu-0.95", "tp=2",
]
FIG4A_CONFIG_LABELS = {
    "default": "Default",
    "mbt=1024": "mbt=1024",
    "mbt=8192": "mbt=8192",
    "cpu-offload": "CPU-Offload",
    "gpu-0.95": "GPU-0.95",
    "tp=2": "TP=2",
}

FIG4B_MODEL = "mistralai/Mixtral-8x7B-v0.1"
FIG4B_CONFIG_ORDER = [
    "default", "mbt=1024", "mbt=8192", "cpu-offload", "gpu-0.95", "tp=4", "ep=4",
]
FIG4B_CONFIG_LABELS = {
    "default": "Default",
    "mbt=1024": "mbt=1024",
    "mbt=8192": "mbt=8192",
    "cpu-offload": "CPU-Offload",
    "gpu-0.95": "GPU-0.95",
    "tp=4": "TP=4",
    "ep=4": "EP=4 (DP=2)",
}

_METADATA_COLUMNS = [
    "hardware", "tp", "dp", "cpu_offloading", "gpu_memory_utilization", "config_tag",
]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


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
    df = df.copy()
    for col in _METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _has_metadata(df: pd.DataFrame) -> bool:
    """Check if DataFrame has non-empty metadata columns."""
    return (
        "hardware" in df.columns
        and df["hardware"].notna().any()
        and (df["hardware"] != "").any()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_rc_params() -> None:
    """Apply publication-quality RC params. Falls back to non-LaTeX if unavailable."""
    params = dict(RC_PARAMS)
    if not shutil.which("latex"):
        params["text.usetex"] = False
    matplotlib.rcParams.update(params)


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure as PDF + PNG at 300 DPI."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = output_path.replace(".pdf", ".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", output_path)


def _aggregate_for_bar_grid(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    group_labels: dict[str, str] | None = None,
) -> tuple[dict, dict, dict, list[str]]:
    """Build data/error_data/dot_data dicts from a DataFrame for _bar_chart_grid.

    Returns (data, error_data, dot_data, present_groups).
    """
    data: dict[str, dict[str, dict[str, float]]] = {}
    error_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    dot_data: dict[str, dict[str, dict[str, list[float]]]] = {}

    for group in group_order:
        gdf = df[df[group_col] == group]
        if gdf.empty:
            continue
        data[group] = {}
        error_data[group] = {}
        dot_data[group] = {}

        for sim in SIMULATOR_ORDER:
            sdf = gdf[gdf["simulator"] == sim]
            if sdf.empty:
                continue
            data[group][sim] = {}
            error_data[group][sim] = {}
            dot_data[group][sim] = {}

            for row in METRICS_GRID:
                for metric_key, _ in row:
                    vals = sdf[sdf["metric_name"] == metric_key]["mape"]
                    if vals.empty:
                        continue
                    med = vals.median()
                    data[group][sim][metric_key] = med
                    if len(vals) > 3:
                        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                        error_data[group][sim][metric_key] = (med - q1, q3 - med)
                    elif len(vals) > 1:
                        dot_data[group][sim][metric_key] = vals.tolist()

    present_groups = [g for g in group_order if g in data]
    return data, error_data, dot_data, present_groups


def _bar_chart_grid(
    data: dict[str, dict[str, dict[str, float | None]]],
    group_order: list[str],
    title: str,
    output_path: str | None,
    group_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    error_data: dict | None = None,
    dot_data: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Render a grouped-bar grid (shared layout for Figures 1-4)."""
    _apply_rc_params()
    n_rows = len(METRICS_GRID)
    figsize = figsize or FIGURE_SIZES["bar_grid"]
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    # Determine which simulators have data in any group
    all_sims: set[str] = set()
    for g_data in data.values():
        all_sims.update(g_data.keys())
    simulators = [s for s in SIMULATOR_ORDER if s in all_sims]
    n_sims = len(simulators)
    n_groups = len(group_order)

    if n_sims == 0 or n_groups == 0:
        if output_path:
            _save_figure(fig, output_path)
        return fig, axes

    bar_width = 0.8 / n_sims
    x = np.arange(n_groups)

    for row_idx, row_metrics in enumerate(METRICS_GRID):
        for col_idx, (metric_key, metric_label) in enumerate(row_metrics):
            ax = axes[row_idx, col_idx]

            for sim_idx, sim in enumerate(simulators):
                offset = (sim_idx - n_sims / 2 + 0.5) * bar_width
                positions = []
                heights = []
                err_lo = []
                err_hi = []

                for g_idx, g in enumerate(group_order):
                    sim_data = data.get(g, {}).get(sim, {})
                    val = sim_data.get(metric_key)

                    if val is not None:
                        positions.append(x[g_idx] + offset)
                        heights.append(val)
                        # IQR error bars
                        e = (error_data or {}).get(g, {}).get(sim, {}).get(metric_key)
                        if e:
                            err_lo.append(e[0])
                            err_hi.append(e[1])
                        else:
                            err_lo.append(0)
                            err_hi.append(0)
                    elif sim_data:
                        # Simulator ran for this group but metric missing → N/A
                        ax.text(
                            x[g_idx] + offset, 0.3, "N/A",
                            ha="center", va="bottom", fontsize=4,
                            color="gray", fontstyle="italic",
                        )

                if not positions:
                    continue

                yerr = None
                if any(lo > 0 or hi > 0 for lo, hi in zip(err_lo, err_hi)):
                    yerr = np.array([err_lo, err_hi])

                label = (
                    SIMULATOR_DISPLAY_NAMES[sim]
                    if row_idx == 0 and col_idx == 0
                    else ""
                )
                ax.bar(
                    positions, heights, bar_width,
                    color=COLOR_PALETTE[sim],
                    hatch=HATCH_PATTERNS[sim],
                    edgecolor="black", linewidth=0.3,
                    label=label,
                    yerr=yerr, capsize=2, error_kw={"linewidth": 0.5},
                )

                # Overlay dots for small-n aggregations
                if dot_data:
                    for g_idx, g in enumerate(group_order):
                        dots = (dot_data or {}).get(g, {}).get(sim, {}).get(
                            metric_key, [],
                        )
                        if dots:
                            dot_x = x[g_idx] + offset
                            ax.scatter(
                                [dot_x] * len(dots), dots,
                                color=COLOR_PALETTE[sim], alpha=0.5,
                                s=8, zorder=5, edgecolors="none",
                            )

            # MAPE threshold line
            ax.axhline(
                MAPE_THRESHOLD, color="gray", linestyle="--",
                linewidth=0.5, zorder=0,
            )

            ax.set_title(metric_label)
            ax.set_xticks(x)
            tick_labels = [
                (group_labels or {}).get(g, g) for g in group_order
            ]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            if col_idx == 0:
                pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
                ax.set_ylabel(f"MAPE ({pct})")

    # Shared legend below bottom row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h_l = [(h, l) for h, l in zip(handles, labels) if l]
    if h_l:
        handles, labels = zip(*h_l)
        fig.legend(
            handles, labels, loc="lower center",
            ncol=len(handles), bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    if output_path:
        _save_figure(fig, output_path)

    return fig, axes


# ---------------------------------------------------------------------------
# Figure Functions
# ---------------------------------------------------------------------------


def plot_model_sensitivity(
    df: pd.DataFrame,
    output_path: str | None = None,
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
    data: dict = {}
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

    present_models = [m for m in MODEL_ORDER if m in data]
    if not present_models:
        warnings.warn("Figure 1: no models with data")
        return None

    fig, _ = _bar_chart_grid(
        data=data,
        group_order=present_models,
        group_labels=MODEL_SHORT_LABELS,
        title="Model Sensitivity",
        output_path=output_path,
    )
    return fig


def plot_hardware_portability(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 2: MAPE across 3 GPU types, aggregated over models."""
    if not _has_metadata(df):
        warnings.warn("Figure 2: skipped (no hardware metadata)")
        return None

    df = df[df["config_tag"] == "default"]
    df = df[df["workload"].isin(("general", "general-lite"))]
    df = df[df["dp"].replace("", 1).fillna(1).astype(float) <= 1]

    if df.empty:
        warnings.warn("Figure 2: no data after filtering")
        return None

    data, error_data, dot_data, present_hw = _aggregate_for_bar_grid(
        df, group_col="hardware", group_order=HARDWARE_ORDER,
    )

    if not present_hw:
        warnings.warn("Figure 2: no hardware groups with data")
        return None

    fig, _ = _bar_chart_grid(
        data=data, group_order=present_hw,
        title="Hardware Portability", output_path=output_path,
        error_data=error_data, dot_data=dot_data,
    )
    return fig


def plot_workload_sensitivity(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 3: MAPE across 4 workload types, aggregated over 4 models."""
    if _has_metadata(df):
        df = df[(df["hardware"] == "H100") & (df["config_tag"] == "default")]
        df = df[df["dp"].replace("", 1).fillna(1).astype(float) <= 1]
    df = df[df["model"].isin(FIGURE3_MODELS)]

    if df.empty:
        warnings.warn("Figure 3: no data after filtering")
        return None

    data, error_data, dot_data, present_wl = _aggregate_for_bar_grid(
        df, group_col="workload", group_order=WORKLOAD_ORDER,
    )

    if not present_wl:
        warnings.warn("Figure 3: no workloads with data")
        return None

    fig, _ = _bar_chart_grid(
        data=data, group_order=present_wl,
        group_labels=WORKLOAD_DISPLAY_NAMES,
        title="Workload Sensitivity", output_path=output_path,
        error_data=error_data, dot_data=dot_data,
    )
    return fig


def _plot_config_sensitivity(
    df: pd.DataFrame,
    model: str,
    config_order: list[str],
    config_labels: dict[str, str],
    title: str,
    output_path: str | None,
) -> plt.Figure | None:
    """Shared implementation for Figures 4a and 4b."""
    if not _has_metadata(df):
        warnings.warn(f"{title}: skipped (no config metadata)")
        return None

    df = df[
        (df["hardware"] == "H100")
        & (df["workload"] == "general")
        & (df["model"] == model)
    ]

    if df.empty:
        warnings.warn(f"{title}: no data after filtering")
        return None

    # Build data dict: {config_tag: {simulator: {metric: mape}}}
    data: dict = {}
    for tag in config_order:
        tdf = df[df["config_tag"] == tag]
        if tdf.empty:
            continue
        data[tag] = {}
        for sim in SIMULATOR_ORDER:
            sdf = tdf[tdf["simulator"] == sim]
            if sdf.empty:
                continue
            data[tag][sim] = dict(zip(sdf["metric_name"], sdf["mape"]))

    present_configs = [c for c in config_order if c in data]
    if not present_configs:
        warnings.warn(f"{title}: no configs with data")
        return None

    fig, _ = _bar_chart_grid(
        data=data, group_order=present_configs,
        group_labels=config_labels,
        title=title, output_path=output_path,
    )
    return fig


def plot_config_sensitivity_dense(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 4a: Config sensitivity for Llama-3.1-8B (dense)."""
    return _plot_config_sensitivity(
        df, model=FIG4A_MODEL,
        config_order=FIG4A_CONFIG_ORDER, config_labels=FIG4A_CONFIG_LABELS,
        title="Config Sensitivity \u2014 Dense (Llama-3.1-8B)",
        output_path=output_path,
    )


def plot_config_sensitivity_moe(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 4b: Config sensitivity for Mixtral-8x7B (MoE)."""
    return _plot_config_sensitivity(
        df, model=FIG4B_MODEL,
        config_order=FIG4B_CONFIG_ORDER, config_labels=FIG4B_CONFIG_LABELS,
        title="Config Sensitivity \u2014 MoE (Mixtral-8x7B)",
        output_path=output_path,
    )


def plot_pareto(
    error_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 5: Accuracy-speed Pareto frontier scatter plot."""
    if error_df.empty or runtime_df.empty:
        warnings.warn("Figure 5: insufficient data")
        return None

    _apply_rc_params()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["pareto"])

    # Per-experiment median MAPE (robust to outlier metrics like Vidur's
    # 2.4M% TTFT on Llama-2-70b), then per-simulator median + IQR
    exp_mape = error_df.groupby(["simulator", "experiment_folder"])["mape"].median()

    sim_stats: dict = {}
    for sim in SIMULATOR_ORDER:
        if sim not in exp_mape.index.get_level_values(0):
            continue
        vals = exp_mape.loc[sim]
        rt = runtime_df[runtime_df["simulator"] == sim]["wall_clock_seconds"]
        if rt.empty:
            continue
        sim_stats[sim] = {
            "mape_med": vals.median(),
            "mape_q1": vals.quantile(0.25),
            "mape_q3": vals.quantile(0.75),
            "n": len(vals),
            "rt_med": rt.median(),
            "rt_q1": rt.quantile(0.25),
            "rt_q3": rt.quantile(0.75),
        }

    if not sim_stats:
        warnings.warn("Figure 5: no simulator data")
        plt.close(fig)
        return None

    # Per-simulator annotation offsets: hand-tuned to avoid overlap in the
    # typical cluster layout (BLIS/LLM-Opt/AIC cluster low-MAPE, Vidur far right).
    _annotation_offsets = {
        "blis-trained-roofline": (-14, -20),
        "blis-roofline": (14, 16),
        "vidur": (12, -18),
        "llm-optimizer-estimate": (14, -18),
        "aiconfigurator-estimate": (-14, 16),
    }

    for sim in SIMULATOR_ORDER:
        if sim not in sim_stats:
            continue
        s = sim_stats[sim]
        marker_size = 120 if sim == "blis-trained-roofline" else 60

        # Clamp error bars to avoid negative values on log scale
        yerr_lo = min(s["rt_med"] - s["rt_q1"], s["rt_med"] * 0.9)
        yerr_hi = s["rt_q3"] - s["rt_med"]

        ax.errorbar(
            s["mape_med"], s["rt_med"],
            xerr=[[s["mape_med"] - s["mape_q1"]], [s["mape_q3"] - s["mape_med"]]],
            yerr=[[max(yerr_lo, 0)], [yerr_hi]],
            fmt="none", ecolor=COLOR_PALETTE[sim], elinewidth=1, capsize=3,
        )
        ax.scatter(
            s["mape_med"], s["rt_med"], s=marker_size,
            color=COLOR_PALETTE[sim], edgecolor="black", linewidth=0.5,
            zorder=5, label=SIMULATOR_DISPLAY_NAMES[sim],
        )
        ox, oy = _annotation_offsets.get(sim, (10, 8))
        ax.annotate(
            f"{SIMULATOR_DISPLAY_NAMES[sim]} (n={s['n']})",
            (s["mape_med"], s["rt_med"]),
            textcoords="offset points", xytext=(ox, oy), fontsize=5.5,
            arrowprops={"arrowstyle": "-", "color": "gray", "linewidth": 0.4},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Tight limits for better spread
    all_mapes = [s["mape_med"] for s in sim_stats.values()]
    ax.set_xlim(min(all_mapes) * 0.5, max(all_mapes) * 3)
    all_rts = [s["rt_med"] for s in sim_stats.values()]
    ax.set_ylim(min(all_rts) * 0.4, max(all_rts) * 4)

    # Pareto-dominated shading (after limits are set)
    if "blis-trained-roofline" in sim_stats:
        bt = sim_stats["blis-trained-roofline"]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between(
            [bt["mape_med"], xlim[1]], bt["rt_med"], ylim[1],
            color="#90E0EF", alpha=0.15, zorder=0,
        )

    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    ax.set_xlabel(f"Median MAPE ({pct})")
    ax.set_ylabel("Median Runtime (s)")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    if output_path:
        _save_figure(fig, output_path)
    return fig


def format_runtime_table_latex(
    runtime_df: pd.DataFrame,
    output_path: str | None = None,
    real_experiment_seconds: float = 1200.0,
) -> str:
    """Table 1: LaTeX tabular with median runtime and speedup vs. real."""
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Simulator & Median Runtime (s) & Speedup vs.\ Real \\",
        r"\midrule",
    ]
    for sim in SIMULATOR_ORDER:
        sdf = runtime_df[runtime_df["simulator"] == sim]
        if sdf.empty:
            continue
        median_t = sdf["wall_clock_seconds"].median()
        speedup = real_experiment_seconds / median_t if median_t > 0 else float("inf")
        lines.append(
            f"{SIMULATOR_DISPLAY_NAMES[sim]} & {median_t:.1f} "
            f"& {speedup:.0f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tex)
        logger.info("Saved %s", output_path)
    return tex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_figure_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from experiment CSVs.",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing error_records.csv and runtime.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Path to experiment_metadata.csv for hardware/config enrichment",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_figure_args(argv)

    error_csv = os.path.join(args.results_dir, "error_records.csv")
    runtime_csv = os.path.join(args.results_dir, "runtime.csv")

    if not os.path.exists(error_csv):
        logger.error("error_records.csv not found in %s", args.results_dir)
        return
    if not os.path.exists(runtime_csv):
        logger.error("runtime.csv not found in %s", args.results_dir)
        return

    error_df = load_error_data(error_csv)
    runtime_df = load_runtime_data(runtime_csv)
    error_df = enrich_with_metadata(error_df, args.metadata)
    runtime_df = enrich_with_metadata(runtime_df, args.metadata)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    figures = [
        ("fig1_model_sensitivity.pdf",
         lambda: plot_model_sensitivity(error_df, os.path.join(out, "fig1_model_sensitivity.pdf"))),
        ("fig2_hardware_portability.pdf",
         lambda: plot_hardware_portability(error_df, os.path.join(out, "fig2_hardware_portability.pdf"))),
        ("fig3_workload_sensitivity.pdf",
         lambda: plot_workload_sensitivity(error_df, os.path.join(out, "fig3_workload_sensitivity.pdf"))),
        ("fig4a_config_dense.pdf",
         lambda: plot_config_sensitivity_dense(error_df, os.path.join(out, "fig4a_config_dense.pdf"))),
        ("fig4b_config_moe.pdf",
         lambda: plot_config_sensitivity_moe(error_df, os.path.join(out, "fig4b_config_moe.pdf"))),
        ("fig5_pareto.pdf",
         lambda: plot_pareto(error_df, runtime_df, os.path.join(out, "fig5_pareto.pdf"))),
    ]

    for name, generate in figures:
        try:
            fig = generate()
            if fig is not None:
                plt.close(fig)
                print(f"  OK: {name}")
            else:
                print(f"  SKIP: {name} (insufficient data)")
        except Exception as e:
            print(f"  FAIL: {name} ({e})")
            logger.exception("Failed to generate %s", name)

    # Table 1
    try:
        tex = format_runtime_table_latex(
            runtime_df, output_path=os.path.join(out, "table1_runtime.tex"),
        )
        print(f"  OK: table1_runtime.tex")
    except Exception as e:
        print(f"  FAIL: table1_runtime.tex ({e})")

    print(f"\nFigures saved to {out}")


if __name__ == "__main__":
    main()
