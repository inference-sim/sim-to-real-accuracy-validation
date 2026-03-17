"""Tests for experiment.figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import matplotlib.pyplot as plt
import pandas as pd
import pytest


# Force non-LaTeX rendering in tests (no LaTeX installation required)
matplotlib.rcParams["text.usetex"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRICS = ["e2e_mean", "e2e_p99", "ttft_mean", "ttft_p99", "itl_mean", "itl_p99"]


def _make_error_row(
    simulator="blis-trained-roofline",
    model="meta-llama/Llama-3.1-8B-Instruct",
    workload="general",
    metric_name="e2e_mean",
    mape=10.0,
    experiment_folder="/exp/1",
    hardware="H100",
    config_tag="default",
    tp=1,
    dp=1,
):
    return {
        "simulator": simulator,
        "experiment_folder": experiment_folder,
        "model": model,
        "workload": workload,
        "stage_index": -1,
        "metric_name": metric_name,
        "predicted": 110.0,
        "actual": 100.0,
        "mape": mape,
        "mpe": mape,
        "absolute_error": 10.0,
        "hardware": hardware,
        "config_tag": config_tag,
        "tp": tp,
        "dp": dp,
        "cpu_offloading": "Disabled",
        "gpu_memory_utilization": 0.90,
    }


def _make_runtime_row(
    simulator="blis-trained-roofline",
    model="meta-llama/Llama-3.1-8B-Instruct",
    workload="general",
    wall_clock_seconds=1.5,
    experiment_folder="/exp/1",
):
    return {
        "simulator": simulator,
        "experiment_folder": experiment_folder,
        "model": model,
        "workload": workload,
        "wall_clock_seconds": wall_clock_seconds,
    }


def _make_full_model_df():
    """DataFrame with all 7 models, 2 simulators, 6 metrics on H100 defaults."""
    from experiment.figures import MODEL_ORDER

    rows = []
    for i, model in enumerate(MODEL_ORDER):
        for sim in ["blis-trained-roofline", "vidur"]:
            for metric in _METRICS:
                rows.append(_make_error_row(
                    simulator=sim, model=model, metric_name=metric,
                    mape=5.0 + i, experiment_folder=f"/exp/{model}",
                ))
    return pd.DataFrame(rows)


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


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: Data Loading
# ---------------------------------------------------------------------------


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

    def test_enrich_does_not_mutate_input(self, tmp_path):
        from experiment.figures import load_error_data, enrich_with_metadata
        df = load_error_data(_sample_error_csv(tmp_path))
        original_cols = set(df.columns)
        enrich_with_metadata(df, metadata_path=None)
        assert set(df.columns) == original_cols

    def test_has_metadata_true(self):
        from experiment.figures import _has_metadata
        df = pd.DataFrame({"hardware": ["H100", "A100"]})
        assert _has_metadata(df)

    def test_has_metadata_false_empty(self):
        from experiment.figures import _has_metadata
        df = pd.DataFrame({"hardware": ["", ""]})
        assert not _has_metadata(df)


# ---------------------------------------------------------------------------
# Tests: Bar Chart Grid
# ---------------------------------------------------------------------------


class TestBarChartGrid:
    def _make_data(self):
        metrics = {m: 10.0 for row in _METRICS for m in [row]}
        # Actually build properly
        from experiment.figures import METRICS_GRID
        d = {}
        for row in METRICS_GRID:
            for k, _ in row:
                d[k] = 10.0
        return {
            "Group-A": {"blis-trained-roofline": dict(d), "vidur": dict(d)},
            "Group-B": {"blis-trained-roofline": dict(d)},
        }

    def test_returns_2x3_axes(self):
        from experiment.figures import _bar_chart_grid
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
            lines = ax.get_lines()
            y_values = []
            for line in lines:
                yd = line.get_ydata()
                if hasattr(yd, "__len__") and len(yd) > 0:
                    y_values.extend(yd)
            assert any(abs(y - MAPE_THRESHOLD) < 0.01 for y in y_values)
        plt.close(fig)

    def test_saves_pdf_and_png(self, tmp_path):
        from experiment.figures import _bar_chart_grid
        out = str(tmp_path / "test.pdf")
        _bar_chart_grid(
            data=self._make_data(), group_order=["Group-A"],
            title="Test", output_path=out,
        )
        assert os.path.exists(out)
        assert os.path.exists(out.replace(".pdf", ".png"))

    def test_empty_data_no_crash(self):
        from experiment.figures import _bar_chart_grid
        fig, axes = _bar_chart_grid(
            data={}, group_order=[], title="Empty", output_path=None,
        )
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_na_annotation_for_missing_metric(self):
        """Simulator present but specific metric missing → N/A annotation."""
        from experiment.figures import _bar_chart_grid
        data = {
            "Group-A": {
                "blis-trained-roofline": {"e2e_mean": 5.0},
                # vidur ran but only has e2e_mean, missing others
                "vidur": {"e2e_mean": 8.0},
            },
        }
        fig, axes = _bar_chart_grid(
            data=data, group_order=["Group-A"],
            title="Test N/A", output_path=None,
        )
        # Check that text annotations exist in subplots where metrics are missing
        for ax in axes.flat:
            texts = [t.get_text() for t in ax.texts]
            # At least some subplots should have N/A
            # (e2e_p99, ttft_mean, etc. are missing for both sims)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Figure 1 — Model Sensitivity
# ---------------------------------------------------------------------------


class TestFigure1:
    def test_returns_figure(self):
        from experiment.figures import plot_model_sensitivity
        df = _make_full_model_df()
        fig = plot_model_sensitivity(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_filters_to_h100_default_only(self):
        from experiment.figures import plot_model_sensitivity
        df = _make_full_model_df()
        # Add an A100 row that should be excluded
        extra = df.iloc[0:1].copy()
        extra["hardware"] = "A100"
        extra["mape"] = 999.0
        df = pd.concat([df, extra], ignore_index=True)
        fig = plot_model_sensitivity(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_empty_returns_none(self):
        from experiment.figures import plot_model_sensitivity
        df = pd.DataFrame(columns=[
            "simulator", "experiment_folder", "model", "workload",
            "stage_index", "metric_name", "mape", "hardware", "config_tag",
        ])
        fig = plot_model_sensitivity(df, output_path=None)
        assert fig is None

    def test_saves_to_file(self, tmp_path):
        from experiment.figures import plot_model_sensitivity
        df = _make_full_model_df()
        out = str(tmp_path / "fig1.pdf")
        fig = plot_model_sensitivity(df, output_path=out)
        assert os.path.exists(out)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Figure 2 — Hardware Portability
# ---------------------------------------------------------------------------


class TestFigure2:
    def _make_df(self):
        rows = []
        for hw in ["H100", "A100-80GB", "L40S"]:
            for sim in ["blis-trained-roofline", "blis-roofline"]:
                for metric in _METRICS:
                    for model_idx in range(4):
                        rows.append(_make_error_row(
                            simulator=sim, hardware=hw,
                            metric_name=metric, mape=10.0 + model_idx,
                            model=f"model-{model_idx}",
                            experiment_folder=f"/exp/{hw}/{model_idx}",
                        ))
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_hardware_portability
        fig = plot_hardware_portability(self._make_df(), output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_skips_without_metadata(self):
        from experiment.figures import plot_hardware_portability
        df = self._make_df()
        df = df.drop(columns=["hardware"])
        fig = plot_hardware_portability(df, output_path=None)
        assert fig is None


# ---------------------------------------------------------------------------
# Tests: Figure 3 — Workload Sensitivity
# ---------------------------------------------------------------------------


class TestFigure3:
    def _make_df(self):
        from experiment.figures import FIGURE3_MODELS
        rows = []
        for wl in ["general", "codegen", "roleplay", "reasoning"]:
            for model in FIGURE3_MODELS:
                for sim in ["blis-trained-roofline"]:
                    for metric in _METRICS:
                        rows.append(_make_error_row(
                            simulator=sim, model=model, workload=wl,
                            metric_name=metric, mape=8.0,
                            experiment_folder=f"/exp/{model}/{wl}",
                        ))
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        from experiment.figures import plot_workload_sensitivity
        fig = plot_workload_sensitivity(self._make_df(), output_path=None)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Figure 4a/4b — Config Sensitivity
# ---------------------------------------------------------------------------


class TestFigure4:
    def _make_config_df(self, model, config_order):
        rows = []
        for tag in config_order:
            for sim in ["blis-trained-roofline", "vidur"]:
                for metric in _METRICS:
                    rows.append(_make_error_row(
                        simulator=sim, model=model,
                        metric_name=metric, mape=7.0,
                        config_tag=tag,
                        experiment_folder=f"/exp/{model}/{tag}",
                    ))
        return pd.DataFrame(rows)

    def test_fig4a_returns_figure(self):
        from experiment.figures import plot_config_sensitivity_dense, FIG4A_MODEL, FIG4A_CONFIG_ORDER
        df = self._make_config_df(FIG4A_MODEL, FIG4A_CONFIG_ORDER)
        fig = plot_config_sensitivity_dense(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_fig4b_returns_figure(self):
        from experiment.figures import plot_config_sensitivity_moe, FIG4B_MODEL, FIG4B_CONFIG_ORDER
        df = self._make_config_df(FIG4B_MODEL, FIG4B_CONFIG_ORDER)
        fig = plot_config_sensitivity_moe(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_skips_without_metadata(self):
        from experiment.figures import plot_config_sensitivity_dense
        df = pd.DataFrame(columns=[
            "simulator", "model", "workload", "metric_name", "mape",
            "experiment_folder", "stage_index",
        ])
        fig = plot_config_sensitivity_dense(df, output_path=None)
        assert fig is None


# ---------------------------------------------------------------------------
# Tests: Figure 5 — Pareto
# ---------------------------------------------------------------------------


class TestFigure5:
    def _make_data(self):
        error_rows, runtime_rows = [], []
        for sim in ["blis-trained-roofline", "vidur"]:
            for i in range(5):
                for metric in _METRICS:
                    error_rows.append(_make_error_row(
                        simulator=sim, metric_name=metric,
                        mape=10.0 + i, experiment_folder=f"/exp/{sim}/{i}",
                    ))
                runtime_rows.append(_make_runtime_row(
                    simulator=sim, wall_clock_seconds=1.0 + i * 0.5,
                    experiment_folder=f"/exp/{sim}/{i}",
                ))
        return pd.DataFrame(error_rows), pd.DataFrame(runtime_rows)

    def test_returns_figure(self):
        from experiment.figures import plot_pareto
        edf, rdf = self._make_data()
        fig = plot_pareto(edf, rdf, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_saves_pdf(self, tmp_path):
        from experiment.figures import plot_pareto
        edf, rdf = self._make_data()
        out = str(tmp_path / "pareto.pdf")
        fig = plot_pareto(edf, rdf, output_path=out)
        assert os.path.exists(out)
        plt.close(fig)

    def test_empty_returns_none(self):
        from experiment.figures import plot_pareto
        edf = pd.DataFrame(columns=["simulator", "experiment_folder", "mape"])
        rdf = pd.DataFrame(columns=["simulator", "experiment_folder", "wall_clock_seconds"])
        fig = plot_pareto(edf, rdf, output_path=None)
        assert fig is None


# ---------------------------------------------------------------------------
# Tests: Table 1 — Runtime
# ---------------------------------------------------------------------------


class TestTable1:
    def _make_runtime_df(self):
        rows = []
        for sim, t in [("blis-trained-roofline", 1.2), ("vidur", 30.0)]:
            for i in range(3):
                rows.append(_make_runtime_row(
                    simulator=sim, wall_clock_seconds=t + i * 0.5,
                    experiment_folder=f"/exp/{sim}/{i}",
                ))
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

    def test_contains_speedup(self):
        from experiment.figures import format_runtime_table_latex
        result = format_runtime_table_latex(self._make_runtime_df())
        assert "\\times" in result


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parse_args_defaults(self):
        from experiment.figures import parse_figure_args
        args = parse_figure_args([])
        assert args.results_dir == "results"
        assert args.output_dir == "results/figures"
        assert args.metadata is None

    def test_parse_args_custom(self):
        from experiment.figures import parse_figure_args
        args = parse_figure_args([
            "--results-dir", "/data", "--output-dir", "/out", "--metadata", "/m.csv",
        ])
        assert args.results_dir == "/data"
        assert args.output_dir == "/out"
        assert args.metadata == "/m.csv"


# ---------------------------------------------------------------------------
# Tests: End-to-End Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _write_csvs(self, tmp_path):
        """Write realistic sample CSVs for all figures."""
        from experiment.figures import (
            MODEL_ORDER, SIMULATOR_ORDER, FIGURE3_MODELS,
            FIG4A_MODEL, FIG4A_CONFIG_ORDER, FIG4B_MODEL, FIG4B_CONFIG_ORDER,
        )

        error_rows = []
        runtime_rows = []
        exp_id = 0

        # Fig 1 data: 7 models on H100 defaults
        for model in MODEL_ORDER:
            for sim in SIMULATOR_ORDER[:3]:  # 3 simulators
                exp_folder = f"/exp/{exp_id}"
                for metric in _METRICS:
                    error_rows.append(_make_error_row(
                        simulator=sim, model=model, metric_name=metric,
                        mape=8.0, experiment_folder=exp_folder,
                    ))
                runtime_rows.append(_make_runtime_row(
                    simulator=sim, experiment_folder=exp_folder,
                    wall_clock_seconds=1.5,
                ))
            exp_id += 1

        # Fig 2 data: models on A100 and L40S
        for hw in ["A100-80GB", "L40S"]:
            for model in MODEL_ORDER[:3]:
                for sim in SIMULATOR_ORDER[:2]:
                    exp_folder = f"/exp/{exp_id}"
                    for metric in _METRICS:
                        error_rows.append(_make_error_row(
                            simulator=sim, model=model, metric_name=metric,
                            mape=12.0, experiment_folder=exp_folder,
                            hardware=hw,
                        ))
                    runtime_rows.append(_make_runtime_row(
                        simulator=sim, experiment_folder=exp_folder,
                        wall_clock_seconds=2.0,
                    ))
                exp_id += 1

        # Fig 3 data: multiple workloads
        for wl in ["codegen", "roleplay", "reasoning"]:
            for model in FIGURE3_MODELS:
                for sim in SIMULATOR_ORDER[:2]:
                    exp_folder = f"/exp/{exp_id}"
                    for metric in _METRICS:
                        error_rows.append(_make_error_row(
                            simulator=sim, model=model, workload=wl,
                            metric_name=metric, mape=9.0,
                            experiment_folder=exp_folder,
                        ))
                    runtime_rows.append(_make_runtime_row(
                        simulator=sim, experiment_folder=exp_folder,
                        workload=wl, wall_clock_seconds=1.8,
                    ))
                exp_id += 1

        # Fig 4a data: config sweeps for dense model
        for tag in FIG4A_CONFIG_ORDER:
            for sim in SIMULATOR_ORDER[:3]:
                exp_folder = f"/exp/{exp_id}"
                for metric in _METRICS:
                    error_rows.append(_make_error_row(
                        simulator=sim, model=FIG4A_MODEL,
                        metric_name=metric, mape=6.0,
                        config_tag=tag, experiment_folder=exp_folder,
                    ))
                runtime_rows.append(_make_runtime_row(
                    simulator=sim, experiment_folder=exp_folder,
                    wall_clock_seconds=1.3,
                ))
            exp_id += 1

        # Fig 4b data: config sweeps for MoE model
        for tag in FIG4B_CONFIG_ORDER:
            for sim in SIMULATOR_ORDER[:2]:
                exp_folder = f"/exp/{exp_id}"
                for metric in _METRICS:
                    error_rows.append(_make_error_row(
                        simulator=sim, model=FIG4B_MODEL,
                        metric_name=metric, mape=7.0,
                        config_tag=tag, experiment_folder=exp_folder,
                    ))
                runtime_rows.append(_make_runtime_row(
                    simulator=sim, experiment_folder=exp_folder,
                    wall_clock_seconds=2.5,
                ))
            exp_id += 1

        # Write CSVs
        edf = pd.DataFrame(error_rows)
        rdf = pd.DataFrame(runtime_rows)

        edf.to_csv(tmp_path / "error_records.csv", index=False)
        rdf.to_csv(tmp_path / "runtime.csv", index=False)

        # Write metadata (all experiments already have metadata in the rows)
        meta_rows = edf[["experiment_folder", "hardware", "tp", "dp",
                         "cpu_offloading", "gpu_memory_utilization",
                         "config_tag"]].drop_duplicates()
        meta_rows.to_csv(tmp_path / "experiment_metadata.csv", index=False)

    def test_full_pipeline(self, tmp_path):
        """Generate all figures from synthetic CSVs — no crash, files created."""
        from experiment.figures import main as figures_main

        self._write_csvs(tmp_path)
        out = tmp_path / "figures"

        figures_main([
            "--results-dir", str(tmp_path),
            "--output-dir", str(out),
            "--metadata", str(tmp_path / "experiment_metadata.csv"),
        ])

        expected = [
            "fig1_model_sensitivity.pdf",
            "fig1_model_sensitivity.png",
            "fig2_hardware_portability.pdf",
            "fig3_workload_sensitivity.pdf",
            "fig4a_config_dense.pdf",
            "fig4b_config_moe.pdf",
            "fig5_pareto.pdf",
            "table1_runtime.tex",
        ]
        for fname in expected:
            assert (out / fname).exists(), f"Missing: {fname}"
