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
    max_num_batched_tokens=2048,
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
        "cpu_offload": False,
        "gpu_mem_util": 0.90,
        "max_num_batched_tokens": max_num_batched_tokens,
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
        "experiment_folder,hardware,tp,dp,cpu_offload,gpu_mem_util,config_tag\n"
        "/exp/1,H100,1,1,False,0.90,default\n"
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
            SIMULATOR_DISPLAY_NAMES, HATCH_PATTERNS, MARKER_STYLES,
        )
        for sim in SIMULATOR_ORDER:
            assert sim in COLOR_PALETTE
            assert sim in SIMULATOR_DISPLAY_NAMES
            assert sim in HATCH_PATTERNS
            assert sim in MARKER_STYLES

    def test_model_order_excludes_llama2_7b(self):
        from experiment.figures import MODEL_ORDER
        assert "meta-llama/Llama-2-7b-hf" not in MODEL_ORDER
        assert len(MODEL_ORDER) == 7

    def test_metrics_grid_is_1x3(self):
        from experiment.figures import METRICS_GRID
        assert len(METRICS_GRID) == 1
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
    def _make_config_sweep_df(self, model, param_col, values):
        """Create a DataFrame where *param_col* takes multiple *values*."""
        rows = []
        for val in values:
            for sim in ["blis-trained-roofline", "vidur"]:
                for metric in _METRICS:
                    row = _make_error_row(
                        simulator=sim, model=model,
                        metric_name=metric, mape=7.0,
                        experiment_folder=f"/exp/{model}/{param_col}_{val}",
                    )
                    row[param_col] = val
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_fig4a_returns_figure(self):
        from experiment.figures import plot_config_sensitivity_dense, FIG4A_MODEL
        df = self._make_config_sweep_df(FIG4A_MODEL, "tp", [1, 2, 4])
        fig = plot_config_sensitivity_dense(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_fig4b_returns_figure(self):
        from experiment.figures import plot_config_sensitivity_moe, FIG4B_MODEL
        df = self._make_config_sweep_df(FIG4B_MODEL, "tp", [2, 4, 8])
        fig = plot_config_sensitivity_moe(df, output_path=None)
        assert fig is not None
        plt.close(fig)

    def test_multiple_varying_params(self):
        """Figure shows one subplot per varying config param."""
        from experiment.figures import plot_config_sensitivity_dense, FIG4A_MODEL
        rows = []
        for tp in [1, 2]:
            for mbt in [1024, 2048, 4096]:
                for sim in ["blis-trained-roofline", "vidur"]:
                    for metric in _METRICS:
                        row = _make_error_row(
                            simulator=sim, model=FIG4A_MODEL,
                            metric_name=metric, mape=7.0,
                            tp=tp, max_num_batched_tokens=mbt,
                        )
                        rows.append(row)
        df = pd.DataFrame(rows)
        fig = plot_config_sensitivity_dense(df, output_path=None)
        assert fig is not None
        # Should have 2 subplots: tp and max_num_batched_tokens
        assert len(fig.axes) == 2
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
        assert args.exclude_simulators == []

    def test_parse_args_custom(self):
        from experiment.figures import parse_figure_args
        args = parse_figure_args([
            "--results-dir", "/data", "--output-dir", "/out", "--metadata", "/m.csv",
        ])
        assert args.results_dir == "/data"
        assert args.output_dir == "/out"
        assert args.metadata == "/m.csv"

    def test_parse_args_exclude_simulators(self):
        from experiment.figures import parse_figure_args
        args = parse_figure_args(["--exclude-simulators", "vidur", "blis-roofline"])
        assert args.exclude_simulators == ["vidur", "blis-roofline"]


# ---------------------------------------------------------------------------
# Tests: End-to-End Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _write_csvs(self, tmp_path):
        """Write realistic sample CSVs for all figures."""
        from experiment.figures import (
            MODEL_ORDER, SIMULATOR_ORDER, FIGURE3_MODELS,
            FIG4A_MODEL, FIG4B_MODEL,
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

        # Fig 4a data: config sweeps for dense model (vary tp and mbt)
        for tp_val in [1, 2, 4]:
            for mbt_val in [1024, 2048, 4096]:
                for sim in SIMULATOR_ORDER[:3]:
                    exp_folder = f"/exp/{exp_id}"
                    for metric in _METRICS:
                        row = _make_error_row(
                            simulator=sim, model=FIG4A_MODEL,
                            metric_name=metric, mape=6.0,
                            experiment_folder=exp_folder,
                            tp=tp_val, max_num_batched_tokens=mbt_val,
                        )
                        error_rows.append(row)
                    runtime_rows.append(_make_runtime_row(
                        simulator=sim, experiment_folder=exp_folder,
                        wall_clock_seconds=1.3,
                    ))
                exp_id += 1

        # Fig 4b data: config sweeps for MoE model (vary tp and dp)
        for tp_val in [2, 4, 8]:
            for dp_val in [1, 2]:
                for sim in SIMULATOR_ORDER[:2]:
                    exp_folder = f"/exp/{exp_id}"
                    for metric in _METRICS:
                        row = _make_error_row(
                            simulator=sim, model=FIG4B_MODEL,
                            metric_name=metric, mape=7.0,
                            experiment_folder=exp_folder,
                            tp=tp_val, dp=dp_val,
                        )
                        error_rows.append(row)
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

    def test_full_pipeline(self, tmp_path):
        """Generate all figures from synthetic CSVs — no crash, files created."""
        from experiment.figures import main as figures_main

        self._write_csvs(tmp_path)
        out = tmp_path / "figures"

        # No --metadata needed: the new pipeline writes metadata inline in CSVs
        figures_main([
            "--results-dir", str(tmp_path),
            "--output-dir", str(out),
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


# ---------------------------------------------------------------------------
# Tests: Config Tag Derivation
# ---------------------------------------------------------------------------


class TestDeriveConfigTag:
    """Tests for _derive_config_tag and _add_config_tags in figures.py."""

    def _make_row(self, **overrides):
        base = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "tp": 1, "dp": 1, "cpu_offload": False,
            "gpu_mem_util": 0.9, "max_num_batched_tokens": 2048,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_default(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row()) == "default"

    def test_mbt_1024(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row(max_num_batched_tokens=1024)) == "mbt=1024"

    def test_cpu_offload(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row(cpu_offload=True)) == "cpu-offload"

    def test_gpu_mem(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row(gpu_mem_util=0.95)) == "gpu-0.95"

    def test_tp_variation(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row(tp=2)) == "tp=2"

    def test_dp_for_dense(self):
        from experiment.figures import _derive_config_tag
        assert _derive_config_tag(self._make_row(dp=2)) == "dp=2"

    def test_ep_for_moe(self):
        from experiment.figures import _derive_config_tag
        row = self._make_row(
            model="mistralai/Mixtral-8x7B-v0.1", tp=2, dp=2,
        )
        assert _derive_config_tag(row) == "ep=4"

    def test_mbt_priority_over_cpu_offload(self):
        from experiment.figures import _derive_config_tag
        row = self._make_row(max_num_batched_tokens=1024, cpu_offload=True)
        assert _derive_config_tag(row) == "mbt=1024"

    def test_add_config_tags_skips_existing(self):
        from experiment.figures import _add_config_tags
        df = pd.DataFrame([{"config_tag": "custom", "max_num_batched_tokens": 1024}])
        result = _add_config_tags(df)
        assert result["config_tag"].iloc[0] == "custom"

    def test_add_config_tags_derives_when_missing(self):
        from experiment.figures import _add_config_tags
        df = pd.DataFrame([{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "tp": 1, "dp": 1, "cpu_offload": False,
            "gpu_mem_util": 0.9, "max_num_batched_tokens": 1024,
        }])
        result = _add_config_tags(df)
        assert result["config_tag"].iloc[0] == "mbt=1024"

    def test_add_config_tags_defaults_without_mbt(self):
        from experiment.figures import _add_config_tags
        df = pd.DataFrame([{"simulator": "vidur", "mape": 10.0}])
        result = _add_config_tags(df)
        assert result["config_tag"].iloc[0] == "default"
