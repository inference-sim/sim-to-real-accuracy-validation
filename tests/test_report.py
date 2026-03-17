"""Tests for experiment.report."""

from __future__ import annotations

import csv
import os

import pytest

from experiment.metrics import ErrorRecord, RuntimeRecord
from experiment.report import (
    format_aggregate_table,
    format_per_model_table,
    format_per_workload_table,
    format_runtime_table,
    format_signed_error_table,
    generate_report,
    save_csv,
    save_runtime_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    simulator="blis-roofline",
    model="llama-7b",
    workload="codegen",
    stage_index=0,
    metric_name="e2e_mean",
    mape=10.0,
    mpe=5.0,
) -> ErrorRecord:
    return ErrorRecord(
        simulator=simulator,
        experiment_folder="/tmp/exp",
        model=model,
        workload=workload,
        stage_index=stage_index,
        metric_name=metric_name,
        predicted=110.0,
        actual=100.0,
        mape=mape,
        mpe=mpe,
        absolute_error=10.0,
    )


def _make_records() -> list[ErrorRecord]:
    """Two simulators, two metrics — 4 records total."""
    return [
        _make_record(simulator="sim-a", metric_name="e2e_mean", mape=10.0, mpe=5.0),
        _make_record(simulator="sim-a", metric_name="ttft_mean", mape=8.0, mpe=-3.0),
        _make_record(simulator="sim-b", metric_name="e2e_mean", mape=15.0, mpe=10.0),
        _make_record(simulator="sim-b", metric_name="ttft_mean", mape=12.0, mpe=7.0),
    ]


# ---------------------------------------------------------------------------
# Tests: format_aggregate_table
# ---------------------------------------------------------------------------


class TestFormatAggregateTable:
    def test_contains_simulator_names(self):
        records = _make_records()
        table = format_aggregate_table(records)
        assert "sim-a" in table
        assert "sim-b" in table

    def test_contains_metric_headers(self):
        records = _make_records()
        table = format_aggregate_table(records)
        assert "e2e_mean" in table
        assert "ttft_mean" in table

    def test_values_are_averages(self):
        records = _make_records()
        table = format_aggregate_table(records)
        # sim-a e2e_mean: 10.0, sim-b e2e_mean: 15.0
        assert "10.00" in table
        assert "15.00" in table


class TestFormatPerModelTable:
    def test_contains_model_names(self):
        records = [
            _make_record(model="llama-7b", mape=10.0),
            _make_record(model="llama-70b", mape=20.0),
        ]
        table = format_per_model_table(records)
        assert "llama-7b" in table
        assert "llama-70b" in table


class TestFormatPerWorkloadTable:
    def test_contains_workload_names(self):
        records = [
            _make_record(workload="codegen", mape=10.0),
            _make_record(workload="roleplay", mape=20.0),
        ]
        table = format_per_workload_table(records)
        assert "codegen" in table
        assert "roleplay" in table


class TestFormatSignedErrorTable:
    def test_uses_mpe_not_mape(self):
        records = [_make_record(simulator="sim-a", metric_name="e2e_mean", mape=10.0, mpe=-5.0)]
        table = format_signed_error_table(records)
        assert "-5.00" in table


# ---------------------------------------------------------------------------
# Tests: save_csv
# ---------------------------------------------------------------------------


class TestSaveCsv:
    def test_creates_csv_file(self, tmp_path):
        records = _make_records()
        path = str(tmp_path / "out.csv")
        save_csv(records, path)
        assert os.path.exists(path)

    def test_csv_has_correct_rows(self, tmp_path):
        records = _make_records()
        path = str(tmp_path / "out.csv")
        save_csv(records, path)

        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["simulator"] == "sim-a"

    def test_csv_has_all_columns(self, tmp_path):
        records = [_make_record()]
        path = str(tmp_path / "out.csv")
        save_csv(records, path)

        with open(path) as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        expected_cols = {
            "simulator", "experiment_folder", "model", "workload",
            "stage_index", "metric_name", "predicted", "actual",
            "mape", "mpe", "absolute_error",
            "exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",
        }
        assert set(row.keys()) == expected_cols


# ---------------------------------------------------------------------------
# Tests: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_creates_csv_in_output_dir(self, tmp_path):
        records = _make_records()
        generate_report(records, str(tmp_path))
        assert os.path.exists(tmp_path / "error_records.csv")

    def test_creates_runtime_csv_when_provided(self, tmp_path):
        records = _make_records()
        runtime_records = _make_runtime_records()
        generate_report(records, str(tmp_path), runtime_records=runtime_records)
        assert os.path.exists(tmp_path / "error_records.csv")
        assert os.path.exists(tmp_path / "runtime.csv")

    def test_no_runtime_csv_when_empty(self, tmp_path):
        records = _make_records()
        generate_report(records, str(tmp_path), runtime_records=[])
        assert os.path.exists(tmp_path / "error_records.csv")
        assert not os.path.exists(tmp_path / "runtime.csv")

    def test_empty_records_prints_message(self, tmp_path, capsys):
        generate_report([], str(tmp_path))
        captured = capsys.readouterr()
        assert "No error records" in captured.out


# ---------------------------------------------------------------------------
# Tests: empty grouped dict
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers: runtime records
# ---------------------------------------------------------------------------


def _make_runtime_records() -> list[RuntimeRecord]:
    """Two simulators, two experiments each — 4 records total."""
    return [
        RuntimeRecord("sim-a", "/exp/1", "llama-7b", "codegen", 1.5),
        RuntimeRecord("sim-a", "/exp/2", "llama-7b", "roleplay", 2.5),
        RuntimeRecord("sim-b", "/exp/1", "llama-7b", "codegen", 3.0),
        RuntimeRecord("sim-b", "/exp/2", "llama-7b", "roleplay", 5.0),
    ]


# ---------------------------------------------------------------------------
# Tests: format_runtime_table
# ---------------------------------------------------------------------------


class TestFormatRuntimeTable:
    def test_contains_simulator_names(self):
        table = format_runtime_table(_make_runtime_records())
        assert "sim-a" in table
        assert "sim-b" in table

    def test_contains_headers(self):
        table = format_runtime_table(_make_runtime_records())
        assert "Mean(s)" in table
        assert "Min(s)" in table
        assert "Max(s)" in table
        assert "Total(s)" in table
        assert "Runs" in table

    def test_mean_value_correct(self):
        table = format_runtime_table(_make_runtime_records())
        # sim-a: mean = (1.5+2.5)/2 = 2.00
        assert "2.00" in table

    def test_empty_returns_no_data(self):
        table = format_runtime_table([])
        assert "no runtime data" in table

    def test_single_record(self):
        records = [RuntimeRecord("sim-x", "/exp/1", "llama-7b", "codegen", 4.25)]
        table = format_runtime_table(records)
        assert "sim-x" in table
        assert "4.25" in table


# ---------------------------------------------------------------------------
# Tests: save_runtime_csv
# ---------------------------------------------------------------------------


class TestSaveRuntimeCsv:
    def test_creates_csv_file(self, tmp_path):
        records = _make_runtime_records()
        path = str(tmp_path / "runtime.csv")
        save_runtime_csv(records, path)
        assert os.path.exists(path)

    def test_csv_has_correct_rows(self, tmp_path):
        records = _make_runtime_records()
        path = str(tmp_path / "runtime.csv")
        save_runtime_csv(records, path)

        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["simulator"] == "sim-a"
        assert float(rows[0]["wall_clock_seconds"]) == 1.5

    def test_csv_has_all_columns(self, tmp_path):
        records = [RuntimeRecord("sim-a", "/exp/1", "llama-7b", "codegen", 1.0)]
        path = str(tmp_path / "runtime.csv")
        save_runtime_csv(records, path)

        with open(path) as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        expected_cols = {
            "simulator", "experiment_folder", "model", "workload", "wall_clock_seconds",
            "exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",
        }
        assert set(row.keys()) == expected_cols


class TestCsvMetadataColumns:
    """Verify that new metadata columns appear in CSV output with correct values."""

    def test_error_csv_includes_metadata(self, tmp_path):
        record = ErrorRecord(
            simulator="sim-a", experiment_folder="/tmp/exp",
            model="llama-7b", workload="codegen", stage_index=0,
            metric_name="e2e_mean", predicted=110.0, actual=100.0,
            mape=10.0, mpe=10.0, absolute_error=10.0,
            exp_id=42, hardware="A100-80GB", dp=2,
            cpu_offload=True, gpu_mem_util=0.85, precision="FP8",
        )
        path = str(tmp_path / "out.csv")
        save_csv([record], path)

        with open(path) as fh:
            row = next(csv.DictReader(fh))
        assert row["exp_id"] == "42"
        assert row["hardware"] == "A100-80GB"
        assert row["dp"] == "2"
        assert row["cpu_offload"] == "True"
        assert row["gpu_mem_util"] == "0.85"
        assert row["precision"] == "FP8"

    def test_error_csv_dp_none_is_empty(self, tmp_path):
        record = ErrorRecord(
            simulator="sim-a", experiment_folder="/tmp/exp",
            model="m", workload="w", stage_index=0,
            metric_name="e2e_mean", predicted=100.0, actual=100.0,
            mape=0.0, mpe=0.0, absolute_error=0.0,
        )
        path = str(tmp_path / "out.csv")
        save_csv([record], path)

        with open(path) as fh:
            row = next(csv.DictReader(fh))
        assert row["dp"] == ""

    def test_runtime_csv_includes_metadata(self, tmp_path):
        record = RuntimeRecord(
            simulator="sim-a", experiment_folder="/exp/1",
            model="llama-7b", workload="codegen", wall_clock_seconds=1.5,
            exp_id=10, hardware="L40S", dp=None,
            cpu_offload=False, gpu_mem_util=0.9, precision="FP16",
        )
        path = str(tmp_path / "runtime.csv")
        save_runtime_csv([record], path)

        with open(path) as fh:
            row = next(csv.DictReader(fh))
        assert row["exp_id"] == "10"
        assert row["hardware"] == "L40S"
        assert row["dp"] == ""
        assert row["cpu_offload"] == "False"
        assert row["precision"] == "FP16"


class TestFormatTableNoneRendering:
    def test_missing_metrics_render_as_na(self):
        """When records only have mean metrics, p90/p99 columns show N/A."""
        records = [
            _make_record(simulator="sim-a", metric_name="e2e_mean", mape=10.0),
        ]
        table = format_aggregate_table(records)
        assert "N/A" in table
        assert "10.00" in table

    def test_all_metrics_missing_show_na(self):
        """A simulator with only one metric shows N/A for all others."""
        records = [
            _make_record(simulator="sim-a", metric_name="ttft_mean", mape=5.0),
        ]
        table = format_aggregate_table(records)
        na_count = table.count("N/A")
        # 9 metric columns, 1 has data → 8 show N/A.
        assert na_count == 8


class TestFormatTableEmpty:
    def test_empty_aggregate_table(self):
        """C3: format_aggregate_table on empty records should not crash."""
        table = format_aggregate_table([])
        assert "no data" in table

    def test_empty_per_model_table(self):
        table = format_per_model_table([])
        assert "no data" in table

    def test_empty_signed_error_table(self):
        table = format_signed_error_table([])
        assert "no data" in table
