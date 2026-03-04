"""Tests for experiment.trace_converter — BLIS trace v2 conversion."""

import csv
import json
import os

import pytest
import yaml

from experiment.trace_converter import convert_to_blis_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_per_request_metrics(num_requests=5, base_start_time=1393.0):
    """Build a synthetic per_request_lifecycle_metrics.json payload."""
    requests = []
    for i in range(num_requests):
        start_time = base_start_time + i * 0.5
        end_time = start_time + 1.8
        request_body = json.dumps({
            "model": "meta-llama/Llama-2-7b-hf",
            "prompt": f"test prompt {i}",
            "max_tokens": 247,
        })
        output_token_times = [start_time + 0.025 + j * 0.007 for j in range(140)]
        requests.append({
            "start_time": start_time,
            "end_time": end_time,
            "request": request_body,
            "info": {
                "input_tokens": 591 + i,
                "output_tokens": 140 + i,
                "output_token_times": output_token_times,
            },
        })
    return requests


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConvertToBLISTrace:
    def test_creates_three_files(self, tmp_path):
        metrics = _make_per_request_metrics()
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        header_path, data_path = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        assert os.path.exists(header_path)
        assert os.path.exists(data_path)
        # Workload spec should also exist
        spec_path = os.path.join(str(tmp_path / "output"), "workload_spec.yaml")
        assert os.path.exists(spec_path)

    def test_header_yaml_content(self, tmp_path):
        metrics = _make_per_request_metrics()
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        header_path, _ = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        with open(header_path) as fh:
            header = yaml.safe_load(fh)

        assert header["trace_version"] == 2
        assert header["time_unit"] == "microseconds"
        assert header["mode"] == "real"
        assert header["warm_up_requests"] == 0

    def test_csv_column_count(self, tmp_path):
        metrics = _make_per_request_metrics(num_requests=3)
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        _, data_path = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        with open(data_path) as fh:
            reader = csv.reader(fh)
            header_row = next(reader)
            assert len(header_row) == 22

            rows = list(reader)
            assert len(rows) == 3
            for row in rows:
                assert len(row) == 22

    def test_arrival_times_relative(self, tmp_path):
        metrics = _make_per_request_metrics(num_requests=3, base_start_time=1393.0)
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        _, data_path = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        with open(data_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # First request should have arrival_time_us = 0
        assert int(rows[0]["arrival_time_us"]) == 0

        # Second request: 0.5s later → 500,000 μs
        assert int(rows[1]["arrival_time_us"]) == 500_000

        # Third request: 1.0s later → 1,000,000 μs
        assert int(rows[2]["arrival_time_us"]) == 1_000_000

    def test_streaming_true(self, tmp_path):
        metrics = _make_per_request_metrics(num_requests=2)
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        _, data_path = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        with open(data_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                assert row["streaming"] == "true"

    def test_token_counts_correct(self, tmp_path):
        metrics = _make_per_request_metrics(num_requests=2)
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        _, data_path = convert_to_blis_trace(input_path, str(tmp_path / "output"))

        with open(data_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # First request: input_tokens=591, output_tokens=140
        assert int(rows[0]["input_tokens"]) == 591
        assert int(rows[0]["output_tokens"]) == 140
        assert int(rows[0]["text_tokens"]) == 591

    def test_workload_spec_content(self, tmp_path):
        metrics = _make_per_request_metrics()
        input_path = str(tmp_path / "per_request_lifecycle_metrics.json")
        with open(input_path, "w") as fh:
            json.dump(metrics, fh)

        convert_to_blis_trace(input_path, str(tmp_path / "output"))

        spec_path = os.path.join(str(tmp_path / "output"), "workload_spec.yaml")
        with open(spec_path) as fh:
            spec = yaml.safe_load(fh)

        assert "workloads" in spec
        assert len(spec["workloads"]) == 1
        assert spec["workloads"][0]["name"] == "trace-replay"
        assert "trace_header" in spec["workloads"][0]
        assert "trace_data" in spec["workloads"][0]
