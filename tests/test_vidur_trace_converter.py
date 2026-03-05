"""Tests for experiment.vidur_trace_converter."""

from __future__ import annotations

import csv
import json
import os

import pytest

from experiment.vidur_trace_converter import convert_to_vidur_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_request_json(tmpdir: str, requests: list[dict]) -> str:
    """Write a synthetic per_request_lifecycle_metrics.json."""
    path = os.path.join(tmpdir, "per_request_lifecycle_metrics.json")
    with open(path, "w") as fh:
        json.dump(requests, fh)
    return path


def _synthetic_requests() -> list[dict]:
    """Return 4 synthetic request records."""
    return [
        {
            "start_time": 100.0,
            "end_time": 101.5,
            "request": json.dumps({"model": "m", "max_tokens": 247}),
            "info": {"input_tokens": 591, "output_tokens": 140, "output_token_times": []},
        },
        {
            "start_time": 100.2,
            "end_time": 102.0,
            "request": json.dumps({"model": "m", "max_tokens": 247}),
            "info": {"input_tokens": 320, "output_tokens": 80, "output_token_times": []},
        },
        {
            "start_time": 100.5,
            "end_time": 103.0,
            "request": json.dumps({"model": "m", "max_tokens": 247}),
            "info": {"input_tokens": 450, "output_tokens": 200, "output_token_times": []},
        },
        {
            "start_time": 101.0,
            "end_time": 104.0,
            "request": json.dumps({"model": "m", "max_tokens": 247}),
            "info": {"input_tokens": 100, "output_tokens": 50, "output_token_times": []},
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvertToVidurTrace:

    def test_creates_csv_file(self, tmp_path):
        src = _make_per_request_json(str(tmp_path), _synthetic_requests())
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        result = convert_to_vidur_trace(src, out_csv)
        assert os.path.isfile(result)
        assert result == out_csv

    def test_csv_has_three_columns(self, tmp_path):
        src = _make_per_request_json(str(tmp_path), _synthetic_requests())
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        convert_to_vidur_trace(src, out_csv)

        with open(out_csv) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == ["arrived_at", "num_prefill_tokens", "num_decode_tokens"]
            rows = list(reader)
            assert len(rows) == 4
            for row in rows:
                assert len(row) == 3

    def test_arrival_times_relative_to_first(self, tmp_path):
        src = _make_per_request_json(str(tmp_path), _synthetic_requests())
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        convert_to_vidur_trace(src, out_csv)

        with open(out_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # First request: start_time=100.0, so relative arrival = 0.0
        assert float(rows[0]["arrived_at"]) == pytest.approx(0.0)
        # Second: 100.2 - 100.0 = 0.2
        assert float(rows[1]["arrived_at"]) == pytest.approx(0.2)
        # Third: 100.5 - 100.0 = 0.5
        assert float(rows[2]["arrived_at"]) == pytest.approx(0.5)
        # Fourth: 101.0 - 100.0 = 1.0
        assert float(rows[3]["arrived_at"]) == pytest.approx(1.0)

    def test_token_counts_correct(self, tmp_path):
        src = _make_per_request_json(str(tmp_path), _synthetic_requests())
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        convert_to_vidur_trace(src, out_csv)

        with open(out_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert int(rows[0]["num_prefill_tokens"]) == 591
        assert int(rows[0]["num_decode_tokens"]) == 140
        assert int(rows[2]["num_prefill_tokens"]) == 450
        assert int(rows[2]["num_decode_tokens"]) == 200

    def test_single_request(self, tmp_path):
        """Single request should have arrived_at=0.0."""
        reqs = [_synthetic_requests()[0]]
        src = _make_per_request_json(str(tmp_path), reqs)
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        convert_to_vidur_trace(src, out_csv)

        with open(out_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert float(rows[0]["arrived_at"]) == pytest.approx(0.0)

    def test_empty_input_creates_header_only(self, tmp_path):
        """Empty request list → CSV with header row but no data rows."""
        src = _make_per_request_json(str(tmp_path), [])
        out_csv = os.path.join(str(tmp_path), "vidur_trace.csv")
        convert_to_vidur_trace(src, out_csv)

        with open(out_csv) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == ["arrived_at", "num_prefill_tokens", "num_decode_tokens"]
            rows = list(reader)
            assert len(rows) == 0
