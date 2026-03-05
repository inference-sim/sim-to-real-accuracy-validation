"""Tests for experiment.adapters.vidur."""

from __future__ import annotations

import csv
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from experiment.adapters.vidur import VidurAdapter, _SUPPORTED_MODELS
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_lat():
    return LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)

def _zero_tp():
    return ThroughputMetrics(0.0, 0.0, 0.0)

def _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1) -> Experiment:
    return Experiment(
        folder="/tmp/test-exp",
        model=model,
        tp=tp,
        workload="codegen",
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        total_kv_blocks=7463,
        cpu_kv_blocks=5,
        stages=[
            StageMetrics(
                stage_index=0, rate=5.0, duration=600.0, num_requests=3000,
                e2e=_zero_lat(), ttft=_zero_lat(), itl=_zero_lat(), throughput=_zero_tp(),
            ),
        ],
        summary=StageMetrics(
            stage_index=-1, rate=0.0, duration=0.0, num_requests=3000,
            e2e=_zero_lat(), ttft=_zero_lat(), itl=_zero_lat(), throughput=_zero_tp(),
        ),
        profile_config={
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        },
    )


def _write_request_metrics_csv(path: str, rows: list[dict]) -> None:
    """Write a synthetic request_metrics.csv."""
    fieldnames = [
        "request_id", "request_e2e_time", "prefill_e2e_time",
        "decode_time_execution_plus_preemption_normalized",
        "request_num_prefill_tokens", "request_num_decode_tokens",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Tests: adapter name and can_run
# ---------------------------------------------------------------------------


class TestVidurAdapterBasics:
    def test_name(self):
        adapter = VidurAdapter("/tmp/vidur")
        assert adapter.name == "vidur"

    def test_can_run_supported_model(self):
        adapter = VidurAdapter("/tmp/vidur")
        for model in _SUPPORTED_MODELS:
            exp = _make_experiment(model=model)
            assert adapter.can_run(exp) is True

    def test_cannot_run_unsupported_model(self):
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment(model="mistralai/Mistral-7B-v0.1")
        assert adapter.can_run(exp) is False


# ---------------------------------------------------------------------------
# Tests: CLI args
# ---------------------------------------------------------------------------


class TestVidurCLIArgs:
    def test_args_include_model_and_tp(self):
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment(model="meta-llama/Llama-2-70b-hf", tp=4)
        args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/output")

        idx = args.index("--replica_config_model_name")
        assert args[idx + 1] == "meta-llama/Llama-2-70b-hf"
        idx = args.index("--replica_config_tensor_parallel_size")
        assert args[idx + 1] == "4"

    def test_args_include_scheduler_config(self):
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/output")

        idx = args.index("--vllm_scheduler_config_batch_size_cap")
        assert args[idx + 1] == "128"
        idx = args.index("--vllm_scheduler_config_max_tokens_in_batch")
        assert args[idx + 1] == "2048"

    def test_args_include_trace_file(self):
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        args = adapter._build_args(exp, "/data/trace.csv", "/tmp/output")

        idx = args.index("--trace_request_generator_config_trace_file")
        assert args[idx + 1] == "/data/trace.csv"

    def test_args_include_device_h100(self):
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/output")

        idx = args.index("--replica_config_device")
        assert args[idx + 1] == "h100"


# ---------------------------------------------------------------------------
# Tests: subprocess error handling
# ---------------------------------------------------------------------------


class TestVidurSubprocessError:
    @patch("experiment.adapters.vidur.convert_to_vidur_trace")
    @patch("experiment.adapters.vidur.subprocess.run")
    def test_wraps_subprocess_error(self, mock_run, mock_convert):
        mock_convert.return_value = "/tmp/trace.csv"
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["vidur"], stderr=b"model not supported"
        )
        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="Vidur failed.*model not supported"):
            adapter.run(exp)


# ---------------------------------------------------------------------------
# Tests: CSV finding
# ---------------------------------------------------------------------------


class TestFindRequestMetricsCsv:
    def test_finds_csv_in_timestamped_subdir(self, tmp_path):
        subdir = tmp_path / "2026-03-04_12-00-00-000000"
        subdir.mkdir()
        csv_path = subdir / "request_metrics.csv"
        csv_path.write_text("request_id\n")

        found = VidurAdapter._find_request_metrics_csv(str(tmp_path))
        assert found == str(csv_path)

    def test_raises_if_no_csv(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No request_metrics.csv"):
            VidurAdapter._find_request_metrics_csv(str(tmp_path))


# ---------------------------------------------------------------------------
# Tests: result parsing
# ---------------------------------------------------------------------------


class TestVidurResultParsing:
    def test_parses_csv_to_simulator_result(self, tmp_path):
        """Parse a synthetic CSV with known values."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        # 10 requests, all in stage 0 (rate=5, duration=600 → expects 3000 reqs)
        rows = []
        for i in range(10):
            rows.append({
                "request_id": i,
                "request_e2e_time": 1.8 + i * 0.01,      # seconds
                "prefill_e2e_time": 0.025 + i * 0.001,    # seconds
                "decode_time_execution_plus_preemption_normalized": 0.0036,  # seconds
                "request_num_prefill_tokens": 566,
                "request_num_decode_tokens": 247,
            })
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)

        assert result.adapter_name == "vidur"
        assert len(result.stages) == 1
        # E2E mean should be ~1845ms (1.845s * 1000)
        assert abs(result.stages[0].e2e.mean - 1845.0) < 1.0
        # TTFT mean should be ~29.5ms
        assert abs(result.stages[0].ttft.mean - 29.5) < 1.0
        # ITL mean should be 3.6ms
        assert abs(result.stages[0].itl.mean - 3.6) < 0.01

    def test_seconds_to_ms_conversion(self, tmp_path):
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        rows = [{
            "request_id": 0,
            "request_e2e_time": 2.0,        # 2000ms
            "prefill_e2e_time": 0.05,        # 50ms
            "decode_time_execution_plus_preemption_normalized": 0.004,  # 4ms
            "request_num_prefill_tokens": 500,
            "request_num_decode_tokens": 200,
        }]
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)

        s0 = result.stages[0]
        assert abs(s0.e2e.mean - 2000.0) < 0.01
        assert abs(s0.ttft.mean - 50.0) < 0.01
        assert abs(s0.itl.mean - 4.0) < 0.01

    def test_empty_csv_produces_zero_metrics(self, tmp_path):
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")
        _write_request_metrics_csv(csv_path, [])

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)

        assert result.stages[0].num_requests == 0
        assert result.stages[0].e2e.mean == 0.0

    def test_summary_uses_total_duration(self, tmp_path):
        """Summary stage should use total duration across all stages."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        rows = [{
            "request_id": 0,
            "request_e2e_time": 1.8,
            "prefill_e2e_time": 0.025,
            "decode_time_execution_plus_preemption_normalized": 0.0036,
            "request_num_prefill_tokens": 566,
            "request_num_decode_tokens": 247,
        }]
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)

        # Summary duration = 600 (single stage)
        assert result.summary.stage_index == -1
        assert result.summary.num_requests == 1

    def test_throughput_computed(self, tmp_path):
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        rows = [{
            "request_id": 0,
            "request_e2e_time": 1.8,
            "prefill_e2e_time": 0.025,
            "decode_time_execution_plus_preemption_normalized": 0.0036,
            "request_num_prefill_tokens": 600,
            "request_num_decode_tokens": 300,
        }]
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)

        s0 = result.stages[0]
        # input_tokens_per_sec = 600 / 600 = 1.0
        assert abs(s0.throughput.input_tokens_per_sec - 1.0) < 0.01
        # output_tokens_per_sec = 300 / 600 = 0.5
        assert abs(s0.throughput.output_tokens_per_sec - 0.5) < 0.01

    def test_multi_stage_splitting(self, tmp_path):
        """T1: Two stages with different rates split rows correctly."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        # Stage 0: rate=2, duration=5 → 10 requests
        # Stage 1: rate=3, duration=5 → 15 requests
        rows = []
        for i in range(25):
            rows.append({
                "request_id": i,
                "request_e2e_time": 1.0 if i < 10 else 2.0,
                "prefill_e2e_time": 0.01,
                "decode_time_execution_plus_preemption_normalized": 0.001,
                "request_num_prefill_tokens": 100,
                "request_num_decode_tokens": 50,
            })
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        # Override profile_config for two stages
        exp = Experiment(
            folder=exp.folder, model=exp.model, tp=exp.tp, workload=exp.workload,
            max_model_len=exp.max_model_len,
            max_num_batched_tokens=exp.max_num_batched_tokens,
            max_num_seqs=exp.max_num_seqs,
            total_kv_blocks=exp.total_kv_blocks, cpu_kv_blocks=exp.cpu_kv_blocks,
            stages=exp.stages, summary=exp.summary,
            profile_config={
                "load": {"stages": [
                    {"duration": 5, "rate": 2},
                    {"duration": 5, "rate": 3},
                ]},
            },
        )

        result = adapter._parse_vidur_results(csv_path, exp)
        assert len(result.stages) == 2
        assert result.stages[0].num_requests == 10
        assert result.stages[1].num_requests == 15
        # Stage 0 rows all have e2e=1.0s → 1000ms
        assert abs(result.stages[0].e2e.mean - 1000.0) < 1.0
        # Stage 1 rows all have e2e=2.0s → 2000ms
        assert abs(result.stages[1].e2e.mean - 2000.0) < 1.0

    def test_missing_columns_filtered(self, tmp_path):
        """C2: Rows missing required columns are silently dropped."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        # Write a CSV with one good row and one row missing _COL_E2E
        os.makedirs(str(subdir), exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            fh.write("request_id,request_e2e_time,prefill_e2e_time,"
                     "decode_time_execution_plus_preemption_normalized,"
                     "request_num_prefill_tokens,request_num_decode_tokens\n")
            fh.write("0,1.8,0.025,0.0036,566,247\n")
            # Row with non-numeric e2e
            fh.write("1,NOT_A_NUMBER,0.025,0.0036,566,247\n")

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)
        # Only the valid row should be parsed
        assert result.stages[0].num_requests == 1

    def test_float_like_token_counts(self, tmp_path):
        """I2: Token columns like '512.0' should parse correctly."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")

        rows = [{
            "request_id": 0,
            "request_e2e_time": 1.8,
            "prefill_e2e_time": 0.025,
            "decode_time_execution_plus_preemption_normalized": 0.0036,
            "request_num_prefill_tokens": "512.0",
            "request_num_decode_tokens": "247.0",
        }]
        _write_request_metrics_csv(csv_path, rows)

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        result = adapter._parse_vidur_results(csv_path, exp)
        # Should not crash, tokens parsed via int(float(...))
        assert result.stages[0].throughput.input_tokens_per_sec > 0

    def test_invalid_profile_config_raises(self, tmp_path):
        """I3: Missing profile_config['load']['stages'] raises RuntimeError."""
        subdir = tmp_path / "ts"
        csv_path = str(subdir / "request_metrics.csv")
        _write_request_metrics_csv(csv_path, [{
            "request_id": 0,
            "request_e2e_time": 1.0,
            "prefill_e2e_time": 0.01,
            "decode_time_execution_plus_preemption_normalized": 0.001,
            "request_num_prefill_tokens": 100,
            "request_num_decode_tokens": 50,
        }])

        adapter = VidurAdapter("/tmp/vidur")
        exp = _make_experiment()
        bad_exp = Experiment(
            folder=exp.folder, model=exp.model, tp=exp.tp, workload=exp.workload,
            max_model_len=exp.max_model_len,
            max_num_batched_tokens=exp.max_num_batched_tokens,
            max_num_seqs=exp.max_num_seqs,
            total_kv_blocks=exp.total_kv_blocks, cpu_kv_blocks=exp.cpu_kv_blocks,
            stages=exp.stages, summary=exp.summary,
            profile_config={},  # Missing 'load' key
        )
        with pytest.raises(RuntimeError, match="missing 'load.stages' key"):
            adapter._parse_vidur_results(csv_path, bad_exp)
