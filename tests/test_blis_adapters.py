"""Tests for the three BLIS adapter subclasses."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml

from experiment.adapters.blis_blackbox import BLISBlackboxAdapter
from experiment.adapters.blis_crossmodel import BLISCrossModelAdapter
from experiment.adapters.blis_roofline import BLISRooflineAdapter
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
            "data": {
                "type": "shared_prefix",
                "shared_prefix": {
                    "num_unique_system_prompts": 5,
                    "num_users_per_system_prompt": 1,
                    "system_prompt_len": 100,
                    "question_len": 466,
                    "output_len": 247,
                },
            },
        },
    )


def _write_defaults_yaml(tmpdir: str, models: list[str]) -> str:
    """Write a synthetic defaults.yaml with given model IDs."""
    data = {"defaults": {m: {"GPU": "H100", "tensor_parallelism": 1} for m in models}}
    path = os.path.join(tmpdir, "defaults.yaml")
    with open(path, "w") as fh:
        yaml.dump(data, fh)
    return path


# ---------------------------------------------------------------------------
# Tests: adapter names
# ---------------------------------------------------------------------------


class TestAdapterNames:
    def test_blackbox_name(self):
        adapter = BLISBlackboxAdapter("/tmp/blis")
        assert adapter.name == "blis-blackbox"

    def test_roofline_name(self):
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter.name == "blis-roofline"

    def test_crossmodel_name(self):
        adapter = BLISCrossModelAdapter("/tmp/blis")
        assert adapter.name == "blis-crossmodel"


# ---------------------------------------------------------------------------
# Tests: can_run()
# ---------------------------------------------------------------------------


class TestBlackboxCanRun:
    def test_exact_match(self, tmp_path):
        defaults = _write_defaults_yaml(str(tmp_path), ["meta-llama/Llama-2-7b-hf"])
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults)
        exp = _make_experiment(model="meta-llama/Llama-2-7b-hf")
        assert adapter.can_run(exp) is True

    def test_case_insensitive_match(self, tmp_path):
        defaults = _write_defaults_yaml(str(tmp_path), ["codellama/codellama-34b-instruct-hf"])
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults)
        exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
        assert adapter.can_run(exp) is True

    def test_no_match(self, tmp_path):
        defaults = _write_defaults_yaml(str(tmp_path), ["meta-llama/llama-3.1-8b-instruct"])
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults)
        exp = _make_experiment(model="meta-llama/Llama-2-7b-hf")
        assert adapter.can_run(exp) is False

    def test_missing_defaults_file(self):
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml="/nonexistent/defaults.yaml")
        exp = _make_experiment()
        assert adapter.can_run(exp) is False


class TestRooflineCanRun:
    def test_always_true(self):
        adapter = BLISRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


class TestCrossModelCanRun:
    def test_always_true(self):
        adapter = BLISCrossModelAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


# ---------------------------------------------------------------------------
# Tests: CLI argument construction
# ---------------------------------------------------------------------------


class TestBLISCLIArgs:
    @patch("experiment.adapters.blis_blackbox.subprocess.run")
    def test_blackbox_no_latency_model_flag(self, mock_run):
        """Blackbox adapter should NOT pass --latency-model."""
        mock_run.return_value = MagicMock()

        adapter = BLISBlackboxAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        assert "--latency-model" not in called_args

    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_roofline_latency_model_flag(self, mock_run):
        """Roofline adapter should pass --latency-model roofline."""
        mock_run.return_value = MagicMock()

        adapter = BLISRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "roofline"

    @patch("experiment.adapters.blis_crossmodel.subprocess.run")
    def test_crossmodel_latency_model_flag(self, mock_run):
        """CrossModel adapter should pass --latency-model crossmodel."""
        mock_run.return_value = MagicMock()

        adapter = BLISCrossModelAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "crossmodel"

    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_kv_offloading_flags_present(self, mock_run):
        """All adapters should include KV offloading flags."""
        mock_run.return_value = MagicMock()

        adapter = BLISRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        assert "--total-kv-blocks" in called_args
        assert "--kv-cpu-blocks" in called_args
        assert "--kv-offload-threshold" in called_args
        assert "--kv-transfer-bandwidth" in called_args

        # Check the actual values
        idx = called_args.index("--total-kv-blocks")
        assert called_args[idx + 1] == "7463"
        idx = called_args.index("--kv-cpu-blocks")
        assert called_args[idx + 1] == "5"

    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_model_and_tp_in_args(self, mock_run):
        """Model and TP should be passed correctly."""
        mock_run.return_value = MagicMock()

        adapter = BLISRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment(model="meta-llama/Llama-2-70b-hf", tp=4)

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--model")
        assert called_args[idx + 1] == "meta-llama/Llama-2-70b-hf"
        idx = called_args.index("--tp")
        assert called_args[idx + 1] == "4"


# ---------------------------------------------------------------------------
# Tests: workload spec generation
# ---------------------------------------------------------------------------


class TestWorkloadSpecGeneration:
    def test_generates_valid_inference_perf_spec(self, tmp_path):
        adapter = BLISRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        spec_path = str(tmp_path / "workload_spec.yaml")
        adapter._write_workload_spec(exp, spec_path)

        with open(spec_path) as fh:
            spec = yaml.safe_load(fh)

        assert spec["version"] == "2"
        assert "inference_perf" in spec
        assert len(spec["inference_perf"]["stages"]) == 1
        assert spec["inference_perf"]["stages"][0]["rate"] == 5.0
        assert spec["inference_perf"]["stages"][0]["duration"] == 600
        assert spec["inference_perf"]["shared_prefix"]["question_len"] == 466

    def test_no_workloads_key(self, tmp_path):
        """The old broken 'workloads' key should not exist."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        spec_path = str(tmp_path / "workload_spec.yaml")
        adapter._write_workload_spec(exp, spec_path)

        with open(spec_path) as fh:
            spec = yaml.safe_load(fh)
        assert "workloads" not in spec


# ---------------------------------------------------------------------------
# Tests: subprocess error handling (C4 fix)
# ---------------------------------------------------------------------------


class TestBLISSubprocessErrors:
    @patch("experiment.adapters.blis_blackbox.subprocess.run")
    def test_blackbox_wraps_subprocess_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"model not found"
        )
        adapter = BLISBlackboxAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS blackbox failed.*model not found"):
            adapter.run(exp)

    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_roofline_wraps_subprocess_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd=["blis"], stderr=b"invalid args"
        )
        adapter = BLISRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS roofline failed.*invalid args"):
            adapter.run(exp)

    @patch("experiment.adapters.blis_crossmodel.subprocess.run")
    def test_crossmodel_wraps_subprocess_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"OOM"
        )
        adapter = BLISCrossModelAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS crossmodel failed.*OOM"):
            adapter.run(exp)

    @patch("experiment.adapters.blis_blackbox.subprocess.run")
    def test_error_includes_model_name(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"err"
        )
        adapter = BLISBlackboxAdapter("/tmp/blis")
        exp = _make_experiment(model="mistral/Mistral-7B")
        with pytest.raises(RuntimeError, match="mistral/Mistral-7B"):
            adapter.run(exp)
