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
from experiment.adapters.blis_trained_roofline import BLISTrainedRooflineAdapter
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


def _write_defaults_yaml(tmpdir: str, models: list[str], tp: int = 1) -> str:
    """Write a synthetic defaults.yaml with given model IDs and trained coefficients."""
    data = {
        "defaults": {m: {"GPU": "H100", "tensor_parallelism": tp} for m in models},
        "models": [
            {
                "id": m,
                "GPU": "H100",
                "tensor_parallelism": tp,
                "vllm_version": "vllm/vllm-openai:v0.8.4",
                "alpha_coeffs": [1.0, 2.0, 3.0],
                "beta_coeffs": [4.0, 5.0, 6.0],
                "total_kv_blocks": 2537,
            }
            for m in models
        ],
    }
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

    def test_trained_roofline_name(self):
        adapter = BLISTrainedRooflineAdapter("/tmp/blis")
        assert adapter.name == "blis-trained-roofline"

    def test_evolved_name(self):
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter
        adapter = BLISEvolvedAdapter("/tmp/blis")
        assert adapter.name == "blis-evolved"

    def test_evolved_name_iter27(self):
        """Iter27 adapter should still report name as blis-evolved."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter
        adapter = BLISEvolvedAdapter("/tmp/blis", iteration=27)
        assert adapter.name == "blis-evolved"
        assert adapter.iteration == 27

    def test_evolved_rejects_invalid_iteration(self):
        """Evolved adapter should reject invalid iteration values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        with pytest.raises(ValueError, match="iteration must be 16, 24, 26, or 27"):
            BLISEvolvedAdapter("/tmp/blis", iteration=99)


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

    def test_tp_mismatch(self, tmp_path):
        """Coefficients exist for TP=1 but experiment uses TP=2 — should reject."""
        defaults = _write_defaults_yaml(str(tmp_path), ["codellama/codellama-34b-instruct-hf"], tp=1)
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults)
        exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf", tp=2)
        assert adapter.can_run(exp) is False

    def test_missing_defaults_file(self):
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml="/nonexistent/defaults.yaml")
        exp = _make_experiment()
        assert adapter.can_run(exp) is False

    def test_gpu_mismatch_rejects(self, tmp_path):
        """Coefficients for H100 should not match A100-80GB experiment."""
        data = {
            "models": [{
                "id": "meta-llama/Llama-2-7b-hf",
                "GPU": "H100",
                "tensor_parallelism": 1,
                "alpha_coeffs": [1.0, 2.0, 3.0],
            }]
        }
        defaults_path = os.path.join(str(tmp_path), "defaults.yaml")
        with open(defaults_path, "w") as fh:
            yaml.dump(data, fh)

        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults_path)

        # H100 should match
        exp_h100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)
        exp_h100.hardware = "H100"
        assert adapter.can_run(exp_h100) is True

        # A100-80GB should NOT match
        exp_a100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)
        exp_a100.hardware = "A100-80GB"
        assert adapter.can_run(exp_a100) is False

    def test_a100_normalization_matches(self, tmp_path):
        """A100-80GB experiment should match A100-80 coefficients via normalization."""
        data = {
            "models": [{
                "id": "meta-llama/Llama-2-7b-hf",
                "GPU": "A100-80",
                "tensor_parallelism": 1,
                "alpha_coeffs": [1.0, 2.0, 3.0],
            }]
        }
        defaults_path = os.path.join(str(tmp_path), "defaults.yaml")
        with open(defaults_path, "w") as fh:
            yaml.dump(data, fh)

        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults_path)
        exp = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)
        exp.hardware = "A100-80GB"
        assert adapter.can_run(exp) is True


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


class TestTrainedRooflineCanRun:
    def test_always_true(self):
        adapter = BLISTrainedRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


class TestEvolvedCanRun:
    def test_always_true(self):
        """Evolved adapter works for any model (iter16 coefficients are cross-model)."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


# ---------------------------------------------------------------------------
# Tests: hardware normalization
# ---------------------------------------------------------------------------


class TestHardwareNormalization:
    def test_a100_80gb_normalization(self):
        """A100-80GB from manifest should normalize to A100-80 for BLIS."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("A100-80GB") == "A100-80"

    def test_h100_passes_through(self):
        """H100 should pass through unchanged."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("H100") == "H100"

    def test_l40s_passes_through(self):
        """L40S should pass through unchanged."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("L40S") == "L40S"


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

    @patch("experiment.adapters.blis_trained_roofline.subprocess.run")
    def test_trained_roofline_latency_model_flag(self, mock_run):
        """Trained-roofline adapter should pass --latency-model trained-roofline."""
        mock_run.return_value = MagicMock()

        adapter = BLISTrainedRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "trained-roofline"

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

        # Check hardware is passed from experiment
        idx = called_args.index("--hardware")
        assert called_args[idx + 1] == "H100"

    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_hardware_from_experiment_normalized(self, mock_run):
        """Hardware should come from experiment and be normalized."""
        mock_run.return_value = MagicMock()
        adapter = BLISRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment()
        exp.hardware = "A100-80GB"

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--hardware")
        assert called_args[idx + 1] == "A100-80"

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

    @patch("experiment.adapters.blis_trained_roofline.subprocess.run")
    def test_trained_roofline_wraps_subprocess_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"coefficients missing"
        )
        adapter = BLISTrainedRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS trained-roofline failed.*coefficients missing"):
            adapter.run(exp)

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_wraps_subprocess_error(self, mock_run):
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"evolved backend not compiled"
        )
        adapter = BLISEvolvedAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS evolved failed.*evolved backend not compiled"):
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


# ---------------------------------------------------------------------------
# Tests: BLISEvolvedAdapter coefficient formatting
# ---------------------------------------------------------------------------


class TestBLISEvolvedCoefficients:
    def test_format_coeffs_three_values(self):
        """Alpha coefficients (3 values) should format with 6 decimals."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([15569.495449697066, 815.0556502348827, 45.705744318725586])
        assert result == "15569.495450,815.055650,45.705744"

    def test_format_coeffs_seven_values(self):
        """Beta coefficients (7 values) should format with 6 decimals."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([
            0.20081681581824434, 1.6173961192042448, 1.3603417361920076,
            0.39579536655780084, 62.19421689224744, 2.937563498958273, 169.37780505091155
        ])
        assert result == "0.200817,1.617396,1.360342,0.395795,62.194217,2.937563,169.377805"

    def test_format_coeffs_no_trailing_zeros(self):
        """Formatted coefficients should maintain 6 decimal places."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([1.0, 2.5, 3.123456789])
        assert result == "1.000000,2.500000,3.123457"


# ---------------------------------------------------------------------------
# Tests: BLISEvolvedAdapter CLI argument construction
# ---------------------------------------------------------------------------


class TestBLISEvolvedCLIArgs:
    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_latency_model_flag(self, mock_run):
        """Evolved adapter should pass --latency-model evolved."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "evolved"

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_alpha_coeffs_flag(self, mock_run):
        """Evolved adapter should pass --alpha-coeffs with iter16 values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--alpha-coeffs")
        alpha_str = called_args[idx + 1]

        # Should be 3 comma-separated values
        assert alpha_str.count(",") == 2
        parts = alpha_str.split(",")
        assert len(parts) == 3

        # Verify first alpha coefficient (QueueingTime ~ 15569.5)
        assert parts[0].startswith("15569.")

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_beta_coeffs_flag(self, mock_run):
        """Evolved adapter should pass --beta-coeffs with iter16 values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--beta-coeffs")
        beta_str = called_args[idx + 1]

        # Should be 7 comma-separated values
        assert beta_str.count(",") == 6
        parts = beta_str.split(",")
        assert len(parts) == 7

        # Verify first beta coefficient (prefill roofline ~ 0.2)
        assert parts[0].startswith("0.2")
