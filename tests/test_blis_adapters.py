"""Tests for the BLIS adapter subclasses (roofline, evolved, trained-physics)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# Tests: adapter names
# ---------------------------------------------------------------------------


class TestAdapterNames:
    def test_roofline_name(self):
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter.name == "blis-roofline"

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

    def test_evolved_name_iter29(self):
        """Iter29 adapter should still report name as blis-evolved."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter
        adapter = BLISEvolvedAdapter("/tmp/blis", iteration=29)
        assert adapter.name == "blis-evolved"
        assert adapter.iteration == 29

    def test_evolved_rejects_invalid_iteration(self):
        """Evolved adapter should reject invalid iteration values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        with pytest.raises(ValueError, match="iteration must be 16, 24, 26, 27, or 29"):
            BLISEvolvedAdapter("/tmp/blis", iteration=99)


# ---------------------------------------------------------------------------
# Tests: can_run()
# ---------------------------------------------------------------------------


class TestRooflineCanRun:
    def test_always_true(self):
        adapter = BLISRooflineAdapter("/tmp/blis")
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
    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_roofline_wraps_subprocess_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd=["blis"], stderr=b"invalid args"
        )
        adapter = BLISRooflineAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS roofline failed.*invalid args"):
            adapter.run(exp)

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_wraps_subprocess_error(self, mock_run):
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"evolved backend not compiled"
        )
        adapter = BLISEvolvedAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS evolved iter26 failed.*evolved backend not compiled"):
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
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis", iteration=16)
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
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis", iteration=16)
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


# ---------------------------------------------------------------------------
# Tests: BLISTrainedPhysicsAdapter
# ---------------------------------------------------------------------------


class TestTrainedPhysicsAdapterName:
    def test_trained_physics_name(self):
        from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
        adapter = BLISTrainedPhysicsAdapter("/tmp/blis")
        assert adapter.name == "blis-trained-physics"


class TestTrainedPhysicsCanRun:
    def test_always_true(self):
        from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
        adapter = BLISTrainedPhysicsAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


class TestTrainedPhysicsCLIArgs:
    @patch("experiment.adapters.blis_trained_physics.subprocess.run")
    def test_trained_physics_latency_model_flag(self, mock_run):
        """Trained-physics adapter should pass --latency-model trained-physics."""
        from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISTrainedPhysicsAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "trained-physics"

    @patch("experiment.adapters.blis_trained_physics.subprocess.run")
    def test_trained_physics_no_coefficient_flags(self, mock_run):
        """Trained-physics adapter should NOT pass --alpha-coeffs or --beta-coeffs."""
        from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISTrainedPhysicsAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        assert "--alpha-coeffs" not in called_args
        assert "--beta-coeffs" not in called_args


class TestTrainedPhysicsSubprocessErrors:
    @patch("experiment.adapters.blis_trained_physics.subprocess.run")
    def test_trained_physics_wraps_subprocess_error(self, mock_run):
        from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"physics model not found"
        )
        adapter = BLISTrainedPhysicsAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS trained-physics failed.*physics model not found"):
            adapter.run(exp)
