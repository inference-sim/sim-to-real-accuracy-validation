"""Tests for experiment.adapters.llm_optimizer_est."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from experiment.adapters.llm_optimizer_est import LLMOptimizerEstimateAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

def _make_experiment(
    model="meta-llama/Llama-2-7b-hf",
    tp=1,
    stages=None,
    summary=None,
    profile_config=None,
    hardware="H100",
    precision="FP16",
) -> Experiment:
    """Build a minimal Experiment for testing."""
    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    zero_tp = ThroughputMetrics(0.0, 0.0, 0.0)

    if stages is None:
        stages = [
            StageMetrics(
                stage_index=0, rate=5.0, duration=600.0, num_requests=3000,
                e2e=LatencyDistribution(mean=1800.0, p90=1926.0, p99=2178.0),
                ttft=LatencyDistribution(mean=25.0, p90=26.75, p99=30.25),
                itl=LatencyDistribution(mean=3.6, p90=3.85, p99=4.36),
                throughput=ThroughputMetrics(2950.0, 966.0, 5.0),
            ),
            StageMetrics(
                stage_index=1, rate=10.0, duration=600.0, num_requests=6000,
                e2e=LatencyDistribution(mean=2100.0, p90=2247.0, p99=2541.0),
                ttft=LatencyDistribution(mean=30.0, p90=32.1, p99=36.3),
                itl=LatencyDistribution(mean=4.6, p90=4.92, p99=5.57),
                throughput=ThroughputMetrics(2800.0, 920.0, 10.0),
            ),
        ]
    if summary is None:
        summary = StageMetrics(
            stage_index=-1, rate=0.0, duration=0.0, num_requests=9000,
            e2e=zero_lat, ttft=zero_lat, itl=zero_lat, throughput=zero_tp,
        )
    if profile_config is None:
        profile_config = {
            "data": {
                "shared_prefix": {
                    "question_len": 466,
                    "system_prompt_len": 100,
                    "output_len": 247,
                },
                "type": "shared_prefix",
            },
            "load": {
                "stages": [
                    {"duration": 600, "rate": 5},
                    {"duration": 600, "rate": 10},
                ],
            },
        }

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
        stages=stages,
        summary=summary,
        profile_config=profile_config,
        hardware=hardware,
        precision=precision,
    )


@dataclass
class _FakePerformanceResult:
    """Mimics llm_optimizer.performance.PerformanceResult."""
    ttft_ms: float
    itl_ms: float
    e2e_latency_s: float
    output_throughput_tps: float
    input_throughput_tps: float
    requests_per_sec: float
    bottleneck_is_memory: bool = False
    concurrency: int = 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMOptimizerName:
    def test_adapter_name(self):
        adapter = LLMOptimizerEstimateAdapter()
        assert adapter.name == "llm-optimizer-estimate"


class TestLLMOptimizerCanRun:
    def test_can_run_shared_prefix(self):
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        assert adapter.can_run(exp) is True

    def test_cannot_run_non_shared_prefix(self):
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment(profile_config={
            "data": {"type": "random"},
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_cannot_run_missing_data_key(self):
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment(profile_config={
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_cannot_run_missing_shared_prefix_subkeys(self):
        """shared_prefix type but missing required sub-keys."""
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment(profile_config={
            "data": {"type": "shared_prefix", "shared_prefix": {"question_len": 100}},
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_can_run_rejects_l40s(self):
        exp = _make_experiment()
        exp.hardware = "L40S"
        assert LLMOptimizerEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_a100_fp8(self):
        exp = _make_experiment()
        exp.hardware = "A100-80GB"
        exp.precision = "FP8"
        assert LLMOptimizerEstimateAdapter().can_run(exp) is False

    def test_can_run_accepts_a100_fp16(self):
        exp = _make_experiment()
        exp.hardware = "A100-80GB"
        exp.precision = "FP16"
        assert LLMOptimizerEstimateAdapter().can_run(exp) is True

    def test_can_run_accepts_h100_fp8(self):
        exp = _make_experiment()
        exp.hardware = "H100"
        exp.precision = "FP8"
        assert LLMOptimizerEstimateAdapter().can_run(exp) is True


class TestConcurrencyDerivation:
    def test_littles_law_stage0(self):
        """Stage 0: rate=5, e2e_mean=1800ms → concurrency = 5 * 1.8 = 9."""
        exp = _make_experiment()
        c = LLMOptimizerEstimateAdapter._derive_concurrency(exp.stages[0], exp.max_num_seqs)
        assert c == 9  # round(5 * 1800 / 1000)

    def test_littles_law_stage1(self):
        """Stage 1: rate=10, e2e_mean=2100ms → concurrency = 10 * 2.1 = 21."""
        exp = _make_experiment()
        c = LLMOptimizerEstimateAdapter._derive_concurrency(exp.stages[1], exp.max_num_seqs)
        assert c == 21  # round(10 * 2100 / 1000)

    def test_minimum_concurrency_is_one(self):
        """Very low rate/latency should still yield concurrency >= 1."""
        zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
        stage = StageMetrics(
            stage_index=0, rate=0.01, duration=60.0, num_requests=1,
            e2e=zero_lat, ttft=zero_lat, itl=zero_lat,
            throughput=ThroughputMetrics(0, 0, 0),
        )
        c = LLMOptimizerEstimateAdapter._derive_concurrency(stage, 128)
        assert c >= 1

    def test_concurrency_clamped_to_max_num_seqs(self):
        """Concurrency should not exceed max_num_seqs."""
        stage = StageMetrics(
            stage_index=0, rate=100.0, duration=600.0, num_requests=60000,
            e2e=LatencyDistribution(mean=50000.0, p90=0.0, p99=0.0),
            ttft=LatencyDistribution(mean=0.0, p90=0.0, p99=0.0),
            itl=LatencyDistribution(mean=0.0, p90=0.0, p99=0.0),
            throughput=ThroughputMetrics(0, 0, 0),
        )
        # raw = round(100 * 50000 / 1000) = 5000, clamped to 32
        c = LLMOptimizerEstimateAdapter._derive_concurrency(stage, 32)
        assert c == 32


class TestInputOutputExtraction:
    def test_extracts_from_shared_prefix(self):
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        inp, out = adapter._extract_lengths(exp)
        # question_len=466 + system_prompt_len=100 = 566
        assert inp == 566
        assert out == 247


class TestRunWithMock:
    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_run_returns_simulator_result(self, mock_get_cfg, mock_estimate):
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0,
            itl_ms=3.5,
            e2e_latency_s=1.8,
            output_throughput_tps=980.0,
            input_throughput_tps=2900.0,
            requests_per_sec=5.2,
            concurrency=9,
        )

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        assert result.adapter_name == "llm-optimizer-estimate"
        assert result.experiment_folder == exp.folder
        assert len(result.stages) == 2

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_latency_unit_conversion(self, mock_get_cfg, mock_estimate):
        """e2e_latency_s is in seconds → should be converted to ms."""
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0,
            itl_ms=3.5,
            e2e_latency_s=1.8,
            output_throughput_tps=980.0,
            input_throughput_tps=2900.0,
            requests_per_sec=5.2,
            concurrency=9,
        )

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        s0 = result.stages[0]
        # e2e mean = 1.8s * 1000 = 1800ms
        assert abs(s0.e2e.mean - 1800.0) < 0.01
        # ttft mean already in ms
        assert abs(s0.ttft.mean - 25.0) < 0.01
        # itl mean already in ms
        assert abs(s0.itl.mean - 3.5) < 0.01

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_percentiles_are_none(self, mock_get_cfg, mock_estimate):
        """P90/P99 should be None (point-estimate simulator)."""
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0,
            itl_ms=3.5,
            e2e_latency_s=1.8,
            output_throughput_tps=980.0,
            input_throughput_tps=2900.0,
            requests_per_sec=5.2,
            concurrency=9,
        )

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        s0 = result.stages[0]
        assert s0.e2e.p90 is None
        assert s0.e2e.p99 is None
        assert s0.ttft.p90 is None
        assert s0.ttft.p99 is None
        assert s0.itl.p90 is None
        assert s0.itl.p99 is None

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_throughput_from_result(self, mock_get_cfg, mock_estimate):
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0,
            itl_ms=3.5,
            e2e_latency_s=1.8,
            output_throughput_tps=980.0,
            input_throughput_tps=2900.0,
            requests_per_sec=5.2,
            concurrency=9,
        )

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        s0 = result.stages[0]
        assert abs(s0.throughput.output_tokens_per_sec - 980.0) < 0.01
        assert abs(s0.throughput.input_tokens_per_sec - 2900.0) < 0.01
        assert abs(s0.throughput.requests_per_sec - 5.2) < 0.01

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_summary_is_weighted_average(self, mock_get_cfg, mock_estimate):
        """Summary E2E mean should be weighted average across stages."""
        mock_get_cfg.return_value = MagicMock()
        # Return different values for stage 0 and stage 1
        mock_estimate.side_effect = [
            _FakePerformanceResult(
                ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
                output_throughput_tps=980.0, input_throughput_tps=2900.0,
                requests_per_sec=5.2, concurrency=9,
            ),
            _FakePerformanceResult(
                ttft_ms=30.0, itl_ms=4.5, e2e_latency_s=2.1,
                output_throughput_tps=920.0, input_throughput_tps=2800.0,
                requests_per_sec=10.1, concurrency=21,
            ),
        ]

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # Latency: request-weighted.  stage0=3000 reqs, stage1=6000 reqs
        # e2e: (1800*3000 + 2100*6000) / 9000 = 2000.0
        assert abs(result.summary.e2e.mean - 2000.0) < 0.01

        # Throughput: duration-weighted.  stage0=600s, stage1=600s (equal duration)
        # requests_per_sec: (5.2*600 + 10.1*600) / 1200 = 7.65
        expected_rps = (5.2 * 600 + 10.1 * 600) / 1200
        assert abs(result.summary.throughput.requests_per_sec - expected_rps) < 0.01

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_model_config_called_once(self, mock_get_cfg, mock_estimate):
        """get_model_config_from_hf should be called once (cached)."""
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
            output_throughput_tps=980.0, input_throughput_tps=2900.0,
            requests_per_sec=5.2, concurrency=9,
        )

        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        adapter.run(exp)

        mock_get_cfg.assert_called_once_with("meta-llama/Llama-2-7b-hf")

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_run_passes_experiment_precision(self, mock_get_cfg, mock_estimate):
        """Precision should come from experiment, not model_config.inferred_precision."""
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
            output_throughput_tps=980.0, input_throughput_tps=2900.0,
            requests_per_sec=5.2, concurrency=9,
        )
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        exp.precision = "FP8"
        adapter.run(exp)
        _, kwargs = mock_estimate.call_args
        assert kwargs["precision"] == "fp8"

    def test_run_rejects_unsupported_hardware(self):
        """run() should raise ValueError for unsupported hardware."""
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment(hardware="L40S")
        with pytest.raises(ValueError, match="Unsupported hardware"):
            adapter.run(exp)

    def test_run_rejects_a100_fp8(self):
        """run() should raise ValueError for A100+FP8 (no FP8 TFLOPS)."""
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment(hardware="A100-80GB", precision="FP8")
        with pytest.raises(ValueError, match="Unsupported precision"):
            adapter.run(exp)

    @patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
    @patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
    def test_run_uses_hardware_gpu_name(self, mock_get_cfg, mock_estimate):
        """gpu_name should come from experiment hardware, not hardcoded H100."""
        mock_get_cfg.return_value = MagicMock()
        mock_estimate.return_value = _FakePerformanceResult(
            ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
            output_throughput_tps=980.0, input_throughput_tps=2900.0,
            requests_per_sec=5.2, concurrency=9,
        )
        adapter = LLMOptimizerEstimateAdapter()
        exp = _make_experiment()
        exp.hardware = "A100-80GB"
        adapter.run(exp)
        _, kwargs = mock_estimate.call_args
        assert kwargs["gpu_name"] == "A100"
