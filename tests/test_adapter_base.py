"""Tests for experiment.adapters.base — SimulatorAdapter ABC and BaseBLISAdapter."""

import json
import os

import pytest

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.adapters.base import BaseBLISAdapter, SimulatorAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment(**overrides):
    """Build a minimal Experiment for testing."""
    defaults = dict(
        folder="/tmp/test-exp",
        model="meta-llama/Llama-2-7b-hf",
        tp=1,
        workload="codegen",
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        total_kv_blocks=7463,
        cpu_kv_blocks=50,
        stages=[],
        summary=StageMetrics(
            stage_index=-1, rate=0, duration=0, num_requests=0,
            e2e=LatencyDistribution(0, 0, 0),
            ttft=LatencyDistribution(0, 0, 0),
            itl=LatencyDistribution(0, 0, 0),
            throughput=ThroughputMetrics(0, 0, 0),
        ),
        profile_config={
            "load": {"stages": [{"duration": 600, "rate": 5}, {"duration": 600, "rate": 10}]}
        },
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def _make_blis_output(num_requests=10, stage_boundary=600.0):
    """Build synthetic BLIS JSON output with per-request data."""
    requests = []
    for i in range(num_requests):
        # First half arrive before boundary, second half after
        arrived_at = (i / num_requests) * 2 * stage_boundary
        requests.append({
            "arrived_at": arrived_at,
            "ttft_ms": 25.0 + i,
            "itl_ms": 3.6 + i * 0.1,
            "e2e_ms": 1800.0 + i * 10,
            "num_prefill_tokens": 566,
            "num_decode_tokens": 247,
        })
    return {
        "e2e_mean_ms": 1845.0,
        "e2e_p90_ms": 1890.0,
        "e2e_p99_ms": 1950.0,
        "ttft_mean_ms": 29.5,
        "ttft_p90_ms": 33.0,
        "ttft_p99_ms": 34.0,
        "itl_mean_ms": 4.05,
        "itl_p90_ms": 4.5,
        "itl_p99_ms": 4.9,
        "responses_per_sec": 5.0,
        "tokens_per_sec": 1235.0,
        "total_input_tokens": 5660,
        "total_output_tokens": 2470,
        "completed_requests": num_requests,
        "requests": requests,
    }


# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseBLISAdapter is abstract)
# ---------------------------------------------------------------------------

class _ConcreteBLISAdapter(BaseBLISAdapter):
    @property
    def name(self) -> str:
        return "test-blis"

    def run(self, experiment):
        raise NotImplementedError("Not used in unit tests")


# ---------------------------------------------------------------------------
# Tests: SimulatorAdapter ABC
# ---------------------------------------------------------------------------

class TestSimulatorAdapterABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SimulatorAdapter()

    def test_can_run_default_true(self):
        class Dummy(SimulatorAdapter):
            @property
            def name(self):
                return "dummy"
            def run(self, experiment):
                pass

        adapter = Dummy()
        exp = _make_experiment()
        assert adapter.can_run(exp) is True


# ---------------------------------------------------------------------------
# Tests: BaseBLISAdapter._build_common_args
# ---------------------------------------------------------------------------

class TestBuildCommonArgs:
    def test_args_include_model_and_tp(self):
        adapter = _ConcreteBLISAdapter(blis_binary="/usr/local/bin/blis")
        exp = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)

        args = adapter._build_common_args(exp, trace_spec="/tmp/spec.yaml", results_path="/tmp/out.json")

        assert "/usr/local/bin/blis" in args
        assert "run" in args
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "meta-llama/Llama-2-7b-hf"

    def test_args_include_kv_offloading_flags(self):
        adapter = _ConcreteBLISAdapter(blis_binary="/usr/local/bin/blis")
        exp = _make_experiment(total_kv_blocks=7463, cpu_kv_blocks=50)

        args = adapter._build_common_args(exp, trace_spec="/tmp/spec.yaml", results_path="/tmp/out.json")

        assert "--total-kv-blocks" in args
        idx = args.index("--total-kv-blocks")
        assert args[idx + 1] == "7463"

        assert "--kv-cpu-blocks" in args
        idx = args.index("--kv-cpu-blocks")
        assert args[idx + 1] == "50"

        assert "--kv-offload-threshold" in args
        assert "--kv-transfer-bandwidth" in args

    def test_args_include_scheduler_limits(self):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        exp = _make_experiment(max_num_seqs=128, max_num_batched_tokens=2048)

        args = adapter._build_common_args(exp, trace_spec="/tmp/spec.yaml", results_path="/tmp/out.json")

        idx = args.index("--max-num-running-reqs")
        assert args[idx + 1] == "128"

        idx = args.index("--max-num-scheduled-tokens")
        assert args[idx + 1] == "2048"

    def test_args_include_workload_spec(self):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        exp = _make_experiment()

        args = adapter._build_common_args(exp, trace_spec="/tmp/trace/spec.yaml", results_path="/tmp/out.json")

        idx = args.index("--workload-spec")
        assert args[idx + 1] == "/tmp/trace/spec.yaml"


# ---------------------------------------------------------------------------
# Tests: BaseBLISAdapter._split_requests_by_stage
# ---------------------------------------------------------------------------

class TestSplitRequestsByStage:
    def test_splits_by_cumulative_boundaries(self):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        stages_config = [{"duration": 600, "rate": 5}, {"duration": 600, "rate": 10}]

        requests = [
            {"arrived_at": 0.0},
            {"arrived_at": 300.0},
            {"arrived_at": 599.9},
            {"arrived_at": 600.0},   # at boundary → stage 0
            {"arrived_at": 600.1},
            {"arrived_at": 1100.0},
            {"arrived_at": 1200.0},  # at boundary → stage 1
        ]

        buckets = adapter._split_requests_by_stage(requests, stages_config)

        assert len(buckets) == 2
        # First 4 requests (arrived_at <= 600) → stage 0
        assert len(buckets[0]) == 4
        # Last 3 requests (600 < arrived_at <= 1200) → stage 1
        assert len(buckets[1]) == 3

    def test_late_arrivals_go_to_last_stage(self):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        stages_config = [{"duration": 100, "rate": 5}]

        requests = [
            {"arrived_at": 50.0},
            {"arrived_at": 200.0},  # past boundary but last stage
        ]

        buckets = adapter._split_requests_by_stage(requests, stages_config)
        assert len(buckets) == 1
        assert len(buckets[0]) == 2


# ---------------------------------------------------------------------------
# Tests: BaseBLISAdapter._parse_blis_results
# ---------------------------------------------------------------------------

class TestParseBLISResults:
    def test_parses_aggregate_summary(self, tmp_path):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        blis_output = _make_blis_output(num_requests=10, stage_boundary=600.0)
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as fh:
            json.dump(blis_output, fh)

        exp = _make_experiment()
        result = adapter._parse_blis_results(results_path, exp)

        assert isinstance(result, SimulatorResult)
        assert result.adapter_name == "test-blis"
        # Summary from top-level keys
        assert abs(result.summary.e2e.mean - 1845.0) < 0.01
        assert abs(result.summary.ttft.mean - 29.5) < 0.01
        assert abs(result.summary.itl.mean - 4.05) < 0.01

    def test_parses_per_stage_metrics(self, tmp_path):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        blis_output = _make_blis_output(num_requests=10, stage_boundary=600.0)
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as fh:
            json.dump(blis_output, fh)

        exp = _make_experiment()
        result = adapter._parse_blis_results(results_path, exp)

        # 10 requests spread over 2 stages
        assert len(result.stages) == 2
        # Each stage should have valid metrics
        for stage in result.stages:
            assert stage.e2e.mean > 0
            assert stage.ttft.mean > 0
            assert stage.itl.mean > 0

    def test_empty_requests_handled(self, tmp_path):
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        blis_output = _make_blis_output(num_requests=0, stage_boundary=600.0)
        blis_output["requests"] = []
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as fh:
            json.dump(blis_output, fh)

        exp = _make_experiment()
        result = adapter._parse_blis_results(results_path, exp)

        # Should still produce 2 stages (with zeros for empty buckets)
        assert len(result.stages) == 2

    def test_summary_throughput_computed(self, tmp_path):
        """Summary input_tokens_per_sec = total_input_tokens / total_duration."""
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        blis_output = _make_blis_output(num_requests=10, stage_boundary=600.0)
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as fh:
            json.dump(blis_output, fh)

        exp = _make_experiment()
        result = adapter._parse_blis_results(results_path, exp)

        # total_input_tokens=5660, total_duration=600+600=1200
        expected_input_tps = 5660 / 1200.0
        assert abs(result.summary.throughput.input_tokens_per_sec - expected_input_tps) < 0.01
        assert abs(result.summary.throughput.output_tokens_per_sec - 1235.0) < 0.01
        assert abs(result.summary.throughput.requests_per_sec - 5.0) < 0.01

    def test_malformed_requests_filtered(self, tmp_path):
        """Requests missing required keys are filtered out, not crashing."""
        adapter = _ConcreteBLISAdapter(blis_binary="blis")
        blis_output = _make_blis_output(num_requests=0, stage_boundary=600.0)
        blis_output["requests"] = [
            # Valid request
            {
                "arrived_at": 100.0, "e2e_ms": 1800.0, "ttft_ms": 25.0,
                "itl_ms": 3.6, "num_prefill_tokens": 566, "num_decode_tokens": 247,
            },
            # Malformed — missing e2e_ms
            {"arrived_at": 200.0, "ttft_ms": 25.0, "itl_ms": 3.6},
            # Malformed — empty dict
            {},
        ]
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as fh:
            json.dump(blis_output, fh)

        exp = _make_experiment()
        result = adapter._parse_blis_results(results_path, exp)

        # Only 1 valid request should be counted in stage 0
        assert result.stages[0].num_requests == 1
        assert result.stages[0].e2e.mean == 1800.0
