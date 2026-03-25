"""Tests for experiment.adapters.llmservingsim."""

from __future__ import annotations

import json
import os

import pytest

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.adapters.llmservingsim import LLMServingSimAdapter, MODEL_MAP


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_experiment(**overrides):
    """Helper to create Experiment with all required fields."""
    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    zero_stage = StageMetrics(
        stage_index=-1,
        rate=0.0,
        duration=0.0,
        num_requests=0,
        e2e=zero_lat,
        ttft=zero_lat,
        itl=zero_lat,
        throughput=ThroughputMetrics(
            input_tokens_per_sec=0.0,
            output_tokens_per_sec=0.0,
            requests_per_sec=0.0,
        ),
    )
    defaults = {
        "folder": "dummy",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tp": 1,
        "workload": "general",
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "total_kv_blocks": 1000,
        "cpu_kv_blocks": 0,
        "stages": [],
        "summary": zero_stage,
        "profile_config": {"load": {"stages": []}},
        "hardware": "H100",
        "precision": "FP16",
        "dp": 1,
        "cpu_offload": False,
    }
    defaults.update(overrides)
    return Experiment(**defaults)


@pytest.fixture
def adapter(tmp_path):
    """Create adapter with mock LLMServingSim directory."""
    llm_dir = tmp_path / "LLMServingSim"
    llm_dir.mkdir()
    (llm_dir / "main.py").touch()

    # Create mock perf model directories
    perf_base = llm_dir / "llm_profile" / "perf_models" / "H100"
    (perf_base / "meta-llama/Llama-3.1-8B" / "tp1").mkdir(parents=True)
    (perf_base / "meta-llama/Llama-3.1-8B" / "tp2").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp1").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp2").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp4").mkdir(parents=True)

    # Create mock cluster config template
    config_dir = llm_dir / "cluster_config"
    config_dir.mkdir()
    template = {
        "nodes": [{
            "num_instances": 1,
            "instances": [{
                "model_name": "placeholder",
                "npu_num": 1,
                "npu_group": 1,
                "npu_mem": {
                    "mem_size": 40.0,
                    "mem_bw": 3350,
                    "mem_latency": 0,
                },
            }],
        }],
    }
    with open(config_dir / "single_node_single_instance_H100.json", "w") as f:
        json.dump(template, f)

    return LLMServingSimAdapter(str(llm_dir))


# ---------------------------------------------------------------------------
# Tests: MODEL_MAP constant
# ---------------------------------------------------------------------------


class TestModelMap:
    def test_model_map_llama(self):
        """Test Llama model mapping strips -Instruct suffix."""
        assert MODEL_MAP["meta-llama/Llama-3.1-8B-Instruct"] == "meta-llama/Llama-3.1-8B"

    def test_model_map_mixtral(self):
        """Test Mixtral model mapping remains unchanged."""
        assert MODEL_MAP["mistralai/Mixtral-8x7B-v0.1"] == "mistralai/Mixtral-8x7B-v0.1"

    def test_model_map_coverage(self):
        """Test MODEL_MAP contains exactly 2 supported models."""
        assert len(MODEL_MAP) == 2


# ---------------------------------------------------------------------------
# Tests: adapter basics
# ---------------------------------------------------------------------------


class TestLLMServingSimAdapterBasics:
    def test_name(self, tmp_path):
        """Adapter name should be 'llmservingsim'."""
        # Create a fake main.py so the constructor doesn't raise
        main_py = tmp_path / "main.py"
        main_py.write_text("")
        adapter = LLMServingSimAdapter(str(tmp_path))
        assert adapter.name == "llmservingsim"

    def test_init_stores_absolute_path(self, tmp_path):
        """Constructor should store absolute path to LLMServingSim dir."""
        main_py = tmp_path / "main.py"
        main_py.write_text("")
        adapter = LLMServingSimAdapter(str(tmp_path))
        assert os.path.isabs(adapter.llmservingsim_dir)

    def test_init_rejects_missing_main_py(self, tmp_path):
        """Constructor should raise ValueError if main.py not found."""
        with pytest.raises(ValueError, match="Invalid LLMServingSim directory"):
            LLMServingSimAdapter(str(tmp_path))


# ---------------------------------------------------------------------------
# Tests: can_run eligibility checking
# ---------------------------------------------------------------------------


class TestCanRun:
    def test_can_run_llama_h100_tp1(self, adapter):
        """Test can_run returns True for supported Llama config."""
        exp = _make_experiment(
            model="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            tp=1,
            precision="FP16",
        )
        assert adapter.can_run(exp) is True

    def test_can_run_mixtral_h100_tp2(self, adapter):
        """Test can_run returns True for supported Mixtral config."""
        exp = _make_experiment(
            model="mistralai/Mixtral-8x7B-v0.1",
            hardware="H100",
            tp=2,
            precision="FP16",
        )
        assert adapter.can_run(exp) is True

    def test_can_run_unsupported_hardware(self, adapter):
        """Test can_run returns False for non-H100 hardware."""
        exp = _make_experiment(hardware="A100-80GB")
        assert adapter.can_run(exp) is False

    def test_can_run_unsupported_model(self, adapter):
        """Test can_run returns False for unsupported model."""
        exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
        assert adapter.can_run(exp) is False

    def test_can_run_unsupported_precision(self, adapter):
        """Test can_run returns False for FP8 precision."""
        exp = _make_experiment(precision="FP8")
        assert adapter.can_run(exp) is False

    def test_can_run_missing_tp_config(self, adapter):
        """Test can_run returns False when TP config doesn't exist."""
        exp = _make_experiment(tp=8)  # tp8 doesn't exist in mock
        assert adapter.can_run(exp) is False


# ---------------------------------------------------------------------------
# Tests: cluster config generation
# ---------------------------------------------------------------------------


class TestGenerateClusterConfig:
    def test_generate_cluster_config_single_instance(self, adapter, tmp_path):
        """Test cluster config generation for single-instance (dp=1)"""
        exp = _make_experiment(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tp=2,
            dp=1,
        )

        output_path = tmp_path / "cluster.json"
        adapter._generate_cluster_config(exp, str(output_path))

        with open(output_path) as f:
            config = json.load(f)

        # Check model name
        assert config["nodes"][0]["instances"][0]["model_name"] == "meta-llama/Llama-3.1-8B"

        # Check TP config
        assert config["nodes"][0]["instances"][0]["npu_num"] == 2
        assert config["nodes"][0]["instances"][0]["npu_group"] == 1

        # Check GPU memory
        assert config["nodes"][0]["instances"][0]["npu_mem"]["mem_size"] == 80.0

        # Check single instance
        assert config["nodes"][0]["num_instances"] == 1
        assert len(config["nodes"][0]["instances"]) == 1

    def test_generate_cluster_config_multi_instance(self, adapter, tmp_path):
        """Test cluster config generation for multi-instance (dp>1)"""
        exp = _make_experiment(
            model="mistralai/Mixtral-8x7B-v0.1",
            tp=4,
            dp=2,
        )

        output_path = tmp_path / "cluster.json"
        adapter._generate_cluster_config(exp, str(output_path))

        with open(output_path) as f:
            config = json.load(f)

        # Check multi-instance setup
        assert config["nodes"][0]["num_instances"] == 2
        assert len(config["nodes"][0]["instances"]) == 2

        # Check both instances have correct config
        for instance in config["nodes"][0]["instances"]:
            assert instance["model_name"] == "mistralai/Mixtral-8x7B-v0.1"
            assert instance["npu_num"] == 4
            assert instance["npu_group"] == 1
            assert instance["npu_mem"]["mem_size"] == 80.0


# ---------------------------------------------------------------------------
# Tests: arrival generation
# ---------------------------------------------------------------------------


def test_generate_arrivals_single_stage():
    """Test constant-rate arrival times for single stage"""
    from experiment.adapters.llmservingsim import _generate_arrivals

    stages = [{"rate": 10, "duration": 5}]  # 10 req/s for 5 seconds
    arrivals = _generate_arrivals(stages)

    # Should have 50 requests
    assert len(arrivals) == 50

    # Check uniform spacing (0.1s = 100ms)
    for i in range(1, len(arrivals)):
        spacing = arrivals[i] - arrivals[i - 1]
        assert abs(spacing - 0.1) < 1e-9  # Constant 100ms spacing


def test_generate_arrivals_multi_stage():
    """Test constant-rate arrivals for multiple stages"""
    from experiment.adapters.llmservingsim import _generate_arrivals

    stages = [
        {"rate": 8, "duration": 2},   # 16 requests, 0-2s
        {"rate": 12, "duration": 3},  # 36 requests, 2-5s
    ]
    arrivals = _generate_arrivals(stages)

    assert len(arrivals) == 52

    # Check stage 1 arrivals (0-2s)
    stage1 = [a for a in arrivals if a < 2.0]
    assert len(stage1) == 16
    for i in range(1, len(stage1)):
        assert abs((stage1[i] - stage1[i - 1]) - 0.125) < 1e-9  # 1/8 = 0.125s

    # Check stage 2 arrivals (2-5s)
    stage2 = [a for a in arrivals if a >= 2.0]
    assert len(stage2) == 36
    for i in range(1, len(stage2)):
        assert abs((stage2[i] - stage2[i - 1]) - (1.0 / 12)) < 1e-9  # 1/12 = 0.0833s


# ---------------------------------------------------------------------------
# Tests: workload file generation
# ---------------------------------------------------------------------------


def test_generate_workload_file(adapter, tmp_path):
    """Test workload .jsonl generation from ground-truth"""
    # Create mock ground-truth data
    gt_dir = tmp_path / "test-exp"
    results_dir = gt_dir / "results"
    results_dir.mkdir(parents=True)

    metrics = [
        {"info": {"input_tokens": 100, "output_tokens": 50}},
        {"info": {"input_tokens": 120, "output_tokens": 60}},
        {"info": {"input_tokens": 110, "output_tokens": 55}},
    ]
    with open(results_dir / "per_request_lifecycle_metrics.json", "w") as f:
        json.dump(metrics, f)

    exp = _make_experiment(
        folder=str(gt_dir),
        profile_config={
            "load": {
                "stages": [{"rate": 1, "duration": 3}]  # 3 requests
            }
        },
    )

    output_path = tmp_path / "workload.jsonl"
    adapter._generate_workload(exp, str(output_path))

    # Read generated workload
    with open(output_path) as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 3

    # Check first request
    assert lines[0]["input_toks"] == 100
    assert lines[0]["output_toks"] == 50
    assert lines[0]["arrival_time_ns"] == 0
    assert len(lines[0]["input_tok_ids"]) == 100

    # Check arrivals are spaced at 1s intervals
    assert lines[1]["arrival_time_ns"] == 1_000_000_000
    assert lines[2]["arrival_time_ns"] == 2_000_000_000


# ---------------------------------------------------------------------------
# Tests: CLI arguments builder
# ---------------------------------------------------------------------------


def test_build_cli_args_single_instance(adapter):
    """Test CLI args for single-instance experiment"""
    exp = _make_experiment(
        profile_config={
            "load": {"stages": [{"rate": 10, "duration": 5}]}
        },
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        dp=1,
    )

    args = adapter._build_cli_args(
        exp,
        cluster_config="/path/to/cluster.json",
        workload="/path/to/workload.jsonl",
        output="/path/to/output.csv",
    )

    assert args[0:2] == ["python", "main.py"]
    assert "--cluster-config" in args
    assert "/path/to/cluster.json" in args
    assert "--dataset" in args
    assert "/path/to/workload.jsonl" in args
    assert "--output" in args
    assert "/path/to/output.csv" in args
    assert "--fp" in args
    assert "16" in args
    assert "--block-size" in args
    assert "16" in args
    assert "--max-batch" in args
    assert "256" in args
    assert "--max-num-batched-tokens" in args
    assert "8192" in args
    assert "--num-req" in args
    assert "50" in args  # 10 req/s * 5s = 50

    # Should NOT have routing policy for single instance
    assert "--request-routing-policy" not in args


def test_build_cli_args_multi_instance(adapter):
    """Test CLI args for multi-instance experiment"""
    exp = _make_experiment(
        profile_config={
            "load": {"stages": [{"rate": 8, "duration": 3}]}
        },
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        dp=3,
    )

    args = adapter._build_cli_args(
        exp,
        cluster_config="/path/to/cluster.json",
        workload="/path/to/workload.jsonl",
        output="/path/to/output.csv",
    )

    # Should have routing policy for multi-instance
    assert "--request-routing-policy" in args
    assert "RR" in args
