# tests/test_llmservingsim_integration.py
"""Integration tests for LLMServingSim adapter (requires LLMServingSim installed)."""

import pytest
import os
import json
from experiment.adapters.llmservingsim import LLMServingSimAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)


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
            requests_per_sec=0.0
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


@pytest.mark.skipif(
    not os.path.exists("LLMServingSim/main.py"),
    reason="LLMServingSim not installed"
)
def test_e2e_llama_h100_tp1(tmp_path):
    """End-to-end test with real LLMServingSim on small workload"""
    # Create mock ground-truth data
    gt_dir = tmp_path / "llama-h100-tp1"
    results_dir = gt_dir / "results"
    results_dir.mkdir(parents=True)

    # Small workload (5 requests)
    metrics = [
        {"info": {"input_tokens": 10, "output_tokens": 5}},
        {"info": {"input_tokens": 12, "output_tokens": 6}},
        {"info": {"input_tokens": 11, "output_tokens": 5}},
        {"info": {"input_tokens": 13, "output_tokens": 7}},
        {"info": {"input_tokens": 10, "output_tokens": 6}},
    ]
    with open(results_dir / "per_request_lifecycle_metrics.json", "w") as f:
        json.dump(metrics, f)

    exp = _make_experiment(
        folder=str(gt_dir),
        profile_config={
            "load": {"stages": [{"rate": 5, "duration": 1}]}  # 5 req/s for 1s
        },
    )

    adapter = LLMServingSimAdapter("LLMServingSim")

    # Verify can_run
    assert adapter.can_run(exp) is True

    # Run simulation (this will take ~30 seconds)
    result = adapter.run(exp)

    # Sanity checks on results
    assert result.adapter_name == "llmservingsim"
    assert result.experiment_folder == str(gt_dir)
    assert result.summary.e2e.mean > 0
    assert result.summary.ttft.mean > 0
    assert result.summary.itl.mean > 0
    assert result.summary.ttft.mean < result.summary.e2e.mean
    assert result.summary.itl.mean < result.summary.ttft.mean
    assert len(result.stages) == 1
    assert result.stages[0].e2e.mean > 0
