"""Tests for experiment.ground_truth — discovery and parsing of ground-truth experiments."""

import json
import os
import textwrap

import pytest
import yaml

from experiment.data_model import Experiment, LatencyDistribution, StageMetrics, ThroughputMetrics
from experiment.ground_truth import (
    _discover_experiments_legacy,
    discover_experiments,
    parse_experiment,
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic experiment directory
# ---------------------------------------------------------------------------

def _make_exp_dir(
    tmp_path,
    folder_name="20260217-155451-llama-2-7b-tp1-codegen",
    model="meta-llama/Llama-2-7b-hf",
    tp=1,
    max_model_len=4096,
    max_num_batched_tokens=2048,
    max_num_seqs=128,
    stages_config=None,
    num_stages=2,
    vllm_log_tokens=119408,
    kv_events_lines=None,
):
    """Create a synthetic experiment directory that mirrors the real structure."""
    exp_dir = tmp_path / folder_name
    exp_dir.mkdir()

    # exp-config.yaml
    exp_config = {
        "model": model,
        "tensor_parallelism": tp,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "app": "inference-perf",
    }
    (exp_dir / "exp-config.yaml").write_text(yaml.dump(exp_config))

    # profile.yaml (single-line JSON style)
    if stages_config is None:
        stages_config = [{"duration": 600, "rate": 5}, {"duration": 600, "rate": 10}]
    profile = {
        "api": {"streaming": True, "type": "completion"},
        "data": {
            "shared_prefix": {
                "question_len": 466,
                "system_prompt_len": 100,
                "output_len": 247,
                "enable_multi_turn_chat": True,
                "num_unique_system_prompts": 11,
                "num_users_per_system_prompt": 4,
            },
            "type": "shared_prefix",
        },
        "load": {"stages": stages_config, "type": "constant"},
        "report": {"request_lifecycle": {"per_request": True, "per_stage": True, "summary": True}},
        "server": {"base_url": "http://10.128.6.76:8000", "model_name": model, "type": "vllm"},
        "storage": {"local_storage": {"path": "/workspace/data/results"}},
        "tokenizer": {"pretrained_model_name_or_path": model},
    }
    (exp_dir / "profile.yaml").write_text(json.dumps(profile))

    # vllm.log
    vllm_log = (
        "INFO 2026-02-17 some line\n"
        f"INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: {vllm_log_tokens:,} tokens\n"
        "INFO 2026-02-17 another line\n"
    )
    (exp_dir / "vllm.log").write_text(vllm_log)

    # kv_events.jsonl
    if kv_events_lines is None:
        kv_events_lines = [
            json.dumps([0.1, [["TransferCompleted", 1, "r1", "GPU", "CPU", 5, True, 0]], {}, {}]),
            json.dumps([0.2, [["TransferCompleted", 2, "r1", "CPU", "GPU", 2, True, 1]], {}, {}]),
        ]
    (exp_dir / "kv_events.jsonl").write_text("\n".join(kv_events_lines) + "\n")

    # inference-perf-data/
    perf_dir = exp_dir / "inference-perf-data"
    perf_dir.mkdir()

    # Stage lifecycle metrics
    for i in range(num_stages):
        stage_data = _make_stage_metrics(
            count=3000 * (i + 1),
            rate=stages_config[i]["rate"],
            duration=stages_config[i]["duration"],
            e2e_mean=1.8 + i * 0.3,
            ttft_mean=0.025 + i * 0.005,
            itl_mean=0.0036 + i * 0.001,
        )
        (perf_dir / f"stage_{i}_lifecycle_metrics.json").write_text(json.dumps(stage_data, indent=2))

    # Summary lifecycle metrics (no send_duration / requested_rate)
    summary_data = _make_stage_metrics(
        count=9000,
        rate=None,  # summary lacks rate
        duration=None,  # summary lacks duration
        e2e_mean=2.08,
        ttft_mean=0.028,
        itl_mean=0.0041,
        is_summary=True,
    )
    (perf_dir / "summary_lifecycle_metrics.json").write_text(json.dumps(summary_data, indent=2))

    return str(exp_dir)


def _make_stage_metrics(count, rate, duration, e2e_mean, ttft_mean, itl_mean, is_summary=False):
    """Build a minimal stage/summary lifecycle metrics JSON structure.

    ``count`` is the number of *successful* requests.  ``load_summary.count``
    is deliberately set higher to verify that parsing uses ``successes.count``.
    """
    load_summary = {"count": count + 100}  # total sent (includes failures)
    if not is_summary:
        load_summary["send_duration"] = duration
        load_summary["requested_rate"] = rate
        load_summary["achieved_rate"] = rate * 0.999
    load_summary["schedule_delay"] = {
        "mean": 0.0004, "min": -0.001, "p0.1": -0.001, "p1": -0.0008,
        "p5": -0.0005, "p10": -0.0003, "p25": 5e-05, "median": 0.0004,
        "p75": 0.0008, "p90": 0.0012, "p95": 0.0014, "p99": 0.0017,
        "p99.9": 0.002, "max": 0.007,
    }

    def _latency_dist(mean_val):
        return {
            "mean": mean_val, "min": mean_val * 0.85,
            "p0.1": mean_val * 0.86, "p1": mean_val * 0.88,
            "p5": mean_val * 0.91, "p10": mean_val * 0.93,
            "p25": mean_val * 0.96, "median": mean_val * 0.99,
            "p75": mean_val * 1.04, "p90": mean_val * 1.07,
            "p95": mean_val * 1.10, "p99": mean_val * 1.21,
            "p99.9": mean_val * 1.26, "max": mean_val * 1.27,
        }

    return {
        "load_summary": load_summary,
        "successes": {
            "count": count,
            "latency": {
                "request_latency": _latency_dist(e2e_mean),
                "time_to_first_token": _latency_dist(ttft_mean),
                "inter_token_latency": _latency_dist(itl_mean),
            },
            "throughput": {
                "input_tokens_per_sec": 2950.0,
                "output_tokens_per_sec": 966.0,
                "total_tokens_per_sec": 3916.0,
                "requests_per_sec": 5.0,
            },
        },
        "failures": {"count": 0, "request_latency": None, "prompt_len": None},
    }


# ---------------------------------------------------------------------------
# Tests: discover_experiments
# ---------------------------------------------------------------------------

class TestDiscoverExperiments:
    """Tests for the legacy regex-based discovery (now _discover_experiments_legacy)."""

    def test_discovers_matching_dirs(self, tmp_path):
        (tmp_path / "20260217-155451-llama-2-7b-tp1-codegen").mkdir()
        (tmp_path / "20260218-120914-mixtral-8x7b-v0-1-tp2-codegen").mkdir()
        (tmp_path / "SCHEMA.md").write_text("schema docs")
        (tmp_path / "random_file.txt").write_text("noise")

        result = _discover_experiments_legacy(str(tmp_path))

        assert len(result) == 2
        assert all(os.path.isabs(p) for p in result)
        # Sorted
        assert "155451" in result[0]
        assert "120914" in result[1]

    def test_returns_empty_for_no_matches(self, tmp_path):
        (tmp_path / "SCHEMA.md").write_text("schema docs")
        result = _discover_experiments_legacy(str(tmp_path))
        assert result == []

    def test_excludes_non_directory(self, tmp_path):
        (tmp_path / "20260217-155451-llama-2-7b-tp1-codegen").write_text("file, not dir")
        result = _discover_experiments_legacy(str(tmp_path))
        assert result == []


# ---------------------------------------------------------------------------
# Tests: parse_experiment
# ---------------------------------------------------------------------------

class TestParseExperiment:
    def test_parses_basic_fields(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        assert exp.model == "meta-llama/Llama-2-7b-hf"
        assert exp.tp == 1
        assert exp.workload == "codegen"
        assert exp.max_model_len == 4096
        assert exp.max_num_batched_tokens == 2048
        assert exp.max_num_seqs == 128

    def test_extracts_workload_from_folder_name(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path, folder_name="20260217-202857-llama-2-70b-tp4-general")
        exp = parse_experiment(exp_path)
        assert exp.workload == "general"

    def test_extracts_workload_roleplay(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path, folder_name="20260217-162547-llama-2-7b-tp1-roleplay")
        exp = parse_experiment(exp_path)
        assert exp.workload == "roleplay"

    def test_kv_blocks_extracted(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path, vllm_log_tokens=119408)
        exp = parse_experiment(exp_path)
        assert exp.total_kv_blocks == 119408 // 16  # 7463

    def test_cpu_kv_blocks_extracted(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)
        # Synthetic: 5 GPU→CPU, then 2 CPU→GPU → peak = 5, final = 3
        assert exp.cpu_kv_blocks == 5

    def test_stages_parsed(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        assert len(exp.stages) == 2

        s0 = exp.stages[0]
        assert s0.stage_index == 0
        assert s0.rate == 5.0
        assert s0.duration == 600.0
        assert s0.num_requests == 3000

    def test_latency_converted_to_ms(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        s0 = exp.stages[0]
        # The synthetic e2e_mean was 1.8 seconds → should be 1800.0 ms
        assert abs(s0.e2e.mean - 1800.0) < 0.01
        # p90 = 1.8 * 1.07 seconds = 1.926 → 1926.0 ms
        assert abs(s0.e2e.p90 - 1800.0 * 1.07) < 0.01
        # p99 = 1.8 * 1.21 seconds → ms
        assert abs(s0.e2e.p99 - 1800.0 * 1.21) < 0.01

    def test_ttft_converted_to_ms(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        s0 = exp.stages[0]
        # ttft_mean 0.025s → 25.0 ms
        assert abs(s0.ttft.mean - 25.0) < 0.01

    def test_itl_converted_to_ms(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        s0 = exp.stages[0]
        # itl_mean 0.0036s → 3.6 ms
        assert abs(s0.itl.mean - 3.6) < 0.01

    def test_summary_parsed(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        assert exp.summary.stage_index == -1
        assert exp.summary.rate == 0.0
        assert exp.summary.duration == 0.0
        assert exp.summary.num_requests == 9000
        # e2e_mean 2.08s → 2080.0 ms
        assert abs(exp.summary.e2e.mean - 2080.0) < 0.01

    def test_throughput_parsed(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        s0 = exp.stages[0]
        assert abs(s0.throughput.input_tokens_per_sec - 2950.0) < 0.01
        assert abs(s0.throughput.output_tokens_per_sec - 966.0) < 0.01
        assert abs(s0.throughput.requests_per_sec - 5.0) < 0.01

    def test_profile_config_preserved(self, tmp_path):
        exp_path = _make_exp_dir(tmp_path)
        exp = parse_experiment(exp_path)

        assert "load" in exp.profile_config
        assert exp.profile_config["load"]["stages"][0]["rate"] == 5


# ---------------------------------------------------------------------------
# Tests: manifest-driven discover_experiments
# ---------------------------------------------------------------------------

class TestManifestDiscovery:
    """Tests for the new manifest-driven discover_experiments."""

    def test_discovers_safe_experiments(self, tmp_path):
        """Only safe experiments with directories should be returned."""
        manifest = [
            {"id": 13, "model": "Qwen3-14B", "precision": "FP16", "hw": "H100",
             "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "safe": "safe", "done": True, "notes": ""},
            {"id": 1, "model": "Codellama-34b", "precision": "FP16", "hw": "H100",
             "workload": "general", "mbt": 2048, "cpu_offload": True,
             "gpu_mem": 0.9, "tp": 2, "dp": None, "safe": "unsafe", "done": True, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "13-qwen3-14b-tp1-general").mkdir()
        (tmp_path / "1-codellama-34b-tp2-general").mkdir()

        result = discover_experiments(str(tmp_path))
        assert len(result) == 1
        entry, path = result[0]
        assert entry["id"] == 13
        assert "13-qwen3-14b" in path

    def test_discovers_all_when_safe_only_false(self, tmp_path):
        """safe_only=False returns all experiments with directories."""
        manifest = [
            {"id": 1, "safe": "unsafe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
            {"id": 2, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "codegen", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "1-model-tp1-general").mkdir()
        (tmp_path / "2-model-tp1-codegen").mkdir()

        result = discover_experiments(str(tmp_path), safe_only=False)
        assert len(result) == 2

    def test_skips_missing_directories(self, tmp_path):
        """Experiments without directories are skipped with warning."""
        manifest = [
            {"id": 47, "safe": "safe", "done": False, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        # No directory for id=47

        result = discover_experiments(str(tmp_path))
        assert len(result) == 0

    def test_sorted_by_id(self, tmp_path):
        """Results are sorted by experiment id."""
        manifest = [
            {"id": 20, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "a", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
            {"id": 3, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "b", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "20-model-tp1-a").mkdir()
        (tmp_path / "3-model-tp1-b").mkdir()

        result = discover_experiments(str(tmp_path))
        assert result[0][0]["id"] == 3
        assert result[1][0]["id"] == 20
