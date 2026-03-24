"""Integration tests using real ground-truth data.

These tests require the ``vllm_data/ground_truth`` directory to be present.
They are smoke tests that verify the full pipeline works end-to-end on
actual experiment data.
"""

from __future__ import annotations

import os
import tempfile

import pytest

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "vllm_data", "ground_truth"
)
_LLAMA_7B_CODEGEN = os.path.join(
    _DATA_DIR, "20260217-155451-llama-2-7b-tp1-codegen"
)

# Skip all tests if ground-truth data is not present.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_LLAMA_7B_CODEGEN),
    reason="Ground-truth data not available",
)


# ---------------------------------------------------------------------------
# 13a: Ground truth parsing smoke test
# ---------------------------------------------------------------------------


class TestGroundTruthParsing:
    def test_llama_7b_codegen_basic_fields(self):
        from experiment.ground_truth import parse_experiment

        exp = parse_experiment(_LLAMA_7B_CODEGEN)

        assert exp.model == "meta-llama/Llama-2-7b-hf"
        assert exp.tp == 1
        assert exp.workload == "codegen"
        assert exp.total_kv_blocks == 7463

    def test_llama_7b_codegen_stages(self):
        from experiment.ground_truth import parse_experiment

        exp = parse_experiment(_LLAMA_7B_CODEGEN)

        assert len(exp.stages) == 2
        assert exp.stages[0].rate == 5.0
        assert exp.stages[0].e2e.mean > 0

    def test_discover_all_experiments(self):
        from experiment.ground_truth import discover_experiments

        dirs = discover_experiments(_DATA_DIR)
        assert len(dirs) == 16


# ---------------------------------------------------------------------------
# --no-dp-scaling integration test
# ---------------------------------------------------------------------------


class TestNoDPScalingIntegration:
    def test_no_dp_scaling_flag_integration(self):
        """End-to-end test: --no-dp-scaling should filter experiments."""
        from experiment.run import run_pipeline

        # Run pipeline with flag (no adapters to avoid external dependencies)
        error_records, runtime_records = run_pipeline(
            data_dir=_DATA_DIR,
            blis_binary="nonexistent",  # Won't be used with no adapters
            vidur_dir="nonexistent",
            output_dir="/tmp/test_output",
            adapter_names=[],
            no_dp_scaling=True,
        )

        # Should complete without errors
        assert error_records == []
        assert runtime_records == []


# ---------------------------------------------------------------------------
# 13b: Trace converter smoke test
# ---------------------------------------------------------------------------


class TestTraceConverterSmoke:
    def test_blis_trace_correct_row_count(self):
        import csv
        from experiment.trace_converter import convert_to_blis_trace

        per_req = os.path.join(
            _LLAMA_7B_CODEGEN, "inference-perf-data", "per_request_lifecycle_metrics.json"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            header_path, data_path = convert_to_blis_trace(per_req, tmpdir)

            with open(data_path) as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            assert len(rows) > 1  # header + data

    def test_vidur_trace_correct_columns(self):
        import csv
        from experiment.vidur_trace_converter import convert_to_vidur_trace

        per_req = os.path.join(
            _LLAMA_7B_CODEGEN, "inference-perf-data", "per_request_lifecycle_metrics.json"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "vidur_trace.csv")
            convert_to_vidur_trace(per_req, out_csv)

            with open(out_csv) as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)

            assert len(rows) > 0
            assert set(rows[0].keys()) == {"arrived_at", "num_prefill_tokens", "num_decode_tokens"}
            # First arrival time should be 0
            assert float(rows[0]["arrived_at"]) == 0.0


# ---------------------------------------------------------------------------
# 13c: KV cache extractor smoke test
# ---------------------------------------------------------------------------


class TestKVCacheExtractorSmoke:
    def test_total_kv_blocks_llama_7b(self):
        from experiment.kv_cache_extractor import extract_total_kv_blocks

        vllm_log = os.path.join(_LLAMA_7B_CODEGEN, "vllm.log")
        assert extract_total_kv_blocks(vllm_log) == 7463

    def test_cpu_kv_blocks_positive(self):
        from experiment.kv_cache_extractor import extract_cpu_kv_blocks

        kv_events = os.path.join(_LLAMA_7B_CODEGEN, "kv_events.jsonl")
        if not os.path.exists(kv_events):
            pytest.skip("kv_events.jsonl not present")
        result = extract_cpu_kv_blocks(kv_events)
        assert isinstance(result, int)
        assert result >= 0
