"""Tests for experiment.metrics."""

from __future__ import annotations

import pytest

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.metrics import (
    ErrorRecord,
    RuntimeRecord,
    compute_absolute_error,
    compute_errors,
    compute_mape,
    compute_mpe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lat(mean, p90, p99):
    return LatencyDistribution(mean=mean, p90=p90, p99=p99)


def _make_stage(index, e2e_mean, ttft_mean, itl_mean):
    return StageMetrics(
        stage_index=index,
        rate=5.0,
        duration=600.0,
        num_requests=3000,
        e2e=_make_lat(e2e_mean, e2e_mean * 1.1, e2e_mean * 1.3),
        ttft=_make_lat(ttft_mean, ttft_mean * 1.1, ttft_mean * 1.3),
        itl=_make_lat(itl_mean, itl_mean * 1.1, itl_mean * 1.3),
        throughput=ThroughputMetrics(100.0, 50.0, 5.0),
    )


# ---------------------------------------------------------------------------
# Tests: compute_mape
# ---------------------------------------------------------------------------


class TestRuntimeRecord:
    def test_constructable(self):
        rec = RuntimeRecord(
            simulator="blis-roofline",
            experiment_folder="/tmp/exp",
            model="llama-7b",
            workload="codegen",
            wall_clock_seconds=3.14,
        )
        assert rec.simulator == "blis-roofline"
        assert rec.wall_clock_seconds == 3.14

    def test_fields_accessible(self):
        rec = RuntimeRecord("sim", "/exp", "model", "wl", 1.0)
        assert rec.experiment_folder == "/exp"
        assert rec.model == "model"
        assert rec.workload == "wl"


class TestComputeMAPE:
    def test_exact_match(self):
        assert compute_mape(100.0, 100.0) == 0.0

    def test_10_percent_over(self):
        assert abs(compute_mape(110.0, 100.0) - 10.0) < 0.01

    def test_10_percent_under(self):
        assert abs(compute_mape(90.0, 100.0) - 10.0) < 0.01

    def test_actual_zero_predicted_nonzero_returns_inf(self):
        assert compute_mape(42.0, 0.0) == float("inf")

    def test_both_zero(self):
        assert compute_mape(0.0, 0.0) == 0.0

    def test_large_error(self):
        assert abs(compute_mape(200.0, 100.0) - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: compute_mpe
# ---------------------------------------------------------------------------


class TestComputeMPE:
    def test_exact_match(self):
        assert compute_mpe(100.0, 100.0) == 0.0

    def test_positive_when_over(self):
        result = compute_mpe(110.0, 100.0)
        assert result > 0
        assert abs(result - 10.0) < 0.01

    def test_negative_when_under(self):
        result = compute_mpe(90.0, 100.0)
        assert result < 0
        assert abs(result - (-10.0)) < 0.01

    def test_actual_zero_predicted_positive_returns_inf(self):
        assert compute_mpe(42.0, 0.0) == float("inf")

    def test_actual_zero_predicted_negative_returns_neg_inf(self):
        assert compute_mpe(-42.0, 0.0) == float("-inf")

    def test_both_zero_returns_zero(self):
        assert compute_mpe(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_absolute_error
# ---------------------------------------------------------------------------


class TestComputeAbsoluteError:
    def test_exact_match(self):
        assert compute_absolute_error(100.0, 100.0) == 0.0

    def test_positive_difference(self):
        assert compute_absolute_error(110.0, 100.0) == 10.0

    def test_negative_difference(self):
        assert compute_absolute_error(90.0, 100.0) == 10.0


# ---------------------------------------------------------------------------
# Tests: compute_errors
# ---------------------------------------------------------------------------


class TestComputeErrors:
    def test_produces_records_for_all_metrics(self):
        """One stage + summary = 2 stages × 9 metrics = 18 records."""
        gt_stage = _make_stage(0, 1800.0, 25.0, 3.6)
        gt_summary = _make_stage(-1, 1800.0, 25.0, 3.6)
        experiment = Experiment(
            folder="/tmp/exp",
            model="meta-llama/Llama-2-7b-hf",
            tp=1,
            workload="codegen",
            max_model_len=4096,
            max_num_batched_tokens=2048,
            max_num_seqs=128,
            total_kv_blocks=7463,
            cpu_kv_blocks=5,
            stages=[gt_stage],
            summary=gt_summary,
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )

        pred_stage = _make_stage(0, 1900.0, 27.0, 4.0)
        pred_summary = _make_stage(-1, 1900.0, 27.0, 4.0)
        result = SimulatorResult(
            adapter_name="test-sim",
            experiment_folder="/tmp/exp",
            stages=[pred_stage],
            summary=pred_summary,
        )

        records = compute_errors(experiment, result)
        assert len(records) == 18  # 9 metrics × 2 (stage + summary)

    def test_metric_names_present(self):
        gt_stage = _make_stage(0, 1800.0, 25.0, 3.6)
        gt_summary = _make_stage(-1, 1800.0, 25.0, 3.6)
        experiment = Experiment(
            folder="/tmp/exp", model="m", tp=1, workload="w",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt_stage], summary=gt_summary,
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        result = SimulatorResult(
            adapter_name="s", experiment_folder="/tmp/exp",
            stages=[_make_stage(0, 1800.0, 25.0, 3.6)],
            summary=_make_stage(-1, 1800.0, 25.0, 3.6),
        )

        records = compute_errors(experiment, result)
        names = {r.metric_name for r in records}
        expected = {
            "e2e_mean", "e2e_p90", "e2e_p99",
            "ttft_mean", "ttft_p90", "ttft_p99",
            "itl_mean", "itl_p90", "itl_p99",
        }
        assert names == expected

    def test_mape_values_correct(self):
        """Known: pred=1900, actual=1800 → MAPE ≈ 5.56%."""
        gt_stage = _make_stage(0, 1800.0, 25.0, 3.6)
        experiment = Experiment(
            folder="/tmp/exp", model="m", tp=1, workload="w",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt_stage], summary=_make_stage(-1, 0, 0, 0),
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        result = SimulatorResult(
            adapter_name="s", experiment_folder="/tmp/exp",
            stages=[_make_stage(0, 1900.0, 25.0, 3.6)],
            summary=_make_stage(-1, 0, 0, 0),
        )

        records = compute_errors(experiment, result)
        e2e_mean_rec = next(
            r for r in records if r.metric_name == "e2e_mean" and r.stage_index == 0
        )
        assert abs(e2e_mean_rec.mape - 5.556) < 0.01
        assert abs(e2e_mean_rec.mpe - 5.556) < 0.01
        assert abs(e2e_mean_rec.absolute_error - 100.0) < 0.01

    def test_experiment_metadata_propagated(self):
        gt = _make_stage(0, 100, 10, 1)
        experiment = Experiment(
            folder="/data/exp-1", model="llama-7b", tp=2, workload="codegen",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt], summary=_make_stage(-1, 0, 0, 0),
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        result = SimulatorResult(
            adapter_name="blis-roofline", experiment_folder="/data/exp-1",
            stages=[_make_stage(0, 110, 11, 1.1)],
            summary=_make_stage(-1, 0, 0, 0),
        )

        records = compute_errors(experiment, result)
        for r in records:
            assert r.simulator == "blis-roofline"
            assert r.model == "llama-7b"
            assert r.workload == "codegen"

    def test_summary_with_nontrivial_values(self):
        """T2: Summary comparison produces correct MAPE for non-zero values."""
        gt_stage = _make_stage(0, 1800.0, 25.0, 3.6)
        gt_summary = _make_stage(-1, 1800.0, 25.0, 3.6)
        experiment = Experiment(
            folder="/tmp/exp", model="m", tp=1, workload="w",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt_stage], summary=gt_summary,
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        pred_stage = _make_stage(0, 1800.0, 25.0, 3.6)  # exact match
        pred_summary = _make_stage(-1, 2000.0, 30.0, 4.0)  # different
        result = SimulatorResult(
            adapter_name="s", experiment_folder="/tmp/exp",
            stages=[pred_stage], summary=pred_summary,
        )

        records = compute_errors(experiment, result)
        summary_records = [r for r in records if r.stage_index == -1]
        assert len(summary_records) == 9
        e2e_summary = next(r for r in summary_records if r.metric_name == "e2e_mean")
        # 2000 vs 1800 → MAPE = 200/1800 * 100 ≈ 11.11%
        assert abs(e2e_summary.mape - 11.11) < 0.1
        assert e2e_summary.mpe > 0  # overprediction

    def test_stage_mismatch_logs_warning(self, caplog):
        """I4: Mismatched stage counts should produce a warning."""
        import logging

        gt = _make_stage(0, 100, 10, 1)
        experiment = Experiment(
            folder="/tmp/exp", model="m", tp=1, workload="w",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt], summary=_make_stage(-1, 0, 0, 0),
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        result = SimulatorResult(
            adapter_name="s", experiment_folder="/tmp/exp",
            stages=[_make_stage(0, 110, 11, 1.1), _make_stage(1, 120, 12, 1.2)],
            summary=_make_stage(-1, 0, 0, 0),
        )

        with caplog.at_level(logging.WARNING, logger="experiment.metrics"):
            compute_errors(experiment, result)
        assert "Stage count mismatch" in caplog.text

    def test_mismatched_stage_count_skips_extra(self):
        """Simulator has 2 stages but experiment has 1 — extra stage skipped."""
        gt = _make_stage(0, 100, 10, 1)
        experiment = Experiment(
            folder="/tmp/exp", model="m", tp=1, workload="w",
            max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
            total_kv_blocks=100, cpu_kv_blocks=0,
            stages=[gt], summary=_make_stage(-1, 0, 0, 0),
            profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
        )
        result = SimulatorResult(
            adapter_name="s", experiment_folder="/tmp/exp",
            stages=[_make_stage(0, 110, 11, 1.1), _make_stage(1, 120, 12, 1.2)],
            summary=_make_stage(-1, 0, 0, 0),
        )

        records = compute_errors(experiment, result)
        stage_indices = {r.stage_index for r in records}
        assert 1 not in stage_indices  # stage 1 skipped (no GT)
