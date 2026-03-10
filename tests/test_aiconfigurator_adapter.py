"""Tests for experiment.adapters.aiconfigurator_est."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiment.adapters.aiconfigurator_est import AIConfiguratorEstimateAdapter
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
    )


def _make_pareto_df(tp: int = 1) -> pd.DataFrame:
    """Build a synthetic Pareto DataFrame mimicking AIConfigurator output.

    Contains rows for concurrency values 1, 5, 10, 20, 50 at the given TP.
    """
    rows = []
    for conc in [1, 5, 10, 20, 50]:
        rows.append({
            "model": "LLAMA2_7B",
            "isl": 566,
            "osl": 247,
            "concurrency": conc,
            "ttft": 20.0 + conc * 0.5,       # increases with concurrency
            "tpot": 3.0 + conc * 0.1,         # increases with concurrency
            "tp": tp,
            "pp": 1,
            "dp": 1,
            "num_total_gpus": tp,
            "seq/s": 5.0 + conc * 0.2,
            "tokens/s": 1000.0 + conc * 10.0,
            "tokens/s/gpu": 1000.0 + conc * 10.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdapterName:
    def test_adapter_name(self):
        adapter = AIConfiguratorEstimateAdapter()
        assert adapter.name == "aiconfigurator-estimate"


class TestCanRun:
    def test_can_run_shared_prefix(self):
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        assert adapter.can_run(exp) is True

    def test_cannot_run_non_shared_prefix(self):
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(profile_config={
            "data": {"type": "random"},
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_cannot_run_missing_data_key(self):
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(profile_config={
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_cannot_run_missing_shared_prefix_subkeys(self):
        """shared_prefix type but missing required sub-keys."""
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(profile_config={
            "data": {"type": "shared_prefix", "shared_prefix": {"question_len": 100}},
            "load": {"stages": [{"duration": 600, "rate": 5}]},
        })
        assert adapter.can_run(exp) is False

    def test_cannot_run_moe_model(self):
        """Mixtral (MoE) models are skipped — vllm backend unsupported."""
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(model="mistralai/Mixtral-8x7B-v0.1")
        assert adapter.can_run(exp) is False

    def test_can_run_non_moe_model(self):
        """CodeLlama is not MoE — should pass."""
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
        assert adapter.can_run(exp) is True


class TestModelNameMapping:
    def test_known_model_mapped(self):
        assert AIConfiguratorEstimateAdapter._resolve_model_name(
            "meta-llama/Llama-2-7b-hf"
        ) == "LLAMA2_7B"

    def test_known_model_70b_mapped(self):
        assert AIConfiguratorEstimateAdapter._resolve_model_name(
            "meta-llama/Llama-2-70b-hf"
        ) == "LLAMA2_70B"

    def test_unknown_model_passthrough(self):
        hf_id = "codellama/CodeLlama-34b-Instruct-hf"
        assert AIConfiguratorEstimateAdapter._resolve_model_name(hf_id) == hf_id


class TestConcurrencyDerivation:
    def test_littles_law_stage0(self):
        """Stage 0: rate=5, e2e_mean=1800ms → concurrency = 5 * 1.8 = 9."""
        exp = _make_experiment()
        c = AIConfiguratorEstimateAdapter._derive_concurrency(exp.stages[0], exp.max_num_seqs)
        assert c == 9

    def test_littles_law_stage1(self):
        """Stage 1: rate=10, e2e_mean=2100ms → concurrency = 10 * 2.1 = 21."""
        exp = _make_experiment()
        c = AIConfiguratorEstimateAdapter._derive_concurrency(exp.stages[1], exp.max_num_seqs)
        assert c == 21

    def test_minimum_concurrency_is_one(self):
        zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
        stage = StageMetrics(
            stage_index=0, rate=0.01, duration=60.0, num_requests=1,
            e2e=zero_lat, ttft=zero_lat, itl=zero_lat,
            throughput=ThroughputMetrics(0, 0, 0),
        )
        c = AIConfiguratorEstimateAdapter._derive_concurrency(stage, 128)
        assert c >= 1

    def test_concurrency_clamped_to_max_num_seqs(self):
        stage = StageMetrics(
            stage_index=0, rate=100.0, duration=600.0, num_requests=60000,
            e2e=LatencyDistribution(mean=50000.0, p90=0.0, p99=0.0),
            ttft=LatencyDistribution(mean=0.0, p90=0.0, p99=0.0),
            itl=LatencyDistribution(mean=0.0, p90=0.0, p99=0.0),
            throughput=ThroughputMetrics(0, 0, 0),
        )
        c = AIConfiguratorEstimateAdapter._derive_concurrency(stage, 32)
        assert c == 32


class TestInputOutputExtraction:
    def test_extracts_from_shared_prefix(self):
        exp = _make_experiment()
        inp, out = AIConfiguratorEstimateAdapter._extract_lengths(exp)
        assert inp == 566   # 466 + 100
        assert out == 247


class TestFindNearestConcurrency:
    def test_exact_match(self):
        df = _make_pareto_df()
        row = AIConfiguratorEstimateAdapter._find_nearest_concurrency(df, 10)
        assert row["concurrency"] == 10

    def test_nearest_below(self):
        """Concurrency 9 should snap to 10 (nearest)."""
        df = _make_pareto_df()
        row = AIConfiguratorEstimateAdapter._find_nearest_concurrency(df, 9)
        assert row["concurrency"] == 10

    def test_nearest_above(self):
        """Concurrency 21 should snap to 20 (nearest)."""
        df = _make_pareto_df()
        row = AIConfiguratorEstimateAdapter._find_nearest_concurrency(df, 21)
        assert row["concurrency"] == 20


class TestRunWithMock:
    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_run_returns_simulator_result(self, mock_create, mock_run):
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        assert result.adapter_name == "aiconfigurator-estimate"
        assert result.experiment_folder == exp.folder
        assert len(result.stages) == 2

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_task_config_args(self, mock_create, mock_run):
        """Verify that TaskConfig receives the correct mapped arguments."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        adapter.run(exp)

        mock_create.assert_called_once_with(
            serving_mode="agg",
            model_name="LLAMA2_7B",
            system_name="h100_sxm",
            backend_name="vllm",
            total_gpus=1,
            isl=566,
            osl=247,
            ttft=5000.0,
            tpot=200.0,
        )

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_e2e_computation(self, mock_create, mock_run):
        """E2E = ttft + tpot × output_length."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # Stage 0: concurrency=9 → snaps to row with concurrency=10
        # ttft = 20.0 + 10*0.5 = 25.0, tpot = 3.0 + 10*0.1 = 4.0
        # e2e = 25.0 + 4.0 * 247 = 25.0 + 988.0 = 1013.0
        s0 = result.stages[0]
        assert abs(s0.ttft.mean - 25.0) < 0.01
        assert abs(s0.itl.mean - 4.0) < 0.01
        assert abs(s0.e2e.mean - 1013.0) < 0.01

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_percentiles_are_none(self, mock_create, mock_run):
        """P90/P99 should be None (point-estimate simulator)."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        s0 = result.stages[0]
        assert s0.e2e.p90 is None
        assert s0.e2e.p99 is None
        assert s0.ttft.p90 is None
        assert s0.ttft.p99 is None
        assert s0.itl.p90 is None
        assert s0.itl.p99 is None

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_throughput_from_dataframe(self, mock_create, mock_run):
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # Stage 0: concurrency=9 → row concurrency=10
        # seq/s = 5.0 + 10*0.2 = 7.0, tokens/s = 1000 + 10*10 = 1100
        s0 = result.stages[0]
        assert abs(s0.throughput.requests_per_sec - 7.0) < 0.01
        assert abs(s0.throughput.output_tokens_per_sec - 1100.0) < 0.01
        # input_tokens_per_sec = seq/s * isl = 7.0 * 566 = 3962.0
        assert abs(s0.throughput.input_tokens_per_sec - 3962.0) < 0.01

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_summary_is_weighted_average(self, mock_create, mock_run):
        """Summary E2E mean should be request-weighted average across stages."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # Stage 0 (3000 reqs): concurrency=9 → row 10 → e2e=1013.0
        # Stage 1 (6000 reqs): concurrency=21 → row 20 → ttft=30, tpot=5.0
        #   e2e = 30.0 + 5.0 * 247 = 1265.0
        # Weighted mean = (1013*3000 + 1265*6000) / 9000
        expected_e2e = (1013.0 * 3000 + 1265.0 * 6000) / 9000
        assert abs(result.summary.e2e.mean - expected_e2e) < 0.01

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_hf_passthrough_model(self, mock_create, mock_run):
        """Models not in _MODEL_MAP should be passed through to AIConfigurator."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
        adapter.run(exp)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model_name"] == "codellama/CodeLlama-34b-Instruct-hf"

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_run_called_once(self, mock_create, mock_run):
        """TaskRunner.run should be called exactly once (not per-stage)."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        adapter.run(exp)

        mock_run.assert_called_once()

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_run_task_returns_none(self, mock_create, mock_run):
        """TaskRunner returning None should raise RuntimeError."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = None

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="returned None"):
            adapter.run(exp)

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_empty_pareto_df(self, mock_create, mock_run):
        """Empty pareto_df should raise RuntimeError."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": pd.DataFrame(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="empty pareto_df"):
            adapter.run(exp)

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_no_matching_tp(self, mock_create, mock_run):
        """No rows matching experiment TP should raise RuntimeError."""
        mock_create.return_value = MagicMock()
        # DataFrame has tp=1, but experiment uses tp=4
        mock_run.return_value = {"pareto_df": _make_pareto_df(tp=1), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(tp=4)
        with pytest.raises(RuntimeError, match="No AIConfigurator results for tp=4"):
            adapter.run(exp)
