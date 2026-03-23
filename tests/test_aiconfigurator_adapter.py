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


def _make_pareto_df(tp: int = 1) -> pd.DataFrame:
    """Build a synthetic Pareto DataFrame mimicking AIConfigurator output.

    Contains rows with different concurrency and throughput values.
    Throughput (seq/s) is designed to match test stage rates (5.0, 10.0).
    """
    rows = []
    # Concurrency and corresponding throughput (seq/s)
    configs = [
        (1, 5.0),    # Matches stage 0 rate=5.0
        (2, 10.0),   # Matches stage 1 rate=10.0
        (4, 18.0),   # Higher throughput
        (8, 30.0),   # Even higher
        (16, 45.0),  # Saturating
    ]
    for conc, seq_per_s in configs:
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
            "seq/s": seq_per_s,
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

    def test_can_run_rejects_non_h100(self):
        exp = _make_experiment()
        exp.hardware = "A100-80GB"
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_l40s(self):
        exp = _make_experiment()
        exp.hardware = "L40S"
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_mixtral_8x22b_instruct(self):
        """_MOE_MODELS should include the Instruct variant."""
        exp = _make_experiment(model="mistralai/Mixtral-8x22B-Instruct-v0.1")
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_mixtral_8x22b_base(self):
        """_MOE_MODELS should include the base variant."""
        exp = _make_experiment(model="mistralai/Mixtral-8x22B-v0.1")
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_llama4_scout(self):
        exp = _make_experiment(model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic")
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False

    def test_can_run_rejects_unknown_precision(self):
        exp = _make_experiment(precision="INT8")
        assert AIConfiguratorEstimateAdapter().can_run(exp) is False


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


class TestThroughputMatching:
    def test_match_throughput_exact_match(self):
        """Exact match: stage_rate=5.0, predicted seq/s=5.0."""
        df = _make_pareto_df()
        row = AIConfiguratorEstimateAdapter._match_throughput(5.0, df)
        assert row["concurrency"] == 1
        assert row["seq/s"] == 5.0

    def test_match_throughput_exact_match_stage1(self):
        """Exact match: stage_rate=10.0, predicted seq/s=10.0."""
        df = _make_pareto_df()
        row = AIConfiguratorEstimateAdapter._match_throughput(10.0, df)
        assert row["concurrency"] == 2
        assert row["seq/s"] == 10.0

    def test_match_throughput_closest_match(self):
        """No exact match: select closest by relative error."""
        df = _make_pareto_df()
        # stage_rate=12.0, closest is conc=2 with seq/s=10.0 (17% error)
        # vs conc=4 with seq/s=18.0 (50% error)
        row = AIConfiguratorEstimateAdapter._match_throughput(12.0, df)
        assert row["concurrency"] == 2

    def test_match_throughput_skips_invalid(self):
        """Should skip rows with zero or negative seq/s."""
        df = _make_pareto_df()
        # Add a row with invalid seq/s
        bad_row = df.iloc[0].copy()
        bad_row["seq/s"] = 0.0
        bad_row["concurrency"] = 999
        df = pd.concat([pd.DataFrame([bad_row]), df], ignore_index=True)

        # Should still match valid row
        row = AIConfiguratorEstimateAdapter._match_throughput(5.0, df)
        assert row["concurrency"] == 1

    def test_match_throughput_raises_if_no_valid(self):
        """Should raise if all rows have invalid seq/s."""
        df = _make_pareto_df()
        # Make all seq/s values invalid
        df["seq/s"] = 0.0

        with pytest.raises(RuntimeError, match="No valid row found"):
            AIConfiguratorEstimateAdapter._match_throughput(5.0, df)

    def test_match_throughput_raises_on_zero_rate(self):
        """Should raise ValueError if stage_rate is zero."""
        df = _make_pareto_df()
        with pytest.raises(ValueError, match="Invalid stage rate"):
            AIConfiguratorEstimateAdapter._match_throughput(0.0, df)

    def test_match_throughput_raises_on_negative_rate(self):
        """Should raise ValueError if stage_rate is negative."""
        df = _make_pareto_df()
        with pytest.raises(ValueError, match="Invalid stage rate"):
            AIConfiguratorEstimateAdapter._match_throughput(-5.0, df)


class TestInputOutputExtraction:
    def test_extracts_from_shared_prefix(self):
        exp = _make_experiment()
        inp, out = AIConfiguratorEstimateAdapter._extract_lengths(exp)
        assert inp == 566   # 466 + 100
        assert out == 247


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
            ttft=150000.0,
            tpot=200.0,
            profiles=["float16_default"],
        )

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_e2e_computed_from_formula(self, mock_create, mock_run):
        """E2E is computed as: ttft + itl * output_length."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # Stage 0: rate=5.0 → matches row with seq/s=5.0, concurrency=1
        # ttft = 20.0 + 1*0.5 = 20.5, itl = 3.0 + 1*0.1 = 3.1
        # E2E = ttft + itl * output_length = 20.5 + 3.1 * 247 = 786.2
        s0 = result.stages[0]
        assert abs(s0.ttft.mean - 20.5) < 0.01
        assert abs(s0.itl.mean - 3.1) < 0.01
        expected_e2e = 20.5 + 3.1 * 247
        assert abs(s0.e2e.mean - expected_e2e) < 0.01

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

        # Stage 0: rate=5.0 → matches row with seq/s=5.0, concurrency=1
        # seq/s = 5.0, tokens/s = 1000 + 1*10 = 1010
        s0 = result.stages[0]
        assert abs(s0.throughput.requests_per_sec - 5.0) < 0.01
        assert abs(s0.throughput.output_tokens_per_sec - 1010.0) < 0.01
        # input_tokens_per_sec = seq/s * isl = 5.0 * 566 = 2830.0
        assert abs(s0.throughput.input_tokens_per_sec - 2830.0) < 0.01

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_summary_is_weighted_average(self, mock_create, mock_run):
        """Summary E2E/TTFT/ITL means should be request-weighted average across stages."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment()
        result = adapter.run(exp)

        # output_length = 247
        # Stage 0 (3000 reqs): rate=5.0 → row conc=1 → ttft=20.5, itl=3.1
        #   E2E = 20.5 + 3.1*247 = 786.2
        # Stage 1 (6000 reqs): rate=10.0 → row conc=2 → ttft=21.0, itl=3.2
        #   E2E = 21.0 + 3.2*247 = 811.4
        # Weighted mean for TTFT = (20.5*3000 + 21.0*6000) / 9000
        expected_ttft = (20.5 * 3000 + 21.0 * 6000) / 9000
        assert abs(result.summary.ttft.mean - expected_ttft) < 0.01

        # Weighted mean for E2E = (786.2*3000 + 811.4*6000) / 9000
        e2e_0 = 20.5 + 3.1 * 247
        e2e_1 = 21.0 + 3.2 * 247
        expected_e2e = (e2e_0 * 3000 + e2e_1 * 6000) / 9000
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

    @patch("experiment.adapters.aiconfigurator_est._run_task")
    @patch("experiment.adapters.aiconfigurator_est._create_task_config")
    def test_fp8_passes_empty_profiles(self, mock_create, mock_run):
        """FP8 experiments should pass profiles=[] to AIConfigurator."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(precision="FP8")
        adapter.run(exp)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["profiles"] == []

    def test_run_rejects_unsupported_hardware(self):
        """run() should raise ValueError for unsupported hardware."""
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(hardware="L40S")
        with pytest.raises(ValueError, match="Unsupported hardware"):
            adapter.run(exp)

    def test_run_rejects_unknown_precision(self):
        """run() should raise ValueError for unknown precision values."""
        adapter = AIConfiguratorEstimateAdapter()
        exp = _make_experiment(precision="INT8")
        with pytest.raises(ValueError, match="Unsupported precision"):
            adapter.run(exp)
