"""Tests for experiment.run (orchestrator)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.run import (
    ALL_ADAPTER_NAMES,
    build_adapter_registry,
    parse_args,
    run_pipeline,
)


_MANIFEST_STUB = {
    "id": 1, "hw": "H100", "dp": None, "cpu_offload": False,
    "gpu_mem": 0.9, "precision": "FP16", "safe": "safe",
    "workload": "codegen", "model": "m", "mbt": 2048, "done": True, "notes": "",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage(idx, e2e=100.0, ttft=10.0, itl=1.0):
    return StageMetrics(
        stage_index=idx, rate=5.0, duration=600.0, num_requests=3000,
        e2e=LatencyDistribution(e2e, e2e * 1.1, e2e * 1.3),
        ttft=LatencyDistribution(ttft, ttft * 1.1, ttft * 1.3),
        itl=LatencyDistribution(itl, itl * 1.1, itl * 1.3),
        throughput=ThroughputMetrics(100.0, 50.0, 5.0),
    )


def _make_experiment(folder="/tmp/exp", model="llama-7b"):
    return Experiment(
        folder=folder, model=model, tp=1, workload="codegen",
        max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
        total_kv_blocks=7463, cpu_kv_blocks=5,
        stages=[_make_stage(0)],
        summary=_make_stage(-1),
        profile_config={"load": {"stages": [{"duration": 600, "rate": 5}]}},
    )


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_default_values(self):
        args = parse_args([])
        assert args.data_dir == "vllm_data/ground_truth"
        assert args.blis_binary == "inference-sim/blis"
        assert args.vidur_dir == "vidur"
        assert args.output_dir == "results"
        assert args.adapters == ALL_ADAPTER_NAMES

    def test_custom_values(self):
        args = parse_args([
            "--data-dir", "/data",
            "--blis-binary", "/bin/blis",
            "--vidur-dir", "/opt/vidur",
            "--output-dir", "/out",
            "--adapters", "vidur", "blis-roofline",
        ])
        assert args.data_dir == "/data"
        assert args.blis_binary == "/bin/blis"
        assert args.adapters == ["vidur", "blis-roofline"]

    def test_no_dp_scaling_default_false(self):
        args = parse_args([])
        assert args.no_dp_scaling is False

    def test_no_dp_scaling_flag_present(self):
        args = parse_args(["--no-dp-scaling"])
        assert args.no_dp_scaling is True


# ---------------------------------------------------------------------------
# Tests: build_adapter_registry
# ---------------------------------------------------------------------------


class TestBuildAdapterRegistry:
    def test_all_adapters_present(self):
        registry = build_adapter_registry("/bin/blis", "/opt/vidur")
        assert set(registry.keys()) == set(ALL_ADAPTER_NAMES)

    def test_adapter_names_match(self):
        registry = build_adapter_registry("/bin/blis", "/opt/vidur")
        for name, adapter in registry.items():
            assert adapter.name == name

    def test_only_requested_adapters_instantiated(self):
        """I5: Only instantiate requested adapters."""
        registry = build_adapter_registry(
            "/bin/blis", "/opt/vidur", adapter_names=["vidur"]
        )
        assert set(registry.keys()) == {"vidur"}

    def test_unknown_adapter_name_ignored(self):
        registry = build_adapter_registry(
            "/bin/blis", "/opt/vidur", adapter_names=["nonexistent"]
        )
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Tests: run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @patch("experiment.run.generate_report")
    @patch("experiment.run.parse_experiment")
    @patch("experiment.run.discover_experiments")
    @patch("experiment.run.build_adapter_registry")
    def test_runs_matching_adapters(
        self, mock_registry, mock_discover, mock_parse, mock_report
    ):
        exp = _make_experiment()
        mock_discover.return_value = [(_MANIFEST_STUB, "/tmp/exp")]
        mock_parse.side_effect = lambda path, manifest_entry=None: exp

        # Create a mock adapter that can_run and returns a result
        mock_adapter = MagicMock()
        mock_adapter.name = "mock-sim"
        mock_adapter.can_run.return_value = True
        mock_adapter.run.return_value = SimulatorResult(
            adapter_name="mock-sim",
            experiment_folder="/tmp/exp",
            stages=[_make_stage(0, e2e=110.0)],
            summary=_make_stage(-1, e2e=110.0),
        )

        mock_registry.return_value = {"mock-sim": mock_adapter}

        error_records, runtime_records = run_pipeline(
            data_dir="/data",
            blis_binary="/bin/blis",
            vidur_dir="/opt/vidur",
            output_dir="/out",
            adapter_names=["mock-sim"],
        )

        assert len(error_records) > 0
        assert len(runtime_records) == 1
        assert runtime_records[0].simulator == "mock-sim"
        assert runtime_records[0].wall_clock_seconds > 0
        mock_adapter.run.assert_called_once_with(exp)
        mock_report.assert_called_once()

    @patch("experiment.run.generate_report")
    @patch("experiment.run.parse_experiment")
    @patch("experiment.run.discover_experiments")
    @patch("experiment.run.build_adapter_registry")
    def test_skips_when_cant_run(
        self, mock_registry, mock_discover, mock_parse, mock_report
    ):
        exp = _make_experiment()
        mock_discover.return_value = [(_MANIFEST_STUB, "/tmp/exp")]
        mock_parse.side_effect = lambda path, manifest_entry=None: exp

        mock_adapter = MagicMock()
        mock_adapter.name = "mock-sim"
        mock_adapter.can_run.return_value = False

        mock_registry.return_value = {"mock-sim": mock_adapter}

        error_records, runtime_records = run_pipeline(
            data_dir="/data",
            blis_binary="/bin/blis",
            vidur_dir="/opt/vidur",
            output_dir="/out",
            adapter_names=["mock-sim"],
        )

        assert len(error_records) == 0
        assert len(runtime_records) == 0
        mock_adapter.run.assert_not_called()

    @patch("experiment.run.generate_report")
    @patch("experiment.run.parse_experiment")
    @patch("experiment.run.discover_experiments")
    @patch("experiment.run.build_adapter_registry")
    def test_continues_on_adapter_failure(
        self, mock_registry, mock_discover, mock_parse, mock_report
    ):
        exp = _make_experiment()
        mock_discover.return_value = [(_MANIFEST_STUB, "/tmp/exp")]
        mock_parse.side_effect = lambda path, manifest_entry=None: exp

        mock_adapter = MagicMock()
        mock_adapter.name = "failing-sim"
        mock_adapter.can_run.return_value = True
        mock_adapter.run.side_effect = RuntimeError("boom")

        mock_registry.return_value = {"failing-sim": mock_adapter}

        # Should not raise — failures are caught and printed
        error_records, runtime_records = run_pipeline(
            data_dir="/data",
            blis_binary="/bin/blis",
            vidur_dir="/opt/vidur",
            output_dir="/out",
            adapter_names=["failing-sim"],
        )

        assert len(error_records) == 0
        assert len(runtime_records) == 0
        mock_report.assert_called_once()

    @patch("experiment.run.generate_report")
    @patch("experiment.run.parse_experiment")
    @patch("experiment.run.discover_experiments")
    def test_passes_manifest_entry_to_parse(
        self, mock_discover, mock_parse, mock_report
    ):
        """Pipeline should pass manifest_entry to parse_experiment."""
        calls = []

        def mock_parse_fn(folder_path, manifest_entry=None):
            calls.append({"path": folder_path, "manifest_entry": manifest_entry})
            return _make_experiment(folder=folder_path)

        mock_discover.return_value = [(_MANIFEST_STUB, "/tmp/exp")]
        mock_parse.side_effect = mock_parse_fn

        run_pipeline(
            data_dir="/data",
            blis_binary="/bin/blis",
            vidur_dir="/opt/vidur",
            output_dir="/out",
            adapter_names=[],
        )

        assert len(calls) == 1
        assert calls[0]["path"] == "/tmp/exp"
        assert calls[0]["manifest_entry"]["id"] == 1

    @patch("experiment.run.generate_report")
    @patch("experiment.run.parse_experiment")
    @patch("experiment.run.discover_experiments")
    @patch("experiment.run.build_adapter_registry")
    def test_runtime_records_include_metadata(
        self, mock_registry, mock_discover, mock_parse, mock_report
    ):
        """RuntimeRecord should carry experiment metadata fields."""
        exp = _make_experiment()
        exp.exp_id = 42
        exp.hardware = "A100-80GB"
        exp.dp = 2
        exp.cpu_offload = True
        exp.gpu_mem_util = 0.85
        exp.precision = "FP8"

        manifest = {**_MANIFEST_STUB, "id": 42, "hw": "A100-80GB", "dp": 2}
        mock_discover.return_value = [(manifest, "/tmp/exp")]
        mock_parse.side_effect = lambda path, manifest_entry=None: exp

        mock_adapter = MagicMock()
        mock_adapter.name = "mock-sim"
        mock_adapter.can_run.return_value = True
        mock_adapter.run.return_value = SimulatorResult(
            adapter_name="mock-sim", experiment_folder="/tmp/exp",
            stages=[_make_stage(0)], summary=_make_stage(-1),
        )
        mock_registry.return_value = {"mock-sim": mock_adapter}

        _, runtime_records = run_pipeline(
            data_dir="/data", blis_binary="/bin/blis",
            vidur_dir="/opt/vidur", output_dir="/out",
            adapter_names=["mock-sim"],
        )

        assert len(runtime_records) == 1
        rr = runtime_records[0]
        assert rr.exp_id == 42
        assert rr.hardware == "A100-80GB"
        assert rr.dp == 2
        assert rr.cpu_offload is True
        assert rr.gpu_mem_util == 0.85
        assert rr.precision == "FP8"

    @patch("experiment.run.generate_report")
    @patch("experiment.run.discover_experiments")
    def test_no_experiments_returns_empty(self, mock_discover, mock_report):
        mock_discover.return_value = []

        error_records, runtime_records = run_pipeline(
            data_dir="/empty",
            blis_binary="/bin/blis",
            vidur_dir="/opt/vidur",
            output_dir="/out",
        )

        assert error_records == []
        assert runtime_records == []
        mock_report.assert_not_called()
