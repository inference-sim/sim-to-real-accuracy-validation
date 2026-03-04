"""Tests for experiment.data_model dataclasses."""

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)


# ---------------------------------------------------------------------------
# Helpers to build reusable fixtures
# ---------------------------------------------------------------------------


def _make_latency(mean: float = 100.0, p90: float = 150.0, p99: float = 200.0) -> LatencyDistribution:
    return LatencyDistribution(mean=mean, p90=p90, p99=p99)


def _make_throughput(
    input_tps: float = 500.0,
    output_tps: float = 300.0,
    rps: float = 10.0,
) -> ThroughputMetrics:
    return ThroughputMetrics(
        input_tokens_per_sec=input_tps,
        output_tokens_per_sec=output_tps,
        requests_per_sec=rps,
    )


def _make_stage(stage_index: int = 0) -> StageMetrics:
    return StageMetrics(
        stage_index=stage_index,
        rate=5.0,
        duration=60.0,
        num_requests=300,
        e2e=_make_latency(100.0, 150.0, 200.0),
        ttft=_make_latency(20.0, 30.0, 40.0),
        itl=_make_latency(10.0, 15.0, 20.0),
        throughput=_make_throughput(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLatencyDistribution:
    def test_construction_and_field_access(self) -> None:
        ld = LatencyDistribution(mean=100.5, p90=150.2, p99=200.8)
        assert ld.mean == 100.5
        assert ld.p90 == 150.2
        assert ld.p99 == 200.8

    def test_equality(self) -> None:
        a = LatencyDistribution(mean=1.0, p90=2.0, p99=3.0)
        b = LatencyDistribution(mean=1.0, p90=2.0, p99=3.0)
        assert a == b

    def test_inequality(self) -> None:
        a = LatencyDistribution(mean=1.0, p90=2.0, p99=3.0)
        b = LatencyDistribution(mean=1.0, p90=2.0, p99=4.0)
        assert a != b


class TestThroughputMetrics:
    def test_construction_and_field_access(self) -> None:
        tm = ThroughputMetrics(
            input_tokens_per_sec=500.0,
            output_tokens_per_sec=300.0,
            requests_per_sec=10.0,
        )
        assert tm.input_tokens_per_sec == 500.0
        assert tm.output_tokens_per_sec == 300.0
        assert tm.requests_per_sec == 10.0

    def test_equality(self) -> None:
        a = _make_throughput()
        b = _make_throughput()
        assert a == b


class TestStageMetrics:
    def test_construction_and_field_access(self) -> None:
        sm = _make_stage(stage_index=2)
        assert sm.stage_index == 2
        assert sm.rate == 5.0
        assert sm.duration == 60.0
        assert sm.num_requests == 300

    def test_nested_latency_fields(self) -> None:
        sm = _make_stage()
        assert sm.e2e.mean == 100.0
        assert sm.ttft.p90 == 30.0
        assert sm.itl.p99 == 20.0

    def test_nested_throughput_fields(self) -> None:
        sm = _make_stage()
        assert sm.throughput.input_tokens_per_sec == 500.0
        assert sm.throughput.output_tokens_per_sec == 300.0
        assert sm.throughput.requests_per_sec == 10.0

    def test_equality(self) -> None:
        a = _make_stage(0)
        b = _make_stage(0)
        assert a == b


class TestExperiment:
    def test_construction_and_field_access(self) -> None:
        stage = _make_stage(0)
        summary = _make_stage(stage_index=-1)
        profile_cfg = {"model": "meta-llama/Llama-2-7b-hf", "tp": 1}

        exp = Experiment(
            folder="/data/experiments/run_001",
            model="meta-llama/Llama-2-7b-hf",
            tp=1,
            workload="general",
            max_model_len=4096,
            max_num_batched_tokens=2048,
            max_num_seqs=256,
            total_kv_blocks=1000,
            cpu_kv_blocks=50,
            stages=[stage],
            summary=summary,
            profile_config=profile_cfg,
        )

        assert exp.folder == "/data/experiments/run_001"
        assert exp.model == "meta-llama/Llama-2-7b-hf"
        assert exp.tp == 1
        assert exp.workload == "general"
        assert exp.max_model_len == 4096
        assert exp.max_num_batched_tokens == 2048
        assert exp.max_num_seqs == 256
        assert exp.total_kv_blocks == 1000
        assert exp.cpu_kv_blocks == 50
        assert len(exp.stages) == 1
        assert exp.stages[0].stage_index == 0
        assert exp.summary.stage_index == -1
        assert exp.profile_config == profile_cfg

    def test_multiple_stages(self) -> None:
        stages = [_make_stage(i) for i in range(3)]
        summary = _make_stage(stage_index=-1)

        exp = Experiment(
            folder="/data/experiments/run_002",
            model="meta-llama/Llama-2-70b-hf",
            tp=4,
            workload="codegen",
            max_model_len=8192,
            max_num_batched_tokens=4096,
            max_num_seqs=128,
            total_kv_blocks=2000,
            cpu_kv_blocks=100,
            stages=stages,
            summary=summary,
            profile_config={},
        )

        assert len(exp.stages) == 3
        assert [s.stage_index for s in exp.stages] == [0, 1, 2]


class TestSimulatorResult:
    def test_construction_and_field_access(self) -> None:
        stage = _make_stage(0)
        summary = _make_stage(stage_index=-1)

        sr = SimulatorResult(
            adapter_name="vidur",
            experiment_folder="/data/experiments/run_001",
            stages=[stage],
            summary=summary,
        )

        assert sr.adapter_name == "vidur"
        assert sr.experiment_folder == "/data/experiments/run_001"
        assert len(sr.stages) == 1
        assert sr.stages[0].stage_index == 0
        assert sr.summary.stage_index == -1

    def test_multiple_stages(self) -> None:
        stages = [_make_stage(i) for i in range(5)]
        summary = _make_stage(stage_index=-1)

        sr = SimulatorResult(
            adapter_name="inference-sim",
            experiment_folder="/data/experiments/run_003",
            stages=stages,
            summary=summary,
        )

        assert len(sr.stages) == 5
        assert sr.adapter_name == "inference-sim"


class TestImportsFromPackage:
    """Verify that dataclasses are importable from the experiment package."""

    def test_import_from_package(self) -> None:
        from experiment import (
            Experiment,
            LatencyDistribution,
            SimulatorResult,
            StageMetrics,
            ThroughputMetrics,
        )

        # Verify they are the same classes
        from experiment.data_model import (
            Experiment as Exp2,
            LatencyDistribution as LD2,
            SimulatorResult as SR2,
            StageMetrics as SM2,
            ThroughputMetrics as TM2,
        )

        assert LatencyDistribution is LD2
        assert ThroughputMetrics is TM2
        assert StageMetrics is SM2
        assert Experiment is Exp2
        assert SimulatorResult is SR2
