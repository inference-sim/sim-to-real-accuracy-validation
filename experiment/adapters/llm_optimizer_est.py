"""LLM-Optimizer analytical estimate adapter.

Calls ``estimate_llm_performance`` from the ``llm-optimizer`` package once
per stage, deriving concurrency via Little's Law and mapping the roofline
result to :class:`StageMetrics`.

Since llm-optimizer produces only **mean** latency estimates, P90 and P99
are left as ``None`` (the metrics layer skips comparisons where the
simulator does not provide a value).
"""

from __future__ import annotations

from experiment.adapters.base import SimulatorAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)

_HW_TO_LLM_OPT: dict[str, str] = {"H100": "H100", "A100-80GB": "A100"}

def get_model_config_from_hf(model_id: str):
    """Lazy import wrapper — allows mocking without requiring llm_optimizer installed."""
    from llm_optimizer.common import get_model_config_from_hf as _fn
    return _fn(model_id)


def estimate_llm_performance(**kwargs):
    """Lazy import wrapper — allows mocking without requiring llm_optimizer installed."""
    from llm_optimizer.performance import estimate_llm_performance as _fn
    return _fn(**kwargs)


class LLMOptimizerEstimateAdapter(SimulatorAdapter):
    """Adapter wrapping the llm-optimizer roofline performance estimator."""

    @property
    def name(self) -> str:
        return "llm-optimizer-estimate"

    def can_run(self, experiment: Experiment) -> bool:
        """True only when hardware is supported and profile config uses ``shared_prefix``."""
        if experiment.hardware not in _HW_TO_LLM_OPT:
            return False
        if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
            return False
        try:
            data = experiment.profile_config["data"]
            if data["type"] != "shared_prefix":
                return False
            sp = data["shared_prefix"]
            return all(k in sp for k in ("question_len", "system_prompt_len", "output_len"))
        except (KeyError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_concurrency(stage: StageMetrics, max_num_seqs: int) -> int:
        """Estimate in-flight concurrency via Little's Law: L = λ × W.

        Clamped to ``max_num_seqs`` to respect the scheduler's batch cap.
        """
        raw = max(1, round(stage.rate * stage.e2e.mean / 1000))
        return min(raw, max_num_seqs)

    @staticmethod
    def _extract_lengths(experiment: Experiment) -> tuple[int, int]:
        """Extract input/output token lengths from profile config."""
        data_cfg = experiment.profile_config["data"]["shared_prefix"]
        input_length = data_cfg["question_len"] + data_cfg["system_prompt_len"]
        output_length = data_cfg["output_len"]
        return input_length, output_length

    @staticmethod
    def _make_latency_dist(mean: float) -> LatencyDistribution:
        return LatencyDistribution(mean=mean)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, experiment: Experiment) -> SimulatorResult:
        if experiment.hardware not in _HW_TO_LLM_OPT:
            raise ValueError(
                f"Unsupported hardware '{experiment.hardware}' for {self.name} "
                f"(supported: {sorted(_HW_TO_LLM_OPT)})"
            )
        if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
            raise ValueError(
                f"Unsupported precision '{experiment.precision}' on "
                f"'{experiment.hardware}' for {self.name} (A100 has no FP8 TFLOPS)"
            )
        model_config = get_model_config_from_hf(experiment.model)
        precision = experiment.precision.lower()  # "fp16" or "fp8"
        input_length, output_length = self._extract_lengths(experiment)

        stages: list[StageMetrics] = []
        for gt_stage in experiment.stages:
            concurrency = self._derive_concurrency(gt_stage, experiment.max_num_seqs)
            perf = estimate_llm_performance(
                num_gpus=experiment.tp,
                gpu_name=_HW_TO_LLM_OPT[experiment.hardware],
                model_config=model_config,
                precision=precision,
                concurrency=concurrency,
                input_length=input_length,
                output_length=output_length,
            )
            e2e_mean_ms = perf.e2e_latency_s * 1000
            stages.append(StageMetrics(
                stage_index=gt_stage.stage_index,
                rate=gt_stage.rate,
                duration=gt_stage.duration,
                num_requests=gt_stage.num_requests,
                e2e=self._make_latency_dist(e2e_mean_ms),
                ttft=self._make_latency_dist(perf.ttft_ms),
                itl=self._make_latency_dist(perf.itl_ms),
                throughput=ThroughputMetrics(
                    input_tokens_per_sec=perf.input_throughput_tps,
                    output_tokens_per_sec=perf.output_throughput_tps,
                    requests_per_sec=perf.requests_per_sec,
                ),
            ))

        summary = self._weighted_summary(stages)
        return SimulatorResult(
            adapter_name=self.name,
            experiment_folder=experiment.folder,
            stages=stages,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_summary(stages: list[StageMetrics]) -> StageMetrics:
        """Compute a weighted-average summary across stages.

        Latency metrics use request-weighted averaging (correct for per-request
        quantities).  Throughput metrics use duration-weighted averaging
        (correct for rates — tokens/sec, requests/sec).
        """
        total_reqs = sum(s.num_requests for s in stages)
        total_duration = sum(s.duration for s in stages)
        if total_reqs == 0:
            zero = LatencyDistribution(0)
            return StageMetrics(
                stage_index=-1, rate=0, duration=0, num_requests=0,
                e2e=zero, ttft=zero, itl=zero,
                throughput=ThroughputMetrics(0, 0, 0),
            )

        def _req_wavg(getter):
            return sum(getter(s) * s.num_requests for s in stages) / total_reqs

        def _dur_wavg(getter):
            if total_duration <= 0:
                return 0.0
            return sum(getter(s) * s.duration for s in stages) / total_duration

        e2e_mean = _req_wavg(lambda s: s.e2e.mean)
        ttft_mean = _req_wavg(lambda s: s.ttft.mean)
        itl_mean = _req_wavg(lambda s: s.itl.mean)

        return StageMetrics(
            stage_index=-1,
            rate=0.0,
            duration=0.0,
            num_requests=total_reqs,
            e2e=LatencyDistribution(e2e_mean),
            ttft=LatencyDistribution(ttft_mean),
            itl=LatencyDistribution(itl_mean),
            throughput=ThroughputMetrics(
                input_tokens_per_sec=_dur_wavg(lambda s: s.throughput.input_tokens_per_sec),
                output_tokens_per_sec=_dur_wavg(lambda s: s.throughput.output_tokens_per_sec),
                requests_per_sec=_dur_wavg(lambda s: s.throughput.requests_per_sec),
            ),
        )
