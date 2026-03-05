from dataclasses import dataclass


@dataclass
class LatencyDistribution:
    mean: float  # milliseconds
    p90: float
    p99: float


@dataclass
class ThroughputMetrics:
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    requests_per_sec: float


@dataclass
class StageMetrics:
    stage_index: int
    rate: float  # req/s
    duration: float  # seconds
    num_requests: int
    e2e: LatencyDistribution
    ttft: LatencyDistribution
    itl: LatencyDistribution
    throughput: ThroughputMetrics


@dataclass
class Experiment:
    folder: str
    model: str  # HuggingFace ID, e.g. "meta-llama/Llama-2-7b-hf"
    tp: int
    workload: str  # "general", "codegen", "roleplay", "reasoning"
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    total_kv_blocks: int  # GPU blocks: "GPU KV cache size" tokens / 16
    cpu_kv_blocks: int  # CPU blocks: peak from kv_events.jsonl
    stages: list[StageMetrics]
    summary: StageMetrics
    profile_config: dict  # Raw parsed profile.yaml


@dataclass
class SimulatorResult:
    adapter_name: str
    experiment_folder: str
    stages: list[StageMetrics]
    summary: StageMetrics
    wall_clock_seconds: float = 0.0
