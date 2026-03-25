"""LLMServingSim adapter for validation against vLLM ground-truth experiments."""

from __future__ import annotations

import copy
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment.data_model import Experiment, SimulatorResult

from experiment.adapters.base import SimulatorAdapter


# Map ground-truth model IDs (with suffixes) to LLMServingSim perf model names
# (without suffixes).  The Experiment.model field contains the full HuggingFace
# model ID from exp-config.yaml (e.g. "meta-llama/Llama-3.1-8B-Instruct").
# LLMServingSim's perf model directories drop the "-Instruct" suffix.
MODEL_MAP: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B",
    "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
}


def _generate_arrivals(stages: list[dict]) -> list[float]:
    """Generate constant-rate arrival times from stage config.

    Args:
        stages: List of {"rate": req/s, "duration": seconds}

    Returns:
        List of arrival times in seconds
    """
    arrivals: list[float] = []
    t = 0.0

    for stage in stages:
        rate = stage["rate"]
        duration = stage["duration"]
        num_requests = round(rate * duration)
        inter_arrival = 1.0 / rate

        for _ in range(num_requests):
            arrivals.append(t)
            t += inter_arrival

    return arrivals


class LLMServingSimAdapter(SimulatorAdapter):
    """Adapter for the LLMServingSim discrete-event simulator.

    Supports H100 hardware with Llama-3.1-8B and Mixtral-8x7B models.
    Generates workloads from ground-truth token counts with constant-rate
    arrivals, executes LLMServingSim via subprocess, and parses results
    into standardised metrics for comparison with vLLM ground truth.
    """

    def __init__(self, llmservingsim_dir: str):
        """Initialize adapter.

        Args:
            llmservingsim_dir: Path to LLMServingSim directory containing
                main.py.

        Raises:
            ValueError: If the directory does not contain main.py.
        """
        self.llmservingsim_dir = os.path.abspath(llmservingsim_dir)
        if not os.path.exists(os.path.join(self.llmservingsim_dir, "main.py")):
            raise ValueError(
                f"Invalid LLMServingSim directory: {llmservingsim_dir}. "
                "Must contain main.py"
            )

    @property
    def name(self) -> str:
        """Return adapter name for identification."""
        return "llmservingsim"

    def can_run(self, experiment: Experiment) -> bool:
        """Check if this adapter supports the given experiment.

        Returns True only if:
        - Hardware is H100
        - Model is in MODEL_MAP
        - Performance model exists for the TP configuration
        - Precision is FP16
        """
        # Check hardware
        if experiment.hardware != "H100":
            return False

        # Check model mapping
        model_sim = MODEL_MAP.get(experiment.model)
        if not model_sim:
            return False

        # Check perf model exists
        perf_model_path = os.path.join(
            self.llmservingsim_dir,
            f"llm_profile/perf_models/H100/{model_sim}/tp{experiment.tp}",
        )
        if not os.path.exists(perf_model_path):
            return False

        # Check precision
        if experiment.precision != "FP16":
            return False

        return True

    def _generate_cluster_config(self, experiment: "Experiment", output_path: str) -> None:
        """Generate cluster config JSON for the experiment.

        Args:
            experiment: Experiment configuration
            output_path: Where to write the cluster config JSON
        """
        # Load H100 template
        template_path = os.path.join(
            self.llmservingsim_dir,
            "cluster_config/single_node_single_instance_H100.json",
        )
        with open(template_path) as f:
            config = json.load(f)

        # Get LLMServingSim model name
        model_sim = MODEL_MAP[experiment.model]

        # Modify instance config
        instance = config["nodes"][0]["instances"][0]
        instance["model_name"] = model_sim
        instance["npu_num"] = experiment.tp
        instance["npu_group"] = 1  # npus_per_group = npu_num / npu_group = TP
        instance["npu_mem"]["mem_size"] = 80.0  # H100 HBM3

        # Handle multi-instance (dp > 1) - use deep copy to avoid shared dicts
        if experiment.dp and experiment.dp > 1:
            config["nodes"][0]["num_instances"] = experiment.dp
            config["nodes"][0]["instances"] = [
                copy.deepcopy(instance) for _ in range(experiment.dp)
            ]

        # Write config
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    def _generate_workload(self, experiment: "Experiment", output_path: str) -> None:
        """Generate workload .jsonl file from ground-truth token counts.

        Args:
            experiment: Experiment configuration
            output_path: Where to write the workload .jsonl
        """
        from experiment.ground_truth import resolve_perf_dir

        # Read ground-truth metrics
        perf_dir = resolve_perf_dir(experiment.folder)
        metrics_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Ground-truth metrics not found: {metrics_path}. "
                "Cannot generate workload without token counts."
            )

        with open(metrics_path) as f:
            requests = json.load(f)

        # Extract token counts
        token_pairs = [
            (req["info"]["input_tokens"], req["info"]["output_tokens"])
            for req in requests
        ]

        # Generate constant-rate arrivals
        stages = experiment.profile_config["load"]["stages"]
        arrivals = _generate_arrivals(stages)

        # Check for mismatch
        if len(token_pairs) < len(arrivals):
            raise ValueError(
                f"Not enough ground-truth requests ({len(token_pairs)}) "
                f"for generated arrivals ({len(arrivals)})"
            )

        # Write workload .jsonl
        with open(output_path, "w") as f:
            for (input_toks, output_toks), arrival_sec in zip(token_pairs, arrivals):
                # Generate dummy token IDs (LLMServingSim only needs counts)
                input_tok_ids = list(range(1, input_toks + 1))

                record = {
                    "input_toks": input_toks,
                    "output_toks": output_toks,
                    "arrival_time_ns": int(arrival_sec * 1e9),
                    "input_tok_ids": input_tok_ids,
                }
                f.write(json.dumps(record) + "\n")

    def _build_cli_args(
        self,
        experiment: "Experiment",
        cluster_config: str,
        workload: str,
        output: str,
    ) -> list[str]:
        """Build LLMServingSim CLI arguments.

        Args:
            experiment: Experiment configuration
            cluster_config: Path to cluster config JSON
            workload: Path to workload .jsonl
            output: Path to output CSV

        Returns:
            List of CLI arguments
        """
        # Calculate total requests
        total_requests = sum(
            round(stage["rate"] * stage["duration"])
            for stage in experiment.profile_config["load"]["stages"]
        )

        args = [
            "python",
            "main.py",
            "--cluster-config",
            cluster_config,
            "--dataset",
            workload,
            "--output",
            output,
            "--fp",
            "16",
            "--block-size",
            "16",
            "--max-batch",
            str(experiment.max_num_seqs),
            "--max-num-batched-tokens",
            str(experiment.max_num_batched_tokens),
            "--num-req",
            str(total_requests),
            "--log-level",
            "WARNING",
        ]

        # Add routing policy for multi-instance
        if experiment.dp and experiment.dp > 1:
            args.extend(["--request-routing-policy", "RR"])

        return args

    def run(self, experiment: Experiment) -> SimulatorResult:
        """Execute LLMServingSim and return predicted metrics.

        Raises:
            NotImplementedError: Until the full execution flow is wired up.
        """
        # TODO: implement main execution flow (Task 7)
        raise NotImplementedError("run() not yet implemented")
