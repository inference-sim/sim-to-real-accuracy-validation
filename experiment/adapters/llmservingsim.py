"""LLMServingSim adapter for validation against vLLM ground-truth experiments."""

from __future__ import annotations

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

    def run(self, experiment: Experiment) -> SimulatorResult:
        """Execute LLMServingSim and return predicted metrics.

        Raises:
            NotImplementedError: Until the full execution flow is wired up.
        """
        # TODO: implement main execution flow (Task 7)
        raise NotImplementedError("run() not yet implemented")
