"""BLIS evolved adapter -- physics-informed latency with learned coefficients.

Uses the ``evolved`` latency backend which combines roofline basis functions
with learned correction terms. Supports both iter16 (7-beta) and iter24 (10-beta)
coefficient sets.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISEvolvedAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model evolved``.

    Passes alpha/beta coefficients on the command line via ``--alpha-coeffs``
    and ``--beta-coeffs``. Supports iter16 (7 betas) and iter24 (10 betas).

    Iter16 training summary
    -----------------------
    * Dataset  : 15 experiments (H100 / FP16)
    * Best MAPE: 60.19 % (trained-roofline architecture)
    * Betas    : 7 (no prefill/decode split, no MoE term)

    Iter24 training summary
    -----------------------
    * Dataset  : 15 experiments (H100 / FP16)
    * Best MAPE: 39.18 % (overall loss: TTFT=24.13%, E2E=15.05%)
    * Method   : 2D grid search (144 pts) + golden section polish
    * Betas    : 10 (prefill/decode split + MoE term)
    * Key finding: Decode is memory-dominated (β₂ₐ=0), prefill is compute-only (β₁ᵦ=0)

    Coefficient semantics (iter24)
    -------------------------------
    Alpha (3 values):
        α₀ : QueueingTime — fixed API overhead (~15.6ms)
        α₁ : PostDecodeFixedOverhead — per-request completion (~0.8ms)
        α₂ : OutputTokenProcessingTime — per-output-token streaming cost

    Beta (10 values):
        β₁ₐ : Prefill compute correction (0.139 — FlashAttention reduces FLOPs 7.2×)
        β₂ₐ : Decode compute correction (0.0 — dropped, decode is memory-bound)
        β₃  : Weight loading correction (1.363 — 36% overhead above roofline)
        β₄  : TP communication correction
        β₅  : Per-layer overhead (62.3 µs/layer)
        β₆  : Per-request scheduling (2.8 µs/req)
        β₇  : Per-step constant (169.4 µs/step)
        β₈  : Per-MoE-layer overhead (427.3 µs/MoE-layer)
        β₁ᵦ : Prefill memory correction (0.0 — dropped, prefill is compute-bound)
        β₂ᵦ : Decode memory correction (1.263 — 26% overhead above roofline)
    """

    # Iter16 optimised coefficients (7 betas)
    ITER16_ALPHA: list[float] = [
        15569.495449697066,   # α₀: QueueingTime
        815.0556502348827,    # α₁: PostDecodeFixedOverhead
        45.705744318725586,   # α₂: OutputTokenProcessingTime
    ]

    ITER16_BETA: list[float] = [
        0.20081681581824434,  # β₀: Prefill roofline correction
        1.6173961192042448,   # β₁: Weight loading
        1.3603417361920076,   # β₂: TP communication
        0.39579536655780084,  # β₃: Decode roofline correction
        62.19421689224744,    # β₄: Per-layer overhead (µs/layer)
        2.937563498958273,    # β₅: Per-request scheduling (µs/req)
        169.37780505091155,   # β₆: Per-step constant (µs/step)
    ]

    # Iter24 optimised coefficients (10 betas)
    ITER24_ALPHA: list[float] = [
        15561.959717498621,  # α₀: QueueingTime (~15.6ms fixed API overhead)
        776.243476414174,    # α₁: PostDecodeFixedOverhead (~0.8ms per-request)
        45.910232684500556,  # α₂: OutputTokenProcessingTime (µs/token streaming)
    ]

    ITER24_BETA: list[float] = [
        0.138541,            # β₁ₐ: Prefill compute (7.2× FlashAttention discount)
        0.0,                 # β₂ₐ: Decode compute (dropped — memory-bound)
        1.363060401466404,   # β₃: Weight loading (36% overhead)
        0.3960942587233032,  # β₄: TP communication
        62.28932987355146,   # β₅: Per-layer overhead (µs/layer)
        2.7976795228174027,  # β₆: Per-request scheduling (µs/req)
        169.36568163371626,  # β₇: Per-step constant (µs/step)
        427.3,               # β₈: Per-MoE-layer overhead (µs/MoE-layer)
        0.0,                 # β₁ᵦ: Prefill memory (dropped — compute-bound)
        1.2632,              # β₂ᵦ: Decode memory (26% overhead)
    ]

    def __init__(self, blis_binary: str, iteration: int = 24):
        """Initialize BLIS evolved adapter.

        Parameters
        ----------
        blis_binary : str
            Path to BLIS binary
        iteration : int, default=24
            Which iteration coefficients to use (16 or 24)
        """
        super().__init__(blis_binary)
        if iteration not in (16, 24):
            raise ValueError(f"iteration must be 16 or 24, got {iteration}")
        self.iteration = iteration

    @staticmethod
    def _format_coeffs(coeffs: list[float]) -> str:
        """Format a list of floats as a comma-separated string with 6 decimal places.

        >>> BLISEvolvedAdapter._format_coeffs([1.0, 2.5, 3.123456789])
        '1.000000,2.500000,3.123457'
        """
        return ",".join(f"{c:.6f}" for c in coeffs)

    @property
    def name(self) -> str:
        return "blis-evolved"

    def run(self, experiment: Experiment) -> SimulatorResult:
        # Select coefficients based on iteration
        if self.iteration == 16:
            alpha = self.ITER16_ALPHA
            beta = self.ITER16_BETA
        else:  # iteration == 24
            alpha = self.ITER24_ALPHA
            beta = self.ITER24_BETA

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)
            args.extend(["--latency-model", "evolved"])
            args.extend(["--alpha-coeffs", self._format_coeffs(alpha)])
            args.extend(["--beta-coeffs", self._format_coeffs(beta)])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS evolved iter{self.iteration} failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
