"""BLIS evolved adapter -- physics-informed latency with iter16 coefficients.

Uses the ``evolved`` latency backend which combines roofline basis functions
with learned correction terms.  The coefficients were optimised during iter16
training (60.19% MAPE across 15 experiments on H100/FP16).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISEvolvedAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model evolved``.

    Passes static iter16 alpha/beta coefficients on the command line via
    ``--alpha-coeffs`` and ``--beta-coeffs``.  Works for any model (no
    per-model profiling required).

    Iter16 training summary
    -----------------------
    * Dataset  : 15 experiments (H100 / FP16)
    * Best MAPE: 60.19 %
    * Optimiser: differential-evolution (inner loop)

    Coefficient semantics
    ---------------------
    Alpha (3 values):
        alpha_0 : QueueingTime scale
        alpha_1 : Prefill attention scale
        alpha_2 : Decode attention scale

    Beta (7 values):
        beta_0 : Prefill roofline correction
        beta_1 : Prefill correction term 1
        beta_2 : Prefill correction term 2
        beta_3 : Decode roofline correction
        beta_4 : Decode correction term 1
        beta_5 : Decode correction term 2
        beta_6 : Scheduling overhead
    """

    # Iter16 optimised coefficients from inner_loop_results.json
    ITER16_ALPHA: list[float] = [
        15569.495449697066,   # alpha_0: QueueingTime scale
        815.0556502348827,    # alpha_1: Prefill attention scale
        45.705744318725586,   # alpha_2: Decode attention scale
    ]

    ITER16_BETA: list[float] = [
        0.20081681581824434,  # beta_0: Prefill roofline correction
        1.6173961192042448,   # beta_1: Prefill correction term 1
        1.3603417361920076,   # beta_2: Prefill correction term 2
        0.39579536655780084,  # beta_3: Decode roofline correction
        62.19421689224744,    # beta_4: Decode correction term 1
        2.937563498958273,    # beta_5: Decode correction term 2
        169.37780505091155,   # beta_6: Scheduling overhead
    ]

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
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)
            args.extend(["--latency-model", "evolved"])
            args.extend(["--alpha-coeffs", self._format_coeffs(self.ITER16_ALPHA)])
            args.extend(["--beta-coeffs", self._format_coeffs(self.ITER16_BETA)])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS evolved failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
