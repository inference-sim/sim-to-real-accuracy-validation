"""BLIS trained-physics adapter — roofline basis functions with learned corrections."""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISTrainedPhysicsAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model trained-physics``.

    Uses globally-fitted roofline basis functions with architecture-aware
    corrections from ``defaults.yaml`` (loaded by the BLIS binary automatically).
    Generalizes across model architectures, workloads, and TP configurations
    without per-model calibration.

    The trained-physics model uses 13 coefficients (10 beta + 3 alpha):
    - Beta coefficients: prefill compute/memory split, decode compute/memory split,
      weight loading, TP communication, layer overhead, batch overhead, step overhead,
      MoE-layer overhead (architecture-aware)
    - Alpha coefficients: API queueing, post-decode fixed overhead, per-token overhead
    """

    @property
    def name(self) -> str:
        return "blis-trained-physics"

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)
            args.extend(["--latency-model", "trained-physics"])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS trained-physics failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
