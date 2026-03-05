"""BLIS blackbox adapter — uses trained alpha/beta regression coefficients."""

from __future__ import annotations

import os
import subprocess
import tempfile

import yaml

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISBlackboxAdapter(BaseBLISAdapter):
    """BLIS simulator in blackbox mode (trained coefficients).

    ``can_run()`` returns True only when the experiment's model appears in
    ``defaults.yaml`` (case-insensitive match), meaning profiled coefficients
    exist for it.
    """

    def __init__(self, blis_binary: str, defaults_yaml: str | None = None):
        super().__init__(blis_binary)
        if defaults_yaml is None:
            defaults_yaml = os.path.join(os.path.dirname(blis_binary), "defaults.yaml")
        self._defaults_yaml = defaults_yaml

    @property
    def name(self) -> str:
        return "blis-blackbox"

    def can_run(self, experiment: Experiment) -> bool:
        """True if defaults.yaml has an entry for this model (case-insensitive)."""
        try:
            with open(self._defaults_yaml) as fh:
                data = yaml.safe_load(fh)
        except (FileNotFoundError, PermissionError, yaml.YAMLError):
            return False

        defaults = (data or {}).get("defaults", {})
        model_lower = experiment.model.lower()
        return any(key.lower() == model_lower for key in defaults)

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS blackbox failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
