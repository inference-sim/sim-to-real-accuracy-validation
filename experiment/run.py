"""Orchestrator — run all simulators against all experiments and generate report.

Usage::

    python -m experiment.run --data-dir vllm_data/ground_truth --output-dir results
"""

from __future__ import annotations

import argparse
import time
import traceback

from experiment.adapters.aiconfigurator_est import AIConfiguratorEstimateAdapter
from experiment.adapters.base import SimulatorAdapter
from experiment.adapters.blis_blackbox import BLISBlackboxAdapter
from experiment.adapters.blis_crossmodel import BLISCrossModelAdapter
from experiment.adapters.blis_roofline import BLISRooflineAdapter
from experiment.adapters.blis_trained_roofline import BLISTrainedRooflineAdapter
from experiment.adapters.llm_optimizer_est import LLMOptimizerEstimateAdapter
from experiment.adapters.vidur import VidurAdapter
from experiment.ground_truth import discover_experiments, parse_experiment
from experiment.metrics import ErrorRecord, RuntimeRecord, compute_errors
from experiment.report import generate_report

ALL_ADAPTER_NAMES = [
    "blis-blackbox",
    "blis-roofline",
    "blis-crossmodel",
    "blis-trained-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
]


def build_adapter_registry(
    blis_binary: str,
    vidur_dir: str,
    adapter_names: list[str] | None = None,
) -> dict[str, SimulatorAdapter]:
    """Build a name → adapter instance mapping.

    Only instantiates adapters listed in *adapter_names* (default: all).
    """
    factories: dict[str, callable] = {
        "blis-blackbox": lambda: BLISBlackboxAdapter(blis_binary),
        "blis-roofline": lambda: BLISRooflineAdapter(blis_binary),
        "blis-crossmodel": lambda: BLISCrossModelAdapter(blis_binary),
        "blis-trained-roofline": lambda: BLISTrainedRooflineAdapter(blis_binary),
        "vidur": lambda: VidurAdapter(vidur_dir),
        "llm-optimizer-estimate": lambda: LLMOptimizerEstimateAdapter(),
        "aiconfigurator-estimate": lambda: AIConfiguratorEstimateAdapter(),
    }
    if adapter_names is None:
        adapter_names = list(factories.keys())
    return {name: factories[name]() for name in adapter_names if name in factories}


def run_pipeline(
    data_dir: str,
    blis_binary: str,
    vidur_dir: str,
    output_dir: str,
    adapter_names: list[str] | None = None,
) -> tuple[list[ErrorRecord], list[RuntimeRecord]]:
    """Core pipeline: discover → run → compute errors → report.

    Returns (error_records, runtime_records).
    """
    if adapter_names is None:
        adapter_names = ALL_ADAPTER_NAMES

    # 1. Discover experiments
    experiment_dirs = discover_experiments(data_dir)
    if not experiment_dirs:
        print(f"No experiments found in {data_dir}")
        return [], []

    print(f"Found {len(experiment_dirs)} experiments")

    # 2. Parse experiments
    experiments = []
    for d in experiment_dirs:
        try:
            experiments.append(parse_experiment(d))
        except Exception:
            print(f"  SKIP (parse error): {d}")
            traceback.print_exc()

    print(f"Parsed {len(experiments)} experiments successfully")

    # 3. Build adapter registry (only requested adapters)
    adapters = build_adapter_registry(blis_binary, vidur_dir, adapter_names)

    # 4. Run all (experiment, adapter) pairs
    all_records: list[ErrorRecord] = []
    runtime_records: list[RuntimeRecord] = []
    for exp in experiments:
        for adapter_name, adapter in adapters.items():
            if not adapter.can_run(exp):
                continue
            try:
                t0 = time.perf_counter()
                result = adapter.run(exp)
                elapsed = time.perf_counter() - t0
                result.wall_clock_seconds = elapsed
                records = compute_errors(exp, result)
                all_records.extend(records)
                runtime_records.append(RuntimeRecord(
                    simulator=adapter_name,
                    experiment_folder=exp.folder,
                    model=exp.model,
                    workload=exp.workload,
                    wall_clock_seconds=elapsed,
                ))
                print(f"  OK: {adapter_name} × {exp.model} ({exp.workload}) [{elapsed:.2f}s]")
            except Exception:
                print(f"  FAIL: {adapter_name} × {exp.model} ({exp.workload})")
                traceback.print_exc()

    # 5. Generate report
    generate_report(all_records, output_dir, runtime_records=runtime_records)

    return all_records, runtime_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sim-to-real accuracy validation."
    )
    parser.add_argument(
        "--data-dir",
        default="vllm_data/ground_truth",
        help="Directory containing ground-truth experiment folders.",
    )
    parser.add_argument(
        "--blis-binary",
        default="inference-sim/blis",
        help="Path to the BLIS simulator binary.",
    )
    parser.add_argument(
        "--vidur-dir",
        default="vidur",
        help="Path to the cloned Vidur repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where reports and CSV will be saved.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=ALL_ADAPTER_NAMES,
        choices=ALL_ADAPTER_NAMES,
        help="Which adapters to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        data_dir=args.data_dir,
        blis_binary=args.blis_binary,
        vidur_dir=args.vidur_dir,
        output_dir=args.output_dir,
        adapter_names=args.adapters,
    )


if __name__ == "__main__":
    main()
