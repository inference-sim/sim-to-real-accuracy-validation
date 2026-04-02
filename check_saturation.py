#!/usr/bin/env python3
"""
Check all ground truth experiments for saturation indicators.

Saturation indicators:
1. High TTFT tail latency (p99 > 0.5s or p99.9 > 1s)
2. High normalized TPOT variance (p99 > 0.05s)
3. Truncated outputs (p1 output_len < 0.9 * median)
4. High schedule delays (p99 > 0.01s)
5. Failed requests
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_experiment_config(exp_dir: Path) -> Dict:
    """Load experiment metadata from experiments.json."""
    with open(exp_dir.parent / "experiments.json") as f:
        experiments = json.load(f)

    # Parse experiment ID from directory name
    exp_id = int(exp_dir.name.split("-")[0])
    for exp in experiments:
        if exp["id"] == exp_id:
            return exp
    return {}


def check_saturation(metrics: Dict, stage_name: str) -> Tuple[bool, List[str]]:
    """Check if metrics indicate saturation."""
    issues = []
    saturated = False

    # Check for failures
    if metrics["failures"]["count"] > 0:
        failure_rate = metrics["failures"]["count"] / (metrics["successes"]["count"] + metrics["failures"]["count"])
        issues.append(f"FAILURES: {metrics['failures']['count']} requests failed ({failure_rate:.1%})")
        saturated = True

    # Check TTFT tail latency
    ttft = metrics["successes"]["latency"]["time_to_first_token"]
    if ttft["p99"] > 0.5:
        issues.append(f"High TTFT p99: {ttft['p99']:.3f}s")
        saturated = True
    if ttft.get("p99.9", 0) > 1.0:
        issues.append(f"High TTFT p99.9: {ttft['p99.9']:.3f}s")
        saturated = True
    if ttft["max"] > 2.0:
        issues.append(f"High TTFT max: {ttft['max']:.3f}s")

    # Check normalized TPOT variance
    norm_tpot = metrics["successes"]["latency"]["normalized_time_per_output_token"]
    if norm_tpot["p99"] > 0.05:
        issues.append(f"High norm_TPOT p99: {norm_tpot['p99']:.3f}s")
        saturated = True
    if norm_tpot.get("p99.9", 0) > 0.2:
        issues.append(f"High norm_TPOT p99.9: {norm_tpot['p99.9']:.3f}s")
        saturated = True

    # Check for truncated outputs
    output_len = metrics["successes"]["output_len"]
    if output_len["median"] > 0:
        truncation_threshold = 0.5 * output_len["median"]
        if output_len["p1"] < truncation_threshold:
            pct = output_len["p1"] / output_len["median"] * 100
            issues.append(f"Truncated outputs: p1={output_len['p1']:.0f} ({pct:.0f}% of median)")
            saturated = True

    # Check schedule delays
    schedule_delay = metrics["load_summary"]["schedule_delay"]
    if schedule_delay["p99"] > 0.01:
        issues.append(f"High schedule delay p99: {schedule_delay['p99']:.3f}s")

    # Check if achieved rate << requested rate (for stages with rate info)
    if "requested_rate" in metrics["load_summary"]:
        requested = metrics["load_summary"]["requested_rate"]
        achieved = metrics["load_summary"]["achieved_rate"]
        if requested > 0 and (achieved / requested) < 0.95:
            pct = achieved / requested * 100
            issues.append(f"Low achieved rate: {achieved:.1f}/{requested:.1f} RPS ({pct:.0f}%)")
            saturated = True

    return saturated, issues


def main():
    ground_truth_dir = Path("vllm_data/ground_truth")

    all_saturated = []
    partially_saturated = []

    for exp_dir in sorted(ground_truth_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        results_dir = exp_dir / "results"
        if not results_dir.exists():
            continue

        # Load experiment config
        exp_config = load_experiment_config(exp_dir)
        exp_name = f"{exp_dir.name}"

        # Check summary metrics
        summary_file = results_dir / "summary_lifecycle_metrics.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            summary = json.load(f)

        saturated, summary_issues = check_saturation(summary, "summary")

        # Check individual stages
        stage_results = {}
        for stage_file in sorted(results_dir.glob("stage_*_lifecycle_metrics.json")):
            stage_name = stage_file.stem.replace("_lifecycle_metrics", "")
            with open(stage_file) as f:
                stage_metrics = json.load(f)
            stage_saturated, stage_issues = check_saturation(stage_metrics, stage_name)
            stage_results[stage_name] = (stage_saturated, stage_issues)

        # Determine overall saturation
        all_issues = summary_issues.copy()
        any_stage_saturated = any(sat for sat, _ in stage_results.values())

        if saturated or any_stage_saturated:
            print(f"\n{'='*80}")
            print(f"🔴 SATURATED: {exp_name}")
            if exp_config:
                print(f"   Model: {exp_config['model']}, TP: {exp_config['tp']}, "
                      f"Workload: {exp_config['workload']}, MBT: {exp_config['mbt']}")
            print(f"{'='*80}")

            if summary_issues:
                print("\nSummary (all stages):")
                for issue in summary_issues:
                    print(f"  • {issue}")

            for stage_name, (stage_sat, stage_issues) in stage_results.items():
                if stage_issues:
                    print(f"\n{stage_name}:")
                    for issue in stage_issues:
                        print(f"  • {issue}")

            all_saturated.append((exp_name, exp_config, summary_issues, stage_results))

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments checked: {len(list(ground_truth_dir.iterdir()))}")
    print(f"Saturated experiments: {len(all_saturated)}")

    if all_saturated:
        print("\nSaturated experiments:")
        for exp_name, exp_config, _, _ in all_saturated:
            if exp_config:
                print(f"  • {exp_name}: {exp_config['model']} (tp{exp_config['tp']}, "
                      f"{exp_config['workload']}, mbt={exp_config['mbt']})")
            else:
                print(f"  • {exp_name}")


if __name__ == "__main__":
    main()
