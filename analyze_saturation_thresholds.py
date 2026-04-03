#!/usr/bin/env python3
"""
Analyze reasonable saturation thresholds by comparing metrics across experiments.

Key insight: Saturation manifests as TAIL LATENCY BLOWUP, not absolute values.
We should look at tail/median ratios and variance.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_metrics(ground_truth_dir: Path) -> pd.DataFrame:
    """Load all experiment metrics into a dataframe."""
    records = []

    with open(ground_truth_dir / "experiments.json") as f:
        exp_configs = {e["id"]: e for e in json.load(f)}

    for exp_dir in sorted(ground_truth_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        results_dir = exp_dir / "results"
        summary_file = results_dir / "summary_lifecycle_metrics.json"
        if not summary_file.exists():
            continue

        exp_id = int(exp_dir.name.split("-")[0])
        exp_config = exp_configs.get(exp_id, {})

        with open(summary_file) as f:
            metrics = json.load(f)

        s = metrics["successes"]

        # Extract key metrics
        record = {
            "exp_id": exp_id,
            "exp_name": exp_dir.name,
            "model": exp_config.get("model", ""),
            "tp": exp_config.get("tp", 0),
            "workload": exp_config.get("workload", ""),
            "mbt": exp_config.get("mbt", 0),
            "hw": exp_config.get("hw", ""),
            "safe": exp_config.get("safe", ""),

            # Throughput
            "rps": s["throughput"]["requests_per_sec"],
            "output_tps": s["throughput"]["output_tokens_per_sec"],

            # TTFT
            "ttft_median": s["latency"]["time_to_first_token"]["median"],
            "ttft_p99": s["latency"]["time_to_first_token"]["p99"],
            "ttft_p99.9": s["latency"]["time_to_first_token"].get("p99.9", 0),
            "ttft_max": s["latency"]["time_to_first_token"]["max"],

            # Normalized TPOT (per-token latency)
            "norm_tpot_median": s["latency"]["normalized_time_per_output_token"]["median"],
            "norm_tpot_p99": s["latency"]["normalized_time_per_output_token"]["p99"],
            "norm_tpot_p99.9": s["latency"]["normalized_time_per_output_token"].get("p99.9", 0),

            # Output length (to check for truncation)
            "output_len_median": s["output_len"]["median"],
            "output_len_p1": s["output_len"]["p1"],
            "output_len_p99": s["output_len"]["p99"],
        }

        # Compute ratios
        record["ttft_p99_ratio"] = record["ttft_p99"] / max(record["ttft_median"], 0.001)
        record["ttft_p99.9_ratio"] = record["ttft_p99.9"] / max(record["ttft_median"], 0.001)
        record["norm_tpot_p99_ratio"] = record["norm_tpot_p99"] / max(record["norm_tpot_median"], 0.001)
        record["norm_tpot_p99.9_ratio"] = record["norm_tpot_p99.9"] / max(record["norm_tpot_median"], 0.001)
        record["output_truncation_ratio"] = record["output_len_p1"] / max(record["output_len_median"], 1)

        records.append(record)

    return pd.DataFrame(records)


def main():
    ground_truth_dir = Path("vllm_data/ground_truth")
    df = load_metrics(ground_truth_dir)

    print("="*80)
    print("SATURATION THRESHOLD ANALYSIS")
    print("="*80)

    # Separate by safety label (from experimenter classification)
    safe_df = df[df["safe"] == "safe"]
    unsafe_df = df[df["safe"] == "unsafe"]

    print(f"\nDataset size:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Safe: {len(safe_df)}")
    print(f"  Unsafe: {len(unsafe_df)}")

    # Analyze TTFT ratios
    print(f"\n{'='*80}")
    print("TTFT (Time to First Token) Ratios (p99/median)")
    print(f"{'='*80}")
    print(f"\nSafe experiments:")
    print(f"  Median ratio: {safe_df['ttft_p99_ratio'].median():.2f}x")
    print(f"  Mean ratio: {safe_df['ttft_p99_ratio'].mean():.2f}x")
    print(f"  p90 ratio: {safe_df['ttft_p99_ratio'].quantile(0.9):.2f}x")
    print(f"  Max ratio: {safe_df['ttft_p99_ratio'].max():.2f}x")

    if len(unsafe_df) > 0:
        print(f"\nUnsafe experiments:")
        print(f"  Median ratio: {unsafe_df['ttft_p99_ratio'].median():.2f}x")
        print(f"  Mean ratio: {unsafe_df['ttft_p99_ratio'].mean():.2f}x")
        print(f"  Max ratio: {unsafe_df['ttft_p99_ratio'].max():.2f}x")

    # Find outliers
    print(f"\nTTFT p99.9 / median > 10x:")
    outliers = df[df["ttft_p99.9_ratio"] > 10]
    for _, row in outliers.iterrows():
        print(f"  {row['exp_name']}: {row['ttft_p99.9_ratio']:.0f}x "
              f"(median={row['ttft_median']*1000:.1f}ms, p99.9={row['ttft_p99.9']:.2f}s)")

    # Analyze normalized TPOT ratios
    print(f"\n{'='*80}")
    print("Normalized TPOT Ratios (p99/median)")
    print(f"{'='*80}")
    print(f"\nSafe experiments:")
    print(f"  Median ratio: {safe_df['norm_tpot_p99_ratio'].median():.2f}x")
    print(f"  Mean ratio: {safe_df['norm_tpot_p99_ratio'].mean():.2f}x")
    print(f"  p90 ratio: {safe_df['norm_tpot_p99_ratio'].quantile(0.9):.2f}x")
    print(f"  Max ratio: {safe_df['norm_tpot_p99_ratio'].max():.2f}x")

    if len(unsafe_df) > 0:
        print(f"\nUnsafe experiments:")
        print(f"  Median ratio: {unsafe_df['norm_tpot_p99_ratio'].median():.2f}x")
        print(f"  Mean ratio: {unsafe_df['norm_tpot_p99_ratio'].mean():.2f}x")
        print(f"  Max ratio: {unsafe_df['norm_tpot_p99_ratio'].max():.2f}x")

    print(f"\nnorm_TPOT p99.9 / median > 10x:")
    outliers = df[df["norm_tpot_p99.9_ratio"] > 10]
    for _, row in outliers.iterrows():
        print(f"  {row['exp_name']}: {row['norm_tpot_p99.9_ratio']:.0f}x "
              f"(median={row['norm_tpot_median']*1000:.1f}ms, p99.9={row['norm_tpot_p99.9']:.2f}s)")

    # Analyze output truncation
    print(f"\n{'='*80}")
    print("Output Truncation (p1/median)")
    print(f"{'='*80}")
    print(f"\nSafe experiments:")
    print(f"  Median ratio: {safe_df['output_truncation_ratio'].median():.2f}")
    print(f"  Mean ratio: {safe_df['output_truncation_ratio'].mean():.2f}")
    print(f"  Min ratio: {safe_df['output_truncation_ratio'].min():.2f}")

    if len(unsafe_df) > 0:
        print(f"\nUnsafe experiments:")
        print(f"  Median ratio: {unsafe_df['output_truncation_ratio'].median():.2f}")
        print(f"  Mean ratio: {unsafe_df['output_truncation_ratio'].mean():.2f}")
        print(f"  Min ratio: {unsafe_df['output_truncation_ratio'].min():.2f}")

    print(f"\nOutput truncation < 50% of median:")
    outliers = df[df["output_truncation_ratio"] < 0.5]
    for _, row in outliers.iterrows():
        print(f"  {row['exp_name']}: {row['output_truncation_ratio']*100:.0f}% "
              f"(p1={row['output_len_p1']:.0f}, median={row['output_len_median']:.0f})")

    # Propose thresholds
    print(f"\n{'='*80}")
    print("PROPOSED SATURATION THRESHOLDS")
    print(f"{'='*80}")

    # Use safe experiments as baseline
    ttft_p99_baseline = safe_df["ttft_p99_ratio"].quantile(0.9)
    ttft_p99_9_baseline = safe_df["ttft_p99.9_ratio"].quantile(0.9)
    norm_tpot_p99_baseline = safe_df["norm_tpot_p99_ratio"].quantile(0.9)
    norm_tpot_p99_9_baseline = safe_df["norm_tpot_p99.9_ratio"].quantile(0.9)
    output_trunc_baseline = safe_df["output_truncation_ratio"].quantile(0.1)

    print(f"\nBased on p90 of 'safe' experiments:")
    print(f"  1. TTFT p99/median > {ttft_p99_baseline:.1f}x")
    print(f"  2. TTFT p99.9/median > {ttft_p99_9_baseline:.1f}x")
    print(f"  3. norm_TPOT p99/median > {norm_tpot_p99_baseline:.1f}x")
    print(f"  4. norm_TPOT p99.9/median > {norm_tpot_p99_9_baseline:.1f}x")
    print(f"  5. Output p1/median < {output_trunc_baseline:.2f}")

    print(f"\nConservative thresholds (catching severe saturation):")
    print(f"  1. TTFT p99.9/median > 10x")
    print(f"  2. norm_TPOT p99.9/median > 10x")
    print(f"  3. Output p1/median < 0.5")
    print(f"  4. TTFT p99.9 > 2s (absolute)")

    print(f"\nSensitive thresholds (catching mild saturation):")
    print(f"  1. TTFT p99/median > 3x")
    print(f"  2. norm_TPOT p99/median > 3x")
    print(f"  3. Output p1/median < 0.8")

    # Count how many would be flagged
    print(f"\n{'='*80}")
    print("THRESHOLD APPLICATION")
    print(f"{'='*80}")

    # Conservative
    conservative = (
        (df["ttft_p99.9_ratio"] > 10) |
        (df["norm_tpot_p99.9_ratio"] > 10) |
        (df["output_truncation_ratio"] < 0.5) |
        (df["ttft_p99.9"] > 2.0)
    )
    print(f"\nConservative thresholds: {conservative.sum()}/{len(df)} experiments saturated")

    # Sensitive
    sensitive = (
        (df["ttft_p99_ratio"] > 3) |
        (df["norm_tpot_p99_ratio"] > 3) |
        (df["output_truncation_ratio"] < 0.8)
    )
    print(f"Sensitive thresholds: {sensitive.sum()}/{len(df)} experiments saturated")

    # Export statistics
    print(f"\n{'='*80}")
    print("DETAILED STATISTICS")
    print(f"{'='*80}")

    metrics_to_show = ["ttft_p99_ratio", "ttft_p99.9_ratio", "norm_tpot_p99_ratio",
                       "norm_tpot_p99.9_ratio", "output_truncation_ratio"]

    print("\nPercentiles across all experiments:")
    for metric in metrics_to_show:
        print(f"\n{metric}:")
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            val = df[metric].quantile(p)
            print(f"  p{int(p*100):2d}: {val:8.2f}x" if "ratio" in metric and "truncation" not in metric
                  else f"  p{int(p*100):2d}: {val:8.2%}")


if __name__ == "__main__":
    main()
