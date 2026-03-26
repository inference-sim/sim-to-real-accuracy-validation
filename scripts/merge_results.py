#!/usr/bin/env python3
"""Merge local and cluster evaluation results.

Usage:
    python scripts/merge_results.py

Expects:
    - results/error_records.csv (local simulators)
    - results/runtime.csv (local simulators)
    - cluster_results/error_records.csv (LLMServingSim from cluster)
    - cluster_results/runtime.csv (LLMServingSim from cluster)

Produces:
    - results/error_records.csv (merged, 8 simulators)
    - results/runtime.csv (merged, 8 simulators)
    - results/error_records_local.csv (backup of original local results)
    - results/runtime_local.csv (backup of original local results)
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Install with: pip install pandas")
    sys.exit(1)


def merge_results():
    """Merge local and cluster CSV results."""

    # Define paths
    local_errors = Path("results/error_records.csv")
    local_runtime = Path("results/runtime.csv")
    cluster_errors = Path("cluster_results/error_records.csv")
    cluster_runtime = Path("cluster_results/runtime.csv")

    # Check all files exist
    missing = []
    for f in [local_errors, local_runtime, cluster_errors, cluster_runtime]:
        if not f.exists():
            missing.append(str(f))

    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nExpected directory structure:")
        print("  results/error_records.csv (local)")
        print("  results/runtime.csv (local)")
        print("  cluster_results/error_records.csv (cluster)")
        print("  cluster_results/runtime.csv (cluster)")
        sys.exit(1)

    print("Loading CSV files...")

    # Read local results
    try:
        local_err_df = pd.read_csv(local_errors)
        local_rt_df = pd.read_csv(local_runtime)
    except Exception as e:
        print(f"ERROR reading local results: {e}")
        sys.exit(1)

    # Read cluster results
    try:
        cluster_err_df = pd.read_csv(cluster_errors)
        cluster_rt_df = pd.read_csv(cluster_runtime)
    except Exception as e:
        print(f"ERROR reading cluster results: {e}")
        sys.exit(1)

    print(f"Local results: {len(local_err_df)} error records, {len(local_rt_df)} runtime records")
    print(f"Cluster results: {len(cluster_err_df)} error records, {len(cluster_rt_df)} runtime records")

    # Check for simulator overlap (shouldn't happen)
    local_sims = set(local_err_df['simulator'].unique())
    cluster_sims = set(cluster_err_df['simulator'].unique())
    overlap = local_sims & cluster_sims

    if overlap:
        print(f"WARNING: Simulators present in both local and cluster results: {overlap}")
        print("This may indicate duplicate runs. Duplicates will be removed.")

    # Merge error records
    print("\nMerging error records...")
    merged_err = pd.concat([local_err_df, cluster_err_df], ignore_index=True)

    # Remove duplicates (keep first occurrence)
    before_dedup = len(merged_err)
    merged_err = merged_err.drop_duplicates(
        subset=['simulator', 'experiment_folder', 'stage_index', 'metric_name'],
        keep='first'
    )
    after_dedup = len(merged_err)

    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate error records")

    # Merge runtime records
    print("Merging runtime records...")
    merged_rt = pd.concat([local_rt_df, cluster_rt_df], ignore_index=True)

    # Remove duplicates
    before_dedup = len(merged_rt)
    merged_rt = merged_rt.drop_duplicates(
        subset=['simulator', 'experiment_folder'],
        keep='first'
    )
    after_dedup = len(merged_rt)

    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate runtime records")

    # Backup originals
    print("\nBacking up original local results...")
    local_errors.rename(local_errors.parent / "error_records_local.csv")
    local_runtime.rename(local_runtime.parent / "runtime_local.csv")
    print("  - results/error_records_local.csv")
    print("  - results/runtime_local.csv")

    # Save merged results
    print("\nSaving merged results...")
    merged_err.to_csv(local_errors, index=False)
    merged_rt.to_csv(local_runtime, index=False)

    # Summary
    print("\nMerge complete!")
    print(f"  - {len(merged_err)} total error records")
    print(f"  - {len(merged_rt)} total runtime records")
    print(f"  - {len(merged_err['simulator'].unique())} unique simulators:")
    for sim in sorted(merged_err['simulator'].unique()):
        count = len(merged_rt[merged_rt['simulator'] == sim])
        print(f"    - {sim}: {count} experiments")

    print("\nNext steps:")
    print("  python -m experiment.figures --results-dir results --output-dir results/figures")


if __name__ == "__main__":
    merge_results()
