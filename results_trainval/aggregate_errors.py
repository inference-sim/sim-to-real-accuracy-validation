#!/usr/bin/env python3
"""
Aggregate simulator errors organized by experiment.
"""
import pandas as pd
import json
import numpy as np

# Read the error records
df = pd.read_csv('error_records.csv')

# Skip the header row if it got included as data
df = df[df['simulator'] != 'simulator']

# Convert numeric columns
numeric_cols = ['mape', 'mpe', 'absolute_error', 'predicted', 'actual', 'tp', 'max_num_batched_tokens', 'stage_index']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter to only summary stage measurements (stage_index == -1)
print(f"Total rows before filtering: {len(df)}")
df = df[df['stage_index'] == -1]
print(f"Total rows after filtering to summary stage (stage_index == -1): {len(df)}")
print()

# Get unique experiments
experiments = {}

# Group by experiment
for exp_id in df['exp_id'].unique():
    exp_df = df[df['exp_id'] == exp_id]

    # Get experiment metadata (should be same across all rows for this exp_id)
    first_row = exp_df.iloc[0]

    exp_key = f"exp_{exp_id}"
    experiments[exp_key] = {
        "metadata": {
            "exp_id": int(exp_id),
            "model": first_row['model'],
            "workload": first_row['workload'],
            "hardware": first_row['hardware'] if pd.notna(first_row['hardware']) else None,
            "tp": int(first_row['tp']) if pd.notna(first_row['tp']) else None,
            "precision": first_row['precision'] if pd.notna(first_row['precision']) else None,
            "max_num_batched_tokens": int(first_row['max_num_batched_tokens']) if pd.notna(first_row['max_num_batched_tokens']) else None,
        },
        "simulators": {}
    }

    # For each simulator that tested this experiment
    for simulator in exp_df['simulator'].unique():
        sim_exp_df = exp_df[exp_df['simulator'] == simulator]

        experiments[exp_key]["simulators"][simulator] = {}

        # For each metric, store only the MPE
        for _, row in sim_exp_df.iterrows():
            metric_name = row['metric_name']
            experiments[exp_key]["simulators"][simulator][f"{metric_name}_mpe"] = float(row['mpe'])

results = {
    "summary": {
        "total_experiments": int(df['exp_id'].nunique()),
        "total_measurements": len(df),
        "simulators": sorted(df['simulator'].unique().tolist()),
        "models": sorted(df['model'].unique().tolist()),
        "workloads": sorted(df['workload'].unique().tolist()),
        "metrics": sorted(df['metric_name'].unique().tolist())
    },
    "experiments": experiments
}

# Write to JSON file
output_file = 'aggregated_errors.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Aggregated error analysis written to {output_file}")
print(f"\nSummary:")
print(f"  Total experiments: {results['summary']['total_experiments']}")
print(f"  Total measurements: {results['summary']['total_measurements']}")
print(f"  Simulators: {len(results['summary']['simulators'])}")
print(f"  Models: {len(results['summary']['models'])}")
print(f"  Workloads: {len(results['summary']['workloads'])}")
print(f"  Metrics tracked: {results['summary']['metrics']}")
print(f"\n  Experiment IDs: {sorted([int(k.split('_')[1]) for k in results['experiments'].keys()])}")
