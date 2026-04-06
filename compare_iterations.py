#!/usr/bin/env python3
"""Generate comparison plot: BLIS-Evolved Iter16 vs Iter24 vs Iter26 vs Iter27 vs Iter29."""

import pandas as pd
import sys
import os

# Add experiment module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment.figures import (
    load_error_data,
    enrich_with_metadata,
    _add_config_tags,
    plot_simulator_comparison,
)
import experiment.figures as figures_module


def main():
    """Generate iter16 vs iter24 vs iter26 vs iter27 vs iter29 comparison figure.

    Note: This script requires results_iter16/, results_iter24/, results_iter26/,
    results_iter27/, and results_iter29/ directories with error_records.csv files.
    Generate them by running:
        python -m experiment.run --adapters blis-evolved --blis-evolved-iteration {16,24,26,27,29}
    """

    # Load iter16 results
    print("Loading iter16 results...")
    try:
        iter16_errors = load_error_data("results_iter16/error_records.csv")
    except FileNotFoundError:
        print("✗ results_iter16/error_records.csv not found")
        print("  Run: python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 16")
        return 1
    # Rename simulator to distinguish from iter24/iter26
    iter16_errors["simulator"] = iter16_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter16"
    )

    # Load iter24 results
    print("Loading iter24 results...")
    try:
        iter24_errors = load_error_data("results_iter24/error_records.csv")
    except FileNotFoundError:
        print("✗ results_iter24/error_records.csv not found")
        print("  Run: python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 24")
        return 1
    # Rename simulator to distinguish from iter16/iter26
    iter24_errors["simulator"] = iter24_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter24"
    )

    # Load iter26 results
    print("Loading iter26 results...")
    try:
        iter26_errors = load_error_data("results_iter26/error_records.csv")
    except FileNotFoundError:
        print("✗ results_iter26/error_records.csv not found")
        print("  Run: python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 26")
        return 1
    # Rename simulator to distinguish from iter16/iter24
    iter26_errors["simulator"] = iter26_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter26"
    )

    # Load iter27 results
    print("Loading iter27 results...")
    try:
        iter27_errors = load_error_data("results_iter27/error_records.csv")
    except FileNotFoundError:
        print("✗ results_iter27/error_records.csv not found")
        print("  Run: python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 27")
        return 1
    # Rename simulator to distinguish from iter16/iter24/iter26
    iter27_errors["simulator"] = iter27_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter27"
    )

    # Load iter29 results
    print("Loading iter29 results...")
    try:
        iter29_errors = load_error_data("results_iter29/error_records.csv")
    except FileNotFoundError:
        print("✗ results_iter29/error_records.csv not found")
        print("  Run: python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 29")
        return 1
    # Rename simulator to distinguish from iter16/iter24/iter26/iter27
    iter29_errors["simulator"] = iter29_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter29"
    )

    # Filter to only blis-evolved from each
    iter16_errors = iter16_errors[iter16_errors["simulator"] == "blis-evolved-iter16"]
    iter24_errors = iter24_errors[iter24_errors["simulator"] == "blis-evolved-iter24"]
    iter26_errors = iter26_errors[iter26_errors["simulator"] == "blis-evolved-iter26"]
    iter27_errors = iter27_errors[iter27_errors["simulator"] == "blis-evolved-iter27"]
    iter29_errors = iter29_errors[iter29_errors["simulator"] == "blis-evolved-iter29"]

    print(f"  Iter16: {len(iter16_errors)} records, {iter16_errors['experiment_folder'].nunique()} experiments")
    print(f"  Iter24: {len(iter24_errors)} records, {iter24_errors['experiment_folder'].nunique()} experiments")
    print(f"  Iter26: {len(iter26_errors)} records, {iter26_errors['experiment_folder'].nunique()} experiments")
    print(f"  Iter27: {len(iter27_errors)} records, {iter27_errors['experiment_folder'].nunique()} experiments")
    print(f"  Iter29: {len(iter29_errors)} records, {iter29_errors['experiment_folder'].nunique()} experiments")

    # Combine
    combined = pd.concat([iter16_errors, iter24_errors, iter26_errors, iter27_errors, iter29_errors], ignore_index=True)

    # Try to add metadata and config tags (optional)
    print("Enriching with metadata...")
    try:
        combined = enrich_with_metadata(combined, metadata_path=None)
        combined = _add_config_tags(combined)
    except Exception as e:
        print(f"  Warning: Could not enrich metadata: {e}")
        print("  Continuing without metadata enrichment...")

    # Check for common experiments
    iter16_exps = set(combined[combined["simulator"] == "blis-evolved-iter16"]["experiment_folder"].unique())
    iter24_exps = set(combined[combined["simulator"] == "blis-evolved-iter24"]["experiment_folder"].unique())
    iter26_exps = set(combined[combined["simulator"] == "blis-evolved-iter26"]["experiment_folder"].unique())
    iter27_exps = set(combined[combined["simulator"] == "blis-evolved-iter27"]["experiment_folder"].unique())
    iter29_exps = set(combined[combined["simulator"] == "blis-evolved-iter29"]["experiment_folder"].unique())
    common_exps = iter16_exps & iter24_exps & iter26_exps & iter27_exps & iter29_exps

    print(f"\nExperiment overlap:")
    print(f"  Common experiments: {len(common_exps)}")
    print(f"  Iter16 only: {len(iter16_exps - iter24_exps - iter26_exps - iter27_exps - iter29_exps)}")
    print(f"  Iter24 only: {len(iter24_exps - iter16_exps - iter26_exps - iter27_exps - iter29_exps)}")
    print(f"  Iter26 only: {len(iter26_exps - iter16_exps - iter24_exps - iter27_exps - iter29_exps)}")
    print(f"  Iter27 only: {len(iter27_exps - iter16_exps - iter24_exps - iter26_exps - iter29_exps)}")
    print(f"  Iter29 only: {len(iter29_exps - iter16_exps - iter24_exps - iter26_exps - iter27_exps)}")

    if len(common_exps) == 0:
        print("✗ No common experiments found - cannot generate comparison")
        return 1

    # Generate comparison plot
    print("\nGenerating comparison figure...")
    output_path = "ITER16_vs_ITER24_vs_ITER26_vs_ITER27_vs_ITER29_comparison.pdf"

    # Temporarily add renamed simulators to SIMULATOR_ORDER for plotting
    original_sim_order = figures_module.SIMULATOR_ORDER.copy()
    original_display_names = figures_module.SIMULATOR_DISPLAY_NAMES.copy()
    original_color_palette = figures_module.COLOR_PALETTE.copy()
    original_hatch_patterns = figures_module.HATCH_PATTERNS.copy()

    figures_module.SIMULATOR_ORDER.extend(["blis-evolved-iter16", "blis-evolved-iter24", "blis-evolved-iter26", "blis-evolved-iter27", "blis-evolved-iter29"])
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter16"] = "BLIS-Evolved (Iter16)"
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter24"] = "BLIS-Evolved (Iter24)"
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter26"] = "BLIS-Evolved (Iter26)"
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter27"] = "BLIS-Evolved (Iter27)"
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter29"] = "BLIS-Evolved (Iter29)"
    # Use distinct colors: blue for iter16, orange for iter24, purple for iter26, green for iter27, red for iter29
    figures_module.COLOR_PALETTE["blis-evolved-iter16"] = "#4C72B0"  # Blue
    figures_module.COLOR_PALETTE["blis-evolved-iter24"] = "#DD8452"  # Orange
    figures_module.COLOR_PALETTE["blis-evolved-iter26"] = "#D946EF"  # Purple/magenta
    figures_module.COLOR_PALETTE["blis-evolved-iter27"] = "#55A868"  # Green
    figures_module.COLOR_PALETTE["blis-evolved-iter29"] = "#C44E52"  # Red
    figures_module.HATCH_PATTERNS["blis-evolved-iter16"] = "//"
    figures_module.HATCH_PATTERNS["blis-evolved-iter24"] = "\\\\"
    figures_module.HATCH_PATTERNS["blis-evolved-iter26"] = "||"
    figures_module.HATCH_PATTERNS["blis-evolved-iter27"] = ".."
    figures_module.HATCH_PATTERNS["blis-evolved-iter29"] = "xx"

    try:
        fig = plot_simulator_comparison(
            combined,
            sim1=["blis-evolved-iter16", "blis-evolved-iter24", "blis-evolved-iter26", "blis-evolved-iter27"],
            sim2="blis-evolved-iter29",
            output_path=output_path,
        )
    finally:
        # Restore original values
        figures_module.SIMULATOR_ORDER = original_sim_order
        figures_module.SIMULATOR_DISPLAY_NAMES = original_display_names
        figures_module.COLOR_PALETTE = original_color_palette
        figures_module.HATCH_PATTERNS = original_hatch_patterns

    if fig:
        print(f"✓ Figure saved to {output_path}")
        # Also save PNG
        png_path = output_path.replace(".pdf", ".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved to {png_path}")
    else:
        print("✗ Failed to generate figure")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
