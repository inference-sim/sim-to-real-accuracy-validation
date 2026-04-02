#!/usr/bin/env python3
"""Generate comparison plot: BLIS-Evolved Iter16 vs Iter24."""

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
    """Generate iter16 vs iter24 comparison figure."""

    # Load iter16 results
    print("Loading iter16 results...")
    iter16_errors = load_error_data("results_iter16/error_records.csv")
    # Rename simulator to distinguish from iter24
    iter16_errors["simulator"] = iter16_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter16"
    )

    # Load iter24 results
    print("Loading iter24 results...")
    iter24_errors = load_error_data("results_iter24/error_records.csv")
    # Rename simulator to distinguish from iter16
    iter24_errors["simulator"] = iter24_errors["simulator"].replace(
        "blis-evolved", "blis-evolved-iter24"
    )

    # Filter to only blis-evolved from each
    iter16_errors = iter16_errors[iter16_errors["simulator"] == "blis-evolved-iter16"]
    iter24_errors = iter24_errors[iter24_errors["simulator"] == "blis-evolved-iter24"]

    print(f"  Iter16: {len(iter16_errors)} records, {iter16_errors['experiment_folder'].nunique()} experiments")
    print(f"  Iter24: {len(iter24_errors)} records, {iter24_errors['experiment_folder'].nunique()} experiments")

    # Combine
    combined = pd.concat([iter16_errors, iter24_errors], ignore_index=True)

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
    common_exps = iter16_exps & iter24_exps

    print(f"\nExperiment overlap:")
    print(f"  Common experiments: {len(common_exps)}")

    if len(common_exps) == 0:
        print("✗ No common experiments found - cannot generate comparison")
        return 1

    # Generate comparison plot
    print("\nGenerating comparison figure...")
    output_path = "ITER16_vs_ITER24_direct_comparison.pdf"

    # Temporarily add renamed simulators to SIMULATOR_ORDER for plotting
    original_sim_order = figures_module.SIMULATOR_ORDER.copy()
    original_display_names = figures_module.SIMULATOR_DISPLAY_NAMES.copy()
    original_color_palette = figures_module.COLOR_PALETTE.copy()
    original_hatch_patterns = figures_module.HATCH_PATTERNS.copy()

    figures_module.SIMULATOR_ORDER.extend(["blis-evolved-iter16", "blis-evolved-iter24"])
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter16"] = "BLIS-Evolved (Iter16)"
    figures_module.SIMULATOR_DISPLAY_NAMES["blis-evolved-iter24"] = "BLIS-Evolved (Iter24)"
    # Use distinct colors: blue for iter16, purple/magenta for iter24
    figures_module.COLOR_PALETTE["blis-evolved-iter16"] = "#4C72B0"  # Blue
    figures_module.COLOR_PALETTE["blis-evolved-iter24"] = "#D946EF"  # Purple/magenta
    figures_module.HATCH_PATTERNS["blis-evolved-iter16"] = "//"
    figures_module.HATCH_PATTERNS["blis-evolved-iter24"] = "||"

    try:
        fig = plot_simulator_comparison(
            combined,
            sim1="blis-evolved-iter16",
            sim2="blis-evolved-iter24",
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
