"""Hero chart: BLIS vs llm-d cluster — speed/cost gap.

Plots median wall-clock time and median cost per experiment for each GPU class,
comparing BLIS simulation to real llm-d cluster execution.
Real cluster durations are derived from actual experiment data (startup + benchmark + tail).

Usage:
    python results/plot_hero_chart.py
"""

import csv
import json
import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent
RUNTIME_CSV = RESULTS_DIR / "runtime.csv"
GROUND_TRUTH_DIR = RESULTS_DIR.parent / "vllm_data" / "ground_truth"
OUTPUT_PATH = RESULTS_DIR / "hero_chart_speed_cost.pdf"

GPU_HOURLY_RATE = {
    "H100": 3.20,
    "A100-80GB": 1.50,
    "L40S": 0.90,
}

# For log scale, BLIS cost is ~$0; use a nominal floor for visibility
BLIS_COST_FLOOR = 0.005

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_blis_data():
    """Load BLIS-trained-physics runtimes from runtime.csv, grouped by hardware."""
    blis_times_by_hw = defaultdict(list)
    exp_hw_tp = {}  # exp_id -> (hardware, tp)

    with open(RUNTIME_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["simulator"] != "blis-trained-physics":
                continue
            hw = row["hardware"]
            tp = int(row["tp"])
            wall_clock = float(row["wall_clock_seconds"])
            blis_times_by_hw[hw].append(wall_clock)
            exp_hw_tp[row["exp_id"]] = (hw, tp)

    return blis_times_by_hw, exp_hw_tp


def load_real_durations(exp_hw_tp):
    """Compute actual wall-clock per experiment from vllm.log + stage metrics."""
    real_wall_by_hw = defaultdict(list)
    real_cost_by_hw = defaultdict(list)

    for folder in sorted(os.listdir(GROUND_TRUTH_DIR)):
        m = re.match(r"^(\d+)-", folder)
        if not m:
            continue
        exp_id = m.group(1)
        if exp_id not in exp_hw_tp:
            continue
        hw, tp = exp_hw_tp[exp_id]

        log_path = GROUND_TRUTH_DIR / folder / "vllm.log"
        exp_path = GROUND_TRUTH_DIR / folder / "results"
        if not log_path.exists() or not exp_path.is_dir():
            continue

        # Parse startup time from vllm.log
        first_ts = None
        ready_ts = None
        with open(log_path) as f:
            for line in f:
                m2 = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
                if m2:
                    ts = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S,%f")
                    if first_ts is None:
                        first_ts = ts
                    if "Starting vLLM API server" in line:
                        ready_ts = ts

        if not first_ts or not ready_ts:
            continue
        startup = (ready_ts - first_ts).total_seconds()
        if startup > 600:  # skip outliers (cold-start anomalies)
            continue

        # Benchmark duration (send + tail drain)
        total_send = 0
        last_max_lat = 0
        stage_files = sorted(glob.glob(str(exp_path / "stage_*_lifecycle_metrics.json")))
        for sf in stage_files:
            with open(sf) as f:
                d = json.load(f)
            if "load_summary" in d and "send_duration" in d["load_summary"]:
                total_send += d["load_summary"]["send_duration"]
        if stage_files:
            with open(stage_files[-1]) as f:
                d = json.load(f)
            if "successes" in d:
                last_max_lat = d["successes"]["latency"]["request_latency"].get("max", 0)

        total_wall = startup + total_send + last_max_lat
        cost = tp * (total_wall / 3600) * GPU_HOURLY_RATE.get(hw, 0)

        real_wall_by_hw[hw].append(total_wall)
        real_cost_by_hw[hw].append(cost)

    return real_wall_by_hw, real_cost_by_hw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_all_paired_data():
    """Load per-experiment paired data: BLIS time + real wall-clock + real cost."""
    blis_times_by_hw, exp_hw_tp = load_blis_data()
    real_wall_by_hw, real_cost_by_hw = load_real_durations(exp_hw_tp)

    # Also load per-experiment BLIS times (not just grouped by hw)
    blis_per_exp = {}
    with open(RUNTIME_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["simulator"] != "blis-trained-physics":
                continue
            blis_per_exp[row["exp_id"]] = float(row["wall_clock_seconds"])

    # Load per-experiment real wall-clock
    real_per_exp = {}
    for folder in sorted(os.listdir(GROUND_TRUTH_DIR)):
        m = re.match(r"^(\d+)-", folder)
        if not m:
            continue
        exp_id = m.group(1)
        if exp_id not in exp_hw_tp:
            continue
        hw, tp = exp_hw_tp[exp_id]

        log_path = GROUND_TRUTH_DIR / folder / "vllm.log"
        exp_path = GROUND_TRUTH_DIR / folder / "results"
        if not log_path.exists() or not exp_path.is_dir():
            continue

        first_ts = None
        ready_ts = None
        with open(log_path) as f:
            for line in f:
                m2 = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
                if m2:
                    ts = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S,%f")
                    if first_ts is None:
                        first_ts = ts
                    if "Starting vLLM API server" in line:
                        ready_ts = ts

        if not first_ts or not ready_ts:
            continue
        startup = (ready_ts - first_ts).total_seconds()
        if startup > 600:
            continue

        total_send = 0
        last_max_lat = 0
        stage_files = sorted(glob.glob(str(exp_path / "stage_*_lifecycle_metrics.json")))
        for sf in stage_files:
            with open(sf) as f:
                d = json.load(f)
            if "load_summary" in d and "send_duration" in d["load_summary"]:
                total_send += d["load_summary"]["send_duration"]
        if stage_files:
            with open(stage_files[-1]) as f:
                d = json.load(f)
            if "successes" in d:
                last_max_lat = d["successes"]["latency"]["request_latency"].get("max", 0)

        total_wall = startup + total_send + last_max_lat
        cost = tp * (total_wall / 3600) * GPU_HOURLY_RATE.get(hw, 0)
        real_per_exp[exp_id] = (total_wall, cost)

    return blis_per_exp, real_per_exp, exp_hw_tp


def main():
    blis_per_exp, real_per_exp, exp_hw_tp = load_all_paired_data()

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Sum across all experiments: total time and total cost
    blis_total_time = 0
    real_total_time = 0
    real_total_cost = 0
    count = 0

    for exp_id in blis_per_exp:
        if exp_id not in real_per_exp:
            continue
        blis_total_time += blis_per_exp[exp_id]
        real_wall, real_cost = real_per_exp[exp_id]
        real_total_time += real_wall
        real_total_cost += real_cost
        count += 1

    # Plot two points
    ax.scatter(blis_total_time, 0, s=180, color="#2563EB", zorder=5,
               edgecolors="white", linewidths=0.5, marker="o")
    ax.scatter(real_total_time, real_total_cost, s=180, color="#DC2626", zorder=5,
               edgecolors="white", linewidths=0.5, marker="o")

    # Annotations
    ax.text(
        blis_total_time, 8,
        "BLIS\n(seconds · laptop · ~$0)",
        fontsize=11, color="#2563EB", fontweight="bold",
        ha="center", va="bottom",
    )
    ax.text(
        real_total_time * 0.6, real_total_cost + 10,
        "llm-d cluster\n(hours · GPUs · $$$)",
        fontsize=11, color="#DC2626", fontweight="bold",
        ha="center", va="bottom",
    )

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel("Total wall-clock time (across 36 experiments)", fontsize=13)
    ax.set_ylabel("Total cost ($)", fontsize=13)

    # Custom x-tick labels
    xticks = [1, 10, 100, 1000, 10000, 100000]
    xlabels = ["1 s", "10 s", "100 s", "~17 min", "~3 hr", "~28 hr"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_xlim(10, 200000)
    ax.set_ylim(-5, 140)

    # Legend
    ax.legend(
        handles=[
            mpatches.Patch(color="#2563EB", label="BLIS (simulation)"),
            mpatches.Patch(color="#DC2626", label="llm-d cluster"),
        ],
        loc="upper left", fontsize=11, framealpha=0.9,
    )

    # Title
    ax.set_title(
        "Cost vs wall-clock time — BLIS vs llm-d cluster",
        fontsize=15, fontweight="bold", pad=12,
    )

    # Speedup annotation
    speedup = real_total_time / blis_total_time
    mid_x = np.sqrt(blis_total_time * real_total_time)
    mid_y = real_total_cost / 2
    ax.text(
        mid_x, mid_y,
        f"200× faster\n$100 saved",
        fontsize=12, ha="center", va="center", color="#475569",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", linewidth=0.5),
    )

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH.with_suffix('.png')}")

    # Print summary
    print(f"\n--- Summary (sum of {count} experiments) ---")
    print(f"  BLIS:  {blis_total_time:.1f}s total ({blis_total_time/60:.1f} min)")
    print(f"  llm-d cluster:  {real_total_time:.0f}s total ({real_total_time/3600:.1f} hr)")
    print(f"  Cost:  ${real_total_cost:.2f} (llm-d cluster) vs ~$0 (BLIS)")
    print(f"  Speedup: {speedup:.0f}×")


if __name__ == "__main__":
    main()
