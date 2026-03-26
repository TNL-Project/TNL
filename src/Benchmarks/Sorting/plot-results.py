#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""Visualize sorting benchmark results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TNL import BenchmarkLogs

MARKERS = ["o", "s", "^", "v", "D", "p", "*", "h", "X", "P"]
LINESTYLES = ["-", "--", ":", "-."]
COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sorting benchmark results")
    parser.add_argument("log_file", help="Path to the benchmark log file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory for output plots (default: current directory)",
    )
    return parser.parse_args()


def load_data(log_file):
    df = BenchmarkLogs.get_benchmark_dataframe(log_file)
    metadata = BenchmarkLogs.get_benchmark_metadata(log_file)
    return df, metadata


def prepare_data(df):
    df["algo_device"] = df["performer"] + " (" + df["device"] + ")"
    return df


def get_algo_data(subset, algo_device):
    algo_data = subset[subset["algo_device"] == algo_device].copy()
    if algo_data.empty:
        return None
    algo_data = algo_data.sort_values("size")
    sizes = algo_data["size"].astype(float).to_numpy()
    times = algo_data["time"].to_numpy()
    stddev_vals = algo_data.get("time_stddev")
    if stddev_vals is not None and len(stddev_vals) > 0:
        stddev = stddev_vals.to_numpy()
    else:
        stddev = np.zeros(len(times))
    return sizes, times, stddev


def setup_axes(ax, title, xlabel="Array size", ylabel="Time [s]"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.grid(True, alpha=0.3)


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_distribution(df, value_type, distribution, algo_devices, output_dir):
    subset = df[(df["value_type"] == value_type) & (df["distribution"] == distribution)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, algo_dev in enumerate(algo_devices):
        data = get_algo_data(subset, algo_dev)
        if data is None:
            continue
        sizes, times, stddev = data
        ax.errorbar(
            sizes,
            times,
            yerr=stddev,
            label=algo_dev,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            capsize=3,
            markersize=6,
            linewidth=1.5,
        )

    setup_axes(ax, f"Sorting Benchmark: {distribution}, {value_type}")
    filename = output_dir / f"sort-{value_type}-{distribution.replace('-', '_')}.pdf"
    save_plot(fig, filename)


def plot_all_results(df, value_types, algo_devices, output_dir):
    fig, ax = plt.subplots(figsize=(14, 9))

    for i, algo_dev in enumerate(algo_devices):
        for j, value_type in enumerate(value_types):
            subset = df[
                (df["algo_device"] == algo_dev) & (df["value_type"] == value_type)
            ]
            if subset.empty:
                continue

            grouped = subset.groupby("size")["time"].agg(["mean", "std"]).reset_index()
            grouped = grouped.sort_values("size")

            ax.errorbar(
                grouped["size"].astype(float).to_numpy(),
                grouped["mean"].to_numpy(),
                yerr=grouped["std"].to_numpy(),
                label=f"{algo_dev}, {value_type}",
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[j % len(LINESTYLES)],
                capsize=3,
                linewidth=1.5,
                markersize=5,
            )

    setup_axes(ax, "Sorting Benchmark: All Results")
    ax.legend(fontsize=8)
    filename = output_dir / "sort-all.pdf"
    save_plot(fig, filename)


def main():
    args = parse_args()

    df, _metadata = load_data(args.log_file)
    if df.empty:
        print("No data found in log file")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(df)

    distributions = df["distribution"].unique()
    value_types = df["value_type"].unique()
    algo_devices = df["algo_device"].unique()

    print(f"Loaded {len(df)} benchmark results")
    print(f"Algorithms: {list(df['performer'].unique())}")
    print(f"Distributions: {list(distributions)}")
    print(f"Value types: {list(value_types)}")

    for value_type in value_types:
        for distribution in distributions:
            plot_distribution(df, value_type, distribution, algo_devices, output_dir)

    plot_all_results(df, value_types, algo_devices, output_dir)


if __name__ == "__main__":
    main()
