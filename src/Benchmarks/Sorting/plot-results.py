#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""Visualize sorting benchmark results."""

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from TNL import BenchmarkLogs

sns.set_theme()

DEFAULT_OUTPUT_DIR = Path("sorting-plots")

_DEVICE_ORDER = {"sequential": 0, "host": 1, "cuda": 2, "hip": 3}


def _algo_device_sort_key(algo_device: str) -> tuple[int, str]:
    """Sort key: device first (sequential→host→cuda→hip), then algorithm name."""
    algo, device = algo_device.rsplit(" (", 1)
    device = device.rstrip(")")
    return (_DEVICE_ORDER.get(device, len(_DEVICE_ORDER)), algo)


def _load_dataframes(filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate sorting benchmark log files."""
    frames: list[pd.DataFrame] = []
    for filename in filenames:
        path = Path(filename)
        if not path.exists():
            print(f"Skipping non-existing input file {filename} ...")
            continue
        frames.append(BenchmarkLogs.get_benchmark_dataframe(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    numeric_cols = ["size", "time", "time_stddev", "bandwidth"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["algo_device"] = df["performer"] + " (" + df["device"] + ")"
    return df


def writeFigure(
    df: pd.DataFrame,
    *,
    x: str = "size",
    y: str = "time",
    y_err: str | None = None,
    hue: str,
    hue_title: str = "",
    hue_order: list[str] | None = None,
    style: str | None = None,
    title: str = "",
    y_label: str = "Time [s]",
    file_path: Path,
) -> None:
    """Write a comparison figure using seaborn lineplot with optional error band."""
    # Rename the hue column so seaborn uses it as the legend section title
    # instead of the raw column name
    if hue_title:
        df = df.rename(columns={hue: hue_title})
        hue = hue_title
    if hue_order is None:
        hue_order = sorted(df[hue].unique())
    palette = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        style=style,
        marker="o",
        errorbar=None,
        palette=palette,
        ax=ax,
    )
    if y_err and y_err in df.columns:
        # Aggregate before fill_between — seaborn averages internally but
        # raw fill_between doesn't, so we must groupby([hue, x]) first
        agg = df.groupby([hue, x]).agg({y: "mean", y_err: "max"}).reset_index()
        for name, group in agg.groupby(hue, sort=True):
            sg = group.sort_values(x)
            xs = sg[x].to_numpy(dtype=float)
            ys = sg[y].to_numpy(dtype=float)
            errs = sg[y_err].to_numpy(dtype=float)
            ax.fill_between(
                xs, ys - errs, ys + errs, alpha=0.15, color=palette[str(name)]
            )
    ax.set_xlabel("Array size")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {file_path}")


def writeDistributionFigures(
    df: pd.DataFrame,
    value_types: list[str],
    distributions: list[str],
    hue_order: list[str],
    output_dir: Path,
) -> None:
    """Write figures comparing algorithms per distribution and value type."""
    for value_type, distribution in product(value_types, distributions):
        sub: pd.DataFrame = df[
            (df["value type"] == value_type) & (df["distribution"] == distribution)
        ]  # pyright: ignore[reportAssignmentType]
        if sub.empty:
            continue
        file_path = (
            output_dir / f"sort-{value_type}-{distribution.replace('-', '_')}.svg"
        )
        title = f"Sorting Benchmark: {distribution}, {value_type}"
        writeFigure(
            sub,
            y="time",
            y_err="time_stddev",
            hue="algo_device",
            hue_title="Algorithm (Device)",
            hue_order=hue_order,
            title=title,
            file_path=file_path,
        )


def writeAllResultsFigure(
    df: pd.DataFrame,
    algo_order: list[str],
    output_dir: Path,
) -> None:
    """Write figure with all algorithms and value types,
    averaged across all distributions.
    Fill-between shows min-to-max range across distributions.
    """
    # Rename columns so seaborn uses them as legend section titles
    # instead of raw column names
    df = df.rename(
        columns={"algo_device": "Algorithm (Device)", "value type": "Value type"}
    )
    algo_col = "Algorithm (Device)"
    hue_order = [a for a in algo_order if a in df[algo_col].unique()]
    # Pre-aggregate across distributions: mean, min, max
    # per (algorithm, value type, size)
    agg_df = (
        df.groupby([algo_col, "Value type", "size"])
        .agg(
            time_mean=("time", "mean"),
            time_min=("time", "min"),
            time_max=("time", "max"),
        )
        .reset_index()
    )
    palette = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.lineplot(
        data=agg_df,
        x="size",
        y="time_mean",
        hue=algo_col,
        hue_order=hue_order,
        style="Value type",
        marker="o",
        errorbar=None,
        palette=palette,
        ax=ax,
    )
    # Fill bands: min/max of time across distributions per (algorithm, value type, size)
    for _name, group in agg_df.groupby([algo_col, "Value type"], sort=True):
        sg = group.sort_values("size")
        algo = str(sg[algo_col].iloc[0])
        xs = sg["size"].to_numpy(dtype=float)
        y_lo = sg["time_min"].to_numpy(dtype=float)
        y_hi = sg["time_max"].to_numpy(dtype=float)
        ax.fill_between(xs, y_lo, y_hi, alpha=0.15, color=palette[algo])
    ax.set_xlabel("Array size")
    ax.set_ylabel("Time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sorting Benchmark: Average across all distributions")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    file_path = output_dir / "sort-all.svg"
    fig.savefig(file_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sorting benchmark results")
    parser.add_argument(
        "input",
        nargs="+",
        help="Input log files (JSON lines format)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframes(args.input)
    if df.empty:
        print("No data found in log files")
        return

    distributions = sorted(df["distribution"].unique())
    value_types = sorted(df["value type"].unique())
    algo_order = sorted(df["algo_device"].unique(), key=_algo_device_sort_key)

    print(f"Loaded {len(df)} benchmark results")
    print(f"Algorithms: {list(df['performer'].unique())}")
    print(f"Distributions: {list(distributions)}")
    print(f"Value types: {list(value_types)}")

    writeDistributionFigures(
        df, value_types, distributions, algo_order, args.output_dir
    )
    writeAllResultsFigure(df, algo_order, args.output_dir)


if __name__ == "__main__":
    main()
