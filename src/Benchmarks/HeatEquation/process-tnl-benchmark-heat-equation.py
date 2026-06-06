#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
import sys
from pathlib import Path

import pandas as pd

from TNL import BenchmarkLogs

DEFAULT_OUTPUT_DIR = Path("heat-equation-plots")

_DEVICE_ORDER = {"sequential": 0, "host": 1, "cuda": 2, "hip": 3}


def _sorted_devices(devices: list[str]) -> list[str]:
    """Sort devices by conventional order: sequential → host → cuda → hip."""
    return sorted(devices, key=lambda d: _DEVICE_ORDER.get(d, len(_DEVICE_ORDER)))


def _load_dataframes(filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate heat equation benchmark log files."""
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

    numeric_cols = ["x size", "y size", "z size", "time", "bandwidth"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_multiindex(input_df: pd.DataFrame, implementations: list[str]) -> pd.MultiIndex:
    """Create multi-index columns for the pivot table based on the actual data."""
    level1: list[str] = ["x size", "y size"]
    level2: list[str] = ["", ""]
    level3: list[str] = ["", ""]

    available_devices = _sorted_devices(list(input_df["device"].unique()))

    for impl in implementations:
        for device in available_devices:
            values = ["time"]
            if impl != "parallel-for":
                values.append("parallel-for speed-up")
            if device == "cuda":
                values.append("CPU speed-up")
            for value in values:
                level1.append(impl)
                level2.append(device)
                level3.append(value)

    return pd.MultiIndex.from_arrays([level1, level2, level3])


def compute_speedup(
    df: pd.DataFrame, available_devices: list[str], implementations: list[str]
) -> None:
    """Compute speedup columns from baselines (vectorized)."""
    have_cuda = "cuda" in available_devices and "host" in available_devices
    if have_cuda:
        for impl in implementations:
            cuda_key = (impl, "cuda", "time")
            host_key = (impl, "host", "time")
            if cuda_key in df.columns and host_key in df.columns:
                df[(impl, "cuda", "CPU speed-up")] = df[host_key] / df[cuda_key]

    for impl in implementations:
        if impl == "parallel-for":
            continue
        for device in available_devices:
            pf_key = ("parallel-for", device, "time")
            impl_key = (impl, device, "time")
            speedup_key = (impl, device, "parallel-for speed-up")
            if pf_key in df.columns and impl_key in df.columns:
                df[speedup_key] = df[pf_key] / df[impl_key]


def build_wide_table(
    input_df: pd.DataFrame, multicolumns: pd.MultiIndex, implementations: list[str]
) -> pd.DataFrame:
    """Convert flat input table to a structured one using multi-index columns."""
    available_devices = _sorted_devices(list(input_df["device"].unique()))
    frames: list[pd.DataFrame] = []
    out_idx = 0

    x_sizes = sorted(set(input_df["x size"]))
    y_sizes = sorted(set(input_df["y size"]))

    for x_size in x_sizes:
        for y_size in y_sizes:
            subset = input_df[
                (input_df["x size"] == x_size) & (input_df["y size"] == y_size)
            ]
            row_data: dict[tuple[str, ...], object] = {
                col: float("nan") for col in multicolumns
            }
            row_data[("x size", "", "")] = x_size
            row_data[("y size", "", "")] = y_size
            for _index, row in subset.iterrows():
                impl = str(row["implementation"])
                device = str(row["device"])
                time_val = pd.to_numeric(row["time"], errors="coerce")
                row_data[(impl, device, "time")] = time_val
            aux_df = pd.DataFrame(
                [row_data], columns=multicolumns, index=pd.Index([out_idx])
            )
            frames.append(aux_df)
            out_idx += 1

    result = pd.concat(frames)
    result.replace("", float("nan"), inplace=True)
    compute_speedup(result, available_devices, implementations)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for processing TNL benchmark heat equation results."
    )
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
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_df = _load_dataframes(args.input)
    if input_df.empty:
        print("No data found in log files", file=sys.stderr)
        return

    raw_path = args.output_dir / "tnl-benchmark-heat-equation-raw.html"
    input_df.to_html(raw_path)
    print(f"Wrote {raw_path}")

    precisions: list[str] = (
        sorted(input_df["precision"].unique())
        if "precision" in input_df.columns
        else [""]
    )

    for precision in precisions:
        print(f"Processing precision {precision} ...")
        if precision:
            subset = input_df.loc[input_df["precision"] == precision]
        else:
            subset = input_df
        if subset.empty:
            continue

        implementations = sorted(subset["implementation"].unique())

        multicolumns = get_multiindex(subset, implementations)
        result = build_wide_table(subset, multicolumns, implementations)

        suffix = precision if precision else "all"
        html_path = args.output_dir / f"tnl-benchmark-heat-equation-{suffix}.html"
        result.to_html(html_path)
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
