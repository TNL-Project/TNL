#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from TNL import BenchmarkLogs
from TNL import MultiindexCreator as mic

sns.set_theme()

DEFAULT_OUTPUT_DIR = Path("segments-plots")

functions = [
    "forElements",
    "forElements with indexes stride 2",
    "forElements with indexes stride 4",
    "forElements with indexes stride 8",
    "forElementsIf stride 2",
    "forElementsIf stride 4",
    "forElementsIf stride 8",
    "forSelectedElements with stride 2",
    "forSelectedElements with stride 4",
    "forSelectedElements with stride 8",
    "reduceSegments",
    "reduceSegmentsWithIndexes stride 2",
    "reduceSegmentsWithIndexes stride 4",
    "reduceSegmentsWithIndexes stride 8",
    "reduceSegmentIf stride 2",
    "reduceSegmentIf stride 4",
    "reduceSegmentIf stride 8",
]

all_segments = [
    "CSR",
    "Ellpack",
    "SlicedEllpack",
    "BiEllpack",
    "ChunkedEllpack",
]

threads_mappings_translation = {
    "": "",
    "Single": "Single",
    "N/A": "N/A",
    "1 TPS": "Single",
    "2 TPS": "2 TPS",
    "4 TPS": "4 TPS",
    "8 TPS": "8 TPS",
    "16 TPS": "16 TPS",
    "32 TPS": "32 TPS",
    "64 TPS": "64 TPS",
    "128 TPS": "128 TPS",
    "256 TPS": "256 TPS",
    "BlockMerged 1 TPS": "Block Merge 1",
    "BlockMerged 2 TPS": "Block Merge 2",
    "BlockMerged 4 TPS": "Block Merge 4",
    "BlockMerged 8 TPS": "Block Merge 8",
    "DynamicGrouping 1 TPS": "Dynamic Grouping 1",
    "DynamicGrouping 2 TPS": "Dynamic Grouping 2",
    "DynamicGrouping 4 TPS": "Dynamic Grouping 4",
    "DynamicGrouping 8 TPS": "Dynamic Grouping 8",
    "DynamicGrouping 16 TPS": "Dynamic Grouping 16",
    "Light CSR": "Light CSR",
    "Hybrid CSR": "Hybrid CSR",
}

_DEVICE_ORDER = {"sequential": 0, "host": 1, "cuda": 2, "hip": 3}

tests = ["constant", "linear", "quadratic"]


def _numeric_sort_key(s: str) -> tuple[str, int]:
    """Sort key for threads mapping labels: group by category, then numeric order."""
    if s == "Single":
        return ("0_Single", 0)
    if s == "N/A":
        return ("9_N/A", 0)
    parts = s.rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    parts = s.split(maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit():
        return ("1_TPS", int(parts[0]))
    return (s, 0)


def _sorted_hue(values: list[str]) -> list[str]:
    """Sort hue values with numeric awareness for X TPS patterns."""
    return sorted(values, key=_numeric_sort_key)


def _load_dataframes(filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate segments benchmark log files."""
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

    numeric_cols = [
        "segments count",
        "max segment size",
        "elements count",
        "time",
        "time_stddev",
        "bandwidth",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "threads mapping" in df.columns:
        df["threads mapping"] = (
            df["threads mapping"]
            .map(threads_mappings_translation)  # pyright: ignore[reportArgumentType]
            .fillna(df["threads mapping"])
        )

    if "time_stddev" in df.columns and "time" in df.columns:
        df["bandwidth_stddev"] = df["bandwidth"] * (df["time_stddev"] / df["time"])

    return df


def get_multiindex(input_df: pd.DataFrame) -> tuple[pd.MultiIndex, list[list[str]]]:
    """Create multi-index columns for the pivot table based on the actual data."""
    mc = mic.MultiindexCreator(5)
    mc.add_entries(
        [
            ["segments setup"],
            ["segments count"],
            ["max segment size"],
            ["elements count"],
        ]
    )

    available_devices = sorted(
        input_df["device"].unique(),
        key=lambda d: _DEVICE_ORDER.get(d, len(_DEVICE_ORDER)),
    )

    for function in functions:
        for device in available_devices:
            if device == "sequential" or device == "host":
                segments_list = ["CSR"]
            else:
                segments_list = all_segments
            for segments in segments_list:
                sub_df = input_df[
                    (input_df["function"] == function)
                    & (input_df["device"] == device)
                    & (input_df["segments type"] == segments)
                ]
                if sub_df.empty:
                    mappings = ["Single"]
                else:
                    mappings = _sorted_hue(list(sub_df["threads mapping"].unique()))  # pyright: ignore[reportAttributeAccessIssue]
                for mapping in mappings:
                    mc.add_entry([function, device, segments, mapping, "time"])
                    mc.add_entry([function, device, segments, mapping, "bandwidth"])
                    mc.add_entry(
                        [function, device, segments, mapping, "bandwidth_stddev"]
                    )
                    if device == "cuda":
                        mc.add_entry([function, device, segments, mapping, "speedup"])

    return mc.get_multiindex()


def convert_data_frame(
    input_df: pd.DataFrame,
    multicolumns: pd.MultiIndex,
) -> pd.DataFrame:
    """Convert flat input table to a structured one using multiindex columns."""
    frames: list[pd.DataFrame] = []
    in_idx = 0
    out_idx = 0
    total = len(input_df.index)
    while in_idx < total:
        segments_setup = input_df.iloc[in_idx]["segments setup"]
        segments_count = input_df.iloc[in_idx]["segments count"]
        max_segment_size = input_df.iloc[in_idx]["max segment size"]
        df_graph = input_df[
            (input_df["segments setup"] == segments_setup)
            & (input_df["segments count"] == segments_count)
            & (input_df["max segment size"] == max_segment_size)
        ]
        print(
            f"{out_idx} : {in_idx} / {total} : "
            f"{segments_setup} {segments_count} {max_segment_size}"
        )
        row_data: dict[tuple[str, ...], object] = {
            col: float("nan") for col in multicolumns
        }
        row_data[("segments setup", "", "", "", "")] = segments_setup
        row_data[("segments count", "", "", "", "")] = segments_count
        row_data[("max segment size", "", "", "", "")] = max_segment_size
        for _index, row in df_graph.iterrows():
            cur_type = str(row["segments type"])
            cur_func = str(row["function"])
            cur_device = str(row["device"])
            cur_mapping = str(row["threads mapping"])
            time_val = pd.to_numeric(row["time"], errors="coerce")
            bandwidth_val = pd.to_numeric(row["bandwidth"], errors="coerce")
            bw_stddev_val = pd.to_numeric(
                row.get("bandwidth_stddev", float("nan")), errors="coerce"
            )
            row_data[("elements count", "", "", "", "")] = row["elements count"]
            key_time = (cur_func, cur_device, cur_type, cur_mapping, "time")
            key_bw = (cur_func, cur_device, cur_type, cur_mapping, "bandwidth")
            key_bw_stddev = (
                cur_func,
                cur_device,
                cur_type,
                cur_mapping,
                "bandwidth_stddev",
            )
            row_data[key_time] = time_val
            row_data[key_bw] = bandwidth_val
            row_data[key_bw_stddev] = bw_stddev_val
        aux_df = pd.DataFrame(
            [row_data], columns=multicolumns, index=pd.Index([out_idx])
        )
        frames.append(aux_df)
        out_idx += 1
        in_idx += len(df_graph.index)
    result = pd.concat(frames)
    result.replace("", float("nan"), inplace=True)
    return result


def compute_speedup(df: pd.DataFrame) -> None:
    """Compute speedup columns from sequential baseline (vectorized)."""
    for function in functions:
        for segments_type in all_segments:
            seq_key = (function, "sequential", "CSR", "Single", "time")
            if seq_key not in df.columns:
                continue
            for col in df.columns:
                if (
                    col[0] == function
                    and col[1] == "cuda"
                    and col[2] == segments_type
                    and col[4] == "time"
                ):
                    speedup_key = (col[0], col[1], col[2], col[3], "speedup")
                    df[speedup_key] = df[seq_key] / df[col]


def _prepare_tidy_df(
    flat_df: pd.DataFrame,
    function: str,
    segments_type: str,
) -> pd.DataFrame | None:
    """Extract tidy DataFrame for FacetGrid from flat input data."""
    mask = (
        (flat_df["function"] == function)
        & (flat_df["device"] == "cuda")
        & (flat_df["segments type"] == segments_type)
    )
    cols = ["segments count", "max segment size", "bandwidth"]
    if "bandwidth_stddev" in flat_df.columns:
        cols.append("bandwidth_stddev")
    if "threads mapping" in flat_df.columns:
        cols.append("threads mapping")
    sub = flat_df.loc[mask, cols].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=["bandwidth"])
    if sub.empty:
        return None
    return sub


def write_facet_grid(
    tidy_df: pd.DataFrame,
    *,
    title: str,
    y_label: str = "Bandwidth [GiB/s]",
    y_lim: tuple[float, float] | None = None,
    file_path: Path,
) -> None:
    """Write a FacetGrid figure with max segment size as columns."""
    x = "segments count"
    y = "bandwidth"
    y_err_col = "bandwidth_stddev"
    hue = "threads mapping"
    hue_title = "Threads mapping"
    col = "max segment size"

    agg_cols = [y]
    if y_err_col in tidy_df.columns:
        agg_cols.append(y_err_col)
    agg_df = tidy_df.groupby([hue, x, col], sort=True)[agg_cols].mean().reset_index()
    agg_df = agg_df.sort_values(x)  # pyright: ignore[reportCallIssue]

    agg_df = agg_df.rename(columns={hue: hue_title})
    hue = hue_title

    col_order = sorted(agg_df[col].unique())
    hue_order = _sorted_hue(list(agg_df[hue].unique()))

    g = sns.FacetGrid(
        agg_df,
        col=col,
        col_order=col_order,
        hue=hue,
        hue_order=hue_order,
        sharey=True,
        sharex=True,
        height=4,
        aspect=1.2,
    )

    g.map_dataframe(sns.lineplot, x=x, y=y, marker="o", errorbar=None)

    if y_err_col in agg_df.columns:
        for ax_i, mss_val in zip(g.axes.flat, col_order):
            facet_df = agg_df[agg_df[col] == mss_val].sort_values(x)  # pyright: ignore[reportCallIssue]
            for _name, group in facet_df.groupby(hue, sort=True):
                xs = group[x].to_numpy(dtype=float)
                ys = group[y].to_numpy(dtype=float)
                errs = group[y_err_col].to_numpy(dtype=float)
                ax_i.fill_between(xs, ys - errs, ys + errs, alpha=0.15)

    for ax_i in g.axes.flat:
        ax_i.set_xscale("log")
        ax_i.set_yscale("log")
        ax_i.set_xlabel("Segments count")
        ax_i.set_ylabel(y_label)
        if y_lim:
            ax_i.set_ylim(y_lim)
        ax_i.grid(True, alpha=0.3)

    g.set_titles(col_template="max segment size = {col_name}")
    g.fig.suptitle(title, y=1.02)
    g.add_legend()

    g.fig.savefig(file_path, format="svg", bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved: {file_path}")


def write_results(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write HTML tables and comparison plots."""
    raw_path = output_dir / "tnl-benchmark-segments-raw.html"
    original_df.to_html(raw_path)
    print(f"Wrote {raw_path}")

    df.to_html(output_dir / "tnl-benchmark-segments.html")

    plot_tasks: list[tuple[pd.DataFrame, Path, str, str]] = []
    for test in tests:
        test_flat_df = original_df[original_df["segments setup"] == test]
        if not isinstance(test_flat_df, pd.DataFrame):
            continue
        test_subdir = output_dir / test
        test_subdir.mkdir(parents=True, exist_ok=True)

        for function in functions:
            for segments in all_segments:
                tidy = _prepare_tidy_df(test_flat_df, function, segments)
                if tidy is None or tidy.empty:
                    continue
                safe_name = function.replace(" ", "_")
                file_path = test_subdir / f"{safe_name}_{segments}_bandwidth.svg"
                title = f"{function}, cuda, {segments}"
                plot_tasks.append((tidy, file_path, title, test))

    if not plot_tasks:
        print("No plot data found")
        return
    all_bw = pd.concat([t["bandwidth"] for t, _, _, _ in plot_tasks]).dropna()
    all_bw = all_bw[all_bw > 0]
    if len(all_bw) == 0:
        print("No bandwidth data found in plot data")
        return
    min_bw = float(all_bw.min())
    max_bw = float(all_bw.max())
    min_bw = max(0, math.floor(min_bw / 50) * 50)
    max_bw = math.ceil(max_bw / 50) * 50
    print(f"Min bandwidth: {min_bw:.3f} GiB/s")
    print(f"Max bandwidth: {max_bw:.3f} GiB/s")

    for test in tests:
        print(f"Processing test: {test}")
        test_df = df[df[("segments setup", "", "", "", "")] == test]
        test_df.to_html(output_dir / f"tnl-benchmark-segments-{test}.html")

        test_subdir = output_dir / test
        test_subdir.mkdir(parents=True, exist_ok=True)

        for function in functions:
            base_cols = [
                ("segments count", "", "", "", ""),
                ("max segment size", "", "", "", ""),
                ("elements count", "", "", "", ""),
            ]
            func_cols = [c for c in df.columns if c[0] == function]
            function_df = pd.DataFrame(test_df[base_cols + func_cols])
            function_df.to_html(test_subdir / f"{function}.html")

        for tidy, file_path, title, task_test in plot_tasks:
            if task_test != test:
                continue
            print(f"    Writing plot: {title}")

            write_facet_grid(
                tidy,
                title=title,
                y_label="Bandwidth [GiB/s]",
                y_lim=(min_bw, max_bw),
                file_path=file_path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for processing TNL benchmark segments results."
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

    df = _load_dataframes(args.input)
    if df.empty:
        print("No data found in log files")
        return

    multicolumns, _data = get_multiindex(df)
    result = convert_data_frame(df, multicolumns)
    compute_speedup(result)

    write_results(result, df, args.output_dir)


if __name__ == "__main__":
    main()
