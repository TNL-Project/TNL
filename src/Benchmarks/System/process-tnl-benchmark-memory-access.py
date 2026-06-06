#!/usr/bin/python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from TNL import BenchmarkLogs

sns.set_theme()

BW_LABEL: str = "Effective bandwidth in GiB/s"
CYCLES_LABEL: str = "CPU cycles per element"
DEFAULT_OUTPUT_DIR = Path("memory-access-plots")

_ACCESS_TYPE_ORDER = {"sequential": 0, "interleaved": 1, "random": 2}
_TEST_TYPE_ORDER = {"read": 0, "write": 1, "read-write": 2}

METRICS: list[tuple[str, str, str, str]] = [
    ("bandwidth", "bandwidth_stddev", BW_LABEL, "bw"),
    ("cycles/op", "cycles/op_stddev", CYCLES_LABEL, "cycles"),
]


def _load_dataframes(filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate log files, then derive stddev columns."""
    frames: list[pd.DataFrame] = []
    for filename in filenames:
        path = Path(filename)
        if not path.exists():
            print(f"Skipping non-existing input file {filename} ...")
            continue
        frames.append(BenchmarkLogs.get_benchmark_dataframe(path))
    df = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "array size",
        "time",
        "time_stddev",
        "bandwidth",
        "cycles/op",
        "cycles_stddev",
        "cycles",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    if "time_stddev" in df.columns and "time" in df.columns:
        df["bandwidth_stddev"] = df["bandwidth"] * (df["time_stddev"] / df["time"])

    if "cycles_stddev" in df.columns and "cycles" in df.columns:
        df["cycles/op_stddev"] = df["cycles/op"] * (df["cycles_stddev"] / df["cycles"])

    df["threads"] = df["threads"].astype(str)
    df["element size"] = df["element size"].astype(str)
    return df


def _make_filename(
    *parts: str, suffix: str, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """Build a dash-separated filename with subdirectory."""
    name = "-".join(str(p) for p in parts if p is not None) + f"-{suffix}.svg"
    return output_dir / name


def writeFigure(
    df: pd.DataFrame,
    *,
    x: str = "array size",
    y: str,
    y_err: str | None = None,
    hue: str,
    hue_title: str = "",
    hue_order: list[str] | None = None,
    title: str = "",
    y_label: str,
    y_lim: tuple[float, float] | None = None,
    file_path: Path,
) -> None:
    """Write a comparison figure using seaborn lineplot with optional error band."""
    fig, ax = plt.subplots(1, 1)
    order = hue_order or (sorted(df[hue].unique()) if hue_title else None)
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        hue_order=order,
        marker="o",
        errorbar=None,
        ax=ax,
    )
    # fill_between is used instead of seaborn's errorbar because we have pre-computed
    # stddev values rather than multiple observations per x-point
    if y_err and y_err in df.columns:
        for _name, group in df.groupby(hue, sort=True):
            sg = group.sort_values(x)
            xs = sg[x].to_numpy(dtype=float)
            ys = sg[y].to_numpy(dtype=float)
            errs = sg[y_err].to_numpy(dtype=float)
            ax.fill_between(xs, ys - errs, ys + errs, alpha=0.15)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Array size in bytes")
    if title:
        ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    if y_lim:
        ax.set_ylim(y_lim)
    if hue_title:
        ax.legend(title=hue_title, loc="best")
    else:
        ax.legend(loc="best")
    ax.grid()
    fig.savefig(file_path, format="svg")
    plt.close(fig)


def _filter_df(
    df: pd.DataFrame,
    *,
    threads: str | None = None,
    access_type: str | None = None,
    test_type: str | None = None,
    element_size: str | None = None,
) -> pd.DataFrame:
    """Filter dataframe by given metadata columns."""
    mask = pd.Series(True, index=df.index)
    if threads is not None:
        mask &= df["threads"] == threads
    if access_type is not None:
        mask &= df["access type"] == access_type
    if test_type is not None:
        mask &= df["test type"] == test_type
    if element_size is not None:
        mask &= df["element size"] == element_size
    result: pd.DataFrame = df[mask]  # pyright: ignore[reportAssignmentType]
    return result


def writeAccessTypeComparisonFigures(
    df: pd.DataFrame,
    threads: list[str],
    access_types: list[str],
    element_sizes: list[str],
    test_types: list[str],
    output_dir: Path,
) -> None:
    """Write figures comparing bandwidth across access types."""
    if len(access_types) < 2:
        return
    for threads_count, test_type, element_size in product(
        threads, test_types, element_sizes
    ):
        sub: pd.DataFrame = _filter_df(
            df,
            threads=threads_count,
            test_type=test_type,
            element_size=element_size,
        )
        file_parts = [
            "access-type-comparison",
            f"{threads_count}-threads",
            test_type,
            f"element-size-{element_size}",
        ]
        title = (
            f"Access type comparison: {threads_count} threads, "
            f"{test_type}, el. size {element_size}"
        )
        print(
            f"Writing figure for access type comparison: "
            f"{threads_count} threads {test_type} "
            f"element size = {element_size}:"
        )
        for y_col, y_err_col, y_label, suffix in METRICS:
            writeFigure(
                sub,
                y=y_col,
                y_err=y_err_col,
                hue="access type",
                hue_title="Access type",
                title=title,
                y_label=y_label,
                file_path=_make_filename(
                    *file_parts, suffix=suffix, output_dir=output_dir
                ),
            )


def writeThreadsCountComparisonFigures(
    df: pd.DataFrame,
    threads: list[str],
    access_types: list[str],
    element_sizes: list[str],
    test_types: list[str],
    output_dir: Path,
) -> None:
    """Write figures comparing bandwidth across different thread counts."""
    if len(threads) < 2:
        return
    threads_sorted = sorted(threads, key=int)
    for access_type, test_type, element_size in product(
        access_types, test_types, element_sizes
    ):
        sub: pd.DataFrame = _filter_df(
            df,
            access_type=access_type,
            test_type=test_type,
            element_size=element_size,
        ).copy()
        file_parts = [
            "threads-comparison",
            access_type,
            test_type,
            f"element-size-{element_size}",
        ]
        title = (
            f"Threads comparison: {access_type}, {test_type}, el. size {element_size}"
        )
        print(
            f"Writing figure for threads count comparison: "
            f"{access_type} {test_type} "
            f"element size = {element_size}:"
        )
        for y_col, y_err_col, y_label, suffix in METRICS:
            writeFigure(
                sub,
                y=y_col,
                y_err=y_err_col,
                hue="threads",
                hue_title="Threads",
                hue_order=threads_sorted,
                title=title,
                y_label=y_label,
                file_path=_make_filename(
                    *file_parts, suffix=suffix, output_dir=output_dir
                ),
            )


def writeTestTypeComparisonFigures(
    df: pd.DataFrame,
    threads: list[str],
    access_types: list[str],
    element_sizes: list[str],
    test_types: list[str],
    output_dir: Path,
) -> None:
    """Write figures comparing bandwidth across different test types."""
    if len(test_types) < 2:
        return
    for threads_count, access_type, element_size in product(
        threads, access_types, element_sizes
    ):
        sub: pd.DataFrame = _filter_df(
            df,
            threads=threads_count,
            access_type=access_type,
            element_size=element_size,
        )
        file_parts = [
            "test-type-comparison",
            access_type,
            f"{threads_count}-threads",
            f"element-size-{element_size}",
        ]
        title = (
            f"Test type comparison: {access_type}, "
            f"{threads_count} threads, el. size {element_size}"
        )
        print(
            f"Writing figure for test type comparison: "
            f"{access_type} {threads_count} threads "
            f"element size = {element_size}:"
        )
        for y_col, y_err_col, y_label, suffix in METRICS:
            writeFigure(
                sub,
                y=y_col,
                y_err=y_err_col,
                hue="test type",
                hue_title="Test type",
                title=title,
                y_label=y_label,
                file_path=_make_filename(
                    *file_parts, suffix=suffix, output_dir=output_dir
                ),
            )


def writeElementSizeComparisonFigures(
    df: pd.DataFrame,
    threads: list[str],
    access_types: list[str],
    element_sizes: list[str],
    test_types: list[str],
    output_dir: Path,
) -> None:
    """Write figures comparing bandwidth across different element sizes."""
    if len(element_sizes) < 2:
        return
    element_sizes_sorted = sorted(element_sizes, key=int)
    for threads_count, access_type, test_type in product(
        threads, access_types, test_types
    ):
        sub: pd.DataFrame = _filter_df(
            df,
            threads=threads_count,
            access_type=access_type,
            test_type=test_type,
        ).copy()
        file_parts = [
            "element-size-comparison",
            f"{threads_count}-threads",
            access_type,
            test_type,
        ]
        title = (
            f"Element size comparison: {access_type}, "
            f"{threads_count} threads, {test_type}"
        )
        print(
            f"Writing figure for element size comparison: "
            f"{threads_count} threads {access_type} {test_type}:"
        )
        for y_col, y_err_col, y_label, suffix in METRICS:
            writeFigure(
                sub,
                y=y_col,
                y_err=y_err_col,
                hue="element size",
                hue_title="Element size",
                hue_order=element_sizes_sorted,
                title=title,
                y_label=y_label,
                file_path=_make_filename(
                    *file_parts, suffix=suffix, output_dir=output_dir
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for processing TNL benchmark memory access results."
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
        help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframes(args.input)

    threads = sorted(df["threads"].unique(), key=int)
    access_types = sorted(
        df["access type"].unique(),
        key=lambda x: _ACCESS_TYPE_ORDER.get(x, len(_ACCESS_TYPE_ORDER)),
    )
    element_sizes = sorted(df["element size"].unique(), key=int)
    test_types = sorted(
        df["test type"].unique(),
        key=lambda x: _TEST_TYPE_ORDER.get(x, len(_TEST_TYPE_ORDER)),
    )

    df.to_html(args.output_dir / "tnl-benchmark-memory-access-raw.html")

    writeAccessTypeComparisonFigures(
        df, threads, access_types, element_sizes, test_types, args.output_dir
    )
    writeThreadsCountComparisonFigures(
        df, threads, access_types, element_sizes, test_types, args.output_dir
    )
    writeTestTypeComparisonFigures(
        df, threads, access_types, element_sizes, test_types, args.output_dir
    )
    writeElementSizeComparisonFigures(
        df, threads, access_types, element_sizes, test_types, args.output_dir
    )


if __name__ == "__main__":
    main()
