#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from TNL import BenchmarkLogs

_CSS = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
    }
    th, td {
        border: 1px solid #ddd;
        text-align: left;
        padding: 6px;
        min-width: 125px;
    }
    th {
        background-color: #f2f2f2;
        color: black;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #f1f1f1;
    }
</style>
"""

_DEVICE_ORDER: dict[str, int] = {"sequential": 0, "host": 1, "cuda": 2, "hip": 3}


def _device_sort_key(device: str) -> int:
    return _DEVICE_ORDER.get(device, len(_DEVICE_ORDER))


@dataclass
class ModeConfig:
    index_candidates: list[str]
    title: str
    primary_algorithms: list[str]
    secondary_algorithms: list[str]
    default_output: str


MULTIPLICATION = ModeConfig(
    index_candidates=["matrix1 size", "matrix2 size", "device", "algorithm"],
    title="Dense Matrix Multiplication",
    primary_algorithms=["cuBLAS", "MAGMA", "Cutlass", "BLAS"],
    secondary_algorithms=[
        "TNL",
        "cuBLAS A",
        "cuBLAS B",
        "cuBLAS AB",
        "MAGMA A",
        "MAGMA B",
        "MAGMA AB",
        "TNL A",
        "TNL B",
        "TNL AB",
    ],
    default_output="dense_matrix_multiplication.html",
)

TRANSPOSITION = ModeConfig(
    index_candidates=["matrix size", "device", "algorithm"],
    title="Dense Matrix Transposition",
    primary_algorithms=["MAGMA"],
    secondary_algorithms=["Kernel 2.1", "Kernel 2.2", "Kernel 2.3", "Kernel 2.4"],
    default_output="dense_matrix_transposition.html",
)


def _load_dataframes(filenames: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate benchmark log files."""
    frames: list[pd.DataFrame] = []
    for filename in filenames:
        path = Path(filename)
        if not path.exists():
            print(f"Skipping non-existing input file {filename} ...", file=sys.stderr)
            continue
        try:
            frames.append(BenchmarkLogs.get_benchmark_dataframe(path))
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _numeric_size_key(size_str: str) -> tuple[int, ...]:
    """Parse 'NxM' string into a numeric tuple for sorting."""
    return tuple(int(x) for x in size_str.split("x"))


def _prepare_df(df: pd.DataFrame, config: ModeConfig) -> pd.DataFrame:
    """Set index, sort numerically by size, keep data numeric."""
    index_cols = [c for c in config.index_candidates if c in df.columns]
    size_cols = [c for c in index_cols if c not in ("algorithm", "device")]

    result = df.copy()

    for col in size_cols:
        unique_vals = list(result[col].dropna().unique())
        sorted_vals = sorted(unique_vals, key=_numeric_size_key)
        result[col] = pd.Categorical(result[col], categories=sorted_vals, ordered=True)

    if "device" in index_cols:
        all_devices = list(result["device"].dropna().unique())
        sorted_devices = sorted(all_devices, key=_device_sort_key)
        result["device"] = pd.Categorical(
            result["device"], categories=sorted_devices, ordered=True
        )

    result = result.sort_values(index_cols)
    result = result.set_index(index_cols)

    return result


def _compute_speedups(df: pd.DataFrame, config: ModeConfig) -> pd.DataFrame:
    """Add speedup columns: secondary time / primary time (vectorized)."""
    size_cols = [c for c in df.index.names if c not in ("algorithm", "device")]
    join_cols = [*size_cols, "device"]
    for p_algo in config.primary_algorithms:
        try:
            p_times = df.xs(p_algo, level="algorithm")[["time"]]
        except KeyError:
            continue
        p_times = p_times.rename(columns={"time": f"__{p_algo}_time"})  # pyright: ignore[reportCallIssue]
        df = df.join(p_times, how="left", on=join_cols)
        df[f"speedup vs {p_algo}"] = df[f"__{p_algo}_time"] / df["time"]
        mask = df.index.get_level_values("algorithm") == p_algo
        df.loc[mask, f"speedup vs {p_algo}"] = float("nan")
        df = df.drop(columns=[f"__{p_algo}_time"])
    return df


_METRIC_RENAMES = {
    "time": "Time",
}
_METRIC_PATTERN = re.compile(r"^(Diff\.(L2|Max)|speedup)\s+vs\s+(.+)$")


def _rename_metric(col: str) -> str:
    """Rename a raw metric column to its display name."""
    if col in _METRIC_RENAMES:
        return _METRIC_RENAMES[col]
    m = _METRIC_PATTERN.match(col)
    if m:
        prefix, kind, ref = m.group(1), m.group(2), m.group(3)
        if prefix == "speedup":
            return f"vs {ref}"
        return f"{kind} vs {ref}"
    return col


_METRIC_ORDER = {"Time": 0, "vs": 1, "L2": 2, "Max": 3}


def _metric_sort_key(col: str) -> tuple[int, str]:
    """Sort key for metric sub-columns: Time → speedups → L2 → Max."""
    display = _rename_metric(col)
    first_word = display.split()[0]
    return (_METRIC_ORDER.get(first_word, len(_METRIC_ORDER)), display)


def _format_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Format numeric values in the wide DataFrame for HTML display."""
    result = wide.copy()
    for col in result.columns:
        metric = col[-1] if isinstance(col, tuple) else col
        display = _rename_metric(metric)
        if display == "Time":
            result[col] = result[col].apply(
                lambda v: f"{v:.5e}" if pd.notna(v) else "N/A"
            )
        elif display.startswith(("L2", "Max")):
            result[col] = result[col].apply(
                lambda v: f"{v:.2e}" if pd.notna(v) else "N/A"
            )
        elif display.startswith("vs"):
            result[col] = result[col].apply(
                lambda v: f"{v:.2f}x" if pd.notna(v) else "N/A"
            )
    return result


def _build_html_table(df: pd.DataFrame, config: ModeConfig) -> str:
    """Build HTML table from processed (indexed, numeric) DataFrame.

    Uses unstack to pivot device and algorithm into columns, producing a
    (device, algorithm, metric) column MultiIndex. Columns with only NaN
    values are auto-removed.
    """
    present_algos = [
        a
        for a in config.primary_algorithms + config.secondary_algorithms
        if a in df.index.get_level_values("algorithm")
    ]
    present_devices = sorted(
        df.index.get_level_values("device").unique().tolist(), key=_device_sort_key
    )
    df = df.loc[
        df.index.get_level_values("algorithm").isin(present_algos)
        & df.index.get_level_values("device").isin(present_devices)
    ]
    df = df.groupby(level=list(range(df.index.nlevels)), sort=False).mean(  # pyright: ignore[reportAssignmentType]
        numeric_only=True
    )

    keep_cols = [c for c in df.columns if c == "time" or _METRIC_PATTERN.match(c)]
    df = df[keep_cols]  # pyright: ignore[reportAssignmentType]

    wide = df.unstack(["device", "algorithm"])
    wide.columns = wide.columns.reorder_levels([1, 2, 0])  # pyright: ignore[reportAttributeAccessIssue]

    parts: list[pd.DataFrame] = []
    for device in present_devices:
        for algo in present_algos:
            mask = (wide.columns.get_level_values(0) == device) & (
                wide.columns.get_level_values(1) == algo
            )
            if not mask.any():
                continue
            algo_cols = wide.loc[:, mask]
            algo_cols = algo_cols.dropna(axis=1, how="all")
            sorted_cols = sorted(
                algo_cols.columns, key=lambda c: _metric_sort_key(c[-1])
            )
            parts.append(algo_cols[sorted_cols])
    wide = pd.concat(parts, axis=1) if parts else pd.DataFrame()

    wide = _format_wide(wide)
    return _CSS + f"<h2>{config.title}</h2>" + wide.to_html()


def _filter_for_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame | None:
    """Filter DataFrame for the given mode. Returns None if no matching data."""
    if mode == "multiplication":
        if "matrix size" in df.columns:
            df = df.loc[df["matrix size"].isna()].copy()
        if "matrix1 size" not in df.columns or "matrix2 size" not in df.columns:
            return None
        return df
    else:
        if "matrix1 size" in df.columns:
            df = df.loc[df["matrix1 size"].isna()].copy()
        if "matrix size" not in df.columns:
            return None
        return df.loc[df["matrix size"].notna()].copy()


def process_mode(
    input_df: pd.DataFrame,
    config: ModeConfig,
    mode: str,
    output_dir: Path,
) -> None:
    """Process one mode (multiplication or transposition) and write HTML."""
    df = _filter_for_mode(input_df, mode)
    if df is None or df.empty:
        return

    print(f"Processing {config.title} ...")

    df = _prepare_df(df, config)
    df = _compute_speedups(df, config)
    html = _build_html_table(df, config)

    output_path = output_dir / config.default_output
    output_path.write_text(html)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML tables from dense matrix benchmark logs"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input log file paths (JSON lines format)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_df = _load_dataframes(args.input)
    if input_df.empty:
        print("No log entries found in input files.", file=sys.stderr)
        sys.exit(1)

    process_mode(input_df, MULTIPLICATION, "multiplication", args.output_dir)
    process_mode(input_df, TRANSPOSITION, "transposition", args.output_dir)


if __name__ == "__main__":
    main()
