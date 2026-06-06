#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from TNL import BenchmarkLogs


def _compute_color_range(df: pd.DataFrame) -> tuple[float, float]:
    """Compute vmin/vmax from bandwidth data, rounded to multiples of 50."""
    bw = pd.to_numeric(df["bandwidth"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
    if len(bw) == 0:
        return 0.0, 100.0
    bw_min = float(bw.min())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    bw_max = float(bw.max())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    vmin = math.floor(bw_min / 50) * 50
    vmax = math.ceil(bw_max / 50) * 50
    if vmax == vmin:
        vmax = vmin + 50
    return float(vmin), float(vmax)


def display_heatmap(
    raw_df: pd.DataFrame,
    title: str,
    vmin: float,
    vmax: float,
    output_path: Path,
) -> None:
    """Display and save a bandwidth heatmap from the NDArray benchmark dataframe."""
    df = raw_df.copy()
    df["axis"] = pd.to_numeric(df["axis"], errors="coerce")
    df["bandwidth"] = pd.to_numeric(df["bandwidth"], errors="coerce")
    df = df.dropna(subset=["axis", "permutation", "bandwidth"])  # pyright: ignore[reportAssignmentType]
    if df.empty:
        print(
            "No valid data for heatmap after dropping rows"
            " with missing axis/permutation."
        )
        return
    df["axis"] = df["axis"].astype(int)
    df = df.groupby(["axis", "permutation"], as_index=False)["bandwidth"].max()
    heatmap_data = df.pivot(index="axis", columns="permutation", values="bandwidth")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=15)
    im = ax.imshow(
        heatmap_data,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(im, ax=ax, label="Bandwidth")
    colorbar.ax.yaxis.label.set_fontsize(14)
    colorbar.ax.tick_params(labelsize=14)

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=90, fontsize=14)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    ax.set_xlabel("Permutations", fontsize=14)
    ax.set_ylabel("Axis", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", dpi=300)
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


_COLUMNS = [
    "device",
    "axis",
    "permutation",
    "size",
    "m",
    "n",
    "o",
    "p",
    "q",
    "time",
    "bandwidth",
]


def _load_dataframes(filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate NDArray benchmark log files, keeping relevant columns."""
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
    available = [c for c in _COLUMNS if c in df.columns]
    result: pd.DataFrame = df[available]  # pyright: ignore[reportAssignmentType]
    if "permutation" in result.columns:
        result["dimension"] = (
            result["permutation"]
            .apply(lambda p: len(p.split()) if isinstance(p, str) else 0)
            .astype("Int64")
        )
    return result


def _filter_by_dim(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Filter dataframe to only rows matching the given dimension.

    Rows are kept if their axis is less than dim and their dimension
    column equals dim.
    """
    df = df.copy()
    df["axis"] = pd.to_numeric(df["axis"], errors="coerce")
    mask = df["axis"].notna() & (df["axis"] < dim)
    if "dimension" in df.columns:
        mask &= df["dimension"] == dim
    result: pd.DataFrame = df[mask]  # pyright: ignore[reportAssignmentType]
    return result.copy()


def _available_dims(df: pd.DataFrame) -> list[int]:
    """Return sorted list of dimensions present in the data."""
    if "dimension" not in df.columns:
        return []
    dims = df["dimension"].dropna().unique()
    return sorted(int(d) for d in dims if d >= 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NDArray benchmark bandwidth heatmaps from log files."
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
        default=Path("."),
        help="Output directory for SVG files (default: current directory)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframes(args.input)

    if "device" not in df.columns:
        devices: list[str] = []
    else:
        devices = sorted(df["device"].dropna().unique())

    dims = _available_dims(df)

    if not dims or "axis" not in df.columns:
        print("No heatmap data (missing axis/permutation columns)")
        return

    mask = df["axis"].notna()
    if "dimension" in df.columns:
        mask &= df["dimension"] >= 2
    dimmed: pd.DataFrame = df[mask].copy()  # pyright: ignore[reportAssignmentType]
    vmin, vmax = _compute_color_range(dimmed)

    for dim in dims:
        for device in devices or [None]:
            sub: pd.DataFrame = _filter_by_dim(df, dim)
            if device is not None and "device" in sub.columns:
                sub = sub[sub["device"] == device].copy()  # pyright: ignore[reportAssignmentType]
            if sub.empty:
                dev_label = f" ({device})" if device else ""
                print(f"No data for {dim}D{dev_label}, skipping...")
                continue
            device_suffix = f"-{device}" if device else ""
            output_path = args.output_dir / f"heatmap-{dim}D{device_suffix}.svg"
            device_label = f" ({device})" if device else ""
            title = f"NDArray {dim}D{device_label} Bandwidth Heatmap"
            display_heatmap(sub, title, vmin, vmax, output_path)


if __name__ == "__main__":
    main()
