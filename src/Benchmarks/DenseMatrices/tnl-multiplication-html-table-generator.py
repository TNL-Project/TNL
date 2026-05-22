#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
import sys

import pandas as pd
from TNL.BenchmarkLogs import get_benchmark_dataframe


def read_log_files(file_paths):
    frames = []
    for path in file_paths:
        try:
            df = get_benchmark_dataframe(path)
            frames.append(df)
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def process_data(df):
    # collect all non-benchmark columns for the index
    index_candidates = [
        "matrix1 size",
        "matrix2 size",
        "algorithm",
    ]
    index_cols = [c for c in index_candidates if c in df.columns]

    header_elements = ["time"] + [c for c in df.columns if c.startswith("Diff")]

    result = df.copy()
    result = result.set_index(index_cols)

    # format time
    result["time"] = result["time"].apply(
        lambda t: f"{t:.5e}" if pd.notna(t) else "N/A"
    )

    # format Diff columns as scientific notation
    for col in [c for c in result.columns if c.startswith("Diff")]:
        result[col] = result[col].apply(
            lambda v: f"{float(v):.2e}" if pd.notna(v) and v != "N/A" else v
        )

    return result, header_elements


def calculate_speedups(df, primary_algorithms, secondary_algorithms):
    size_cols = [c for c in df.index.names if c != "algorithm"]
    speedup_frames = []

    for sizes, group in df.groupby(level=size_cols):
        rows = {}
        for algo in secondary_algorithms:
            try:
                sec_time = group.xs(algo, level="algorithm")["time"].iloc[0]
            except (KeyError, IndexError):
                continue
            sec_time_val = float(sec_time) if sec_time != "N/A" else None
            for p_algo in primary_algorithms:
                try:
                    prim_time = group.xs(p_algo, level="algorithm")["time"].iloc[0]
                except (KeyError, IndexError):
                    rows.setdefault(algo, {})[f"speedup vs {p_algo}"] = "N/A"
                    continue
                prim_time_val = float(prim_time) if prim_time != "N/A" else None
                if prim_time_val and sec_time_val:
                    rows.setdefault(algo, {})[f"speedup vs {p_algo}"] = (
                        f"{prim_time_val / sec_time_val:.2f}x"
                    )
                else:
                    rows.setdefault(algo, {})[f"speedup vs {p_algo}"] = "N/A"

        speedup_df = pd.DataFrame.from_dict(rows, orient="index")
        speedup_df.index.name = "algorithm"
        # add back size index levels
        for i, name in enumerate(size_cols):
            speedup_df[name] = sizes[i] if isinstance(sizes, tuple) else sizes
        speedup_df = speedup_df.set_index(size_cols, append=True)
        speedup_df = speedup_df.reorder_levels(df.index.names)
        speedup_frames.append(speedup_df)

    if not speedup_frames:
        return df
    speedups = pd.concat(speedup_frames)
    return df.join(speedups, how="left")


def create_html_table(df, primary_algorithms, secondary_algorithms):
    style = """
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
    title = "<h2>Dense Matrix Multiplication</h2>"

    # build multi-column header using pandas MultiIndex
    all_algorithms = primary_algorithms + secondary_algorithms
    columns = []
    for algo in all_algorithms:
        if algo in primary_algorithms:
            columns.append((algo, "Time"))
        else:
            columns.append((algo, "Time"))
            for p in primary_algorithms:
                columns.append((algo, f"vs {p}"))
            columns.append((algo, "l2Norm"))
            columns.append((algo, "MaxNorm"))

    col_index = pd.MultiIndex.from_tuples(columns)

    # build data rows
    size_cols = [c for c in df.index.names if c != "algorithm"]
    rows = []
    row_indices = []

    for sizes, group in df.groupby(level=size_cols):
        row = []
        for algo in all_algorithms:
            try:
                algo_row = group.xs(algo, level="algorithm")
            except KeyError:
                algo_row = None

            if algo_row is not None and not algo_row.empty:
                time_val = algo_row["time"].iloc[0]
            else:
                time_val = "N/A"
            row.append(time_val)

            if algo in secondary_algorithms:
                for p in primary_algorithms:
                    col = f"speedup vs {p}"
                    if algo_row is not None and col in algo_row.columns:
                        row.append(algo_row[col].iloc[0])
                    else:
                        row.append("N/A")
                # l2Norm
                if algo_row is not None:
                    diff_l2 = []
                    for c in ["Diff.L2 1", "Diff.L2 2", "Diff.L2 3"]:
                        if c in algo_row.columns:
                            diff_l2.append(algo_row[c].iloc[0])
                    row.append(" ".join(diff_l2) if diff_l2 else "N/A")
                    # MaxNorm
                    diff_max = []
                    for c in ["Diff.Max 1", "Diff.Max 2", "Diff.Max 3"]:
                        if c in algo_row.columns:
                            diff_max.append(algo_row[c].iloc[0])
                    row.append(" ".join(diff_max) if diff_max else "N/A")
                else:
                    row.append("N/A")
                    row.append("N/A")

        rows.append(row)
        if isinstance(sizes, tuple):
            row_indices.append(sizes)
        else:
            row_indices.append((sizes,))

    row_index = pd.MultiIndex.from_tuples(row_indices, names=size_cols)
    table_df = pd.DataFrame(rows, index=row_index, columns=col_index)

    html_table = table_df.to_html()
    return style + title + html_table


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML table from dense matrix multiplication benchmark "
        "logs"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input log file paths (JSON lines format)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML file path (default: dense_matrix_multiplication.html)",
    )
    args = parser.parse_args()

    df = read_log_files(args.inputs)
    if df.empty:
        print("No log entries found in input files.", file=sys.stderr)
        sys.exit(1)

    # drop rows with "matrix size" key (transposition entries use different schema)
    if "matrix size" in df.columns:
        df = df[df["matrix size"].isna()]

    # filter to multiplication entries (must have matrix1 size and matrix2 size)
    if "matrix1 size" not in df.columns or "matrix2 size" not in df.columns:
        print("Input does not contain multiplication benchmark data.", file=sys.stderr)
        sys.exit(1)

    primary_algorithms = ["cuBLAS", "Magma", "Cutlass", "BLAS"]
    secondary_algorithms = [
        "Final",
        "cublasA",
        "cublasB",
        "cublasAB",
        "magmaA",
        "magmaB",
        "magmaAB",
        "tnlA",
        "tnlB",
        "tnlAB",
    ]

    processed_df, _ = process_data(df)
    processed_df = calculate_speedups(
        processed_df, primary_algorithms, secondary_algorithms
    )
    html = create_html_table(processed_df, primary_algorithms, secondary_algorithms)

    output_path = args.output or "dense_matrix_multiplication.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"HTML table saved as {output_path}")


if __name__ == "__main__":
    main()
