#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""
Generates HTML tables from the log of tnl-benchmark-scan (or tnl-benchmark-scan-cuda,
tnl-benchmark-scan-hip). For each benchmarked operation, e.g. "inclusive scan
(inplace)", it produces:

- a table of measured times for all devices and performers,
- a table of effective bandwidths,
- GPU vs CPU speedups of the same performer (and of TNL vs the best CPU result),
- speedups of TNL vs the other performers (libraries) on the same device.

When the log contains the explicitly selected GPU kernels (--gpu-kernels), it also
produces the tuning tables for each operation and GPU device:

- the times of all kernels and values per thread,
- the best number of values per thread of each kernel for each value type and size,
- the recommended number of values per thread of each kernel for each value type,
  i.e. the one with the smallest geometric mean of the slowdown against the best
  number of values per thread over all sizes.
"""

import argparse
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from TNL import BenchmarkLogs

_CSS = """
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 16px;
    }
    table {
        border-collapse: collapse;
        margin-bottom: 16px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 4px 8px;
        text-align: right;
        white-space: nowrap;
    }
    th {
        background-color: #f2f2f2;
        text-align: center;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #f1f1f1;
    }
    td.best {
        font-weight: bold;
    }
    td.faster {
        background-color: #d4f4d4;
    }
    td.slower {
        background-color: #f8d7d7;
    }
    nav ul {
        columns: 2;
    }
    .note {
        color: #555;
        font-size: 90%;
    }
</style>
"""

# order of devices in the tables, CPU devices first
_DEVICE_ORDER = ["sequential", "host", "cuda", "hip"]
_GPU_DEVICES = {"cuda", "hip"}
_CPU_DEVICES = {"sequential", "host"}
# order of value types in the tables (as logged by the benchmark)
_VALUE_TYPE_ORDER = ["char", "short", "int", "long", "float", "double"]
# the TNL scan via the public API
_TNL_PERFORMER = "TNL"
# prefix of the performers of the explicitly selected GPU kernels
_KERNEL_PREFIX = "TNL "
# columns identifying a benchmark configuration (rows of the tables)
_KEY_COLUMNS = ["value type", "size"]
_VPT_COLUMN = "values per thread"


def _load_dataframe(filenames: list[Path]) -> pd.DataFrame:
    frames = []
    for path in filenames:
        if not path.is_file():
            print(f"Skipping non-existing input file {path}", file=sys.stderr)
            continue
        frames.append(BenchmarkLogs.get_benchmark_dataframe(str(path)))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    missing = [
        c for c in ["operation", "device", "performer", *_KEY_COLUMNS] if c not in df
    ]
    if missing:
        sys.exit(
            f"The log is not a log of the scan benchmark (missing columns {missing})."
        )
    if _VPT_COLUMN not in df.columns:
        df[_VPT_COLUMN] = "default"
    df[_VPT_COLUMN] = df[_VPT_COLUMN].astype(str)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").astype("Int64")
    for column in ["time", "time_median", "bandwidth"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _is_kernel_row(df: pd.DataFrame) -> pd.Series:
    """Rows of the explicitly selected GPU kernels (numeric values per thread)."""
    return df[_VPT_COLUMN].str.fullmatch(r"\d+")


def _value_type_key(value_type: str) -> tuple[int, str]:
    if value_type in _VALUE_TYPE_ORDER:
        return (_VALUE_TYPE_ORDER.index(value_type), "")
    return (len(_VALUE_TYPE_ORDER), value_type)


def _sort_index(table: pd.DataFrame) -> pd.DataFrame:
    """Sorts the rows of a table indexed by (value type, size)."""
    return table.reindex(
        sorted(table.index, key=lambda k: (_value_type_key(k[0]), *k[1:]))
    )


def _device_key(device: str) -> int:
    return (
        _DEVICE_ORDER.index(device) if device in _DEVICE_ORDER else len(_DEVICE_ORDER)
    )


def _is_gpu(device: str) -> bool:
    return device in _GPU_DEVICES


def _is_cpu(device: str) -> bool:
    return device in _CPU_DEVICES


def _format_value(value: object, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    return fmt.format(value)


def _render_table(
    table: pd.DataFrame,
    fmt: str,
    highlight: str | None = None,
) -> str:
    """
    Renders a dataframe whose index are configuration keys and whose columns are
    (group, label) pairs.

    :param highlight: "min" or "max" highlights the best value in each row,
        "speedup" colors values greater/less than 1.
    """
    if table.empty:
        return "<p class='note'>No data.</p>\n"

    out = ["<table>\n<thead>\n"]
    index_names = [str(n) for n in table.index.names]
    groups: list[tuple[str, int]] = []
    for group, _ in table.columns:
        if groups and groups[-1][0] == group:
            groups[-1] = (group, groups[-1][1] + 1)
        else:
            groups.append((group, 1))

    out.append("<tr>")
    for name in index_names:
        out.append(f"<th rowspan='2'>{html.escape(name)}</th>")
    for group, span in groups:
        out.append(f"<th colspan='{span}'>{html.escape(str(group))}</th>")
    out.append("</tr>\n<tr>")
    for _, label in table.columns:
        out.append(f"<th>{html.escape(str(label))}</th>")
    out.append("</tr>\n</thead>\n<tbody>\n")

    for key, row in table.iterrows():
        keys = key if isinstance(key, tuple) else (key,)
        values = list(row)
        numeric = [
            v for v in values if isinstance(v, int | float) and not math.isnan(v)
        ]
        best = None
        if highlight == "min" and numeric:
            best = min(numeric)
        elif highlight == "max" and numeric:
            best = max(numeric)
        out.append("<tr>")
        for k in keys:
            out.append(f"<th>{html.escape(str(k))}</th>")
        for value in values:
            classes = []
            if best is not None and value == best:
                classes.append("best")
            if (
                highlight == "speedup"
                and isinstance(value, int | float)
                and not math.isnan(value)
            ):
                if value > 1:
                    classes.append("faster")
                elif value < 1:
                    classes.append("slower")
            cls = f" class='{' '.join(classes)}'" if classes else ""
            out.append(f"<td{cls}>{html.escape(_format_value(value, fmt))}</td>")
        out.append("</tr>\n")
    out.append("</tbody>\n</table>\n")
    return "".join(out)


def _pivot(df: pd.DataFrame, columns: list[str], value: str) -> pd.DataFrame:
    """Pivot so that rows are (value type, size) and columns are given by `columns`."""
    table = df.pivot_table(
        index=_KEY_COLUMNS, columns=columns, values=value, aggfunc="first"
    )
    return _sort_index(table)


def _pivot_comparison(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot so that columns are (device, performer), TNL first on each device."""
    table = _pivot(df, ["device", "performer"], value)
    order = list(dict.fromkeys(zip(df["device"], df["performer"], strict=True)))
    order = [c for c in order if c in table.columns]
    order.sort(key=lambda c: (_device_key(c[0]), c[1] != _TNL_PERFORMER))
    return table[order]


def _gpu_vs_cpu(times: pd.DataFrame) -> pd.DataFrame:
    """Speedup of each GPU performer against the same performer on CPU devices."""
    result = {}
    devices = list(dict.fromkeys(d for d, _ in times.columns))
    gpu_devices = [d for d in devices if _is_gpu(d)]
    cpu_devices = [d for d in devices if _is_cpu(d)]
    cpu_columns = [c for c in times.columns if _is_cpu(c[0])]
    for gpu in gpu_devices:
        for performer in [p for d, p in times.columns if d == gpu]:
            # the libraries have different names on the CPU and the GPU (e.g. CUB),
            # but TNL and Thrust are compared
            for cpu in cpu_devices:
                if (cpu, performer) in times.columns:
                    result[(f"{performer} on {gpu} vs", cpu)] = (
                        times[(cpu, performer)] / times[(gpu, performer)]
                    )
            if performer == _TNL_PERFORMER and cpu_columns:
                best_cpu = times[cpu_columns].min(axis=1)
                result[(f"{performer} on {gpu} vs", "best CPU")] = (
                    best_cpu / times[(gpu, performer)]
                )
    return pd.DataFrame(result, index=times.index)


def _tnl_vs_others(times: pd.DataFrame) -> pd.DataFrame:
    """Speedup of TNL against the other performers on the same device."""
    result = {}
    devices = list(dict.fromkeys(d for d, _ in times.columns))
    for device in devices:
        if (device, _TNL_PERFORMER) not in times.columns:
            continue
        for performer in [p for d, p in times.columns if d == device]:
            if performer != _TNL_PERFORMER:
                result[(f"{_TNL_PERFORMER} on {device} vs", performer)] = (
                    times[(device, performer)] / times[(device, _TNL_PERFORMER)]
                )
    return pd.DataFrame(result, index=times.index)


def _kernel_name(performer: str) -> str:
    return performer.removeprefix(_KERNEL_PREFIX)


def _tuning_times(kernels: pd.DataFrame, default: pd.DataFrame) -> pd.DataFrame:
    """
    Times with columns (kernel, values per thread), the default TNL scan (if measured)
    is the first column.
    """
    kernels = kernels.assign(
        kernel=kernels["performer"].map(_kernel_name),
        vpt=kernels[_VPT_COLUMN].astype(int),
    )
    table = _pivot(kernels, ["kernel", "vpt"], "time_ms")
    order = list(dict.fromkeys(kernels["kernel"]))
    table = table[sorted(table.columns, key=lambda c: (order.index(c[0]), c[1]))]
    if not default.empty:
        default_times = _pivot(default, ["performer"], "time_ms")[_TNL_PERFORMER]
        table.insert(0, (_TNL_PERFORMER, "default"), default_times)
    return table


def _best_values_per_thread(times: pd.DataFrame) -> pd.DataFrame:
    """For each kernel: the best values per thread, its time and its speedup vs TNL."""
    result: dict[tuple[str, str], pd.Series] = {}
    kernels = [
        k for k in dict.fromkeys(k for k, _ in times.columns) if k != _TNL_PERFORMER
    ]
    default = times.get((_TNL_PERFORMER, "default"))
    for kernel in kernels:
        kernel_times = times[kernel]
        valid = kernel_times.notna().any(axis=1)
        best_vpt = kernel_times[valid].idxmin(axis=1).reindex(times.index)
        best_time = kernel_times.min(axis=1)
        result[(kernel, "values per thread")] = best_vpt.map(
            lambda v: "" if pd.isna(v) else str(int(v))
        )
        result[(kernel, "time [ms]")] = best_time
        if default is not None:
            result[(kernel, f"speedup vs {_TNL_PERFORMER}")] = default / best_time
    return pd.DataFrame(result, index=times.index)


def _recommended_values_per_thread(times: pd.DataFrame) -> pd.DataFrame:
    """
    For each value type and kernel: the number of values per thread with the smallest
    geometric mean (over sizes) of the slowdown against the best number of values per
    thread of the same kernel.
    """
    rows = {}
    kernels = [
        k for k in dict.fromkeys(k for k, _ in times.columns) if k != _TNL_PERFORMER
    ]
    value_types = sorted(
        dict.fromkeys(times.index.get_level_values(0)), key=_value_type_key
    )
    for value_type in value_types:
        row: dict[tuple[str, str], object] = {}
        for kernel in kernels:
            kernel_times = times.loc[value_type, kernel].dropna(how="all")
            if kernel_times.empty:
                continue
            slowdown = kernel_times.div(kernel_times.min(axis=1), axis=0)
            geomean = np.exp(np.log(slowdown).mean(axis=0, skipna=False)).dropna()
            if geomean.empty:
                continue
            row[(kernel, "values per thread")] = str(int(geomean.idxmin()))
            row[(kernel, "slowdown")] = geomean.min()
        rows[value_type] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not table.empty:
        table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.index.name = "value type"
    return table


def _anchor(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()


def _comparison_html(op: str, op_df: pd.DataFrame) -> list[str]:
    parts = [f"<h2 id='{_anchor(op)}'>{html.escape(op)}</h2>\n"]

    times = _pivot_comparison(op_df, "time_ms")
    parts.append("<h3>Time [ms]</h3>\n")
    parts.append(_render_table(times, "{:.4g}", highlight="min"))

    if "bandwidth" in op_df.columns and op_df["bandwidth"].notna().any():
        parts.append("<h3>Bandwidth [GB/s]</h3>\n")
        parts.append(
            _render_table(
                _pivot_comparison(op_df, "bandwidth"), "{:.4g}", highlight="max"
            )
        )

    gpu_cpu = _gpu_vs_cpu(times)
    if not gpu_cpu.empty:
        parts.append("<h3>Speedup GPU vs CPU</h3>\n")
        parts.append(_render_table(gpu_cpu, "{:.2f}", highlight="speedup"))

    tnl_others = _tnl_vs_others(times)
    if not tnl_others.empty:
        parts.append("<h3>Speedup TNL vs others</h3>\n")
        parts.append(_render_table(tnl_others, "{:.2f}", highlight="speedup"))
    return parts


def _tuning_html(title: str, kernels: pd.DataFrame, default: pd.DataFrame) -> list[str]:
    parts = [f"<h2 id='{_anchor(title)}'>{html.escape(title)}</h2>\n"]

    times = _tuning_times(kernels, default)
    parts.append("<h3>Time [ms]</h3>\n")
    parts.append(_render_table(times, "{:.4g}", highlight="min"))

    parts.append("<h3>Best values per thread</h3>\n")
    parts.append(_render_table(_best_values_per_thread(times), "{:.4g}"))

    parts.append("<h3>Recommended values per thread</h3>\n")
    parts.append(
        "<p class='note'>The number of values per thread with the smallest geometric "
        "mean (over all sizes) of the slowdown against the best number of values per "
        "thread of the same kernel. The slowdown is the geometric mean, 1 means that "
        "it is the best for all sizes.</p>\n"
    )
    parts.append(_render_table(_recommended_values_per_thread(times), "{:.3f}"))
    return parts


def generate_html(
    df: pd.DataFrame, metadata: dict | None, time_column: str, title: str
) -> str:
    df = df.assign(time_ms=pd.to_numeric(df[time_column], errors="coerce") * 1e3)
    is_kernel = _is_kernel_row(df)
    comparison = df[~is_kernel]
    kernels = df[is_kernel]

    operations = list(dict.fromkeys(comparison["operation"]))
    tunings = [
        (f"Tuning: {op} on {device}", op, device)
        for op, device in dict.fromkeys(
            zip(kernels["operation"], kernels["device"], strict=True)
        )
    ]

    parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n",
        f"<title>{html.escape(title)}</title>\n",
        _CSS,
        "</head>\n<body>\n",
        f"<h1>{html.escape(title)}</h1>\n",
    ]
    if metadata:
        parts.append("<h2>Metadata</h2>\n")
        parts.append(BenchmarkLogs.dict_to_html_table(metadata))

    parts.append("<h2>Operations</h2>\n<nav><ul>\n")
    for heading in operations + [t for t, _, _ in tunings]:
        parts.append(
            f"<li><a href='#{_anchor(heading)}'>{html.escape(heading)}</a></li>\n"
        )
    parts.append("</ul></nav>\n")
    parts.append(
        "<p class='note'>"
        "Times are given in milliseconds (column "
        f"<code>{html.escape(time_column)}</code> of the log), "
        "bandwidth in GB/s. The best value in each row is in bold. "
        "Speedup is the ratio of times: values greater than 1 (green) mean that the "
        "first named performer is faster.</p>\n"
    )

    for op in operations:
        parts += _comparison_html(op, comparison[comparison["operation"] == op])

    for heading, op, device in tunings:
        selected = (df["operation"] == op) & (df["device"] == device)
        default = comparison[
            selected[~is_kernel] & (comparison["performer"] == _TNL_PERFORMER)
        ]
        parts += _tuning_html(heading, kernels[selected[is_kernel]], default)

    parts.append("</body>\n</html>\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=[Path("tnl-benchmark-scan-cuda.log")],
        help="benchmark log file(s) (default: tnl-benchmark-scan-cuda.log)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output HTML file (default: <first log>.html)"
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="append",
        default=[],
        help="process only operations containing this substring "
        "(can be given multiple times, e.g. -f inplace)",
    )
    parser.add_argument(
        "-t",
        "--value-type",
        action="append",
        default=[],
        help="process only this value type "
        "(can be given multiple times, e.g. -t double)",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        choices=["time", "time_median"],
        help="which measured time is used for the tables and speedups (default: time)",
    )
    args = parser.parse_args()

    df = _load_dataframe(args.logs)
    if df.empty:
        sys.exit("No benchmark data found.")
    if args.filter:
        mask = (
            df["operation"]
            .astype(str)
            .apply(lambda op: any(f in op for f in args.filter))
        )
        df = df[mask]
        if df.empty:
            sys.exit(f"No operations match the filter {args.filter}.")
    if args.value_type:
        df = df[df["value type"].isin(args.value_type)]
        if df.empty:
            sys.exit(f"No value types match {args.value_type}.")

    metadata = BenchmarkLogs.get_benchmark_metadata(str(args.logs[0]))
    title = "Scan benchmark: " + ", ".join(p.name for p in args.logs)
    output = args.output or args.logs[0].with_suffix(".html")
    output.write_text(generate_html(df, metadata, args.time_column, title))
    print(f"Written {output}")


if __name__ == "__main__":
    main()
