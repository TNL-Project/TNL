#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""
Generates HTML tables from the log of tnl-benchmark-blas (or tnl-benchmark-blas-cuda,
tnl-benchmark-blas-hip). For each benchmarked operation it produces:

- a table of measured times for all devices and performers,
- a table of effective bandwidths,
- GPU vs CPU speedups of the same performer (and of TNL vs the best CPU result),
- speedups of TNL vs the other performers (libraries) on the same device.
"""

import argparse
import html
import math
import sys
from pathlib import Path

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
        columns: 3;
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
# performers representing the main TNL implementation, in order of preference
_TNL_PERFORMERS = ["ET", "TNL"]
# columns with measured results, all other columns (except operation, device and
# performer) identify the benchmark configuration
_RESULT_COLUMNS = {"speedup", "bandwidth", "loops", "ops_per_loop"}
_RESULT_COLUMN_PREFIXES = ("time", "cycles")


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

    # older logs do not have the "device" column,
    # the device is a prefix of the performer
    if "device" not in df.columns:
        df["device"] = pd.NA
    missing = df["device"].isna()
    if missing.any():
        split = (
            df.loc[missing, "performer"].astype(str).str.split(" ", n=1, expand=True)
        )
        prefix_map = {"CPU": "host", "GPU": "gpu"}
        has_prefix = split[0].isin(prefix_map.keys()) & split[1].notna()
        idx = split.index[has_prefix]
        df.loc[idx, "device"] = split.loc[idx, 0].map(prefix_map)
        df.loc[idx, "performer"] = split.loc[idx, 1]
        df["device"] = df["device"].fillna("unknown")
    return df


def _key_columns(df: pd.DataFrame) -> list[str]:
    """Columns that identify a benchmark configuration (e.g. precision, size)."""
    return [
        c
        for c in df.columns
        if c not in ("operation", "device", "performer")
        and c not in _RESULT_COLUMNS
        and not c.startswith(_RESULT_COLUMN_PREFIXES)
    ]


def _format_key(value: object) -> str:
    # pandas parses integer columns with missing values as floats
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sort_value(value: object) -> tuple[int, float | str]:
    try:
        return (0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (1, str(value))


def _device_key(device: str) -> int:
    return (
        _DEVICE_ORDER.index(device) if device in _DEVICE_ORDER else len(_DEVICE_ORDER)
    )


def _is_gpu(device: str) -> bool:
    return device in _GPU_DEVICES or device == "gpu"


def _is_cpu(device: str) -> bool:
    return device in _CPU_DEVICES


def _format_value(value: float, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
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
        values = row.to_numpy(dtype=float)
        best = None
        valid = values[~pd.isna(values)]
        if highlight == "min" and valid.size > 0:
            best = valid.min()
        elif highlight == "max" and valid.size > 0:
            best = valid.max()
        out.append("<tr>")
        for k in keys:
            out.append(f"<th>{html.escape(str(k))}</th>")
        for value in values:
            classes = []
            if best is not None and value == best:
                classes.append("best")
            if highlight == "speedup" and not math.isnan(value):
                if value > 1:
                    classes.append("faster")
                elif value < 1:
                    classes.append("slower")
            cls = f" class='{' '.join(classes)}'" if classes else ""
            out.append(f"<td{cls}>{_format_value(value, fmt)}</td>")
        out.append("</tr>\n")
    out.append("</tbody>\n</table>\n")
    return "".join(out)


def _pivot(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Pivot so that rows are configurations and columns are (device, performer)."""
    table = df.pivot_table(
        index=keys, columns=["device", "performer"], values=value, aggfunc="first"
    )
    table = table.reindex(
        sorted(table.index, key=lambda k: tuple(_sort_value(v) for v in k))
    )
    if len(keys) == 1:
        table.index = pd.Index(
            [k[0] if isinstance(k, tuple) else k for k in table.index], name=keys[0]
        )
    # keep the order of performers as found in the log, devices by _DEVICE_ORDER
    order = list(dict.fromkeys(zip(df["device"], df["performer"], strict=True)))
    order = [c for c in order if c in table.columns]
    order.sort(key=lambda c: _device_key(c[0]))
    return table[order]


def _tnl_performer(performers: list[str]) -> str | None:
    for p in _TNL_PERFORMERS:
        if p in performers:
            return p
    return None


def _gpu_vs_cpu(times: pd.DataFrame) -> pd.DataFrame:
    """Speedup of each GPU performer against the same performer on CPU devices."""
    result = {}
    devices = list(dict.fromkeys(d for d, _ in times.columns))
    gpu_devices = [d for d in devices if _is_gpu(d)]
    cpu_devices = [d for d in devices if _is_cpu(d)]
    cpu_columns = [c for c in times.columns if _is_cpu(c[0])]
    for gpu in gpu_devices:
        gpu_performers = [p for d, p in times.columns if d == gpu]
        tnl = _tnl_performer(gpu_performers)
        # TNL first so that all its columns form one group in the table header
        if tnl is not None:
            gpu_performers.remove(tnl)
            gpu_performers.insert(0, tnl)
        for performer in gpu_performers:
            for cpu in cpu_devices:
                if (cpu, performer) in times.columns:
                    result[(f"{performer} on {gpu} vs", cpu)] = (
                        times[(cpu, performer)] / times[(gpu, performer)]
                    )
            if performer == tnl and cpu_columns:
                best_cpu = times[cpu_columns].min(axis=1)
                result[(f"{tnl} on {gpu} vs", "best CPU")] = (
                    best_cpu / times[(gpu, tnl)]
                )
    return pd.DataFrame(result, index=times.index)


def _tnl_vs_others(times: pd.DataFrame) -> pd.DataFrame:
    """Speedup of the TNL performer against the other performers on the same device."""
    result = {}
    devices = list(dict.fromkeys(d for d, _ in times.columns))
    for device in devices:
        performers = [p for d, p in times.columns if d == device]
        tnl = _tnl_performer(performers)
        if tnl is None:
            continue
        for performer in performers:
            if performer != tnl:
                result[(f"{tnl} on {device} vs", performer)] = (
                    times[(device, performer)] / times[(device, tnl)]
                )
    return pd.DataFrame(result, index=times.index)


def _anchor(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()


def generate_html(
    df: pd.DataFrame, metadata: dict | None, time_column: str, title: str
) -> str:
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

    operations = list(dict.fromkeys(df["operation"]))
    parts.append("<h2>Operations</h2>\n<nav><ul>\n")
    for op in operations:
        parts.append(f"<li><a href='#{_anchor(op)}'>{html.escape(op)}</a></li>\n")
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
        op_df = df[df["operation"] == op].dropna(axis=1, how="all")
        keys = _key_columns(op_df)
        if not keys:
            op_df = op_df.assign(configuration="-")
            keys = ["configuration"]
        op_df = op_df.copy()
        op_df[keys] = op_df[keys].map(_format_key)
        op_df[time_column] = pd.to_numeric(op_df[time_column], errors="coerce")
        op_df["time_ms"] = op_df[time_column] * 1e3

        parts.append(f"<h2 id='{_anchor(op)}'>{html.escape(op)}</h2>\n")

        times = _pivot(op_df, keys, "time_ms")
        parts.append("<h3>Time [ms]</h3>\n")
        parts.append(_render_table(times, "{:.4g}", highlight="min"))

        if "bandwidth" in op_df.columns:
            op_df["bandwidth"] = pd.to_numeric(op_df["bandwidth"], errors="coerce")
            if op_df["bandwidth"].notna().any():
                parts.append("<h3>Bandwidth [GB/s]</h3>\n")
                parts.append(
                    _render_table(
                        _pivot(op_df, keys, "bandwidth"), "{:.4g}", highlight="max"
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
        default=[Path("tnl-benchmark-blas-cuda.log")],
        help="benchmark log file(s) (default: tnl-benchmark-blas-cuda.log)",
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
        "(can be given multiple times, e.g. -f scan)",
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

    metadata = BenchmarkLogs.get_benchmark_metadata(str(args.logs[0]))
    title = "BLAS benchmark: " + ", ".join(p.name for p in args.logs)
    output = args.output or args.logs[0].with_suffix(".html")
    output.write_text(generate_html(df, metadata, args.time_column, title))
    print(f"Written {output}")


if __name__ == "__main__":
    main()
