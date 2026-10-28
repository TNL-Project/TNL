#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""
Processes the results of tnl-benchmark-matrix-assembly.

It reads one or more JSON log files (e.g. from the host and the GPU executables)
and writes:

- an HTML page with tables of the measured times and speed-ups,
- LaTeX/TikZ (pgfplots) figures with speed-up graphs, optionally compiled to PDF.

The speed-up of TNL over another library is the ratio of the time of the library
and the time of TNL with the same matrix format, device, precision and grid.
"""

import argparse
import html
import shutil
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("matrix-assembly-results")

OPERATIONS = [
    "allocation",
    "filling",
    "assembly",
    "host assembly",
    "transfer",
    "host assembly + transfer",
    "array transfer",
]
LIBRARIES = ["TNL", "Ginkgo", "Hypre", "PETSc", "cuSPARSE"]
FORMATS = ["CSR", "Ellpack", "SlicedEllpack"]
GROUP_COLUMNS = ["precision", "device", "dimension", "grid size"]

# pgfplots colors and marks for the bars of the individual series
BAR_STYLES = [
    "fill=blue!60",
    "fill=red!60",
    "fill=green!50!black",
    "fill=orange!80",
    "fill=violet!60",
    "fill=cyan!60",
    "fill=brown!70",
    "fill=gray!60",
    "fill=magenta!60",
    "fill=olive!70",
]


def load_results(filenames: list[str]) -> pd.DataFrame:
    """Reads the JSON log files and returns one dataframe with times in milliseconds."""
    frames = []
    for filename in filenames:
        print(f"Parsing input file {filename}")
        frames.append(pd.read_json(filename, orient="records", lines=True, dtype=False))
    df = pd.concat(frames, ignore_index=True)

    # rows with error messages do not contain the performer
    if "performer" not in df.columns:
        raise ValueError("The input files do not contain any benchmark results.")
    df = df[df["performer"].notna()].copy()

    df["dimension"] = df["dimension"].astype(int)
    df["rows"] = df["rows"].astype(int)
    df["nonzeros"] = df["nonzeros"].astype(int)
    df["time"] = pd.to_numeric(df["time"], errors="coerce") * 1e3
    df["time_stddev"] = pd.to_numeric(df["time_stddev"], errors="coerce") * 1e3

    # if the same measurement appears in more files, keep the last one
    key = [*GROUP_COLUMNS, "performer", "format", "operation"]
    return df.drop_duplicates(subset=key, keep="last")


def sort_key(column: pd.Series) -> pd.Series:
    """Sorts libraries, formats and operations in the natural order."""
    orders = {"performer": LIBRARIES, "format": FORMATS, "operation": OPERATIONS}
    if column.name in orders:
        order = orders[column.name]
        return column.map(lambda value: order.index(value) if value in order else 99)
    return column


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the column "TNL speed-up" with the speed-up of TNL over the given row."""
    tnl = df[df["performer"] == "TNL"][[*GROUP_COLUMNS, "format", "operation", "time"]]
    tnl = tnl.rename(columns={"time": "TNL time"})
    df = df.merge(tnl, on=[*GROUP_COLUMNS, "format", "operation"], how="left")
    df["TNL speed-up"] = df["time"] / df["TNL time"]
    return df.drop(columns=["TNL time"])


def series_label(row: pd.Series) -> str:
    return f"{row['performer']} {row['format']}"


def grid_label(dimension: int, grid_size: str) -> str:
    return f"{dimension}D {grid_size}"


# --------------------------------------------------------------------------- HTML


HTML_STYLE = """
body { font-family: sans-serif; margin: 2em; }
table { border-collapse: collapse; margin-bottom: 2em; }
th, td { border: 1px solid #999; padding: 0.25em 0.6em; text-align: right; }
th { background: #e8e8e8; }
td.text { text-align: left; }
tr.tnl { background: #eef5ff; }
td.faster { color: #006400; }
td.slower { color: #b00000; }
"""


def format_number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}g}"


def html_results_table(df: pd.DataFrame) -> str:
    """Table of times (ms) of all operations and the TNL speed-up of the assembly."""
    operations = [op for op in OPERATIONS if op in set(df["operation"])]
    times = df.pivot_table(
        index=["dimension", "grid size", "rows", "nonzeros", "performer", "format"],
        columns="operation",
        values="time",
        aggfunc="first",
    )
    assembly = df[df["operation"] == "assembly"].set_index(
        ["dimension", "grid size", "rows", "nonzeros", "performer", "format"]
    )["TNL speed-up"]
    times = times.reset_index().sort_values(
        ["dimension", "grid size", "performer", "format"], key=sort_key
    )

    lines = ["<table>", "<thead><tr>"]
    headers = ["dim", "grid", "rows", "nonzeros", "library", "format"]
    headers += [f"{op} [ms]" for op in operations]
    headers += ["TNL speed-up (assembly)"]
    lines.append("".join(f"<th>{html.escape(h)}</th>" for h in headers))
    lines.append("</tr></thead>")
    lines.append("<tbody>")
    for _, row in times.iterrows():
        index = (
            row["dimension"],
            row["grid size"],
            row["rows"],
            row["nonzeros"],
            row["performer"],
            row["format"],
        )
        speedup = assembly.get(index, float("nan"))
        css = ' class="tnl"' if row["performer"] == "TNL" else ""
        cells = [
            f"<td>{row['dimension']}</td>",
            f'<td class="text">{html.escape(str(row["grid size"]))}</td>',
            f"<td>{row['rows']}</td>",
            f"<td>{row['nonzeros']}</td>",
            f'<td class="text">{html.escape(row["performer"])}</td>',
            f'<td class="text">{html.escape(row["format"])}</td>',
        ]
        cells += [f"<td>{format_number(row.get(op))}</td>" for op in operations]
        if row["performer"] == "TNL" or pd.isna(speedup):
            cells.append("<td></td>")
        else:
            speed_class = "faster" if speedup >= 1 else "slower"
            cells.append(f'<td class="{speed_class}">{format_number(speedup)}</td>')
        lines.append(f"<tr{css}>{''.join(cells)}</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def write_html(df: pd.DataFrame, output_dir: Path) -> Path:
    body = [
        "<h1>Matrix assembly benchmark</h1>",
        (
            "<p>Times are in milliseconds. The TNL speed-up is the ratio of the "
            "assembly time of the library and the assembly time of TNL with the same "
            "matrix format, device, precision and grid. Values greater than 1 mean "
            "that TNL is faster.</p>"
        ),
    ]
    for (precision, device), group in df.groupby(["precision", "device"], sort=True):
        body.append(f"<h2>{html.escape(device)}, {html.escape(precision)}</h2>")
        body.append(html_results_table(group))
    page = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        "<title>Matrix assembly benchmark</title>\n"
        f"<style>{HTML_STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )
    path = output_dir / "matrix-assembly.html"
    path.write_text(page)
    print(f"Written {path}")
    return path


# --------------------------------------------------------------------------- TikZ


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def tikz_bar_chart(
    title: str,
    ylabel: str,
    xlabels: list[str],
    series: dict[str, dict[str, float]],
    log_scale: bool,
) -> str:
    """
    Returns a standalone LaTeX document with a pgfplots bar chart.

    :param xlabels: labels of the groups of bars on the x axis
    :param series: mapping from the legend entry to the mapping from the x label
        to the value
    """
    coords = ",".join(xlabels)
    bar_width = 6.0  # pt
    # each bar takes about 0.25 cm, plus space between the groups of bars
    width = max(12.0, 0.25 * len(xlabels) * len(series) + 1.5 * len(xlabels) + 2)
    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        "   ybar,",
        f"   bar width={bar_width:.1f}pt,",
        f"   width={width:.1f}cm, height=8cm,",
        f"   title={{{latex_escape(title)}}},",
        f"   ylabel={{{latex_escape(ylabel)}}},",
        f"   symbolic x coords={{{coords}}},",
        "   xtick=data,",
        "   enlarge x limits=" + ("0.5" if len(xlabels) == 1 else "0.2") + ",",
        "   legend style={at={(0.5,-0.15)}, anchor=north, legend columns=2,",
        "      legend cell align=left, column sep=1em},",
        "   ymajorgrids,",
        "   extra y ticks={1},",
        "   extra y tick style={grid=major, major grid style={red}},",
    ]
    if log_scale:
        lines.append("   ymode=log, log origin=infty,")
    lines.append("]")
    for idx, (label, values) in enumerate(series.items()):
        style = BAR_STYLES[idx % len(BAR_STYLES)]
        points = " ".join(
            f"({x},{values[x]:.6g})"
            for x in xlabels
            if x in values and pd.notna(values[x]) and values[x] > 0
        )
        lines.append(rf"\addplot[{style}] coordinates {{{points}}};")
        lines.append(rf"\addlegendentry{{{latex_escape(label)}}}")
    lines += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""]
    return "\n".join(lines)


def x_labels(df: pd.DataFrame) -> list[str]:
    grids = df[["dimension", "grid size"]].drop_duplicates()
    grids = grids.sort_values(["dimension", "grid size"])
    # symbolic coordinates must not contain commas
    return [
        grid_label(d, str(g)).replace(",", " ")
        for d, g in grids.itertuples(index=False)
    ]


def speedup_figures(df: pd.DataFrame, operation: str) -> dict[str, str]:
    """Speed-up of TNL over the other libraries for each device and precision."""
    figures = {}
    data = df[(df["operation"] == operation) & (df["performer"] != "TNL")]
    data = data[data["TNL speed-up"].notna()]
    for (precision, device), group in data.groupby(["precision", "device"], sort=True):
        xlabels = x_labels(group)
        series: dict[str, dict[str, float]] = {}
        for _, row in group.sort_values(
            ["performer", "format"], key=sort_key
        ).iterrows():
            x = grid_label(row["dimension"], str(row["grid size"])).replace(",", " ")
            series.setdefault(series_label(row), {})[x] = row["TNL speed-up"]
        name = f"speedup-{operation}-{device}-{precision}".replace(" ", "-")
        figures[name] = tikz_bar_chart(
            title=f"TNL speed-up, {operation}, {device}, {precision}",
            ylabel="speed-up of TNL (same format)",
            xlabels=xlabels,
            series=series,
            log_scale=True,
        )
    return figures


def tikz_stacked_chart(
    title: str,
    ylabel: str,
    xlabels: list[str],
    series: dict[str, dict[str, float]],
) -> str:
    """
    Returns a standalone LaTeX document with a pgfplots stacked bar chart.

    :param xlabels: labels of the bars on the x axis
    :param series: mapping from the legend entry (part of the bar) to the mapping from
        the x label to the value
    """
    coords = ",".join(xlabels)
    width = max(10.0, 1.0 * len(xlabels) + 3)
    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        "   ybar stacked,",
        "   bar width=14pt,",
        f"   width={width:.1f}cm, height=8cm,",
        f"   title={{{latex_escape(title)}}},",
        f"   ylabel={{{latex_escape(ylabel)}}},",
        f"   symbolic x coords={{{coords}}},",
        "   xtick=data,",
        "   xticklabel style={rotate=45, anchor=east},",
        "   enlarge x limits=0.08,",
        "   ymin=0,",
        "   legend style={at={(1.02,1)}, anchor=north west},",
        "   legend cell align=left,",
        "   ymajorgrids,",
        "]",
    ]
    for idx, (label, values) in enumerate(series.items()):
        style = BAR_STYLES[idx % len(BAR_STYLES)]
        points = " ".join(
            f"({x},{values[x]:.6g})"
            for x in xlabels
            if x in values and pd.notna(values[x])
        )
        lines.append(rf"\addplot[{style}] coordinates {{{points}}};")
        lines.append(rf"\addlegendentry{{{latex_escape(label)}}}")
    lines += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""]
    return "\n".join(lines)


def gpu_figures(df: pd.DataFrame) -> dict[str, str]:
    """
    Figures comparing the assembly on the GPU with the assembly on the host followed
    by the transfer to the GPU:

    - speed-up of the assembly on the GPU over the host assembly + transfer for all
      libraries and formats,
    - for each grid, stacked bars with the times of the host assembly and the transfer.
    """
    figures = {}
    gpu = df[df["device"].isin(["cuda", "hip"])]
    for (precision, device), group in gpu.groupby(["precision", "device"], sort=True):
        times = group.pivot_table(
            index=["dimension", "grid size", "performer", "format"],
            columns="operation",
            values="time",
            aggfunc="first",
        ).reset_index()
        times = times.sort_values(
            ["dimension", "grid size", "performer", "format"], key=sort_key
        )
        if "host assembly + transfer" not in times.columns:
            continue

        # speed-up of the assembly on the GPU
        series: dict[str, dict[str, float]] = {}
        for _, row in times.iterrows():
            x = grid_label(row["dimension"], str(row["grid size"])).replace(",", " ")
            speedup = row["host assembly + transfer"] / row.get("assembly")
            if pd.notna(speedup):
                series.setdefault(series_label(row), {})[x] = speedup
        if series:
            figures[f"gpu-speedup-{device}-{precision}"] = tikz_bar_chart(
                title=(
                    f"GPU assembly vs. host assembly + transfer, {device}, {precision}"
                ),
                ylabel="speed-up of the assembly on the GPU",
                xlabels=x_labels(group),
                series=series,
                log_scale=True,
            )

        # time of the host assembly and the transfer for each grid
        measured = times[times["host assembly + transfer"].notna()]
        for (dimension, grid), grid_times in measured.groupby(
            ["dimension", "grid size"], sort=True
        ):
            if grid_times.empty:
                continue
            labels = [series_label(row) for _, row in grid_times.iterrows()]
            parts: dict[str, dict[str, float]] = {
                "host assembly": {},
                "transfer": {},
            }
            for label, (_, row) in zip(labels, grid_times.iterrows(), strict=True):
                for part, values in parts.items():
                    values[label] = row.get(part)
            name = f"host-to-gpu-{device}-{precision}-{dimension}D-{grid}"
            figures[name] = tikz_stacked_chart(
                title=(
                    f"Host assembly + transfer, {grid_label(dimension, str(grid))}, "
                    f"{device}, {precision}"
                ),
                ylabel="time [ms]",
                xlabels=labels,
                series=parts,
            )
    return figures


def write_figures(figures: dict[str, str], output_dir: Path, compile_pdf: bool) -> None:
    latex = shutil.which("pdflatex") if compile_pdf else None
    if compile_pdf and latex is None:
        print("pdflatex was not found, the figures will not be compiled")
    for name, source in figures.items():
        path = output_dir / f"{name}.tex"
        path.write_text(source)
        print(f"Written {path}")
        if latex is not None:
            result = subprocess.run(
                [latex, "-interaction=nonstopmode", "-halt-on-error", path.name],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(f"Compilation of {path} failed:\n{result.stdout[-2000:]}")
            else:
                print(f"Written {path.with_suffix('.pdf')}")
                for ext in (".aux", ".log"):
                    path.with_suffix(ext).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process the results of tnl-benchmark-matrix-assembly."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="JSON log files written by the benchmark (e.g. host and GPU results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["allocation", "filling", "assembly"],
        choices=["allocation", "filling", "assembly"],
        help="Operations for which the speed-up graphs are generated.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Compile the TikZ figures to PDF with pdflatex.",
    )
    args = parser.parse_args()

    df = compute_speedups(load_results(args.log_files))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_html(df, args.output_dir)

    figures = {}
    for operation in args.operations:
        figures.update(speedup_figures(df, operation))
    figures.update(gpu_figures(df))
    write_figures(figures, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
