#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from TNL.BenchmarkLogs import (
    dict_to_html_table,
    gen_dataframes_per_operation,
    get_benchmark_metadata,
)
from TNL.BenchmarkPlots import (
    get_image_html_tag,
    heatmaps_bandwidth,
    plot_bandwidth_vs_size,
)

_CSS = """\
h1, h2 { border-bottom: solid 1px lightgray; }
table { border-collapse: collapse; }
table.benchmark td { text-align: end; }
th, td { padding: 2px; }
"""

_HTML_HEADER = f"""\
<head>
<meta charset="UTF-8">
<style>
{_CSS}
</style>
</head>
<body>"""

_FORMATTERS: dict[str, Any] = {
    "time_stddev": lambda value: f"{value:e}",
    "bandwidth": lambda value: f"{value:.3f}",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TNL benchmark log files to HTML reports."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input log file (one JSON record per line)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output HTML file (default: INPUT with .html extension)",
    )
    args = parser.parse_args()

    log_path: Path = args.input
    html_path: Path = args.output or log_path.with_suffix(".html")

    metadata = get_benchmark_metadata(log_path)
    if metadata is not None and "title" in metadata:
        title = metadata["title"]
    else:
        title = log_path.stem
    dataframes = list(gen_dataframes_per_operation(log_path))

    print(f"Writing output to {html_path}")
    with html_path.open("w") as f:
        print("<html>", file=f)
        print(_HTML_HEADER, file=f)

        print(f"<h1>{title}</h1>", file=f)
        if metadata is not None:
            print(dict_to_html_table(metadata), file=f)

        print("<h2>Table of contents</h2>", file=f)
        print("<ol>", file=f)
        for op, _df in dataframes:
            anchor = op.replace(" ", "_")
            print(f'<li><a href="#{anchor}">{op}</a></li>', file=f)
        print("</ol>", file=f)

        for op, df in dataframes:
            anchor = op.replace(" ", "_")
            print(f'<h2 id="{anchor}">{op}</h2>', file=f)
            print(
                df.to_html(classes="benchmark", formatters=_FORMATTERS),  # pyright: ignore[reportArgumentType]
                file=f,
            )

            size_name: str | None = None
            if "size" in df.index.names:
                size_name = "size"
            elif "DOFs" in df.index.names:
                size_name = "DOFs"
            if size_name is not None:
                fig, _ax = plot_bandwidth_vs_size(df, size_name)
                print(get_image_html_tag(fig, format="png"), file=f)
                plt.close(fig)

            if all(name in df.index.names for name in ["rows", "columns", "bandwidth"]):
                for fig, _ax in heatmaps_bandwidth(df):
                    print(get_image_html_tag(fig, format="png"), file=f)
                    plt.close(fig)

        print("</body>", file=f)
        print("</html>", file=f)


if __name__ == "__main__":
    main()
