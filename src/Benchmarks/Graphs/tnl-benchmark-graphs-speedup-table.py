#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""
Builds a per-graph speedup summary table from tnl-benchmark-graphs log files.

For every TNL measurement executed on a GPU, the speedup is computed relative to:
  - TNL on CPU (the "host" run with plain CSR segments)
  - Boost (sequential)
  - each individual Gunrock launch configuration (thread_mapped, block_mapped, ...)
  - the best (fastest) of the Gunrock launch configurations
Additionally, the speedup of TNL on CPU relative to Boost is computed as well.

Semiring variants of a problem (e.g. "Semiring BFS dir") are compared against the
same Boost/Gunrock baselines as the corresponding plain problem ("BFS dir"), since
Boost and Gunrock do not have semiring implementations.

The result is written as a single table with a structured (MultiIndex) header:
(problem, solver, device, kernel, launch cfg., metric, reference).

For each problem, a performance-profile plot is also produced: for every TNL
kernel (the best time across its launch configurations), the graphs are sorted by
their speedup relative to the best Gunrock kernel, and the number of graphs
reaching a given speedup is plotted as a step curve.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from TNL.BenchmarkLogs import get_benchmark_dataframe
from TNL.MultiindexCreator import MultiindexCreator

GRAPH_COLUMNS = ["Graph name", "nodes", "edges", "edges per node"]
COLUMN_DEPTH = 7  # (problem, solver, device, kernel, launch cfg., metric, reference)

# Okabe-Ito palette: colorblind-safe, used in a fixed order (never cycled/reassigned
# when the set of curves changes). Paired with a linestyle so curves stay
# distinguishable if a plot ever needs more series than the palette provides.
CVD_SAFE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
LINESTYLES = ["-", "--", "-.", ":"]


def load_results(input_files):
    """Parse and concatenate one or more benchmark log files into one dataframe."""
    df = pd.DataFrame()
    for filename in input_files:
        df = pd.concat([df, get_benchmark_dataframe(filename)], ignore_index=True)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["launch cfg."] = df["launch cfg."].fillna("")
    # "mode" distinguishes e.g. top-down/bottom-up BFS variants that share the same
    # kernel and launch cfg.; fold it into the kernel label so such variants don't
    # collapse into a single pivot column.
    mode = df["mode"] if "mode" in df.columns else pd.Series("", index=df.index)
    mode = mode.fillna("")
    df["kernel"] = [
        kernel if m in ("", "N/A") else f"{kernel} [{m}]"
        for kernel, m in zip(df["kernel"], mode)
    ]
    return df


def base_problem(problem):
    """Strip the "Semiring " prefix so semiring variants reuse the plain baselines."""
    return problem.removeprefix("Semiring ")


def display_kernel(kernel):
    """Drop the "[top-down compact]" mode suffix (the default, common mode) from
    a kernel label so plot legends stay short; other modes are kept since they
    denote a genuinely different algorithm variant."""
    return kernel.replace(" [top-down compact]", "")


def pivot_times(df):
    """Wide table: graph name -> time, keyed by
    (problem, solver, performer, kernel, launch cfg.)."""
    return df.pivot_table(
        index="graph name",
        columns=["problem", "solver", "performer", "kernel", "launch cfg."],
        values="time",
        aggfunc="first",
    )


def get_graph_info(df):
    info = df.drop_duplicates(subset=["graph name"])[
        ["graph name", "nodes", "edges"]
    ].copy()
    info["nodes"] = pd.to_numeric(info["nodes"])
    info["edges"] = pd.to_numeric(info["edges"])
    info["edges per node"] = info["edges"] / info["nodes"]
    return info.set_index("graph name")


def get_tnl_gpu_configs(df, problem):
    """(device, kernel, launch cfg.) combos of TNL runs on a GPU for this problem."""
    rows = df[
        (df["problem"] == problem)
        & (df["solver"] == "TNL")
        & (df["performer"].isin(["cuda", "hip"]))
    ]
    return list(
        rows[["performer", "kernel", "launch cfg."]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def get_tnl_cpu_config(df, problem):
    """The TNL "host" reference run (always plain CSR segments), if it exists."""
    rows = df[
        (df["problem"] == problem)
        & (df["solver"] == "TNL")
        & (df["performer"] == "host")
    ]
    configs = rows[["kernel", "launch cfg."]].drop_duplicates()
    if configs.empty:
        return None
    kernel, launch_cfg = configs.iloc[0]
    return "host", kernel, launch_cfg


def get_gunrock_configs(df, base):
    rows = df[
        (df["problem"] == base)
        & (df["solver"] == "Gunrock")
        & (df["performer"] == "cuda")
    ]
    return sorted(rows["launch cfg."].dropna().unique())


def get_boost_time(pivot, base):
    col = (base, "Boost", "sequential", "N/A", "")
    return pivot[col] if col in pivot.columns else None


def get_gunrock_best_time(pivot, base, gunrock_configs):
    """Per-graph minimum time across the Gunrock launch configurations."""
    cols = [
        (base, "Gunrock", "cuda", "N/A", cfg)
        for cfg in gunrock_configs
        if (base, "Gunrock", "cuda", "N/A", cfg) in pivot.columns
    ]
    if not cols:
        return None
    return pivot[cols].min(axis=1)


def get_tnl_kernel_best_times(df, pivot, problem):
    """Per-graph minimum time across launch configurations, for each (device,
    kernel) combination of TNL on the given problem."""
    rows = df[(df["problem"] == problem) & (df["solver"] == "TNL")]
    result = {}
    for device, kernel in (
        rows[["performer", "kernel"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        launch_cfgs = rows.loc[
            (rows["performer"] == device) & (rows["kernel"] == kernel), "launch cfg."
        ].unique()
        cols = [
            (problem, "TNL", device, kernel, launch_cfg)
            for launch_cfg in launch_cfgs
            if (problem, "TNL", device, kernel, launch_cfg) in pivot.columns
        ]
        if not cols:
            continue
        result[(device, kernel)] = pivot[cols].min(axis=1)
    return result


def get_tnl_best_gpu_time(config_times):
    """Per-graph minimum time across all GPU (cuda/hip) configurations, i.e. the
    best TNL can do on the GPU regardless of which configuration is used.
    "host" and "sequential" entries are excluded, since this is a GPU-only
    best. config_times is keyed by tuples whose first element is the device
    ("sequential"/"host"/"cuda"/"hip")."""
    gpu_series = [
        times for key, times in config_times.items() if key[0] in ("cuda", "hip")
    ]
    if not gpu_series:
        return None
    return pd.concat(gpu_series, axis=1).min(axis=1)


def get_tnl_csr_launch_configs(df, pivot, problem):
    """Per-graph time series for each (device, kernel, launch cfg.) of TNL on
    the given problem, restricted to the CSR kernel (Ellpack-family segment
    types are ignored)."""
    rows = df[
        (df["problem"] == problem)
        & (df["solver"] == "TNL")
        & (df["kernel"].str.startswith("CSR"))
    ]
    result = {}
    for device, kernel, launch_cfg in (
        rows[["performer", "kernel", "launch cfg."]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        col = (problem, "TNL", device, kernel, launch_cfg)
        if col in pivot.columns:
            result[(device, kernel, launch_cfg)] = pivot[col]
    return result


def build_table(df, pivot):
    problems = sorted(df["problem"].unique())
    graph_info = get_graph_info(df)

    data = {}
    data[("Graph name", "", "", "", "", "", "")] = pd.Series(
        graph_info.index, index=graph_info.index
    )
    data[("nodes", "", "", "", "", "", "")] = graph_info["nodes"]
    data[("edges", "", "", "", "", "", "")] = graph_info["edges"]
    data[("edges per node", "", "", "", "", "", "")] = graph_info["edges per node"]

    for problem in problems:
        base = base_problem(problem)
        boost_time = get_boost_time(pivot, base)
        gunrock_configs = get_gunrock_configs(df, base)
        gunrock_best_time = get_gunrock_best_time(pivot, base, gunrock_configs)
        cpu_config = get_tnl_cpu_config(df, problem)

        cpu_time = None
        if cpu_config is not None:
            cpu_device, cpu_kernel, cpu_launch_cfg = cpu_config
            cpu_col = (problem, "TNL", cpu_device, cpu_kernel, cpu_launch_cfg)
            if cpu_col in pivot.columns:
                cpu_time = pivot[cpu_col]
                key = (
                    problem,
                    "TNL",
                    cpu_device,
                    cpu_kernel,
                    cpu_launch_cfg,
                    "time",
                    "",
                )
                data.setdefault(key, cpu_time)
                if boost_time is not None:
                    key = (
                        problem,
                        "TNL",
                        cpu_device,
                        cpu_kernel,
                        cpu_launch_cfg,
                        "Speedup",
                        "Boost",
                    )
                    data.setdefault(key, boost_time / cpu_time)
                if gunrock_best_time is not None:
                    key = (
                        problem,
                        "TNL",
                        cpu_device,
                        cpu_kernel,
                        cpu_launch_cfg,
                        "Speedup",
                        "Gunrock best",
                    )
                    data.setdefault(key, gunrock_best_time / cpu_time)

        for device, kernel, launch_cfg in get_tnl_gpu_configs(df, problem):
            col = (problem, "TNL", device, kernel, launch_cfg)
            gpu_time = pivot[col]
            data[(problem, "TNL", device, kernel, launch_cfg, "time", "")] = gpu_time

            if cpu_time is not None:
                key = (problem, "TNL", device, kernel, launch_cfg, "Speedup", "TNL CPU")
                data[key] = cpu_time / gpu_time

            if boost_time is not None:
                key = (problem, "TNL", device, kernel, launch_cfg, "Speedup", "Boost")
                data[key] = boost_time / gpu_time

            if gunrock_best_time is not None:
                key = (
                    problem,
                    "TNL",
                    device,
                    kernel,
                    launch_cfg,
                    "Speedup",
                    "Gunrock best",
                )
                data[key] = gunrock_best_time / gpu_time

            for gunrock_cfg in gunrock_configs:
                gunrock_col = (base, "Gunrock", "cuda", "N/A", gunrock_cfg)
                if gunrock_col not in pivot.columns:
                    continue
                key = (
                    problem,
                    "TNL",
                    device,
                    kernel,
                    launch_cfg,
                    "Speedup",
                    f"Gunrock ({gunrock_cfg})",
                )
                data[key] = pivot[gunrock_col] / gpu_time

        if boost_time is not None:
            key = (base, "Boost", "sequential", "N/A", "", "time", "")
            data.setdefault(key, boost_time)

        for gunrock_cfg in gunrock_configs:
            gunrock_col = (base, "Gunrock", "cuda", "N/A", gunrock_cfg)
            key = (base, "Gunrock", "cuda", "N/A", gunrock_cfg, "time", "")
            data.setdefault(key, pivot[gunrock_col])

        if gunrock_best_time is not None:
            key = (base, "Gunrock", "cuda", "N/A", "best", "time", "")
            data.setdefault(key, gunrock_best_time)

    mic = MultiindexCreator(depth=COLUMN_DEPTH)
    mic.add_entries([list(key) for key in data])
    multicolumns, _ = mic.get_multiindex()

    table = pd.DataFrame(data)
    table = table.reindex(columns=multicolumns)
    table = table.reset_index(drop=True)
    return table


def plot_survival_curve(ax, ref_time, target_time, **step_kwargs):
    """Plot the percentage of graphs that reach a given speedup (ref_time /
    target_time) or better, as a function of that speedup. x[i] is a speedup
    value that was actually observed; since the underlying values are exact
    ("or better" is inclusive), the curve must hold its higher, pre-drop level
    up to and including x[i], hence where="pre" rather than the more usual
    "post"."""
    speedup = (ref_time / target_time).dropna().sort_values()
    if speedup.empty:
        return
    n = len(speedup)
    percentage = [100.0 * (n - i) / n for i in range(n)]
    ax.step(speedup.to_numpy(), percentage, where="pre", **step_kwargs)


def styled_curves(config_times, label_fn):
    """Build (label, series, color, linestyle, linewidth) tuples for each entry
    of config_times, in a fixed color/linestyle order (never reassigned)."""
    curves = []
    for i, key in enumerate(sorted(config_times)):
        curves.append(
            (
                label_fn(key),
                config_times[key],
                CVD_SAFE_COLORS[i % len(CVD_SAFE_COLORS)],
                LINESTYLES[(i // len(CVD_SAFE_COLORS)) % len(LINESTYLES)],
                2,
            )
        )
    return curves


def render_profile_plot(gunrock_best_time, curves, title, xlim_min, xlim_max, filename):
    """Draw one survival-curve figure (see plot_survival_curve) from a list of
    (label, series, color, linestyle, linewidth) curves and save it."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.axvline(1, color="dimgray", linestyle="--", linewidth=2, zorder=0)
    for label, times, color, linestyle, linewidth in curves:
        plot_survival_curve(
            ax,
            gunrock_best_time,
            times,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ax.set_xscale("log")
    ax.set_xlim(xlim_min, xlim_max)
    # Only label the "nice" 1/2/5 ticks per decade; with the minor ticks also
    # labeled the axis becomes too dense and the numbers overlap.
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Speedup of TNL vs best Gunrock kernel")
    ax.set_ylabel("Graphs reaching this speedup or better (%)")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize="small")

    fig.savefig(filename)
    plt.close(fig)
    print(f"Wrote {filename}")


def plot_speedup_profiles(df, pivot, output_dir, xlim_min, xlim_max):
    """For each problem, render two performance-profile plots: x is the speedup
    relative to the best Gunrock launch configuration (log scale), y is the
    percentage of graphs reaching that speedup or better.
      - "*-vs-gunrock-best.pdf": one curve per TNL kernel (best time across its
        launch configurations).
      - "*-vs-gunrock-best-CSR.pdf": one curve per launch configuration of the
        CSR kernel only (Ellpack-family kernels are ignored), i.e. it compares
        traversal/launch strategies rather than segment storage formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    for problem in sorted(df["problem"].unique()):
        base = base_problem(problem)
        gunrock_configs = get_gunrock_configs(df, base)
        gunrock_best_time = get_gunrock_best_time(pivot, base, gunrock_configs)
        if gunrock_best_time is None:
            continue

        kernel_times = get_tnl_kernel_best_times(df, pivot, problem)
        if kernel_times:
            curves = styled_curves(
                kernel_times, lambda k: f"{k[0]} {display_kernel(k[1])}"
            )
            best_gpu_time = get_tnl_best_gpu_time(kernel_times)
            if best_gpu_time is not None:
                curves.append(("best TNL (GPU)", best_gpu_time, "black", "-", 2.5))
            plot_name = f"{problem.replace(' ', '_')}-vs-gunrock-best.pdf"
            render_profile_plot(
                gunrock_best_time,
                curves,
                problem,
                xlim_min,
                xlim_max,
                os.path.join(output_dir, plot_name),
            )

        csr_times = get_tnl_csr_launch_configs(df, pivot, problem)
        if csr_times:
            curves = styled_curves(
                csr_times, lambda k: f"{k[0]} {display_kernel(k[1])} {k[2]}".strip()
            )
            best_gpu_time = get_tnl_best_gpu_time(csr_times)
            if best_gpu_time is not None:
                curves.append(("best TNL (GPU, CSR)", best_gpu_time, "black", "-", 2.5))
            plot_name = f"{problem.replace(' ', '_')}-vs-gunrock-best-CSR.pdf"
            render_profile_plot(
                gunrock_best_time,
                curves,
                f"{problem} (CSR)",
                xlim_min,
                xlim_max,
                os.path.join(output_dir, plot_name),
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Builds a speedup summary table (TNL GPU vs TNL CPU/Boost/Gunrock, and "
            "TNL CPU vs Boost) from tnl-benchmark-graphs log files."
        )
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Input log files (JSON lines)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="graphs-benchmark-speedup",
        help="Base name (without extension) for the output HTML/CSV files.",
    )
    parser.add_argument(
        "--plot-dir",
        default="Plots",
        help="Directory for the per-problem speedup-vs-Gunrock-best profile plots.",
    )
    parser.add_argument(
        "--xlim-min",
        type=float,
        default=0.1,
        help="Lower limit of the (logarithmic) speedup axis in the profile plots.",
    )
    parser.add_argument(
        "--xlim-max",
        type=float,
        default=4.0,
        help="Upper limit of the (logarithmic) speedup axis in the profile plots.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating the profile plots."
    )
    args = parser.parse_args()

    df = load_results(args.input)
    pivot = pivot_times(df)
    table = build_table(df, pivot)

    table.to_html(f"{args.output}.html")
    table.to_csv(f"{args.output}.csv")
    print(f"Wrote {args.output}.html and {args.output}.csv")

    if not args.no_plots:
        plot_speedup_profiles(df, pivot, args.plot_dir, args.xlim_min, args.xlim_max)


if __name__ == "__main__":
    main()
