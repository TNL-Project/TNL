#!/usr/bin/python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import os
import json
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from TNL.BenchmarkLogs import *
import TNL.MultiindexCreator as mic

matplotlib_fixed_colors = ["blue", "green", "red", "cyan", "magenta", "#663300"]


def extract_sorted(df, subset, ascending=False):
    """
    Extract column subset from the dataframe, drop rows with NaN in this column and sort the result by this column.

    :param df: pandas.DataFrame from which to extract the data
    :param subset: column names to extract and sort
    :param ascending: sort order, False for descending, True for ascending
    """
    if not subset in df.columns:
        raise Exception(f"Column {subset} not found in the dataframe")

    filtered_df = df.dropna(subset=[subset]).copy()
    filtered_df.sort_values(
        by=[subset],
        inplace=True,
        ascending=ascending,
    )
    return filtered_df[subset].copy()


def draw_graphs(
    graph_labels,
    graphs,
    xlabel,
    ylabel,
    filename,
    legend_loc="upper right",
    bar="none",
    yscale="linear",
    latex_labels={},
):
    """
    Draw several graphs into one figure
    graph_labels - list of labels of graphs to be drawn
    graphs - dictionary with graphs where key is the label of the graph and value is the data
    xlabel - label of x axis
    ylabel - label of y axis
    filename - name of the output file
    legend_loc - location of the legend
    bar - name of the bar to be drawn. Bar is drawn as a line with value 1.
        It serves for better visualization of speed-up (i.e. where it is larger or smaller than one).
        If bar is set to 'none', no bar is drawn.
    yscale - scale of y axis ('linear' or 'log')
    latex_labels - dictionary with labels in latex format where key is the label of the graph and value is the latex label
    """
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    latexNames = []
    size = 1
    color_idx = 0
    for label in graph_labels:
        if not label in graphs:
            raise RuntimeError(f"Graph {label} not found in graphs")
        t = np.arange(len(graphs[label]))
        if color_idx < len(matplotlib_fixed_colors):
            axs.plot(
                t,
                graphs[label],
                "-o",
                ms=1,
                lw=1,
                color=matplotlib_fixed_colors[color_idx],
            )
            color_idx = color_idx + 1
        else:
            axs.plot(t, graphs[label], "-o", ms=1, lw=1)
        size = len(graphs[label])
        if not latex_labels:
            latexNames.append(label)
        else:
            latexNames.append(latex_labels[label])
    if bar != "none":
        bar_data = np.full(size, 1)
        axs.plot(t, bar_data, "-", ms=1, lw=1.5)
        if bar != "":
            latexNames.append(bar)

    axs.legend(latexNames, loc=legend_loc)
    axs.yaxis.set_label_position("right")
    axs.yaxis.tick_right()
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_yscale(yscale)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            # "font.sans-serif": ["Helvetica"],
            "font.size": 22,
        }
    )
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def get_benchmark_dataframe(logFile):
    """
    Get pandas dataframe with benchmark results stored in the given log file.

    :param logFile: path to the log file
    :returns: pandas.DataFrame instance
    """
    print(f"Parsing input file {logFile}")
    with open(logFile, "r") as file:
        df = pd.read_json(file, orient="records", lines=True)
        # convert "N/A" in the speedup column to nan
        # if "speedup" in df.columns:
        #    df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")

    return df


def get_problems(input_df):
    """
    Get list of problems from the input dataframe
    """
    problems = list(
        set(input_df["problem"].values.tolist())
    )  # list of all problems in the benchmark results
    problems.sort()
    return problems


def get_solvers(input_df):
    """
    Get list of solvers from the input dataframe
    """
    solvers = list(
        set(input_df["solver"].values.tolist())
    )  # list of all solvers in the benchmark results
    solvers.sort()
    solvers.remove("TNL")
    solvers.append("TNL")  # Move TNL to the last position

    devices = {}
    for solver in solvers:
        devices[solver] = list(
            set(input_df.loc[input_df["solver"] == solver, "performer"].values.tolist())
        )
        order = {"sequential": 0, "host": 1, "cuda": 2, "hip": 3}
        devices[solver].sort(key=lambda d: order.get(d, len(order)))
        print(f"Solver {solver} has devices: {devices[solver]}")

    return solvers, devices


def get_kernels(input_df):
    """
    Get list of kernels from the input dataframe
    """
    kernels = list(set(input_df["kernel"].values.tolist()))
    kernels.sort()
    return kernels


def get_launch_configurations(input_df, kernels):
    """
    Get list of launch configurations from the input dataframe
    """
    launch_configs = {}
    in_idx = 0
    while in_idx < len(input_df.index):
        row = input_df.iloc[in_idx]
        problem = row["problem"]
        solver = row["solver"]
        device = row["performer"]
        kernel = row["kernel"]
        launch_cfg = row["launch cfg."]
        if (problem, solver, device, kernel) not in launch_configs:
            launch_configs[(problem, solver, device, kernel)] = []
        if launch_cfg not in launch_configs[(problem, solver, device, kernel)]:
            launch_configs[(problem, solver, device, kernel)].append(launch_cfg)
        in_idx += 1

    for problem in problems:
        for solver in ["TNL"]:
            for device in ["sequential", "host", "cuda", "hip"]:
                for kernel in kernels:
                    if (problem, solver, device, kernel) in launch_configs:
                        order = {
                            "1 TPS": 0,
                            "2 TPS": 1,
                            "4 TPS": 2,
                            "8 TPS": 3,
                            "16 TPS": 4,
                            "32 TPS": 5,
                            "64 TPS": 6,
                            "128 TPS": 7,
                            "256 TPS": 8,
                            "BlockMerged 1 TPS": 9,
                            "BlockMerged 2 TPS": 10,
                            "BlockMerged 4 TPS": 11,
                            "BlockMerged 8 TPS": 12,
                            "DynamicGrouping 1 TPS": 13,
                        }
                        launch_configs[(problem, solver, device, kernel)].sort(
                            key=lambda d: order.get(d, len(order))
                        )
    return launch_configs


def get_multiindex(problems, solvers, kernels, launch_configs):
    """
    Create index for the table.
    """
    mc = mic.MultiindexCreator(7)
    mc.add_entries([["Graph name"], ["nodes"], ["edges"], ["edges per node"]])

    for problem in problems:
        for solver in solvers:
            for device in ["sequential", "host", "cuda", "hip"]:
                for kernel in kernels:
                    if (problem, solver, device, kernel) not in launch_configs:
                        continue
                    for launch_cfg in launch_configs[(problem, solver, device, kernel)]:
                        mc.add_entry(
                            [problem, solver, device, kernel, launch_cfg, "time"]
                        )
                        if solver == "TNL":
                            mc.add_entry(
                                [
                                    problem,
                                    solver,
                                    device,
                                    kernel,
                                    launch_cfg,
                                    "Speedup",
                                    "Boost",
                                ]
                            )
                            if device == "cuda":
                                mc.add_entry(
                                    [
                                        problem,
                                        solver,
                                        device,
                                        kernel,
                                        launch_cfg,
                                        "Speedup",
                                        "Gunrock",
                                    ]
                                )

    return mc.get_multiindex()


def convert_data_frame(input_df, multicolumns, df_data, begin_idx=0, end_idx=-1):
    """
    Convert input table to a better structured one using multiindex
    """
    frames = []
    in_idx = 0
    out_idx = 0
    # max_out_idx = max_rows
    if end_idx == -1:
        end_idx = len(input_df.index)
    while in_idx < len(input_df.index) and out_idx < end_idx:
        graphName = input_df.iloc[in_idx]["graph name"]
        df_graph = input_df.loc[input_df["graph name"] == graphName]
        if out_idx >= begin_idx:
            print(f"{out_idx} : {in_idx} / {len(input_df.index)} : {graphName}")
        else:
            print(f"{out_idx} : {in_idx} / {len(input_df.index)} : {graphName} - SKIP")
        aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[out_idx])
        for index, row in df_graph.iterrows():
            aux_df.loc[out_idx, "Graph name"] = row["graph name"]
            aux_df.loc[out_idx, "nodes"] = row["nodes"]
            aux_df.loc[out_idx, "edges"] = row["edges"]
            aux_df.loc[out_idx, "edges per node"] = float(row["edges"]) / float(
                row["nodes"]
            )

            current_problem = row["problem"]
            current_solver = row["solver"]
            current_device = row["performer"]
            current_kernel = row["kernel"]
            current_launch_cfg = row["launch cfg."]
            time = pd.to_numeric(row["time"], errors="coerce")
            aux_df.loc[
                out_idx,
                (
                    current_problem,
                    current_solver,
                    current_device,
                    current_kernel,
                    current_launch_cfg,
                    "time",
                    "",
                ),
            ] = time
        if out_idx >= begin_idx:
            frames.append(aux_df)
        out_idx = out_idx + 1
        in_idx = in_idx + len(df_graph.index)
        # print(aux_df)
        print(f"out_idx: {out_idx} in_idx: {in_idx}")
    result = pd.concat(frames)
    result.replace("", float("nan"), inplace=True)
    return result


def divide_columns(df, in_colA, in_colB, out_col):
    """
    Compute out_col = in_colA / in_colB
    """
    if not in_colA in df.columns:
        raise Exception(f"Column {in_colA} not found in the dataframe")
    if not in_colB in df.columns:
        raise Exception(f"Column {in_colB} not found in the dataframe")
    if not out_col in df.columns:
        raise Exception(f"Column {out_col} not found in the dataframe")
    in_colA_list = df[in_colA]
    in_colB_list = df[in_colB]
    out_col_list = []

    for A, B in zip(in_colA_list, in_colB_list):
        div = 0
        try:
            div = A / B
        except:
            div = float("nan")
        out_col_list.append(div)
    df[out_col] = out_col_list


def compute_speedup(df, problems, solvers, kernels, launch_configurations):
    """
    Compute speedup column
    """
    for problem in problems:
        for device in ["sequential", "host", "cuda", "hip"]:
            for kernel in kernels:
                if (problem, "TNL", device, kernel) not in launch_configurations:
                    continue
                for launch_cfg in launch_configurations[
                    (problem, "TNL", device, kernel)
                ]:
                    reference_problem = problem.removeprefix("Semiring ")
                    if "Boost" in solvers:
                        divide_columns(
                            df,
                            (
                                reference_problem,
                                "Boost",
                                "sequential",
                                "N/A",
                                "",
                                "time",
                                "",
                            ),
                            (problem, "TNL", device, kernel, launch_cfg, "time", ""),
                            (
                                problem,
                                "TNL",
                                device,
                                kernel,
                                launch_cfg,
                                "Speedup",
                                "Boost",
                            ),
                        )

                    if "Gunrock" in solvers:
                        divide_columns(
                            df,
                            (
                                reference_problem,
                                "Gunrock",
                                "cuda",
                                "N/A",
                                "",
                                "time",
                                "",
                            ),
                            (problem, "TNL", device, kernel, launch_cfg, "time", ""),
                            (
                                problem,
                                "TNL",
                                device,
                                kernel,
                                launch_cfg,
                                "Speedup",
                                "Gunrock",
                            ),
                        )


def draw_speedup_graphs(in_df, kernels, launch_configurations):
    """
    Draw speedup graphs
    """
    for metric in ["Time", "Speedup"]:
        if not os.path.exists(metric):
            os.mkdir(metric)
        for problem in problems:
            if not os.path.exists(f"{metric}/{problem}"):
                os.mkdir(f"{metric}/{problem}")
                for device in ["sequential", "host", "cuda", "hip"]:
                    if not os.path.exists(f"{metric}/{problem}/{device}"):
                        os.mkdir(f"{metric}/{problem}/{device}")
    profiles = {}
    color_idx = 0
    print(kernels)
    df = in_df
    for problem in problems:
        for device in ["sequential", "host", "cuda", "hip"]:
            for kernel in kernels:
                if (problem, "TNL", device, kernel) not in launch_configurations:
                    continue
                for launch_cfg in launch_configurations[
                    (problem, "TNL", device, kernel)
                ]:
                    label = f"{kernel} {launch_cfg}"

                    profiles[label] = extract_sorted(
                        df,
                        (problem, "TNL", device, kernel, launch_cfg, "time", ""),
                        ascending=False,
                    )
                    print(
                        f"Writing time profile of {kernel} on {device} with '{launch_cfg}' launch config "
                    )
                    draw_graphs(
                        [label],
                        profiles,
                        xlabel=f"Graph number",
                        ylabel="Time",
                        filename=f"Time/{problem}/{device}/{kernel}-{launch_cfg}.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="linear",
                    )

                    for reference_solver in ["Boost", "Gunrock"]:
                        if reference_solver == "Gunrock" and device != "cuda":
                            continue
                        profiles[label] = extract_sorted(
                            df,
                            (
                                problem,
                                "TNL",
                                device,
                                kernel,
                                launch_cfg,
                                "Speedup",
                                reference_solver,
                            ),
                            ascending=False,
                        )
                        print(
                            f"Writing speedup profile of {kernel} on {device} with '{launch_cfg}' launch config "
                        )
                        draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Graph number",
                            ylabel="Speedup",
                            filename=f"Speedup/{problem}/{device}/{kernel}-{launch_cfg}-vs-{reference_solver}.pdf",
                            legend_loc="upper right",
                            bar="none",
                            yscale="linear",
                        )
                        draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Graph number",
                            ylabel="Speedup",
                            filename=f"Speedup/{problem}/{device}/{kernel}-{launch_cfg}-vs-{reference_solver}-log.pdf",
                            legend_loc="upper right",
                            bar="none",
                            yscale="log",
                        )
                        copy_df = df.copy()
                        for k in kernels:
                            if k != kernel:
                                mask = copy_df.columns.get_level_values(4) == "ref"
                                copy_df = copy_df.loc[:, ~mask]
                        copy_df.to_html(
                            f"Speedup/{problem}/{device}/{kernel}-{launch_cfg}.html"
                        )


###
# Main part of the script
parser = argparse.ArgumentParser(
    description="Script for parsing log files from tnl-benchmark-graphs."
)
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    help="Input files",
    default=["graphs-benchmark.log"],
)
parser.add_argument(
    "-v", "--verbose", help="Zobrazit více informací", action="store_true"
)

args = parser.parse_args()

print("Parsing input files....")
input_df = pd.DataFrame()
for file in args.input:
    df = get_benchmark_dataframe(file)
    input_df = pd.concat([input_df, df])

problems = get_problems(input_df)
solvers, devices = get_solvers(input_df)
kernels = get_kernels(input_df)
launch_configurations = get_launch_configurations(input_df, kernels)
multicolumns, df_data = get_multiindex(
    problems, solvers, kernels, launch_configurations
)
input_df.to_html("graphs-benchmark-input.html")
df = convert_data_frame(input_df, multicolumns, df_data={})
df.to_html("graphs-benchmark.html")

compute_speedup(df, problems, solvers, kernels, launch_configurations)
df.to_html("graphs-benchmark.html")
draw_speedup_graphs(df, kernels, launch_configurations)
