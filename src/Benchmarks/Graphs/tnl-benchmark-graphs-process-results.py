#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from TNL.BenchmarkLogs import *
import MultiindexCreator as mic

problems = ["BFS dir", "BFS undir", "SSSP dir", "SSSP undir", "MST undir"]
all_formats = [
    "Ellpack",
    "SlicedEllpack",
    "BiEllpack",
    "CSRAdaptive",
    "CSRScalar",
    "CSRVector",
    "CSRLight",
]


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


def get_multiindex():
    """
    Create index for the table.
    """
    mc = mic.MultiindexCreator(6)
    mc.add_entries([["Graph name"], ["nodes"], ["edges"], ["edges per node"]])

    for problem in problems:
        for solver in ["TNL", "Boost", "Gunrock"]:
            devices = {
                "TNL": ["sequential", "host", "cuda"],
                "Boost": ["sequential"],
                "Gunrock": ["cuda"],
            }
            for device in devices[solver]:
                formats = ["N/A"]
                if solver == "TNL":
                    if device == "sequential" or device == "host":
                        formats = [
                            "CSRScalar",
                        ]
                    else:
                        formats = all_formats
                for format in formats:
                    mc.add_entry([problem, solver, device, format, "time"])
                    if solver == "TNL":
                        if device == "cuda":
                            mc.add_entry(
                                [problem, solver, device, format, "speedup", "boost"]
                            )
                            mc.add_entry(
                                [problem, solver, device, format, "speedup", "gunrock"]
                            )
                        if device == "host" or device == "sequential":
                            mc.add_entry(
                                [problem, solver, device, format, "speedup", "boost"]
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
            current_format = row["format"]
            time = pd.to_numeric(row["time"], errors="coerce")
            aux_df.loc[
                out_idx,
                (
                    current_problem,
                    current_solver,
                    current_device,
                    current_format,
                    "time",
                    "",
                ),
            ] = time
        if out_idx >= begin_idx:
            frames.append(aux_df)
        out_idx = out_idx + 1
        in_idx = in_idx + len(df_graph.index)
        print(f"out_idx: {out_idx} in_idx: {in_idx}")
    result = pd.concat(frames)
    result.replace("", float("nan"), inplace=True)
    return result


def divide_columns(df, in_colA, in_colB, out_col):
    """
    Compute out_col = in_colA / in_colB
    """
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


def compute_speedup(df):
    """
    Compute speedup column
    """
    for problem in problems:
        divide_columns(
            df,
            (problem, "Boost", "sequential", "N/A", "time", ""),
            (problem, "TNL", "sequential", "CSRScalar", "time", ""),
            (problem, "TNL", "sequential", "CSRScalar", "speedup", "boost"),
        )

        divide_columns(
            df,
            (problem, "Boost", "sequential", "N/A", "time", ""),
            (problem, "TNL", "host", "CSRScalar", "time", ""),
            (problem, "TNL", "host", "CSRScalar", "speedup", "boost"),
        )

        for format in all_formats:
            divide_columns(
                df,
                (problem, "Gunrock", "cuda", "N/A", "time", ""),
                (problem, "TNL", "cuda", format, "time", ""),
                (problem, "TNL", "cuda", format, "speedup", "gunrock"),
            )
            divide_columns(
                df,
                (problem, "Boost", "sequential", "N/A", "time", ""),
                (problem, "TNL", "cuda", format, "time", ""),
                (problem, "TNL", "cuda", format, "speedup", "boost"),
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

multicolumns, df_data = get_multiindex()
df = convert_data_frame(input_df, multicolumns, df_data={})
compute_speedup(df)
df.to_html("graphs-benchmark.html")
