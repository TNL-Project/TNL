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
import Graphs

functions = [
    "forElements",
    "forElements with indexes stride 2",
    "forElements with indexes stride 4",
    "forElements with indexes stride 8",
    "forElementsIf stride 2",
    "forElementsIf stride 4",
    "forElementsIf stride 8",
    "forElementsIfSparse stride 2",
    "forElementsIfSparse stride 4",
    "forElementsIfSparse stride 8",
    "reduceSegments",
    "reduceSegmentsWithIndexes stride 2",
    "reduceSegmentsWithIndexes stride 4",
    "reduceSegmentsWithIndexes stride 8",
    "reduceSegmentsIf stride 2",
    "reduceSegmentsIf stride 4",
    "reduceSegmentsIf stride 8",
]
all_segments = [
    "CSR",
    "Ellpack",
    "SlicedEllpack",
    "BiEllpack",
    "ChunkedEllpack",
]

threads_mappings = [
    "Single",
    "Warp",
    "Block Merge 1",
    "Block Merge 2",
    "Block Merge 4",
    "Block Merge 8",
]

threads_mappings_translation = {
    "": "",
    "Single": "Single",
    "N/A": "N/A",
    "1 TPS": "Single",
    "Warp per segment": "Warp",
    "BlockMerged 1 TPS": "Block Merge 1",
    "BlockMerged 2 TPS": "Block Merge 2",
    "BlockMerged 4 TPS": "Block Merge 4",
    "BlockMerged 8 TPS": "Block Merge 8",
}

tests = {
    "constant",
    "linear",
    "quadratic",
}


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
    mc = mic.MultiindexCreator(5)
    mc.add_entries(
        [
            ["segments setup"],
            ["segments count"],
            ["max segment size"],
            ["elements count"],
        ]
    )

    for function in functions:
        for device in ["sequential", "host", "cuda"]:
            segments_list = ["N/A"]
            if device == "sequential" or device == "host":
                segments_list = [
                    "CSR",
                ]
            else:
                segments_list = all_segments
            for segments in segments_list:
                mappings = ["Single"]
                if device == "cuda":
                    if segments in ["CSR", "SlicedEllpack"]:
                        mappings = threads_mappings
                for mapping in mappings:
                    mc.add_entry([function, device, segments, mapping, "time"])
                    mc.add_entry([function, device, segments, mapping, "bandwidth"])
                    if device == "cuda":
                        mc.add_entry([function, device, segments, mapping, "speedup"])

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
        segments_setup = input_df.iloc[in_idx]["segments setup"]
        segments_count = input_df.iloc[in_idx]["segments count"]
        max_segment_size = input_df.iloc[in_idx]["max segment size"]
        # df_graph = input_df.loc[
        #    input_df["segments setup"]
        #    == segments_setup & input_df["segments count"]
        #    == segments_count & input_df["max segment size"]
        #    == max_segment_size
        # ]
        df_graph = input_df[
            (input_df["segments setup"] == segments_setup)
            & (input_df["segments count"] == segments_count)
            & (input_df["max segment size"] == max_segment_size)
        ]
        if out_idx >= begin_idx:
            print(
                f"{out_idx} : {in_idx} / {len(input_df.index)} : {segments_setup} {segments_count} {max_segment_size}"
            )
        else:
            print(
                f"{out_idx} : {in_idx} / {len(input_df.index)} : {segments_setup} {segments_count} {max_segment_size} - SKIP"
            )
        aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[out_idx])
        for index, row in df_graph.iterrows():
            aux_df.loc[out_idx, "segments setup"] = row["segments setup"]
            aux_df.loc[out_idx, "segments count"] = row["segments count"]
            aux_df.loc[out_idx, "max segment size"] = row["max segment size"]
            aux_df.loc[out_idx, "elements count"] = row["elements count"]

            current_segments_type = row["segments type"]
            current_function = row["function"]
            current_device = row["performer"]
            current_mapping = row["threads mapping"]
            time = pd.to_numeric(row["time"], errors="coerce")
            aux_df.loc[
                out_idx,
                (
                    current_function,
                    current_device,
                    current_segments_type,
                    threads_mappings_translation[current_mapping],
                    "time",
                ),
            ] = time
            bandwidth = pd.to_numeric(row["bandwidth"], errors="coerce") / (1024**3)
            aux_df.loc[
                out_idx,
                (
                    current_function,
                    current_device,
                    current_segments_type,
                    threads_mappings_translation[current_mapping],
                    "bandwidth",
                ),
            ] = bandwidth
        if out_idx >= begin_idx:
            frames.append(aux_df)
        out_idx = out_idx + 1
        in_idx = in_idx + len(df_graph.index)
        # print(df_graph)
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
    for function in functions:
        for segments_type in all_segments:
            for threads_mapping in threads_mappings:
                try:
                    divide_columns(
                        df,
                        (function, "sequential", "CSR", "Single", "time"),
                        (function, "cuda", segments_type, threads_mapping, "time"),
                        (function, "cuda", segments_type, threads_mapping, "speedup"),
                    )
                except:
                    print(
                        f"Cannot compute speedup for {function} {segments_type} {threads_mapping}"
                    )


def write_results(df):
    """
    Write results to a file
    """
    df.to_html("tnl-benchmark-segments.html")
    bw_df = df.xs("bandwidth", level=4, axis=1)
    bw_df = bw_df.dropna(axis=1, how="all")
    min_bw = bw_df.min().min()
    max_bw = bw_df.max().max()
    print(f"Min bandwidth: {min_bw} GB/s")
    print(f"Max bandwidth: {max_bw} GB/s")
    for test in tests:
        print("Processing test: ", test)
        test_df = df[df["segments setup"] == test]
        test_df.to_html(f"tnl-benchmark-segments-{test}.html")
        try:
            os.mkdir(f"{test}")
        except:
            pass
        for function in functions:
            print("Processing function: ", function)
            function_df = test_df[
                ["segments count", "max segment size", "elements count", function]
            ]
            function_df.to_html(f"{test}/{function}.html")

            for segments in all_segments:
                graphs = {}
                labels = []
                for mapping in threads_mappings:
                    label = f"{segments} {mapping}"
                    print(
                        f"Processing graph for {function}, cuda, {segments}, {mapping}"
                    )
                    try:
                        graphs[label] = function_df[
                            (function, "cuda", segments, mapping, "bandwidth")
                        ].copy()
                        labels.append(label)
                    except:
                        print(
                            f"Cannot process graph for {function}, cuda, {segments}, {mapping}"
                        )
                Graphs.draw_graphs(
                    labels,
                    graphs,
                    "Function",
                    "Bandwidth [GB/s]",
                    f"{test}/{function}_{segments}_bandwidth.pdf",
                    legend_loc="lower right",
                    yscale="log",
                    y_min=min_bw,
                    y_max=max_bw,
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
    default=["tnl-benchmark-segments.log"],
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
input_df.to_html("tnl-benchmark-segments-raw.html")
df = convert_data_frame(input_df, multicolumns, df_data={})
compute_speedup(df)

write_results(df)
