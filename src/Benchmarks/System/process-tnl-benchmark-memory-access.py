#!/usr/bin/python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import os
import argparse
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
from os.path import exists

font_size = "15"
threads = []
accesses = ["sequential", "random"]
element_sizes = ["1", "2", "4", "16", "64", "256"]


####
# Create multiindex for columns
def get_multiindex():
    level1 = ["size"]
    level2 = [
        "",
    ]
    level3 = [
        "",
    ]
    level4 = [
        "",
    ]
    level5 = [
        "",
    ]
    df_data = [[" "]]
    for threads_count in threads:
        for access in accesses:
            for rw in ["read", "write"]:
                orderings = ["blocks"]
                if access == "sequential" and float(threads_count) > 1:
                    orderings.append("interleaving")
                for ordering in orderings:
                    for element_size in element_sizes:
                        values = ["time", "bandwidth", "CPU cycles"]
                        for value in values:
                            level1.append(f"Threads {threads_count} {access}")
                            level2.append(rw)
                            level3.append(ordering)
                            level4.append(element_size)
                            level5.append(value)
                            df_data[0].append("")

    multiColumns = pd.MultiIndex.from_arrays([level1, level2, level3, level4, level5])
    return multiColumns, df_data


####
# Process dataframe
def processDf(df):
    multicolumns, df_data = get_multiindex()

    frames = []
    in_idx = 0
    out_idx = 0

    sizes = list(set(df["array size"]))
    sizes.sort()
    print(f"Sizes = {sizes}")

    for size in sizes:
        aux_df = df.loc[(df["array size"] == size)]
        new_df = pd.DataFrame(df_data, columns=multicolumns, index=[out_idx])
        out_idx += 1
        new_df.iloc[0][("size", "", "", "", "")] = size
        for index, row in aux_df.iterrows():
            threads_count = row["threads"]
            if threads_count not in threads:
                continue
            access_type = row["access type"]
            read_test = row["read test"]
            write_test = row["write test"]
            test_type = ""
            if read_test == "true" and write_test == "false":
                test_type = "read"
            if read_test == "false" and write_test == "true":
                test_type = "write"
            if test_type == "":
                raise RuntimeError("Wrong combination of read test and write test.")
            interleaving = row["interleaving"]
            ordering = "blocks"
            if interleaving == "true":
                ordering = "interleaving"
            element_size = row["element size"]
            time = row["time"]
            bandwidth = row["bandwidth"]
            cpu_cycles = row["cycles/op."]
            print(
                f"Threads {threads_count} \t {access_type} \t {test_type} \t {ordering}  \t {element_size} \t {time} \t {bandwidth} \t {cpu_cycles} \r",
                end="",
            )
            new_df.iloc[0][
                (
                    f"Threads {threads_count} {access_type}",
                    test_type,
                    ordering,
                    element_size,
                    "time",
                )
            ] = time
            new_df.iloc[0][
                (
                    f"Threads {threads_count} {access_type}",
                    test_type,
                    ordering,
                    element_size,
                    "bandwidth",
                )
            ] = bandwidth
            new_df.iloc[0][
                (
                    f"Threads {threads_count} {access_type}",
                    test_type,
                    ordering,
                    element_size,
                    "CPU cycles",
                )
            ] = cpu_cycles
        frames.append(new_df)
    result = pd.concat(frames)
    return result


####
# Extract data with memory bandwidth from a data frame
def get_bandwidth(df, threads_count, access, test_type, ordering, element_size):
    in_bandwidth = df[
        (
            f"Threads {threads_count} {access}",
            test_type,
            ordering,
            element_size,
            "bandwidth",
        )
    ].tolist()
    bandwidth = []
    for bw in in_bandwidth:
        try:
            bandwidth.append(float(bw))
        except ValueError:
            bandwidth.append(0)
            print(
                f"Warning wrong value of bandwidth: {bw} for threads count {threads_count}, access {access}, test type {test_type}, ordering {ordering}, element size {element_size} "
            )
    return bandwidth


###
# Extract data with CPU cycles from a data frame
def get_cpu_cycles(df, threads_count, access, test_type, ordering, element_size):
    in_cpu_cycles = df[
        (
            f"Threads {threads_count} {access}",
            test_type,
            ordering,
            element_size,
            "CPU cycles",
        )
    ].tolist()
    cpu_cycles = []
    for cycles in in_cpu_cycles:
        try:
            cpu_cycles.append(float(cycles))
        except ValueError:
            cpu_cycles.append(0)
            print(
                f"Warning wrong value of CPU cycles: {cycles} for threads count {threads_count}, access {access}, test type {test_type}, ordering {ordering}, element size {element_size} "
            )
    return cpu_cycles


###
# Helper function for writting of figures
def writeFigure(file_name, data_set, legend, legend_location, sizes, y_lim):
    fig, axs = plt.subplots(1, 1)
    for data in data_set:
        axs.plot(sizes, data, "-o", ms=6, lw=2)
    axs.legend(legend, loc=legend_location)
    axs.set_ylabel("Effective bandwidth in GB/sec")
    axs.set_xlabel("Array size in bytes")
    axs.set_xscale("log")
    axs.set_yscale("linear")
    if y_lim:
        axs.set_ylim(y_lim)
    axs.grid()
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "sans-serif", "font.size": font_size}
    )
    plt.savefig(file_name)
    plt.close(fig)


####
# Write figures for particular benchmark tests
def writeGeneralFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    for threads_count in threads:
        for access in accesses:
            for test_type in ["read", "write"]:
                orderings = ["blocks"]
                if access == "sequential" and float(threads_count) > 1:
                    orderings.append("interleaving")
                for ordering in orderings:
                    for element_size in element_sizes:
                        print(
                            f"Writing figures for benchmark: {access} threads={threads_count} {test_type} {ordering} element size = {element_size}:"
                        )
                        bandwidth = get_bandwidth(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                        cpu_cycles = get_cpu_cycles(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                        print(f"   BW: {bandwidth}")
                        print(f"   CPU cycles: {cpu_cycles}")
                        max_bandwidth = max(bandwidth)

                        file_name = f"{access}-{threads_count}-threads-{test_type}-{ordering}-element-size-{element_size}-bw.pdf"
                        data_set = [bandwidth]
                        legend = [f"{access} access {threads_count} threads"]
                        legend_location = "upper right"
                        writeFigure(
                            file_name,
                            data_set,
                            legend,
                            legend_location,
                            sizes,
                            [0, 1.2 * max_bandwidth],
                        )

                        file_name = f"{access}-{threads_count}-threads-{test_type}-{ordering}-element-size-{element_size}-cycles.pdf"
                        data_set = [cpu_cycles]
                        legend = [f"{access} access {threads_count} threads"]
                        legend_location = "upper left"
                        writeFigure(
                            file_name, data_set, legend, legend_location, sizes, []
                        )


####
# Write figures for comparison of sequential and random access
def writeSequentialRandomComparisonFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    for threads_count in threads:
        for test_type in ["read", "write"]:
            ordering = "blocks"
            for element_size in element_sizes:
                print(
                    f"Writing figure for comparison of sequential and random access: {test_type} element size = {element_size}:"
                )

                file_name = f"sequential-random-comparison-{threads_count}-threads-{test_type}-element-size-{element_size}-bw.pdf"
                data_set = []
                legend = []
                for access in accesses:
                    legend.append(access)
                    data_set.append(
                        get_bandwidth(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "lower left"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])

                file_name = f"sequential-random-comparison-{threads_count}-threads-{test_type}-element-size-{element_size}-cycles.pdf"
                data_set = []
                legend = []
                for access in accesses:
                    legend.append(access)
                    data_set.append(
                        get_cpu_cycles(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "upper left"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])


####
# Write figures for comparison with different threads count
def writeThreadsCountComparisonFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    for access in accesses:
        for test_type in ["read", "write"]:
            orderings = ["blocks"]
            if access == "sequential":
                orderings.append("interleaving")
            for ordering in orderings:
                for element_size in element_sizes:
                    print(
                        f"Writing figure for threads count comparison: {access} {test_type} {ordering} element size = {element_size}:"
                    )

                    file_name = f"threads-comparison-{access}-{test_type}-{ordering}-element-size-{element_size}-bw.pdf"
                    data_set = []
                    legend = []
                    for threads_count in threads:
                        if ordering == "interleaving" and threads_count == "1":
                            continue
                        legend.append(f"{threads_count} threads")
                        data_set.append(
                            get_bandwidth(
                                df,
                                threads_count,
                                access,
                                test_type,
                                ordering,
                                element_size,
                            )
                        )
                    legend_location = "upper right"
                    writeFigure(file_name, data_set, legend, legend_location, sizes, [])

                    file_name = f"threads-comparison-{access}-{test_type}-{ordering}-element-size-{element_size}-cycles.pdf"
                    data_set = []
                    legend = []
                    for threads_count in threads:
                        if ordering == "interleaving" and threads_count == "1":
                            continue
                        legend.append(f"{threads_count} threads")
                        data_set.append(
                            get_cpu_cycles(
                                df,
                                threads_count,
                                access,
                                test_type,
                                ordering,
                                element_size,
                            )
                        )
                    legend_location = "upper right"
                    writeFigure(file_name, data_set, legend, legend_location, sizes, [])


####
# Write figures for comparison of read and write access
def writeReadWriteComparisonFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    for threads_count in threads:
        for access in accesses:
            ordering = "blocks"
            for element_size in element_sizes:
                print(
                    f"Writing figure for comparison of read and write access: {access} {ordering} element size = {element_size}:"
                )

                file_name = f"read-write-comparison-{access}-{threads_count}-threads-{ordering}-element-size-{element_size}-bw.pdf"
                data_set = []
                legend = []
                for test_type in ["read", "write"]:
                    legend.append(test_type)
                    data_set.append(
                        get_bandwidth(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "lower left"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])

                file_name = f"read-write-comparison-{access}-{threads_count}-threads-{ordering}-element-size-{element_size}-cycles.pdf"
                data_set = []
                legend = []
                for test_type in ["read", "write"]:
                    legend.append(test_type)
                    data_set.append(
                        get_cpu_cycles(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "upper left"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])


####
# Write figures for comparison of blocks and interleaving for sequential access
def writeBlocksInterleavingComparisonFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    orderings = ["blocks", "interleaving"]
    for threads_count in threads:
        if threads_count == "1":
            continue
        for test_type in ["read", "write"]:
            for element_size in element_sizes:
                access = "sequential"
                print(
                    f"Writing figure for comparison of interleaved and blocked ordering: {test_type} element size = {element_size}:"
                )

                file_name = f"blocked-interleaved-comparison-{threads_count}-threads-{test_type}-element-size-{element_size}-bw.pdf"
                data_set = []
                legend = []
                for ordering in orderings:
                    legend.append(ordering)
                    data_set.append(
                        get_bandwidth(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "upper right"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])

                file_name = f"blocked-interleaved-comparison-{threads_count}-threads-{test_type}-element-size-{element_size}-cycles.pdf"
                data_set = []
                legend = []
                for ordering in orderings:
                    legend.append(ordering)
                    data_set.append(
                        get_cpu_cycles(
                            df, threads_count, access, test_type, ordering, element_size
                        )
                    )
                legend_location = "upper left"
                writeFigure(file_name, data_set, legend, legend_location, sizes, [])


####
# Write figures for comparison with different element sizes
def writeElementSizeComparisonFigures(df):
    sizes = df[("size", "", "", "", "")].tolist()
    for threads_count in threads:
        for access in accesses:
            for test_type in ["read", "write"]:
                orderings = ["blocks"]
                if access == "sequential" and int(threads_count) > 1:
                    orderings.append("interleaving")
                for ordering in orderings:
                    print(
                        f"Writing figure for element size comparison: {threads_count } threads {access} {test_type} {ordering}:"
                    )

                    file_name = f"element-size-comparison-{threads_count}-threads-{access}-{test_type}-{ordering}-bw.pdf"
                    data_set = []
                    legend = []
                    for element_size in element_sizes:
                        legend.append(f"el.size {element_size}")
                        data_set.append(
                            get_bandwidth(
                                df,
                                threads_count,
                                access,
                                test_type,
                                ordering,
                                element_size,
                            )
                        )
                    legend_location = "upper right"
                    writeFigure(file_name, data_set, legend, legend_location, sizes, [])

                    file_name = f"element-size-comparison-{threads_count}-threads-{access}-{test_type}-{ordering}-cycles.pdf"
                    data_set = []
                    legend = []
                    for element_size in element_sizes:
                        legend.append(f"el.size {element_size}")
                        data_set.append(
                            get_cpu_cycles(
                                df,
                                threads_count,
                                access,
                                test_type,
                                ordering,
                                element_size,
                            )
                        )
                    legend_location = "upper left"
                    writeFigure(file_name, data_set, legend, legend_location, sizes, [])


#####
# Parse input files
parser = argparse.ArgumentParser(
    description="Scritp for processing TNL benchmark memory access results."
)
parser.add_argument(
    "-i",
    "--input-file",
    dest="input_files",
    nargs="+",
    required=True,
    help="The input file to be processed",
)
args = parser.parse_args()

parsed_lines = []

for filename in args.input_files:
    if not exists(filename):
        print(f"Skipping non-existing input file {filename} ....")
    print(f"Parsing input file {filename} ....")
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            parsed_line = json.loads(line)
            parsed_lines.append(parsed_line)

df = pd.DataFrame(parsed_lines)
if not threads:
    for threads_count in df["threads"]:
        if threads_count not in threads:
            threads.append(threads_count)

if not element_sizes:
    for element_size in df["element size"]:
        if element_size not in element_sizes:
            element_sizes.append(element_size)

keys = ["array size", "time", "bandwidth", "cycles/op"]

for key in keys:
    if key in df.keys():
        df[key] = pd.to_numeric(df[key])

df.to_html("tnl-benchmark-memory-access-raw.html")
frame = processDf(df)
frame.to_html(f"tnl-benchmark-memory-access.html")
writeGeneralFigures(frame)
writeSequentialRandomComparisonFigures(frame)
writeThreadsCountComparisonFigures(frame)
writeReadWriteComparisonFigures(frame)
writeBlocksInterleavingComparisonFigures(frame)
writeElementSizeComparisonFigures(frame)
