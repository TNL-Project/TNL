#!/usr/bin/env python3
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import json
from os.path import exists

import pandas as pd

dimensions = ["1", "2", "3"]
devices = ["sequential", "host", "cuda"]
precisions = ["float", "double"]
schemes = ["fdm", "fvm"]
tests = ["parallel-for", "nd-array", "grid"]


####
# Create multiindex for columns
def get_multiindex():
    level1 = ["Dimension", "x size", "y size", "z size"]
    level2 = ["", "", "", ""]
    level3 = ["", "", "", ""]
    df_data = [["", "", "", ""]]
    for test in tests:
        for device in devices:
            values = ["time"]
            if test != "parallel-for":
                values.append("parallel-for speed-up")
            if device == "cuda":
                values.append("CPU speed-up")
            for value in values:
                level1.append(test)
                level2.append(device)
                level3.append(value)
                df_data[0].append("")

    multiColumns = pd.MultiIndex.from_arrays([level1, level2, level3])
    return multiColumns, df_data


####
# Process dataframe for given precision - float or double
def processDf(df, scheme, precision, dimension, tests):
    multicolumns, df_data = get_multiindex()

    frames = []
    out_idx = 0

    x_sizes = list(set(df["x size"]))
    x_sizes.sort()
    y_sizes = list(set(df["y size"]))
    y_sizes.sort()
    z_sizes = list(set(df["z size"]))
    z_sizes.sort()

    devices_found = []

    for x_size in x_sizes:
        for y_size in y_sizes:
            for z_size in z_sizes:
                aux_df = df.loc[
                    (df["x size"] == x_size)
                    & (df["y size"] == y_size)
                    & (df["z size"] == z_size)
                ]
                new_df = pd.DataFrame(df_data, columns=multicolumns, index=[out_idx])
                out_idx += 1
                new_df.iloc[0][("dimension", "", "")] = dimension
                new_df.iloc[0][("x size", "", "")] = x_size
                new_df.iloc[0][("y size", "", "")] = y_size
                new_df.iloc[0][("z size", "", "")] = z_size
                for index, row in aux_df.iterrows():
                    test = row["implementation"]
                    # print( test )
                    time = row["time"]
                    new_df.iloc[0][(test, row["device"], "time")] = float(time)
                    devices_found.append(row["device"])
                # print( new_df )
                frames.append(new_df)
    try:
        result = pd.concat(frames)
    except ValueError:
        print(f"No data for processing for {dimension}D in {precision} precision.")
        return

    idx = 0
    have_cuda = devices_found.count("cuda") > 0
    for index, row in result.iterrows():
        for test in tests:
            if have_cuda:
                try:
                    result.iloc[idx][(test, "cuda", "CPU speed-up")] = float(
                        row[(test, "host", "time")]
                    ) / float(row[(test, "cuda", "time")])
                except ValueError as v:
                    print(f"{v}: test = {test}.")

            if test != "parallel-for":
                for device in devices:
                    if device == "cuda" and not have_cuda:
                        continue
                    try:
                        result.iloc[idx][(test, device, "parallel-for speed-up")] = (
                            float(row[("parallel-for", device, "time")])
                            / float(row[(test, device, "time")])
                        )
                    except ValueError:
                        base_time = row[("parallel-for", device, "time")]
                        test_time = row[(test, device, "time")]
                        print(
                            f"Cannot compute speed-up for {test} on {device} device: "
                            f"{base_time} , {test_time}"
                        )
        idx += 1

    result.to_html(
        f"tnl-benchmark-heat-equation-{scheme}-{precision}-{dimension}d.html"
    )


#####
# Parse input files

parsed_lines = []
for dimension in dimensions:
    for device in devices:
        for precision in precisions:
            for scheme in schemes:
                for test in tests:
                    filename = (
                        "tnl-benchmark-heat-equation-"
                        f"{test}-{device}-{precision}-{scheme}-{dimension}d.json"
                    )
                    if not exists(filename):
                        print(f"Skipping non-existing input file {filename} ....")
                        continue
                    print(f"Parsing input file {filename} ....")
                    with open(filename) as f:
                        lines = f.readlines()
                        for line in lines:
                            parsed_line = json.loads(line)
                            parsed_lines.append(parsed_line)

df = pd.DataFrame(parsed_lines)

keys = ["x size", "y size", "z size", "time", "bandwidth"]

for key in keys:
    if key in df.keys():
        df[key] = pd.to_numeric(df[key])
for dimension in dimensions:
    for precision in precisions:
        for scheme in schemes:
            aux_df = df.loc[
                (df["scheme"] == scheme)
                & (df["precision"] == precision)
                & (df["dimension"] == dimension)
            ]
            if scheme == "fvm":
                processDf(
                    aux_df, scheme, precision, dimension, ["parallel-for", "grid"]
                )
            else:
                processDf(
                    aux_df,
                    scheme,
                    precision,
                    dimension,
                    ["parallel-for", "nd-array", "grid"],
                )
