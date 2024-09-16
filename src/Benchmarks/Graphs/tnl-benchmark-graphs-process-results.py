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
        if "speedup" in df.columns:
            df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")

    return df


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

input_df.to_html("graphs-benchmark.html")
