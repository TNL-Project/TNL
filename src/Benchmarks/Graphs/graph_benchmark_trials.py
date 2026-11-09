# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

"""Aggregation of the per-trial rows of the graph benchmarks.

Following the GAP Benchmark Suite, the graph benchmarks log one row per trial
(the "trial" and "source" columns): the single-source kernels (BFS, SSSP) run
from 64 source vertices and every run is timed separately. The rows of one
configuration are aggregated here into one row:

* time        - the mean time of the trials (the "Average Time" of gapbs),
* time_median - the median time of the trials,
* time_min, time_max, time_stddev - the spread of the trials,
* trials      - the number of trials,
* traversed edges - the total number of traversed edges,
* GTEPS       - the total number of traversed edges divided by the total time,
                in 10^9 edges per second (i.e. the edges-weighted rate).

Rows without the "trial" column (benchmarks without the GAP methodology) and
the whole-graph kernels (trial "all") are kept as they are.
"""

import pandas as pd

TRIAL_COLUMNS = ["trial", "source"]
MEASURED_COLUMNS = [
    "time",
    "time_median",
    "bandwidth",
    "cycles/op",
    "cycles",
    "cycles_median",
    "time_stddev",
    "time_stddev/time",
    "cycles_stddev",
    "cycles_stddev/cycles",
    "loops",
    "ops_per_loop",
    "traversed edges",
    "GTEPS",
]


def aggregate_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-trial rows of each configuration into one row."""
    if "trial" not in df.columns:
        return df
    df = df.copy()
    for column in ("time", "traversed edges"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    is_trial = df["trial"].notna() & (df["trial"].astype(str) != "all")
    kept = df[~is_trial]
    trials = df[is_trial]
    if trials.empty:
        return df

    keys = [c for c in trials.columns if c not in TRIAL_COLUMNS + MEASURED_COLUMNS]
    # groupby drops the rows with NaN keys, so the missing metadata are made explicit
    trials = trials.copy()
    trials[keys] = trials[keys].fillna("")
    grouped = trials.groupby(keys, sort=False)
    aggregated = grouped["time"].agg(
        time="mean",
        time_median="median",
        time_min="min",
        time_max="max",
        time_stddev="std",
        trials="size",
    )
    total_time = grouped["time"].sum()
    if "traversed edges" in trials.columns:
        edges = grouped["traversed edges"].sum(min_count=1)
        aggregated["traversed edges"] = edges
        aggregated["GTEPS"] = edges / total_time / 1e9
    aggregated["time_stddev/time"] = aggregated["time_stddev"] / aggregated["time"]
    aggregated["loops"] = grouped["loops"].first() if "loops" in trials else 1
    aggregated = aggregated.reset_index()
    if kept.empty:
        return aggregated
    return pd.concat([kept, aggregated], ignore_index=True)
