# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import pandas as pd
import json

# Read the log file line by line
with open("tnl-benchmark-ndarray-reduction.log", "r") as file:
    lines = file.readlines()

# Initialize empty lists to store extracted values
axis_values = []
permutation_values = []
sizes = []
m_values = []
n_values = []
o_values = []
p_values = []
q_values = []
time_values = []
bandwidth_values = []

# Parse each line as JSON and extract the required values
for line in lines:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue  # Skip lines that are not valid JSON

    axis_values.append(data.get("axis", ""))
    permutation_values.append(data.get("permutation", ""))
    sizes.append(data.get("size", ""))
    m_values.append(data.get("m", ""))
    n_values.append(data.get("n", ""))
    o_values.append(data.get("o", ""))
    p_values.append(data.get("p", ""))
    q_values.append(data.get("q", ""))
    time_values.append(data.get("time", ""))
    bandwidth_values.append(data.get("bandwidth", ""))

# Create a DataFrame using pandas
df = pd.DataFrame(
    {
        "Axis": axis_values,
        "Permutation": permutation_values,
        "Size": sizes,
        "M": m_values,
        "N": n_values,
        "O": o_values,
        "P": p_values,
        "Q": q_values,
        "Time": time_values,
        "Bandwidth": bandwidth_values,
    }
)

# Export the DataFrame to a text file with tab-separated values
df.to_csv("output.txt", index=False, sep="\t")
