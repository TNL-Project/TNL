#!/usr/bin/python3
import pandas as pd
import json
import os


def read_log_files(file_paths):
    all_log_entries = []
    for file_path in file_paths:
        if os.path.exists(file_path):  # Check if the file exists
            with open(file_path, 'r') as file:
                log_entries = []
                for line in file:
                    entry = json.loads(line)
                    # Stop reading further if 'matrix size' is encountered
                    if 'matrix size' in entry:
                        log_entries.append(entry)
                all_log_entries.extend(log_entries)  # Combine entries from all files
    return all_log_entries

def round_scientific(notation, precision=2):
    if isinstance(notation, str):
        try:
            number = float(notation)
            return f"{number:.{precision}e}"
        except ValueError:
            return notation  # Return original string if it's not a number
    else:
        return f"{notation:.{precision}e}"


def process_data(log_data):
    data = {}
    for entry in log_data:
        matrix_size = entry.get("matrix size")
        algorithm = entry.get("algorithm")
        time = entry.get("time", 0)

        # Initialize the algorithm data structure if necessary
        if matrix_size not in data:
            data[matrix_size] = {}
        if algorithm not in data[matrix_size]:
            data[matrix_size][algorithm] = {
                'time': 0,
                'Diff.L2 1': 'N/A',
                'Diff.Max 1': 'N/A',
            }

        # Update time and norms
        data[matrix_size][algorithm]['time'] = float(time)
        if "Diff.L2 1" in entry:
            data[matrix_size][algorithm]['Diff.L2 1'] = round_scientific(entry["Diff.L2 1"])
        if "Diff.Max 1" in entry:
            data[matrix_size][algorithm]['Diff.Max 1'] = round_scientific(entry["Diff.Max 1"])

    return data

def format_time(time):
    return f"{time:.5e}"

def calculate_speedup(base_time, compare_time):
    if base_time is not None and compare_time is not None and compare_time != 0:
        return base_time / compare_time
    else:
        return None

def create_html_table(data):
    style = '''
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #ddd;
            text-align: left;
            padding: 6px;
            min-width: 125px;
        }
        th {
            background-color: #f2f2f2;
            color: black;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
    '''
    title = "<h2>Dense Matrix Transposition</h2>"
    html = style + title
    html += "<table>\n"

    # Define primary and secondary algorithms
    primary_algorithms = ['MAGMA']
    secondary_algorithms = ['Kernel 2.1', 'Kernel 2.2', 'Kernel 2.3']
    all_algorithms = primary_algorithms + secondary_algorithms

    # Header row 1: Algorithm names
    html += "<tr><th rowspan='2'>Matrix Size</th>"
    for algo in all_algorithms:
        colspan = '4' if algo in secondary_algorithms else '1'  # Adjust colspan for secondary algorithms
        html += f"<th colspan='{colspan}'>{algo}</th>"
    html += "</tr>\n"

    # Header row 2: Time and Speedup columns
    html += "<tr>"  # Start the second header row
    for algo in primary_algorithms:
        html += "<th>Time</th>"  # Time columns for primary algorithms
    for algo in secondary_algorithms:
        html += "<th>Time</th>"  # Time column for secondary algorithms
        for primary_algo in primary_algorithms:
            html += f"<th> vs {primary_algo}</th>"  # Speedup columns for secondary algorithms
        html += "<th>l2Norm</th>"
        html += "<th>MaxNorm</th>"
    html += "</tr>\n"

    # Data rows
    for sizes, algos in data.items():
        matrix_size = sizes
        html += f"<tr><td>{matrix_size}</td>"

        for algo in all_algorithms:
            algo_data = algos.get(algo, {'time': 'N/A', 'Diff.L2 1': 'N/A', 'Diff.Max 1':'N/A' })
            # Time for each algorithm
            formatted_time = format_time(algo_data['time']) if algo_data['time'] != 'N/A' else 'N/A'
            html += f"<td>{formatted_time}</td>"

            if algo in secondary_algorithms:
                # Speedup calculations for secondary algorithms
                for primary_algo in primary_algorithms:
                    primary_time = algos.get(primary_algo, {}).get('time', None)
                    speedup = calculate_speedup(primary_time, algo_data['time'])
                    formatted_speedup = f"{speedup:.2f}x" if speedup else 'N/A'
                    html += f"<td>{formatted_speedup}</td>"

                # Adding 'Diff.L2 1' data in the additional cell
                diff_l2_1 = algo_data['Diff.L2 1']
                html += f"<td>Magma:{diff_l2_1}</td>"

                diff_Max_1 = algo_data['Diff.Max 1']
                html += f"<td>Magma:{diff_Max_1}</td>"

        html += "</tr>\n"  # End of the data row

    html += "</table>"  # End of the table
    return html

log_file_paths = ['tnl-benchmark-dense-matrices.log', 'tnl-benchmark-dense-matrices-cpu.log']
log_data = read_log_files(log_file_paths)
processed_data = process_data(log_data)
html_table = create_html_table(processed_data)

# Function to determine the next available file name
def get_next_available_filename(base_path, extension):
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_path}.{extension}"
        else:
            file_name = f"{base_path}{counter}.{extension}"
        if not os.path.exists(file_name):
            return file_name
        counter += 1

# Save the HTML table to a file
output_base_path = 'dense_matrix_transposition'
output_extension = 'html'
output_file_path = get_next_available_filename(output_base_path, output_extension)

with open(output_file_path, 'w') as file:
    file.write(html_table)

print(f"HTML table saved as {output_file_path}")
