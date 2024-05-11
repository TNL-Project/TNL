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
                        break
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
        matrix_sizes = (entry["matrix1 size"], entry["matrix2 size"])
        algorithm = entry["algorithm"]
        time = float(entry["time"])

        if matrix_sizes not in data:
            data[matrix_sizes] = {}
        if algorithm not in data[matrix_sizes]:
            data[matrix_sizes][algorithm] = {'time': time}
        #l2Norm
        data[matrix_sizes][algorithm]['Diff.L2 1'] = round_scientific(entry.get("Diff.L2 1", 'N/A'))
        data[matrix_sizes][algorithm]['Diff.L2 2']= round_scientific(entry.get("Diff.L2 2", 'N/A'))
        data[matrix_sizes][algorithm]['Diff.L2 3'] = round_scientific(entry.get("Diff.L2 3", 'N/A'))

        #MaxNorm
        data[matrix_sizes][algorithm]['Diff.Max 1'] = round_scientific(entry.get("Diff.Max 1", 'N/A'))
        data[matrix_sizes][algorithm]['Diff.Max 2']= round_scientific(entry.get("Diff.Max 2", 'N/A'))
        data[matrix_sizes][algorithm]['Diff.Max 3'] = round_scientific(entry.get("Diff.Max 3", 'N/A'))

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
    title = "<h2>Dense Matrix Multiplication</h2>"
    html = style + title
    html += "<table>\n"

    # Define primary and secondary algorithms
    primary_algorithms = ['cuBLAS', 'Magma', 'Cutlass','BLAS']
    secondary_algorithms = ['Kernel 1.1', 'Kernel 1.2', 'Kernel 1.3', 'Kernel 1.4', 'Kernel 1.5', 'Kernel 1.6']
    all_algorithms = primary_algorithms + secondary_algorithms

    # Header row 1: Algorithm names
    html += "<tr><th rowspan='2'>Matrix 1</th><th rowspan='2'>Matrix 2</th>"
    for algo in all_algorithms:
        colspan = '7' if algo in secondary_algorithms else '1'  # Adjust colspan for secondary algorithms
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
        html += "<th>l2Norm</th>"  # Additional cell for Diff.L2
        html += "<th>MaxNorm</th>"
    html += "</tr>\n"

    # Data rows
    for sizes, algos in data.items():
        matrix1_size, matrix2_size = sizes
        html += f"<tr><td>{matrix1_size}</td><td>{matrix2_size}</td>"

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
                diff_l2_2 = algo_data['Diff.L2 2']
                diff_l2_3 = algo_data['Diff.L2 3']
                html += f"<td>Cublas:{diff_l2_1} Magma:{diff_l2_2} Cutlass:{diff_l2_3}</td>"

                diff_Max_1 = algo_data['Diff.Max 1']
                diff_Max_2 = algo_data['Diff.Max 2']
                diff_Max_3 = algo_data['Diff.Max 3']
                html += f"<td>Cublas:{diff_Max_1} Magma:{diff_Max_2} Cutlass:{diff_Max_3}</td>"

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
output_base_path = 'dense_matrix_multiplication'
output_extension = 'html'
output_file_path = get_next_available_filename(output_base_path, output_extension)

with open(output_file_path, 'w') as file:
    file.write(html_table)

print(f"HTML table saved as {output_file_path}")
