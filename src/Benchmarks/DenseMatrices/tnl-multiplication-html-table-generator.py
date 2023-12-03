import pandas as pd
import json
import os

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_entries = []
        for line in file:
            entry = json.loads(line)
            # Stop reading further if 'matrix size' is encountered
            if 'matrix size' in entry:
                break
            log_entries.append(entry)
        return log_entries

def process_data(log_data):
    data = {}
    for entry in log_data:
        matrix_sizes = (entry["matrix1 size"], entry["matrix2 size"])
        algorithm = entry["algorithm"]
        time = float(entry["time"])

        if matrix_sizes not in data:
            data[matrix_sizes] = {}
        data[matrix_sizes][algorithm] = time

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
            padding: 8px;
            min-width: 120px; /* Set minimum width for each column */
        }
        th {
            background-color: #f2f2f2;
            color: black.
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
    html += "<table>\n<tr><th>Matrix 1</th><th>Matrix 2</th>"

    # Define primary and secondary algorithms
    primary_algorithms = ['cuBLAS', 'Magma', 'Cutlass']
    secondary_algorithms = ['TNL', 'TNL2', '2D SMA', 'Warptiling', 'Warptiling2']

    # Add primary algorithm columns and their comparisons
    html += "<th>cuBLAS</th><th>Magma</th><th>cuBLAS vs Magma</th><th>Cutlass</th><th>cuBLAS vs Cutlass</th>"

    # Add secondary algorithms and their comparison with primary algorithms
    for algo in secondary_algorithms:
        html += f"<th>{algo}</th>"
        html += f"<th>cuBLAS vs {algo}</th>"
        html += f"<th>Magma vs {algo}</th>"
        html += f"<th>Cutlass vs {algo}</th>"

    html += "</tr>\n"

    for sizes, algos in data.items():
        matrix1_row, matrix1_col = sizes[0].split('x')
        matrix2_row, matrix2_col = sizes[1].split('x')
        html += f"<tr><td>{matrix1_row}x{matrix1_col}</td><td>{matrix2_row}x{matrix2_col}</td>"

        # Fetch and format times for primary algorithms
        cublas_time = algos.get('cuBLAS')
        magma_time = algos.get('Magma')
        cutlass_time = algos.get('Cutlass')
        formatted_cublas_time = format_time(cublas_time) if cublas_time is not None else 'N/A'
        formatted_magma_time = format_time(magma_time) if magma_time is not None else 'N/A'
        formatted_cutlass_time = format_time(cutlass_time) if cutlass_time is not None else 'N/A'

        # Add times and speedup comparisons for primary algorithms
        html += f"<td>{formatted_cublas_time}</td>"
        html += f"<td>{formatted_magma_time}</td>"
        magma_cublas_speedup = calculate_speedup(cublas_time, magma_time)
        formatted_magma_cublas_speedup = f"{magma_cublas_speedup:.2f}x" if magma_cublas_speedup is not None else 'N/A'
        html += f"<td>{formatted_magma_cublas_speedup}</td>"
        html += f"<td>{formatted_cutlass_time}</td>"
        cutlass_cublas_speedup = calculate_speedup(cublas_time, cutlass_time)
        formatted_cutlass_cublas_speedup = f"{cutlass_cublas_speedup:.2f}x" if cutlass_cublas_speedup is not None else 'N/A'
        html += f"<td>{formatted_cutlass_cublas_speedup}</td>"

        # Add tnl algorithms and their comparison with primary algorithms
        for algo in secondary_algorithms:
            algo_time = algos.get(algo)
            formatted_time = format_time(algo_time) if algo_time is not None else 'N/A'
            html += f"<td>{formatted_time}</td>"

            algo_cublas_speedup = calculate_speedup(cublas_time, algo_time)
            formatted_algo_cublas_speedup = f"{algo_cublas_speedup:.2f}x" if algo_cublas_speedup is not None else 'N/A'
            html += f"<td>{formatted_algo_cublas_speedup}</td>"

            algo_magma_speedup = calculate_speedup(magma_time, algo_time)
            formatted_algo_magma_speedup = f"{algo_magma_speedup:.2f}x" if algo_magma_speedup is not None else 'N/A'
            html += f"<td>{formatted_algo_magma_speedup}</td>"

            algo_cutlass_speedup = calculate_speedup(cutlass_time, algo_time)
            formatted_algo_cutlass_speedup = f"{algo_cutlass_speedup:.2f}x" if algo_cutlass_speedup is not None else 'N/A'
            html += f"<td>{formatted_algo_cutlass_speedup}</td>"

        html += "</tr>\n"

    html += "</table>"
    return html

log_file_path = 'tnl-benchmark-dense-matrices.log'
log_data = read_log_file(log_file_path)
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

