import pandas as pd
import json
import os

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_entries = []
        start_processing = False
        for line in file:
            entry = json.loads(line)
            # Start processing when 'matrix size' is encountered
            if 'matrix size' in entry:
                start_processing = True
            if start_processing:
                log_entries.append(entry)
        return log_entries

def process_data(log_data):
    data = {}
    for entry in log_data:
        matrix_size = (entry["matrix size"])
        algorithm = entry["algorithm"]
        time = float(entry["time"])

        if matrix_size not in data:
            data[matrix_size] = {}
        data[matrix_size][algorithm] = time

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
    title = "<h2>Dense Matrix Transposition</h2>"
    html = style + title
    html += "<table>\n<tr><th>Matrix Size</th><th>MAGMA</th><th>TNL</th><th>MAGMA vs TNL</th></tr>\n"

    for matrix_size, algos in data.items():
        magma_time = algos.get('MAGMA', None)
        tnl_time = algos.get('TNL', None)

        formatted_magma_time = format_time(magma_time) if magma_time is not None else 'N/A'
        formatted_tnl_time = format_time(tnl_time) if tnl_time is not None else 'N/A'

        magma_tnl_speedup = calculate_speedup(magma_time, tnl_time)
        formatted_magma_tnl_speedup = f"{magma_tnl_speedup:.2f}x" if magma_tnl_speedup is not None else 'N/A'

        html += f"<tr><td>{matrix_size}</td><td>{formatted_magma_time}</td><td>{formatted_tnl_time}</td><td>{formatted_magma_tnl_speedup}</td></tr>\n"

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
output_base_path = 'dense_matrix_transposition'
output_extension = 'html'
output_file_path = get_next_available_filename(output_base_path, output_extension)

with open(output_file_path, 'w') as file:
    file.write(html_table)

print(f"HTML table saved as {output_file_path}")

