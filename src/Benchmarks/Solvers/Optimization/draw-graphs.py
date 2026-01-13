#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from pathlib import Path

def read_data(filename):
    print( f"Reading file {filename}...\n")
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def plot_functions(base, restarting, file1, file2, restart_file='restarts.txt'):
    # Read the data
    x1, y1 = read_data(file1+'.txt')
    x2, y2 = read_data(file2+'.txt')

    # Extract labels from filenames
    label1 = os.path.splitext(os.path.basename(file1))[0]
    label2 = os.path.splitext(os.path.basename(file2))[0]

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # First function
    color1 = 'tab:blue'
    ax1.set_xlabel('x')
    ax1.set_ylabel(label1, color=color1)
    ax1.set_yscale('log')
    ax1.plot(x1, y1, color=color1, label=label1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Second y-axis for second function
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_yscale('log')
    ax2.set_ylabel(label2, color=color2)
    ax2.plot(x2, y2, color=color2, label=label2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Plot restart lines if file exists
    if os.path.isfile(restart_file):
        colors = {'ARTIFICIAL': 'red', 'SUFFICIENT': 'green', 'NECESSARY': 'blue'}
        linestyles = {'CURRENT': 'solid', 'AVERAGE': 'dashed'}

        with open(restart_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                iter_num, rtype, target = parts
                try:
                    iter_val = float(iter_num)
                    color = colors.get(rtype, 'gray')
                    linestyle = linestyles.get(target, 'dotted')
                    ax1.axvline(x=iter_val, color=color, linestyle=linestyle, linewidth=1.0)
                except ValueError:
                    continue

    # Title and layout
    plt.title(f'Functions {label1} and {label2}')
    fig.tight_layout()

    # Save to PDF
    output_filename = f"{base}_{restarting}_{label1}_{label2}.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

def process_dir( base, restarting ):
    current = Path.cwd()
    subdir = current / restarting
    os.chdir(subdir)
    plot_functions(base, restarting, 'fast-current-primal-objective', 'kkt-current-primal-objective')
    plot_functions(base, restarting, 'fast-current-dual-objective', 'kkt-current-dual-objective')
    plot_functions(base, restarting, 'fast-current-primal-feasibility', 'kkt-current-primal-feasibility')
    plot_functions(base, restarting, 'fast-current-dual-feasibility', 'kkt-current-dual-feasibility')
    plot_functions(base, restarting, 'fast-current-duality-gap', 'kkt-current-duality-gap')
    plot_functions(base, restarting, 'fast-current-mu', 'kkt-current-mu')
    plot_functions(base, restarting, 'fast-averaged-primal-objective', 'kkt-averaged-primal-objective')
    plot_functions(base, restarting, 'fast-averaged-dual-objective', 'kkt-averaged-dual-objective')
    #plot_functions(base, restarting, 'fast-averaged-primal-feasibility', 'kkt-averaged-primal-feasibility')
    #plot_functions(base, restarting, 'fast-averaged-dual-feasibility', 'kkt-averaged-dual-feasibility')
    plot_functions(base, restarting, 'fast-averaged-duality-gap', 'kkt-averaged-duality-gap')
    plot_functions(base, restarting, 'fast-averaged-mu', 'kkt-averaged-mu')

    plot_functions(base, restarting, 'kkt-current-primal-objective', 'kkt-averaged-primal-objective')
    plot_functions(base, restarting, 'kkt-current-dual-objective', 'kkt-averaged-dual-objective')
    plot_functions(base, restarting, 'kkt-current-primal-feasibility', 'kkt-averaged-primal-feasibility')
    plot_functions(base, restarting, 'kkt-current-dual-feasibility', 'kkt-averaged-dual-feasibility')
    plot_functions(base, restarting, 'kkt-current-duality-gap', 'kkt-averaged-duality-gap')
    plot_functions(base, restarting, 'kkt-current-mu', 'kkt-averaged-mu')

    plot_functions(base, restarting, 'current-gradient', 'kkt-current-mu')
    plot_functions(base, restarting, 'averaged-gradient', 'kkt-averaged-mu')
    os.chdir(current)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dir>")
        sys.exit(1)
    dir = sys.argv[1]

    dir_path = Path(dir)
    for file in dir_path.glob('*.mps'):

        file_path = Path(file)
        base = file_path.stem

        if base == 'set-cover-model' or base == 'qap15':
            continue

        current = Path.cwd()

        try:
            subdir = current / (base + "-results")
        except:
            print( f'Error in dir {subdir}...' )

        os.chdir(subdir)
        print( f'Processing {base}')
        process_dir( base, 'adaptive-restarting' )
        #process_dir( base, '' )
        os.chdir(current)






