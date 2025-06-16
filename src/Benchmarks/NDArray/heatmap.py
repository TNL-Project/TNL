# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Load the CSV file
file_path = "2D CPU.csv"  # Update with your local file path
gpu_6d_data = pd.read_csv(file_path)


# Define the heatmap function
def display_6d_gpu_heatmap_save(
    dataframe, title, vmin=0, vmax=100, output_path="2D_СPU_Heatmap.svg"
):
    # Clean the 'Axis' column and convert it to integer for better labeling
    dataframe["Axis"] = dataframe["Axis"].astype(float).astype(int)

    # Pivot data for heatmap
    heatmap_data = dataframe.pivot(
        index="Axis", columns="Permutation", values="Bandwidth"
    )

    # Plot heatmap
    plt.figure(figsize=(10, 8))  # Adjust figure size for 6D data
    plt.title(title, fontsize=15)  # Title font size 22
    heatmap = plt.imshow(
        heatmap_data,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = plt.colorbar(label="Bandwidth")
    colorbar.ax.yaxis.label.set_size(14)  # Set the font size of the "Bandwidth" label
    colorbar.ax.tick_params(labelsize=14)  # Set the font size of the color range ticks

    plt.xticks(
        range(len(heatmap_data.columns)), heatmap_data.columns, rotation=0, fontsize=14
    )  # Horizontal text, font size 21
    plt.yticks(
        range(len(heatmap_data.index)), heatmap_data.index, fontsize=14
    )  # Uniform font size 21
    plt.xlabel("Permutations", fontsize=14)
    plt.ylabel("Axis", fontsize=14)

    # Add text annotations with white text and black contour, font size 22
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value):  # Ensure valid values for annotation
                text = plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="white",
                    weight="bold",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1.5, foreground="black"),
                        path_effects.Normal(),
                    ]
                )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300)  # Save the heatmap as an SVG file
    plt.show()


# Generate and save the heatmap
output_file_path = "6D_СPU_Heatmap.svg"
display_6d_gpu_heatmap_save(
    gpu_6d_data,
    "6D СPU Bandwidth Heatmap",
    vmin=0,
    vmax=100,
    output_path=output_file_path,
)
