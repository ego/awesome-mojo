#!/usr/bin/env python

import os
import argparse
import json
import matplotlib.pyplot as plt


def plot(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Get the list of items under "results"
    result_items = data['results']

    # Create a single figure for the combined plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lists to store combined data
    combined_times = []

    # Define a list of colors for each line
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Loop through each item and combine "times" data
    for i, item in enumerate(result_items):
        times = item['times']
        combined_times.extend(times)
        label = f'{item["command"]}'  # Label for legend
        color = colors[i % len(colors)]  # Cycle through colors
        ax.plot(times, marker='o', linestyle='-', color=color, label=label)

    # Create a list of indices for the x-axis
    indices = list(range(1, len(combined_times) + 1))

    # Add a legend to the plot
    ax.legend()

    # Set labels and title
    ax.set_title('Combined Execution Times')
    ax.set_xlabel('Runs')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True)

    # Save the combined plot as an image
    plt.savefig(f'{file_path}.all.png')

    # Display the combined plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("file", help="JSON file")
    args = parser.parse_args()
    plot(args.file)
