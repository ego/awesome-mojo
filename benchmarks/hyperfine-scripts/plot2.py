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

    # Create a single figure with multiple subplots
    num_items = len(result_items)
    fig, axes = plt.subplots(num_items, 1, figsize=(10, 6 * num_items), sharex=True)

    # Loop through each item and create a subplot for "times" data
    for idx, item in enumerate(result_items):
        times = item['times']
        
        # Create a list of indices for the x-axis
        indices = list(range(1, len(times) + 1))

        # Create a line plot of the "times" data on the corresponding subplot
        ax = axes[idx]
        ax.plot(indices, times, marker='o', linestyle='-', color='b')
        ax.set_title(f'{item["command"]}')
        ax.set_ylabel('Time sec')
        ax.grid(True)

    # Set a common x-axis label
    axes[-1].set_xlabel('Runs')

    # Adjust subplot layout
    plt.tight_layout()

    # Save the combined plot as an image
    plt.savefig(f'{file_path}.combined.png')

    # Display the combined plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("file", help="JSON file")
    args = parser.parse_args()
    plot(args.file)
