#!/usr/bin/env python

import os
import argparse
import glob
import json
import matplotlib.pyplot as plt


def plot(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Get the list of items under "results"
    result_items = data['results']

    # Loop through each item and create a plot for "times" data
    for idx, item in enumerate(result_items, 1):
        times = item['times']

        # Create a list of indices for the x-axis
        indices = list(range(1, len(times) + 1))

        # Create a line plot of the "times" data
        plt.figure(figsize=(10, 6))
        plt.plot(indices, times, marker='o', linestyle='-', color='b')
        plt.title(f'{item["command"]}')
        plt.xlabel('Runs')
        plt.ylabel('Time sec')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image
        plot_filename = f'{file_path}.{idx}.png'
        plt.savefig(plot_filename)

        # Display the plot
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("file", help="JSON file")
    args = parser.parse_args()
    plot(args.file)
