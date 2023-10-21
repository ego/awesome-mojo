#!/usr/bin/env python

"""This script merge all *.json files in the derectory into one for plots."""

import os
import json
import argparse
import glob


def combine_json_files(input_directory, output_file):
    # Create an empty list to store JSON data from each file
    combined_data = []

    # Use glob to find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))

    # Loop through the JSON files
    for json_file in json_files:
        with open(json_file, "r") as file:
            try:
                data = json.load(file)
                combined_data += data["results"]
            except json.JSONDecodeError:
                print(f"Error reading {json_file}. Skipping...")

    # Combine the JSON data from all files
    combined_json = json.dumps({"results": combined_data}, indent=4)

    # Write the combined JSON data to the output file
    with open(output_file, "w") as outfile:
        outfile.write(combined_json)
    print(f"Combined JSON data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine JSON files in a directory into one JSON file.")
    parser.add_argument("directory", help="Directory containing JSON files")
    parser.add_argument("output_file", help="Output file for combined JSON data")
    args = parser.parse_args()
    combine_json_files(args.directory, args.output_file)
