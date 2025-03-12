import re
import csv
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Parse a log file into CSV format.")
parser.add_argument("-log", required=True, help="Path to the input log file")
parser.add_argument("-csv", help="Path to the output CSV file (optional)")
args = parser.parse_args()

log_file_path = args.log

# If CSV file path is not provided, generate it from the log file's first line
if args.csv:
    csv_file_path = args.csv
else:
    with open(log_file_path, "r") as log_file:
        first_line = log_file.readline()
        match = re.search(r"\.\/logs\/([\w/-]+)", first_line)
        if match:
            safe_filename = match.group(1).replace("/", "_")
            csv_file_path = f"{safe_filename}.csv"
        else:
            csv_file_path = "output.csv"  # Default fallback name

# Regular expression pattern to extract key-value pairs
pattern = re.compile(r"\|\s*(.*?)\s*\|\s*([-+eE0-9\.]+)\s*\|")

data = []
headers = set()

# Read and parse the log file
with open(log_file_path, "r") as log_file:
    current_epoch = {}
    for line in log_file:
        match = pattern.findall(line)
        if match:
            for key, value in match:
                current_epoch[key] = value
                headers.add(key)
        elif "---------------------------------------------" in line and current_epoch:
            data.append(current_epoch)
            current_epoch = {}

# Sort headers to maintain consistency
headers = sorted(headers)

# Write data to CSV file
with open(csv_file_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

print(f"CSV file saved to {csv_file_path}")