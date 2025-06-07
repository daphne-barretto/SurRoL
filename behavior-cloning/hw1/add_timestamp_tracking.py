import sys
import os

# Get the path to the training script
script_path = sys.argv[1]

# Read the current content
with open(script_path, 'r') as f:
    content = f.read()

# Add timestamp import
if 'import datetime' not in content:
    timestamp_import = "import datetime"
    content = content.replace("import time", "import time\n" + timestamp_import)

# Add timestamp code to the main function
timestamp_code = """
    # Add timestamp to save directory for tracking different runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_timestamp:
        args.save_dir = f"{args.save_dir}_{timestamp}"
"""

# Insert the code after parsing arguments
content = content.replace("# Create save directory if it doesn't exist", timestamp_code + "\n    # Create save directory if it doesn't exist")

# Add timestamp argument to the parser
timestamp_arg = """    parser.add_argument('--no_timestamp', action='store_true',
                      help='Do not add timestamp to save directory')
"""

# Insert the argument after the last add_argument call
content = content.replace("    parser.add_argument('--save_every', type=int, default=10,\n                      help='Save model every this many epochs')", 
                          "    parser.add_argument('--save_every', type=int, default=10,\n                      help='Save model every this many epochs')\n" + timestamp_arg)

# Write the modified content back to the file
with open(script_path, 'w') as f:
    f.write(content)

print(f"Added automatic timestamp tracking to {script_path}")
