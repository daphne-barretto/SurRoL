#!/usr/bin/env python3
"""
Wrapper script for Universal BC System that sets correct Python path
"""

import os
import sys

# Add our SurRoL directory to Python path (before SurRol-elsa)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
surrol_path = os.path.join(project_root, 'SurRoL')

# Insert at the beginning to take precedence over SurRol-elsa
if surrol_path not in sys.path:
    sys.path.insert(0, surrol_path)

print(f"🔧 Added SurRoL path: {surrol_path}")
print(f"🔧 Python path now includes:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    if 'SurRoL' in path or 'SurRol' in path:
        print(f"   {i}: {path}")

# Now import and run the universal BC system
from universal_bc_system import main

if __name__ == '__main__':
    main() 