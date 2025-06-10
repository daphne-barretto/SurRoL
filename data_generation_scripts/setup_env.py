#!/usr/bin/env python3
"""
Setup script to configure Python environment for SurRoL (not SurRol-elsa)
"""

import sys
import os

# Remove SurRol-elsa from path if it exists
surrol_elsa_path = '/home/ubuntu/project/SurRol-elsa'
if surrol_elsa_path in sys.path:
    sys.path.remove(surrol_elsa_path)
    print(f"🔧 Removed SurRol-elsa from path: {surrol_elsa_path}")

# Add the correct paths in the right order  
paths_to_add = [
    '/home/ubuntu/project/SurRoL',  # SurRoL should come first
    '/home/ubuntu/project/behavior-cloning/hw1',
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"✅ Added to path: {path}")

# Add SurRol-elsa back at the end as fallback
sys.path.append(surrol_elsa_path)
print(f"🔧 Added SurRol-elsa as fallback: {surrol_elsa_path}")

# Verify the setup
try:
    print("\n🧪 Testing environment setup...")
    
    # Import gymnasium first
    import gymnasium as gym
    print("✅ Gymnasium imported successfully")
    
    # Now import surrol.gym
    import surrol.gym
    print(f"✅ Using SurRoL gym module from: {surrol.gym.__file__}")
    
    # Check which module is being used
    if 'SurRoL' in surrol.gym.__file__:
        print("✅ Correct! Using SurRoL (not SurRol-elsa)")
    else:
        print("❌ Warning! Still using SurRol-elsa")
    
    # Test a few environment registrations
    test_envs = [
        'PegTransferTwoBlocksNoColor-v0',
        'PegTransferTwoBlocksNoColorOneHot-v0',
        'PegTransferTwoBlocksNoColorOneHotTargetPeg-v0'
    ]
    
    working_envs = 0
    for env_name in test_envs:
        try:
            env = gym.make(env_name)
            env.close()
            working_envs += 1
            print(f"✅ {env_name} working")
        except Exception as e:
            print(f"❌ {env_name}: {str(e)[:50]}")
    
    print(f"\n📊 {working_envs}/{len(test_envs)} test environments working")
    
    if working_envs == len(test_envs):
        print("\n🎉 Environment setup complete! You can now use:")
        print("   • All 2-block no-color environments")
        print("   • Universal BC training system")
        print("   • Correct SurRoL module (not SurRol-elsa)")
        
        print("\n📝 Summary of created 2-block no-color environments:")
        all_2block_envs = [
            ('PegTransferTwoBlocksNoColor-v0', 'Base 2-block environment'),
            ('PegTransferTwoBlocksNoColorOneHot-v0', '2-block + one-hot encoding'),
            ('PegTransferTwoBlocksNoColorTargetBlock-v0', '2-block + target block info'),
            ('PegTransferTwoBlocksNoColorTargetPeg-v0', '2-block + target peg info'),
            ('PegTransferTwoBlocksNoColorTargetBlockPeg-v0', '2-block + target block + peg'),
            ('PegTransferTwoBlocksNoColorOneHotTargetPeg-v0', '2-block + one-hot + target peg'),
            ('PegTransferTwoBlocksNoColorFourTuple-v0', '2-block + RGBA color info'),
        ]
        
        for env_name, description in all_2block_envs:
            print(f"   • {env_name:<45} - {description}")
    
except Exception as e:
    print(f"❌ Setup failed: {e}")
    import traceback
    traceback.print_exc()

# Show what's in sys.path
print(f"\n🔍 Current Python path (project-related entries):")
for i, path in enumerate(sys.path):
    if 'project' in path:
        marker = " ← SurRoL" if 'SurRoL' in path and 'elsa' not in path else ""
        marker += " ← SurRol-elsa" if 'elsa' in path else ""
        print(f"   {i}: {path}{marker}") 