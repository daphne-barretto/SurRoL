"""
check_action_dimension.py

Quick script to check the action dimension in the PegTransfer data.
"""

import numpy as np
import argparse

def check_npz_action_dim(npz_path):
    """Check action dimension from npz file"""
    print(f"Checking action dimension in: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    print("Available keys:", list(data.keys()))
    
    if 'acs' in data:
        acs_data = data['acs']
        print(f"Actions data shape: {acs_data.shape}")
        print(f"Number of episodes: {len(acs_data)}")
        
        if len(acs_data) > 0:
            first_episode = acs_data[0]
            print(f"First episode actions shape: {first_episode.shape}")
            
            if len(first_episode) > 0:
                first_action = first_episode[0]
                print(f"First action: {first_action}")
                print(f"Action dimension: {len(first_action)}")
                return len(first_action)
    
    return None

def check_environment_action_space():
    """Check action space directly from environment"""
    try:
        import gymnasium as gym
        import surrol.gym
        
        env = gym.make('PegTransfer-v0')
        action_space = env.action_space
        print(f"Environment action space: {action_space}")
        print(f"Action dimension: {action_space.shape[0]}")
        env.close()
        return action_space.shape[0]
        
    except Exception as e:
        print(f"Could not check environment: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                      help='Path to npz data file to check')
    parser.add_argument('--check_env', action='store_true',
                      help='Check environment action space')
    
    args = parser.parse_args()
    
    print("=== Action Dimension Check ===")
    
    if args.check_env:
        print("\n1. Checking environment action space:")
        env_dim = check_environment_action_space()
    
    if args.data_path:
        print(f"\n2. Checking data file:")
        data_dim = check_npz_action_dim(args.data_path)
        
        if args.check_env and env_dim and data_dim:
            if env_dim == data_dim:
                print(f"\n✓ Action dimensions match: {env_dim}")
            else:
                print(f"\n✗ Action dimension mismatch: env={env_dim}, data={data_dim}")

if __name__ == '__main__':
    main()