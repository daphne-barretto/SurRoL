"""
Script to convert the .npz data to the format expected by the BC agent.
Specifically designed for your data structure.
"""

import numpy as np
import pickle
import os
import argparse

def convert_npz_to_bc_format(npz_file, output_file, use_images=True):
    """
    Convert .npz data to BC agent format based on the structure:
    - acs: (10, 50, 5)
    - obs: (10, 51, 480, 640, 3)
    - infos: (10, 50)
    - rewards: (10, 50)
    - traditional_obs: (10, 51)
    
    Parameters
    ----------
    npz_file : str
        Path to the .npz file
    output_file : str
        Path to save the output pickle file
    use_images : bool
        Whether to use image observations or traditional state-based observations
    """
    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract arrays
    actions = data['acs']               # Shape: (10, 50, 5)
    if 'rewards' in data.files:
        rewards = data['rewards']           # Shape: (10, 50)
    else:
        print("Warning: 'rewards' not found in the .npz file. Filling with zeros.")
        rewards = np.zeros(actions.shape[:2], dtype=np.float32)
    infos = data['infos'] if 'infos' in data.files else None  # Shape: (10, 50) or None
    
    if use_images:
        observations = data['obs']      # Shape: (10, 51, 480, 640, 3)
    else:
        observations = data['traditional_obs']  # Shape: (10, 51)
    
    # Print shapes to verify
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    if use_images:
        print(f"Image observations shape: {observations.shape}")
    else:
        print(f"Traditional observations shape: {observations.shape}")
    
    # Number of episodes
    num_episodes = actions.shape[0]
    print(f"Found {num_episodes} episodes")
    
    paths = []
    for i in range(num_episodes):
        print(f"Processing episode {i+1}/{num_episodes}")
        
        # Get data for this episode
        episode_obs = observations[i]    # Shape: (51, 480, 640, 3) for images or (51,) for traditional
        episode_acts = actions[i]        # Shape: (50, 5)
        episode_rewards = rewards[i]     # Shape: (50,)
        
        # Create path dictionary
        # Note: observations include the initial state, so there are 51 observations for 50 actions
        path = {
            'observation': episode_obs[:50],         # First 50 observations (excluding the last)
            'action': episode_acts,                  # All 50 actions
            'reward': episode_rewards,               # All 50 rewards
            'next_observation': episode_obs[1:51],   # Last 50 observations (excluding the first)
            'terminal': np.zeros(50, dtype=bool)     # Terminal flags
        }
        
        # Set the last step as terminal
        path['terminal'][-1] = True
        
        paths.append(path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save the converted data
    print(f"Saving {num_episodes} episodes to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(paths, f)
    
    print(f"Conversion complete!")
    return paths

def main():
    parser = argparse.ArgumentParser(description='Convert .npz data to BC format')
    parser.add_argument('--input', type=str, required=True, 
                      help='Path to input .npz file')
    parser.add_argument('--output', type=str, required=True, 
                      help='Path to output pickle file')
    parser.add_argument('--traditional_obs', action='store_true',
                      help='Use traditional state-based observations instead of images')
    
    args = parser.parse_args()
    
    convert_npz_to_bc_format(
        args.input, 
        args.output, 
        use_images=not args.traditional_obs
    )

if __name__ == '__main__':
    main()