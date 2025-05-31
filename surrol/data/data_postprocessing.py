"""
Example usage:
python data_postprocessing.py /path/to/npz --output_path /path/to/output/npz --condition_type <one_hot/target_block/...>

e.g.
python data_postprocessing.py data_PegTransfer-v0_random_10000_2025-05-31_14-46-53.npz --output_path data_PegTransfer-v0_random_10000_2025-05-31_14-46-53_targetblock.npz --condition_type target_block
"""

import numpy as np
import argparse
import os

def apply_condition_type(observation, block_encoding, achieved_goal, desired_goal, condition_type):
    """Apply the specified conditioning to the observation"""
    if condition_type == "one_hot":
        return np.hstack([observation, block_encoding])
    elif condition_type == "target_block":
        return np.hstack([observation, achieved_goal])
    elif condition_type == "target_block_and_target_peg":
        return np.hstack([observation, achieved_goal, desired_goal])
    elif condition_type == "one_hot_and_target_peg":
        return np.hstack([observation, block_encoding, desired_goal])
    else:
        print(f"Invalid condition type: {condition_type}")
        exit(1)

def postprocess_demo_data(input_demo_file, output_demo_file, condition_type="one_hot"):
    """Function that postprocesses demo data by concatenating block encoding with observations"""

    demo_data = np.load(input_demo_file, allow_pickle=True)
    
    demo_data_obs = demo_data['obs']
    demo_data_acs = demo_data['acs']
    demo_data_info = demo_data['info']

    postprocessed_obs = []
    postprocessed_acs = []
    postprocessed_info = []

    num_episodes = len(demo_data_obs)
    
    print(f"Processing {num_episodes} episodes...")
    
    for epsd in range(num_episodes):
        episode_obs = []
        episode_acs = []
        episode_info = []
    
        for transition in range(len(demo_data_obs[epsd]) - 1):
            observation = demo_data_obs[epsd][transition].get('observation')
            block_encoding = demo_data_obs[epsd][transition].get('block_encoding')
            achieved_goal = demo_data_obs[epsd][transition].get('achieved_goal') # target block pos
            desired_goal = demo_data_obs[epsd][transition].get('desired_goal') # target peg / goal pos
        
            new_obs = {}
            for key, value in demo_data_obs[epsd][transition].items():
                if key == 'observation':
                    new_obs[key] = apply_condition_type(observation, block_encoding, achieved_goal, desired_goal, condition_type)
                else:
                    new_obs[key] = value
            
            episode_obs.append(new_obs)
            episode_acs.append(demo_data_acs[epsd][transition])
            episode_info.append(demo_data_info[epsd][transition])
        
        # Process the last observation
        last_obs_idx = len(demo_data_obs[epsd]) - 1
        last_observation = demo_data_obs[epsd][last_obs_idx].get('observation')
        last_block_encoding = demo_data_obs[epsd][last_obs_idx].get('block_encoding')
        last_achieved_goal = demo_data_obs[epsd][last_obs_idx].get('achieved_goal') # target block pos
        last_desired_goal = demo_data_obs[epsd][last_obs_idx].get('desired_goal') # target peg / goal pos
        
        new_last_obs = {}
        for key, value in demo_data_obs[epsd][last_obs_idx].items():
            if key == 'observation':
                new_last_obs[key] = apply_condition_type(last_observation, last_block_encoding, last_achieved_goal, last_desired_goal, condition_type)
            else:
                new_last_obs[key] = value
        
        episode_obs.append(new_last_obs)
    
        postprocessed_obs.append(episode_obs)
        postprocessed_acs.append(episode_acs)
        postprocessed_info.append(episode_info)
        
        # print(f"Processed episode {epsd + 1}/{num_episodes}")
        
    postprocessed_data = {
        'obs': np.array(postprocessed_obs, dtype=object),
        'acs': np.array(postprocessed_acs, dtype=object),
        'info': np.array(postprocessed_info, dtype=object)
    }
    
    np.savez_compressed(output_demo_file, **postprocessed_data)
    
    print(f"postprocessed demo data saved to: {output_demo_file}")
    print(f"Original observation shape example: {demo_data_obs[0][0]['observation'].shape}")
    print(f"Block encoding shape example: {demo_data_obs[0][0]['block_encoding'].shape}")
    print(f"New concatenated observation shape example: {postprocessed_obs[0][0]['observation'].shape}")
    
    return postprocessed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='postprocess demo data by concatenating block encoding with observations')
    parser.add_argument('input_path', type=str, help='Path to input .npz demo file')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Path to output .npz file (default: input_path with _postprocessed suffix)')
    parser.add_argument('--condition_type', type=str, default="one_hot",
                       choices=["one_hot", "target_block", "target_block_and_target_peg", "one_hot_and_target_peg"],
                       help='Type of conditioning to apply (default: one_hot)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        exit(1)
    
    # Generate output path if not provided
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        input_name = os.path.basename(args.input_path)
        name_without_ext = os.path.splitext(input_name)[0]
        args.output_path = os.path.join(input_dir, f"{name_without_ext}_postprocessed.npz")
    
    print(f"Input file: {args.input_path}")
    print(f"Output file: {args.output_path}")
    print(f"Condition type: {args.condition_type}")

    try:
        postprocessed_data = postprocess_demo_data(args.input_path, args.output_path, args.condition_type)
        print("postprocessing completed successfully!")
    except Exception as e:
        print(f"Error during postprocessing: {e}")
