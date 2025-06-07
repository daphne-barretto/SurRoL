#!/usr/bin/env python3

"""
Data Generation Script for PegTransferTwoBlocksTargetBlockOnlyNoColor-v0 Environment

This script generates demonstration data for the 2-block peg transfer task
where observations only include information about the target block (no other blocks, no color).

Observation structure: 
- robot_state(7) + target_block_pos(3) + target_block_rel_pos(3) = 13 dims
- No additional conditioning
- No color information
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add SurRoL to path
project_root = os.path.dirname(os.path.abspath(__file__))
surrol_path = os.path.join(project_root, 'SurRoL')
if surrol_path not in sys.path:
    sys.path.insert(0, surrol_path)

import gymnasium as gym
import surrol.gym

def generate_target_block_only_no_color_demos(num_episodes=5000, max_steps_per_episode=50, 
                                             save_path=None, verbose=True):
    """
    Generate demonstration data for the target block only no-color environment
    
    Args:
        num_episodes: Number of episodes to generate
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save the data (auto-generated if None)
        verbose: Whether to print progress
    """
    
    if verbose:
        print(f"🎯 GENERATING TARGET BLOCK ONLY NO-COLOR DEMONSTRATIONS")
        print("="*60)
        print(f"   • Episodes: {num_episodes}")
        print(f"   • Max steps per episode: {max_steps_per_episode}")
        print(f"   • Environment: PegTransferTwoBlocksTargetBlockOnlyNoColor-v0")
    
    # Create environment
    try:
        env = gym.make('PegTransferTwoBlocksTargetBlockOnlyNoColor-v0')
        if verbose:
            print(f"   ✅ Environment created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return None
    
    # Data storage
    observations = []
    actions = []
    episode_data = []
    
    successful_episodes = 0
    total_transitions = 0
    
    # Generate episodes
    for episode in range(num_episodes):
        try:
            # Reset environment
            obs, info = env.reset()
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_success = False
            
            # Get access to the underlying environment for oracle actions
            actual_env = env
            while hasattr(actual_env, '_env') or hasattr(actual_env, 'env'):
                if hasattr(actual_env, '_env'):
                    actual_env = actual_env._env
                elif hasattr(actual_env, 'env'):
                    actual_env = actual_env.env
                else:
                    break
            
            for step in range(max_steps_per_episode):
                # Get oracle action (expert demonstration) - use improved method
                try:
                    # Try multiple approaches to get oracle action
                    if hasattr(actual_env, 'get_oracle_action'):
                        action = actual_env.get_oracle_action(obs)
                    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_oracle_action'):
                        action = env.unwrapped.get_oracle_action(obs)
                    else:
                        # Fallback to random action
                        action = env.action_space.sample()
                except Exception as oracle_error:
                    if verbose and step == 0:  # Only print on first step to avoid spam
                        print(f"   ⚠️  Oracle action failed for episode {episode + 1}: {oracle_error}")
                    # Fallback to random action
                    action = env.action_space.sample()
                
                # Store transition
                episode_obs.append(obs.copy())
                episode_actions.append(action.copy())
                
                # Take step
                next_obs, reward, done, truncated, info = env.step(action)
                episode_rewards.append(reward)
                
                obs = next_obs
                total_transitions += 1
                
                # Check for success
                if info.get('is_success', False):
                    episode_success = True
                
                # Episode termination
                if done or truncated:
                    break
            
            # Store episode data - ONLY KEEP SUCCESSFUL EPISODES
            if len(episode_obs) > 0 and episode_success:  # Only store successful episodes
                observations.append(episode_obs)
                actions.append(episode_actions)
                
                episode_info = {
                    'episode': episode + 1,
                    'length': len(episode_obs),
                    'total_reward': sum(episode_rewards),
                    'success': episode_success,
                    'final_obs_dim': len(episode_obs[-1]['observation'])
                }
                episode_data.append(episode_info)
                
                if episode_success:
                    successful_episodes += 1
            
            # Progress update
            if verbose and (episode + 1) % 10 == 0:
                success_rate = successful_episodes / (episode + 1)
                stored_episodes = len(observations)  # Number of episodes actually stored
                print(f"   Episode {episode + 1:4d}/{num_episodes}: "
                      f"Success rate: {success_rate:.1%} "
                      f"({successful_episodes}/{episode + 1}) | "
                      f"Stored: {stored_episodes}")
        
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Episode {episode + 1} failed: {e}")
            continue
    
    env.close()
    
    # Convert to numpy arrays
    obs_array = np.array(observations, dtype=object)
    acs_array = np.array(actions, dtype=object)
    
    # Generate save path if not provided - save in demo directory with correct format
    if save_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"/home/ubuntu/project/SurRoL/surrol/data/demo/data_PegTransferTwoBlocksTargetBlockOnlyNoColor-v0_{num_episodes}_{timestamp}.npz"
    
    # Save data
    np.savez_compressed(
        save_path,
        obs=obs_array,
        acs=acs_array,
        episode_data=episode_data
    )
    
    # Results summary
    final_success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0
    
    if verbose:
        print(f"\n📊 DATA GENERATION SUMMARY:")
        print(f"   • Total episodes attempted: {num_episodes}")
        print(f"   • Successful episodes: {successful_episodes}")
        print(f"   • Success rate: {final_success_rate:.1%}")
        print(f"   • Episodes stored in dataset: {len(observations)} (successful only)")
        print(f"   • Total transitions: {total_transitions:,}")
        print(f"   • Average episode length: {total_transitions / num_episodes:.1f}")
        
        if len(episode_data) > 0:
            sample_obs_dim = episode_data[0]['final_obs_dim']
            avg_stored_length = sum(ep['length'] for ep in episode_data) / len(episode_data)
            print(f"   • Average stored episode length: {avg_stored_length:.1f}")
            print(f"   • Observation dimension: {sample_obs_dim}")
            print(f"   • Data type: target_block_only_2_no_color")
            print(f"   • Conditioning: none")
            print(f"   • Color information: none")
            print(f"   • Quality: successful episodes only")
        
        print(f"   • Data saved to: {save_path}")
        print(f"\n💡 USAGE WITH UNIVERSAL BC:")
        print(f"   conda activate gcrl && cd /home/ubuntu/project && PYTHONPATH=\"/home/ubuntu/project/SurRoL:$PYTHONPATH\" python behavior-cloning/hw1/cs224r/scripts/universal_bc_system.py --mode train \\")
        print(f"     --data_path {save_path} \\")
        print(f"     --save_dir /home/ubuntu/project/behavior-cloning/hw1/cs224r/experiments/5k_demos/bc_results_2block_targetblockonly_no_color_5k \\")
        print(f"     --base_type goal_conditioned_demos_with_only_target_block_2_blocks_no_color \\")
        print(f"     --conditioning_type none --epochs 150 --batch_size 64 --eval_interval 10 --eval_episodes 20")
    
    return {
        'save_path': save_path,
        'num_episodes': num_episodes,
        'successful_episodes': successful_episodes,
        'success_rate': final_success_rate,
        'total_transitions': total_transitions,
        'obs_dim': episode_data[0]['final_obs_dim'] if episode_data else 0
    }

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate target block only no-color demonstration data')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to generate')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--save_path', type=str, help='Path to save data (auto-generated if not provided)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    result = generate_target_block_only_no_color_demos(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_path=args.save_path,
        verbose=not args.quiet
    )
    
    if result:
        print(f"\n✅ Data generation completed successfully!")
        return 0
    else:
        print(f"\n❌ Data generation failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 