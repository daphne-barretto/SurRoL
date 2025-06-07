"""
Data and Environment Inspector

This script helps you understand the actual structure of your training data
and environment observations to properly configure the BC system.
"""

import numpy as np
import gymnasium as gym
import surrol.gym
import os
import json
from typing import Dict, Any

def analyze_training_data(data_path: str) -> Dict[str, Any]:
    """Analyze the structure of training data"""
    print(f"🔍 ANALYZING TRAINING DATA")
    print("="*50)
    print(f"File: {os.path.basename(data_path)}")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs']
    acs_data = data['acs']
    
    # Basic info
    print(f"\n📊 Basic Info:")
    print(f"   • Episodes: {len(obs_data)}")
    print(f"   • Actions shape: {acs_data[0][0].shape}")
    
    # Analyze first episode, first observation
    first_obs = obs_data[0][0]
    print(f"\n🔬 First Observation Analysis:")
    print(f"   • Type: {type(first_obs)}")
    
    if isinstance(first_obs, dict):
        print(f"   • Keys: {list(first_obs.keys())}")
        for key, value in first_obs.items():
            if isinstance(value, np.ndarray):
                print(f"   • {key}: shape {value.shape}, dtype {value.dtype}")
                if key == 'observation':
                    print(f"      - First 10 values: {value[:10]}")
                    print(f"      - Last 10 values: {value[-10:]}")
                else:
                    print(f"      - Values: {value}")
            else:
                print(f"   • {key}: {value}")
    else:
        print(f"   • Shape: {first_obs.shape}")
        print(f"   • First 10 values: {first_obs[:10]}")
    
    # Analyze a few more observations to see variation
    print(f"\n📈 Observation Variation Analysis:")
    obs_dims = []
    for ep_idx in range(min(5, len(obs_data))):
        for t in range(min(3, len(obs_data[ep_idx]))):
            obs = obs_data[ep_idx][t]
            if isinstance(obs, dict):
                obs_dims.append(len(obs['observation']))
            else:
                obs_dims.append(len(obs))
    
    print(f"   • Observation dimensions across samples: {set(obs_dims)}")
    
    # Try to identify structure patterns
    if isinstance(first_obs, dict) and 'observation' in first_obs:
        obs_vec = first_obs['observation']
        print(f"\n🧩 Observation Vector Structure Analysis:")
        print(f"   • Total dimensions: {len(obs_vec)}")
        
        # Look for patterns that might indicate robot state + block positions
        # Robot state is typically 7 dims (position + quaternion)
        if len(obs_vec) >= 7:
            print(f"   • Potential robot state (first 7): {obs_vec[:7]}")
            remaining = obs_vec[7:]
            print(f"   • Remaining {len(remaining)} dims: {remaining}")
            
            # Check if remaining dims are multiples of 3 (block positions)
            if len(remaining) % 3 == 0:
                num_possible_blocks = len(remaining) // 3
                print(f"   • Could be {num_possible_blocks} blocks (3D positions each)")
                
                # Show potential block positions
                for i in range(num_possible_blocks):
                    start_idx = i * 3
                    end_idx = start_idx + 3
                    block_pos = remaining[start_idx:end_idx]
                    print(f"      - Block {i+1} position: {block_pos}")
            
            # Check for other common patterns
            if len(remaining) % 4 == 0:
                num_possible_rgba = len(remaining) // 4
                print(f"   • Could include {num_possible_rgba} RGBA color encodings")
        
        # Check for goal conditioning info
        if 'achieved_goal' in first_obs:
            print(f"   • Achieved goal: {first_obs['achieved_goal']}")
        if 'desired_goal' in first_obs:
            print(f"   • Desired goal: {first_obs['desired_goal']}")
        if 'block_encoding' in first_obs:
            print(f"   • Block encoding: {first_obs['block_encoding']}")
    
    return {
        'num_episodes': len(obs_data),
        'obs_dim': len(first_obs['observation']) if isinstance(first_obs, dict) else len(first_obs),
        'action_dim': len(acs_data[0][0]),
        'obs_type': type(first_obs).__name__,
        'has_goals': isinstance(first_obs, dict) and ('achieved_goal' in first_obs or 'desired_goal' in first_obs),
        'sample_obs': first_obs
    }

def analyze_environment() -> Dict[str, Any]:
    """Analyze the PegTransfer environment"""
    print(f"\n🎮 ANALYZING ENVIRONMENT")
    print("="*50)
    
    # Suppress PyBullet output
    os.environ['PYBULLET_EGL'] = '0'
    
    try:
        env = gym.make("PegTransfer-v0")
        
        # Reset and get initial observation
        obs, info = env.reset()
        print(f"📊 Environment Info:")
        print(f"   • Observation space: {env.observation_space}")
        print(f"   • Action space: {env.action_space}")
        print(f"   • Reset info keys: {list(info.keys()) if info else 'None'}")
        
        print(f"\n🔬 Initial Observation Analysis:")
        print(f"   • Type: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"   • Keys: {list(obs.keys())}")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"   • {key}: shape {value.shape}, dtype {value.dtype}")
                    if key == 'observation':
                        print(f"      - First 10 values: {value[:10]}")
                        print(f"      - Last 10 values: {value[-10:] if len(value) > 10 else value}")
                    else:
                        print(f"      - Values: {value}")
                else:
                    print(f"   • {key}: {value}")
        
        # Take a few random actions to see how observations change
        print(f"\n🎲 Testing Action Effects:")
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"   Step {i+1}:")
            print(f"      • Action: {action}")
            print(f"      • Reward: {reward}")
            print(f"      • Done/Truncated: {done}/{truncated}")
            
            if isinstance(obs, dict) and 'observation' in obs:
                print(f"      • Obs dims: {len(obs['observation'])}")
                # Look for what might be changing (block positions)
                if i == 0:
                    first_obs_vec = obs['observation'].copy()
                else:
                    diff_indices = np.where(np.abs(obs['observation'] - first_obs_vec) > 0.001)[0]
                    print(f"      • Changed dimensions: {diff_indices.tolist()}")
            
            if done or truncated:
                break
        
        env.close()
        
        return {
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space),
            'obs_dim': len(obs['observation']) if isinstance(obs, dict) else len(obs),
            'action_dim': env.action_space.shape[0],
            'sample_obs': obs
        }
        
    except Exception as e:
        print(f"❌ Error analyzing environment: {e}")
        return {'error': str(e)}

def compare_data_and_env(data_path: str):
    """Compare training data and environment structures"""
    print(f"\n⚖️  COMPARING DATA AND ENVIRONMENT")
    print("="*60)
    
    # Analyze both
    data_analysis = analyze_training_data(data_path)
    env_analysis = analyze_environment()
    
    print(f"\n📊 COMPARISON SUMMARY:")
    print("="*40)
    
    if 'error' not in env_analysis:
        print(f"Training Data Obs Dim: {data_analysis['obs_dim']}")
        print(f"Environment Obs Dim:   {env_analysis['obs_dim']}")
        print(f"Dimension Match:       {'✅ YES' if data_analysis['obs_dim'] == env_analysis['obs_dim'] else '❌ NO'}")
        print()
        
        print(f"Training Data Action Dim: {data_analysis['action_dim']}")
        print(f"Environment Action Dim:   {env_analysis['action_dim']}")
        print(f"Action Match:             {'✅ YES' if data_analysis['action_dim'] == env_analysis['action_dim'] else '❌ NO'}")
        print()
        
        # Recommendations
        print(f"🎯 RECOMMENDATIONS:")
        print("-" * 30)
        
        if data_analysis['obs_dim'] != env_analysis['obs_dim']:
            diff = env_analysis['obs_dim'] - data_analysis['obs_dim']
            print(f"❗ Observation dimension mismatch!")
            print(f"   Environment has {diff} more dimensions than training data.")
            
            if env_analysis['obs_dim'] == 19 and data_analysis['obs_dim'] == 15:
                print(f"   🤔 Possible causes:")
                print(f"      • Environment: 4-block setup (7 robot + 12 block positions)")
                print(f"      • Training: 2-block setup + 2D one-hot encoding")
                print(f"   💡 Solutions:")
                print(f"      • Use 4-block training data, OR")
                print(f"      • Use 2-block environment, OR")
                print(f"      • Truncate env obs to match training data")
            
            elif env_analysis['obs_dim'] == 13 and data_analysis['obs_dim'] == 15:
                print(f"   🤔 Possible causes:")
                print(f"      • Environment: 2-block setup (7 robot + 6 block positions)")
                print(f"      • Training: 2-block + 2D conditioning")
                print(f"   💡 Solution: Add conditioning to env obs during evaluation")
            
            else:
                print(f"   🤔 Unknown mismatch pattern. Manual investigation needed.")
        
        else:
            print(f"✅ Dimensions match! Data and environment are compatible.")
    
    return data_analysis, env_analysis

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect training data and environment structure')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--data_only', action='store_true', help='Only analyze data, skip environment')
    parser.add_argument('--env_only', action='store_true', help='Only analyze environment, skip data')
    
    args = parser.parse_args()
    
    if args.env_only:
        analyze_environment()
    elif args.data_only:
        analyze_training_data(args.data_path)
    else:
        compare_data_and_env(args.data_path)

if __name__ == '__main__':
    main()