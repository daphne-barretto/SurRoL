"""
Evaluate a trained behavior cloning policy on SurRoL environments.
Records videos of the policy in action.
"""

import os
import torch
import numpy as np
import argparse
import gymnasium as gym
from tqdm import tqdm
import imageio

from train_bc_surrol import ConvPolicy
from cs224r.infrastructure import pytorch_util as ptu

def evaluate_policy(
    model_path,
    env_name,
    num_episodes=10,
    max_steps=100,
    record_video=True,
    video_path=None,
    ac_dim=5,
    render_mode='rgb_array'
):
    """
    Evaluate a trained policy in a SurRoL environment.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model
    env_name : str
        Name of the environment
    num_episodes : int
        Number of episodes to evaluate
    max_steps : int
        Maximum number of steps per episode
    record_video : bool
        Whether to record videos of the policy in action
    video_path : str
        Path to save the videos
    ac_dim : int
        Action dimension
    render_mode : str
        Render mode for the environment
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptu.init_gpu(use_gpu=device.type=='cuda')
    
    # Load policy
    print(f"Loading policy from {model_path}...")
    policy = ConvPolicy(ac_dim=ac_dim)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    # Create environment
    print(f"Creating environment {env_name}...")
    env = gym.make(env_name, render_mode=render_mode)
    
    # Evaluate
    episode_returns = []
    episode_lengths = []
    
    # Create video directory if recording
    if record_video and video_path is not None:
        os.makedirs(video_path, exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0
        episode_length = 0
        
        # List to store frames if recording
        frames = []
        
        # Collect trajectory
        for step in range(max_steps):
            # Render if recording
            if record_video:
                frame = env.render()
                frames.append(frame)
            
            # Get action from policy
            action = policy.get_action(obs)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        # Record final frame if recording
        if record_video:
            frame = env.render()
            frames.append(frame)
        
        # Save video if recording
        if record_video and video_path is not None:
            video_file = os.path.join(video_path, f"episode_{episode+1}.mp4")
            imageio.mimsave(video_file, frames, fps=30)
            print(f"Video saved to {video_file}")
        
        # Record metrics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode return: {episode_return:.2f}, Episode length: {episode_length}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Mean episode return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_returns, episode_lengths

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained BC policy on SurRoL environments')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--env', type=str, required=True,
                      help='Name of the environment')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=100,
                      help='Maximum number of steps per episode')
    parser.add_argument('--no_video', action='store_true',
                      help='Disable video recording')
    parser.add_argument('--video_path', type=str, default='videos',
                      help='Path to save the videos')
    
    args = parser.parse_args()
    
    evaluate_policy(
        model_path=args.model,
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        record_video=not args.no_video,
        video_path=args.video_path
    )

if __name__ == '__main__':
    main()