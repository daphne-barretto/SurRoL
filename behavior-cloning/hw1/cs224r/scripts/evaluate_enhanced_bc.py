"""
Enhanced evaluation script with more robust metrics and visualization.
"""

import os
import torch
import numpy as np
import argparse
import gymnasium as gym
from tqdm import tqdm
import imageio
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pickle

import surrol.gym
import gymnasium as gym
env = gym.make("NeedleReach-v0")

from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

def visualize_trajectory(frames, achieved_goals, desired_goals, file_path):
    """Create a visualization of the trajectory showing goal progress"""
    fig = plt.figure(figsize=(15, 7))
    
    # Plot the final frame
    ax1 = fig.add_subplot(121)
    if frames and len(frames) > 0:
        ax1.imshow(frames[-1])
        ax1.set_title('Final State')
        ax1.axis('off')
    
    # Plot the 3D trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    if achieved_goals and desired_goals:
        ax2.plot([g[0] for g in achieved_goals], [g[1] for g in achieved_goals], [g[2] for g in achieved_goals], 'b-', label='Trajectory')
        ax2.scatter(desired_goals[0][0], desired_goals[0][1], desired_goals[0][2], c='r', marker='*', s=100, label='Target')
        ax2.scatter(achieved_goals[0][0], achieved_goals[0][1], achieved_goals[0][2], c='g', marker='o', s=50, label='Start')
        ax2.scatter(achieved_goals[-1][0], achieved_goals[-1][1], achieved_goals[-1][2], c='b', marker='o', s=50, label='End')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('End-Effector Trajectory')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def evaluate_policy(
    model_path,
    env_name,
    ob_dim=7,
    goal_dim=3,
    ac_dim=5,
    num_episodes=10,
    max_steps=100,
    record_video=True,
    video_path=None,
    use_goal=True,
    goal_importance=2.0,
    visualize=True,
    render_mode='rgb_array',
    n_layers=3,
    hidden_size=128,
):
    """
    Enhanced evaluation with detailed metrics and better visualization.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptu.init_gpu(use_gpu=device.type=='cuda')
    print(f"Using device: {ptu.device}")
    
    # Load policy
    print(f"Loading policy from {model_path}...")
    
    # Try to load the model configuration first
    config_path = os.path.join(os.path.dirname(model_path), 'model_config.pkl')
    if os.path.exists(config_path):
        print(f"Found model configuration at {config_path}")
        with open(config_path, 'rb') as f:
            model_config = pickle.load(f)
            
        # Use the saved configuration parameters
        ac_dim = model_config.get('ac_dim', ac_dim)
        ob_dim = model_config.get('ob_dim', ob_dim)
        goal_dim = model_config.get('goal_dim', goal_dim)
        use_goal = model_config.get('use_goal', use_goal)
        goal_importance = model_config.get('goal_importance', goal_importance)
        n_layers = model_config.get('n_layers', n_layers)
        hidden_size = model_config.get('size', hidden_size)
        
        print(f"Using configuration from file: layers={n_layers}, hidden_size={hidden_size}, use_goal={use_goal}")
    else:
        print(f"No model configuration found at {config_path}, using default parameters")
    
    policy = EnhancedDictPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        goal_dim=goal_dim,
        use_goal=use_goal,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size
    )
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print("Policy loaded successfully")
    
    # Create environment
    print(f"Creating environment {env_name}...")
    env = gym.make(env_name, render_mode=render_mode)
    
    # Metrics to track
    episode_returns = []
    episode_lengths = []
    success_count = 0
    goal_distances = []
    initial_distances = []
    normalized_progress = []
    
    # Create directories if recording
    if record_video and video_path is not None:
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(os.path.join(video_path, 'trajectory_plots'), exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0
        episode_length = 0
        
        # Lists to store trajectory data
        frames = []
        achieved_goals = []
        desired_goals = []
        distances = []
        rewards = []
        
        # Collect trajectory
        for step in range(max_steps):
            # Get frame if recording
            if record_video:
                frame = env.render()
                frames.append(frame)
            
            # Store goal information
            if 'achieved_goal' in obs and 'desired_goal' in obs:
                achieved = obs['achieved_goal']
                desired = obs['desired_goal']
                achieved_goals.append(achieved)
                desired_goals.append(desired)
                
                # Calculate distance to goal
                dist = np.linalg.norm(achieved - desired)
                distances.append(dist)
            
            # Get action from policy
            action = policy.get_action(obs)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            
            episode_return += reward
            episode_length += 1

            if 'is_success' in info and info['is_success']:
                success_count += 1
                print(f"Episode successful!")
                done = True
                break
            
            # if done or truncated:
            #     # Check if the episode was successful
            #     if 'is_success' in info and info['is_success']:
            #         success_count += 1
            #         print(f"Episode successful!")
            #         break
            #     break
        
        # Get final frame if recording
        if record_video:
            frame = env.render()
            frames.append(frame)
        
        # Goal progress metrics
        if distances:
            initial_distance = distances[0]
            final_distance = distances[-1]
            
            initial_distances.append(initial_distance)
            goal_distances.append(final_distance)
            
            # Calculate normalized progress (0 = no progress, 1 = reached goal)
            progress = (initial_distance - final_distance) / initial_distance if initial_distance > 0 else 0
            normalized_progress.append(progress)
            
            print(f"Initial distance: {initial_distance:.4f}")
            print(f"Final distance: {final_distance:.4f}")
            print(f"Progress: {progress:.2f} ({progress*100:.1f}%)")
        
        # Save video if recording
        if record_video and video_path is not None and len(frames) > 0:
            # Save video
            video_file = os.path.join(video_path, f"episode_{episode+1}.mp4")
            try:
                # Ensure all frames have the same dimensions
                height, width = frames[0].shape[:2]
                processed_frames = []
                for f in frames:
                    if f.shape[:2] != (height, width):
                        f = cv2.resize(f, (width, height))
                    processed_frames.append(f)
                
                # Save video
                imageio.mimsave(video_file, processed_frames, fps=30)
                print(f"Video saved to {video_file}")
            except Exception as e:
                print(f"Error saving video: {e}")
            
            # Create trajectory visualization
            if visualize and achieved_goals and desired_goals:
                viz_file = os.path.join(video_path, 'trajectory_plots', f"trajectory_{episode+1}.png")
                visualize_trajectory(frames, achieved_goals, desired_goals, viz_file)
                print(f"Trajectory visualization saved to {viz_file}")
        
        # Record metrics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode return: {episode_return:.2f}, length: {episode_length}")
    
    # Compile and save all metrics
    metrics = {
        'returns': episode_returns,
        'lengths': episode_lengths,
        'goal_distances': goal_distances,
        'initial_distances': initial_distances,
        'normalized_progress': normalized_progress,
        'success_rate': success_count / num_episodes
    }
    
    # Create a summary plot of all episodes
    if len(goal_distances) > 0:
        plt.figure(figsize=(15, 10))
        
        # Progress plot
        plt.subplot(2, 2, 1)
        plt.bar(range(1, num_episodes+1), normalized_progress)
        plt.axhline(y=np.mean(normalized_progress), color='r', linestyle='-', label=f'Mean: {np.mean(normalized_progress):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Normalized Progress (0-1)')
        plt.title('Goal Progress by Episode')
        plt.legend()
        
        # Distance plot
        plt.subplot(2, 2, 2)
        for i in range(len(initial_distances)):
            plt.plot([1, 2], [initial_distances[i], goal_distances[i]], 'bo-', alpha=0.3)
        plt.plot([1, 2], [np.mean(initial_distances), np.mean(goal_distances)], 'ro-', linewidth=2, label='Mean')
        plt.xticks([1, 2], ['Initial', 'Final'])
        plt.ylabel('Distance to Goal')
        plt.title('Distance Reduction')
        plt.legend()
        
        # Returns plot
        plt.subplot(2, 2, 3)
        plt.bar(range(1, num_episodes+1), episode_returns)
        plt.axhline(y=np.mean(episode_returns), color='r', linestyle='-', label=f'Mean: {np.mean(episode_returns):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Returns by Episode')
        plt.legend()
        
        # Summary text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
        Evaluation Summary:
        
        Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)
        
        Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}
        Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}
        
        Mean Initial Distance: {np.mean(initial_distances):.4f}
        Mean Final Distance: {np.mean(goal_distances):.4f}
        Mean Distance Reduction: {np.mean(initial_distances) - np.mean(goal_distances):.4f}
        
        Mean Progress: {np.mean(normalized_progress)*100:.1f}%
        """
        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        if video_path:
            summary_path = os.path.join(video_path, 'evaluation_summary.png')
            plt.savefig(summary_path)
            print(f"Evaluation summary saved to {summary_path}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Mean episode return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    if len(goal_distances) > 0:
        print(f"Mean initial distance to goal: {np.mean(initial_distances):.4f}")
        print(f"Mean final distance to goal: {np.mean(goal_distances):.4f}")
        print(f"Mean progress towards goal: {np.mean(normalized_progress)*100:.1f}%")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained BC policy on SurRoL environments')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--env', type=str, required=True,
                      help='Name of the environment')
    parser.add_argument('--ob_dim', type=int, default=7,
                      help='Observation dimension')
    parser.add_argument('--goal_dim', type=int, default=3,
                      help='Goal dimension')
    parser.add_argument('--ac_dim', type=int, default=5,
                      help='Action dimension')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=100,
                      help='Maximum number of steps per episode')
    parser.add_argument('--no_video', action='store_true',
                      help='Disable video recording')
    parser.add_argument('--video_path', type=str, default='videos',
                      help='Path to save the videos')
    parser.add_argument('--no_goal', action='store_true',
                      help='Do not use goal information')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Size of hidden layers')
    
    args = parser.parse_args()
    
    evaluate_policy(
        model_path=args.model,
        env_name=args.env,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        ac_dim=args.ac_dim,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        record_video=not args.no_video,
        video_path=args.video_path,
        use_goal=not args.no_goal,
        goal_importance=args.goal_importance,
        n_layers=args.layers,
        hidden_size=args.hidden_size
    )

if __name__ == '__main__':
    main()
