"""
Evaluate a policy with action smoothing for more stable trajectories.
"""

import os
import torch
import numpy as np
import argparse
import gymnasium as gym

from cs224r.policies.smoothed_policy import SmoothedPolicy
from cs224r.scripts.evaluate_enhanced_bc import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description='Evaluate a smoothed policy')
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
    parser.add_argument('--smoothing', type=float, default=0.7,
                      help='Smoothing factor (0-1, higher = more smoothing)')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--video_path', type=str, default='videos',
                      help='Path to save the videos')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    
    args = parser.parse_args()
    
    # Create smoothed policy
    policy = SmoothedPolicy(
        base_policy_path=args.model,
        ac_dim=args.ac_dim,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        goal_importance=args.goal_importance,
        smoothing_factor=args.smoothing
    )
    
    # Create save path for the smoothed policy
    model_dir = os.path.dirname(args.model)
    smooth_model_path = os.path.join(model_dir, f"smoothed_{args.smoothing}_policy.pt")
    torch.save(policy.state_dict(), smooth_model_path)
    
    # Evaluate the smoothed policy
    smoothed_video_path = os.path.join(args.video_path, f"smoothed_{args.smoothing}")
    
    metrics = evaluate_policy(
        model_path=smooth_model_path,
        env_name=args.env,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        ac_dim=args.ac_dim,
        num_episodes=args.episodes,
        record_video=True,
        video_path=smoothed_video_path,
        use_goal=True,
        goal_importance=args.goal_importance
    )
    
    return metrics

if __name__ == '__main__':
    main()
