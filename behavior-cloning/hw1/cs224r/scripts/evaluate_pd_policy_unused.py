"""
Evaluate a policy with PD control for smoother trajectories.
"""

import os
import torch
import numpy as np
import argparse
import gymnasium as gym

from cs224r.policies.pd_controller_policy import PDControllerPolicy
from cs224r.scripts.evaluate_enhanced_bc import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description='Evaluate a PD-controlled policy')
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
    parser.add_argument('--p_gain', type=float, default=0.8,
                      help='Proportional gain')
    parser.add_argument('--d_gain', type=float, default=0.2,
                      help='Derivative gain')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--video_path', type=str, default='videos',
                      help='Path to save the videos')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    
    args = parser.parse_args()
    
    # Create PD controller policy
    policy = PDControllerPolicy(
        base_policy_path=args.model,
        ac_dim=args.ac_dim,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        goal_importance=args.goal_importance,
        p_gain=args.p_gain,
        d_gain=args.d_gain
    )
    
    # Create save path for the PD policy
    model_dir = os.path.dirname(args.model)
    pd_model_path = os.path.join(model_dir, f"pd_p{args.p_gain}_d{args.d_gain}_policy.pt")
    torch.save(policy.state_dict(), pd_model_path)
    
    # Evaluate the PD policy
    pd_video_path = os.path.join(args.video_path, f"pd_p{args.p_gain}_d{args.d_gain}")
    
    metrics = evaluate_policy(
        model_path=pd_model_path,
        env_name=args.env,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        ac_dim=args.ac_dim,
        num_episodes=args.episodes,
        record_video=True,
        video_path=pd_video_path,
        use_goal=True,
        goal_importance=args.goal_importance
    )
    
    return metrics

if __name__ == '__main__':
    main()
