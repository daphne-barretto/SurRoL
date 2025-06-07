"""
DAgger (Dataset Aggregation) implementation for iterative policy improvement.
"""

import os
import torch
import numpy as np
import pickle
import argparse
import gymnasium as gym
from tqdm import tqdm
import time

from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu
from cs224r.scripts.train_enhanced_bc import train_bc_agent

def collect_rollouts(
    policy,
    env_name,
    num_episodes=10,
    max_steps=50,
    render=False
):
    """
    Collect rollouts using the current policy in the environment.
    
    Parameters
    ----------
    policy : EnhancedDictPolicy
        Policy to collect rollouts with
    env_name : str
        Name of the environment
    num_episodes : int
        Number of episodes to collect
    max_steps : int
        Maximum number of steps per episode
    render : bool
        Whether to render the environment
        
    Returns
    -------
    paths : list
        List of trajectories
    """
    # Create environment
    env = gym.make(env_name, render_mode='rgb_array' if render else None)
    
    # Collect rollouts
    paths = []
    
    for episode in range(num_episodes):
        print(f"Collecting episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Initialize lists to store trajectory data
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        infos = []
        
        # Collect trajectory
        for step in range(max_steps):
            # Render if specified
            if render:
                env.render()
            
            # Store current observation
            observations.append(obs)
            
            # Get action from policy
            action = policy.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store data
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(done or truncated)
            infos.append(info)
            
            # Update observation
            obs = next_obs
            
            if done or truncated:
                break
        
        # Create path dictionary
        path = {
            'observation': np.array(observations),
            'action': np.array(actions),
            'reward': np.array(rewards),
            'next_observation': np.array(next_observations),
            'terminal': np.array(terminals),
            'info': infos
        }
        paths.append(path)
    
    return paths

def label_data_with_expert(
    paths,
    expert_policy,
    relabel_actions=True,
):
    """
    Label data with an expert policy.
    
    Parameters
    ----------
    paths : list
        List of trajectories
    expert_policy : function or object
        Expert policy to label data with. Should take an observation and return an action.
    relabel_actions : bool
        Whether to replace actions with expert actions
        
    Returns
    -------
    labeled_paths : list
        List of trajectories with expert labels
    """
    print("Labeling data with expert policy...")
    
    labeled_paths = []
    
    for path in tqdm(paths):
        # Get observations
        observations = path['observation']
        
        # Get expert actions
        expert_actions = []
        for obs in observations:
            expert_action = expert_policy.get_action(obs)
            expert_actions.append(expert_action)
        
        # Create new path
        labeled_path = path.copy()
        
        # Replace actions with expert actions if specified
        if relabel_actions:
            labeled_path['action'] = np.array(expert_actions)
        
        labeled_paths.append(labeled_path)
    
    return labeled_paths

def run_dagger(
    initial_policy_path,
    expert_policy_path,
    env_name,
    data_path,
    save_dir,
    n_dagger_iterations=5,
    n_rollouts_per_iter=10,
    max_steps_per_rollout=50,
    n_epochs=50,
    batch_size=64,
    learning_rate=1e-4,
    ob_dim=7,
    goal_dim=3,
    ac_dim=5,
    use_goal=True,
    goal_importance=2.0,
    n_layers=3,
    hidden_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Run DAgger algorithm.
    
    Parameters
    ----------
    initial_policy_path : str
        Path to the initial policy
    expert_policy_path : str
        Path to the expert policy
    env_name : str
        Name of the environment
    data_path : str
        Path to the initial dataset
    save_dir : str
        Directory to save the trained models and data
    n_dagger_iterations : int
        Number of DAgger iterations
    n_rollouts_per_iter : int
        Number of rollouts to collect per iteration
    max_steps_per_rollout : int
        Maximum number of steps per rollout
    n_epochs : int
        Number of epochs to train per iteration
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimization
    ob_dim : int
        Observation dimension
    goal_dim : int
        Goal dimension
    ac_dim : int
        Action dimension
    use_goal : bool
        Whether to use goal information
    goal_importance : float
        Weight for goal information
    n_layers : int
        Number of hidden layers
    hidden_size : int
        Size of hidden layers
    device : str
        Device to use for training
    """
    # Set device
    ptu.init_gpu(use_gpu=device=='cuda')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load initial dataset
    print(f"Loading initial dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
    # Load initial policy
    print(f"Loading initial policy from {initial_policy_path}...")
    policy = EnhancedDictPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        goal_dim=goal_dim,
        use_goal=use_goal,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size
    )
    policy.load_state_dict(torch.load(initial_policy_path, map_location=device))
    policy.eval()
    
    # Load expert policy
    print(f"Loading expert policy from {expert_policy_path}...")
    expert_policy = EnhancedDictPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        goal_dim=goal_dim,
        use_goal=use_goal,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size
    )
    expert_policy.load_state_dict(torch.load(expert_policy_path, map_location=device))
    expert_policy.eval()
    
    # Run DAgger iterations
    for i in range(n_dagger_iterations):
        print(f"\n=== DAgger Iteration {i+1}/{n_dagger_iterations} ===")
        iter_start_time = time.time()
        
        # Collect rollouts with current policy
        print(f"Collecting {n_rollouts_per_iter} rollouts with current policy...")
        new_paths = collect_rollouts(
            policy=policy,
            env_name=env_name,
            num_episodes=n_rollouts_per_iter,
            max_steps=max_steps_per_rollout,
            render=False
        )
        
        # Label data with expert policy
        labeled_paths = label_data_with_expert(
            paths=new_paths,
            expert_policy=expert_policy,
            relabel_actions=True
        )
        
        # Combine with existing data
        combined_paths = paths + labeled_paths
        print(f"Combined dataset size: {len(combined_paths)} trajectories")
        
        # Save combined dataset
        combined_data_path = os.path.join(save_dir, f"dagger_data_iter_{i+1}.pkl")
        with open(combined_data_path, 'wb') as f:
            pickle.dump(combined_paths, f)
        print(f"Saved combined dataset to {combined_data_path}")
        
        # Train policy on combined dataset
        print(f"Training policy on combined dataset...")
        save_path = os.path.join(save_dir, f"dagger_policy_iter_{i+1}.pt")
        policy = train_bc_agent(
            data_path=combined_data_path,
            save_path=save_path,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_split=0.1,
            use_goal=use_goal,
            goal_importance=goal_importance,
            n_layers=n_layers,
            hidden_size=hidden_size,
            device=device,
        )
        
        # Update paths for next iteration
        paths = combined_paths
        
        # Iteration timing
        iter_time = time.time() - iter_start_time
        print(f"DAgger iteration {i+1} completed in {iter_time:.2f} seconds")
    
    print("\nDAgger complete!")
    final_policy_path = os.path.join(save_dir, "final_dagger_policy.pt")
    torch.save(policy.state_dict(), final_policy_path)
    print(f"Final policy saved to {final_policy_path}")
    
    return policy, paths

def main():
    parser = argparse.ArgumentParser(description='Run DAgger algorithm on SurRoL environments')
    parser.add_argument('--initial_policy', type=str, required=True,
                      help='Path to the initial policy')
    parser.add_argument('--expert_policy', type=str, required=True,
                      help='Path to the expert policy')
    parser.add_argument('--env', type=str, required=True,
                      help='Name of the environment')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the initial dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the trained models and data')
    parser.add_argument('--dagger_iters', type=int, default=5,
                      help='Number of DAgger iterations')
    parser.add_argument('--rollouts_per_iter', type=int, default=10,
                      help='Number of rollouts to collect per iteration')
    parser.add_argument('--steps_per_rollout', type=int, default=50,
                      help='Maximum number of steps per rollout')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train per iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate for optimization')
    parser.add_argument('--ob_dim', type=int, default=7,
                      help='Observation dimension')
    parser.add_argument('--goal_dim', type=int, default=3,
                      help='Goal dimension')
    parser.add_argument('--ac_dim', type=int, default=5,
                      help='Action dimension')
    parser.add_argument('--no_goal', action='store_true',
                      help='Do not use goal information')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Size of hidden layers')
    
    args = parser.parse_args()
    
    run_dagger(
        initial_policy_path=args.initial_policy,
        expert_policy_path=args.expert_policy,
        env_name=args.env,
        data_path=args.data,
        save_dir=args.save_dir,
        n_dagger_iterations=args.dagger_iters,
        n_rollouts_per_iter=args.rollouts_per_iter,
        max_steps_per_rollout=args.steps_per_rollout,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ob_dim=args.ob_dim,
        goal_dim=args.goal_dim,
        ac_dim=args.ac_dim,
        use_goal=not args.no_goal,
        goal_importance=args.goal_importance,
        n_layers=args.layers,
        hidden_size=args.hidden_size,
    )

if __name__ == '__main__':
    main()
