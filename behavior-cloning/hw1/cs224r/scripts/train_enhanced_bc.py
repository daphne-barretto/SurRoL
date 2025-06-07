"""
Enhanced training script for behavior cloning with data augmentation,
curriculum learning, and better training practices.
"""

import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import datetime
import torch.utils.tensorboard as tensorboard

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure.logger import Logger

def add_noise_to_obs(obs, noise_level=0.01):
    """Add Gaussian noise to observation vectors for data augmentation"""
    if isinstance(obs, dict):
        # Create a copy of the dict to avoid modifying the original
        noisy_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                noise = np.random.normal(0, noise_level, size=value.shape)
                noisy_obs[key] = value + noise
            else:
                noisy_obs[key] = value
        return noisy_obs
    elif isinstance(obs, np.ndarray) and obs.dtype == np.dtype('O'):
        # Handle array of dictionaries
        noisy_obs = np.empty_like(obs)
        for i in range(len(obs)):
            noisy_obs[i] = add_noise_to_obs(obs[i], noise_level)
        return noisy_obs
    else:
        # Handle raw numpy array
        noise = np.random.normal(0, noise_level, size=obs.shape)
        return obs + noise

def add_noise_to_actions(actions, noise_level=0.01):
    """Add Gaussian noise to actions for data augmentation"""
    noise = np.random.normal(0, noise_level, size=actions.shape)
    return actions + noise

def augment_batch(obs_batch, actions_batch, noise_level=0.01):
    """Augment a batch of observations and actions with noise"""
    # Add noise to observations
    noisy_obs = add_noise_to_obs(obs_batch, noise_level)
    
    # Add noise to actions
    noisy_actions = add_noise_to_actions(actions_batch, noise_level)
    
    return noisy_obs, noisy_actions

def curriculum_sort_by_distance(paths):
    """
    Sort paths by goal difficulty (distance between achieved and desired goal)
    to implement curriculum learning
    """
    def get_avg_distance(path):
        # Calculate average distance between achieved and desired goals
        dists = []
        for obs_dict in path['observation']:
            if isinstance(obs_dict, dict):
                achieved = obs_dict['achieved_goal']
                desired = obs_dict['desired_goal']
                dist = np.linalg.norm(achieved - desired)
                dists.append(dist)
        return np.mean(dists) if dists else float('inf')
    
    # Sort paths by average goal distance
    return sorted(paths, key=get_avg_distance)

def evaluate_during_training(policy, env_name, num_episodes=5, max_steps=100):
    """
    Evaluate policy during training to track success rate
    """
    import gymnasium as gym
    import surrol.gym
    
    # Create environment
    env = gym.make(env_name)
    
    # Metrics to track
    success_count = 0
    returns = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0
        
        # Collect trajectory
        for step in range(max_steps):
            # Get action from policy
            action = policy.get_action(obs)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_return += reward
            
            if done or truncated:
                # Check if the episode was successful
                if 'is_success' in info and info['is_success']:
                    success_count += 1
                break
        
        returns.append(episode_return)
    
    # Calculate metrics
    success_rate = success_count / num_episodes
    mean_return = np.mean(returns)
    
    return success_rate, mean_return

def train_bc_agent(
    data_path,
    save_path,
    n_epochs=200,          # More epochs
    batch_size=64,          # Larger batch size
    learning_rate=5e-4,
    val_split=0.1,
    use_goal=True,
    goal_importance=2.0,    # Weight goal information more
    n_layers=3,             # Deeper network
    hidden_size=128,        # Wider network
    use_data_augmentation=True,
    noise_level=0.02,       # Noise level for data augmentation
    curriculum_learning=True,
    curriculum_phases=3,    # Number of curriculum phases
    early_stopping_patience=20,  # Early stopping patience
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ac_dim=5,
    save_every=10,
    log_dir=None,
    eval_interval=10,      # Evaluate every N epochs
    eval_episodes=5,       # Number of episodes for evaluation
    eval_env="NeedleReach-v0",  # Environment for evaluation
):
    """
    Enhanced training for behavior cloning with curriculum learning and data augmentation
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    ptu.init_gpu(use_gpu=device=='cuda')
    print(f"Using device: {ptu.device}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
    # Apply curriculum learning if enabled
    if curriculum_learning:
        print("Applying curriculum learning (sorting paths by difficulty)...")
        paths = curriculum_sort_by_distance(paths)
    
    # Combine all data
    all_obs = []
    all_actions = []
    
    for path in paths:
        all_obs.append(path['observation'])
        all_actions.append(path['action'])
    
    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Data loaded with {len(all_obs)} samples")
    print(f"Observations shape: {all_obs.shape}, dtype: {all_obs.dtype}")
    print(f"Actions shape: {all_actions.shape}, dtype: {all_actions.dtype}")
    
    # Split into train and validation sets
    n_samples = len(all_obs)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_obs = all_obs[train_indices]
    train_actions = all_actions[train_indices]
    
    val_obs = all_obs[val_indices]
    val_actions = all_actions[val_indices]
    
    print(f"Training samples: {len(train_obs)}")
    print(f"Validation samples: {len(val_obs)}")
    
    # Determine observation dimension
    sample_obs = all_obs[0]
    ob_dim = len(sample_obs['observation'])
    goal_dim = len(sample_obs['achieved_goal'])
    print(f"Observation dimension: {ob_dim}")
    print(f"Goal dimension: {goal_dim}")

    # Create logger
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"########################")
    print(f"Logging outputs to {log_dir}")
    print(f"########################")
    logger = Logger(log_dir)

    # Create TensorBoard writer
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs will be saved to {tensorboard_dir}")

    # Create model
    policy = EnhancedDictPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        use_goal=use_goal,
        goal_dim=goal_dim,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size,
        learning_rate=learning_rate
    )

    # Comment out TensorBoard graph addition - PyTorch distributions are not supported by the tracer
    # dummy_obs = policy.process_observation(all_obs[0:2])
    # dummy_obs_tensor = ptu.from_numpy(dummy_obs.astype(np.float32))
    # writer.add_graph(policy, dummy_obs_tensor)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Success tracking
    success_rates = []
    eval_returns = []
    eval_epochs = []
    
    # Learning curriculum phases
    if curriculum_learning:
        # Divide training into phases, gradually increasing difficulty
        curriculum_samples = []
        samples_per_phase = len(train_obs) // curriculum_phases
        
        for phase in range(curriculum_phases):
            start_idx = 0
            end_idx = (phase + 1) * samples_per_phase
            phase_samples = min(end_idx, len(train_obs))
            curriculum_samples.append(phase_samples)
    
    # Time tracking
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Update curriculum phase
        if curriculum_learning:
            curr_phase = min(epoch // (n_epochs // curriculum_phases), curriculum_phases - 1)
            n_curr_samples = curriculum_samples[curr_phase]
            print(f"Curriculum phase {curr_phase + 1}/{curriculum_phases}, using {n_curr_samples} samples")
            curr_train_obs = train_obs[:n_curr_samples]
            curr_train_actions = train_actions[:n_curr_samples]
        else:
            curr_train_obs = train_obs
            curr_train_actions = train_actions
        
        # Display epoch progress
        print(f"Epoch {epoch+1}/{n_epochs} - Time elapsed: {time.time() - start_time:.1f}s")
        
        # Training
        policy.train()
        epoch_losses = []
        
        # Create batches
        n_batches = len(curr_train_obs) // batch_size
        
        for i in tqdm(range(n_batches)):
            # Get batch
            batch_indices = np.random.choice(len(curr_train_obs), batch_size, replace=False)
            obs_batch = curr_train_obs[batch_indices]
            actions_batch = curr_train_actions[batch_indices]
            
            # Data augmentation
            if use_data_augmentation and np.random.random() < 0.5:
                obs_batch, actions_batch = augment_batch(
                    obs_batch, actions_batch, noise_level=noise_level
                )
            
            # Update policy
            loss = policy.update(obs_batch, actions_batch)
            epoch_losses.append(loss)
        
        # Average training loss
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        policy.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_obs), batch_size):
                # Get batch
                obs_batch = val_obs[i:i+batch_size]
                actions_batch = val_actions[i:i+batch_size]
                
                # Forward pass
                processed_obs = policy.process_observation(obs_batch)
                obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
                actions_tensor = ptu.from_numpy(actions_batch.astype(np.float32))
                
                dist = policy(obs_tensor)
                loss = -dist.log_prob(actions_tensor).sum(dim=-1).mean().item()
                
                val_epoch_losses.append(loss)
        
        # Average validation loss
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        current_lr = policy.update_lr_scheduler(avg_val_loss)
        
        # Log
        print(f"Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        logger.log_scalar(avg_train_loss, "train_loss", epoch)
        logger.log_scalar(avg_val_loss, "val_loss", epoch)
        logger.log_scalar(current_lr, "learning_rate", epoch)

        # Log to TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('validation/loss', avg_val_loss, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)

        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(policy.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path), f"checkpoint_{epoch+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Evaluate policy periodically
        if epoch % eval_interval == 0 or epoch == n_epochs - 1:
            print(f"Evaluating policy at epoch {epoch+1}...")
            success_rate, mean_return = evaluate_during_training(
                policy, eval_env, num_episodes=eval_episodes, max_steps=100
            )
            success_rates.append(success_rate)
            eval_returns.append(mean_return)
            eval_epochs.append(epoch)
            
            print(f"Success rate: {success_rate:.2f}, Mean return: {mean_return:.2f}")
            
            # Log to TensorBoard
            writer.add_scalar('evaluation/success_rate', success_rate, epoch)
            writer.add_scalar('evaluation/mean_return', mean_return, epoch)
            
            # Log to logger
            logger.log_scalar(success_rate, "success_rate", epoch)
            logger.log_scalar(mean_return, "eval_return", epoch)
    
    # Training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Plot learning curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(2, 2, 2)
    plt.plot(eval_epochs, success_rates, 'g-o')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.title('Success Rate During Training')
    plt.grid(True)
    
    # Plot evaluation returns
    plt.subplot(2, 2, 3)
    plt.plot(eval_epochs, eval_returns, 'm-o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Return')
    plt.title('Evaluation Returns')
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    if hasattr(policy, 'scheduler'):
        lrs = []
        for i, loss in enumerate(val_losses):
            lr = policy.optimizer.param_groups[0]['lr']
            policy.scheduler.step(loss)
            lrs.append(lr)
        
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(save_path), 'training_curves.png')
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    
    # Load best model for return
    policy.load_state_dict(torch.load(save_path))
    
    # Save model configuration for later evaluation
    model_config = {
        'ac_dim': ac_dim,
        'ob_dim': ob_dim,
        'use_goal': use_goal,
        'goal_dim': goal_dim,
        'goal_importance': goal_importance,
        'n_layers': n_layers,
        'size': hidden_size,
        'learning_rate': learning_rate
    }
    config_path = os.path.join(os.path.dirname(save_path), 'model_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model configuration saved to {config_path}")
    
    # Close TensorBoard writer
    writer.close()

    return policy

def main():
    parser = argparse.ArgumentParser(description='Train an enhanced behavior cloning agent')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the BC format data file')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                      help='Learning rate for optimization')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Fraction of data to use for validation')
    parser.add_argument('--no_goal', action='store_true',
                      help='Do not use goal information')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Size of hidden layers')
    parser.add_argument('--no_augmentation', action='store_true',
                      help='Disable data augmentation')
    parser.add_argument('--noise_level', type=float, default=0.02,
                      help='Noise level for data augmentation')
    parser.add_argument('--no_curriculum', action='store_true',
                      help='Disable curriculum learning')
    parser.add_argument('--patience', type=int, default=20,
                      help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save model every this many epochs')
    parser.add_argument('--no_timestamp', action='store_true',
                      help='Do not add timestamp to save directory')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Evaluate policy every N epochs')
    parser.add_argument('--eval_episodes', type=int, default=5,
                      help='Number of episodes for evaluation during training')
    parser.add_argument('--eval_env', type=str, default="NeedleReach-v0",
                      help='Environment to use for evaluation')

    
    args = parser.parse_args()
    
    
    # Add timestamp to save directory for tracking different runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_timestamp:
        args.save_dir = f"{args.save_dir}_{timestamp}"

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Path to save the final model
    save_path = os.path.join(args.save_dir, 'enhanced_bc_policy.pt')
    
    # Train agent
    policy = train_bc_agent(
        data_path=args.data,
        save_path=save_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        use_goal=not args.no_goal,
        goal_importance=args.goal_importance,
        n_layers=args.layers,
        hidden_size=args.hidden_size,
        use_data_augmentation=not args.no_augmentation,
        noise_level=args.noise_level,
        curriculum_learning=not args.no_curriculum,
        early_stopping_patience=args.patience,
        device=args.device,
        save_every=args.save_every,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_env=args.eval_env,
    )
    
    print(f"Training complete! Best model saved to {save_path}")

if __name__ == '__main__':
    main()
