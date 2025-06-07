"""
Train a behavior cloning agent on SurRoL data with dictionary observations.
"""

import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.dict_mlp_policy import DictMLPPolicy
from cs224r.infrastructure.logger import Logger

def train_bc_agent(
    data_path,
    save_path,
    n_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    val_split=0.1,
    use_goal=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ac_dim=5,
    save_every=10,
    log_dir=None,
):
    """
    Train a behavior cloning agent on data with dictionary observations.
    
    Parameters
    ----------
    data_path : str
        Path to the BC format data file
    save_path : str
        Path to save the trained model
    n_epochs : int
        Number of epochs to train
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimization
    val_split : float
        Fraction of data to use for validation
    use_goal : bool
        Whether to use goal information in the policy
    device : str
        Device to use for training
    ac_dim : int
        Action dimension
    save_every : int
        Save model every this many epochs
    log_dir : str
        Directory to save logs
    """
    # Set device
    ptu.init_gpu(use_gpu=device=='cuda')
    print(f"Using device: {ptu.device}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
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
    # Extract the first observation dictionary to get dimensions
    sample_obs = all_obs[0]
    ob_dim = len(sample_obs['observation'])
    goal_dim = len(sample_obs['achieved_goal'])
    print(f"Observation dimension: {ob_dim}")
    print(f"Goal dimension: {goal_dim}")
    
    # Create model
    policy = DictMLPPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        use_goal=use_goal,
        goal_dim=goal_dim,
        learning_rate=learning_rate,
    )
    
    # Create logger
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"########################")
    print(f"Logging outputs to {log_dir}")
    print(f"########################")
    logger = Logger(log_dir)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        # Training
        policy.train()
        epoch_losses = []
        
        # Create batches
        n_batches = len(train_obs) // batch_size
        
        for i in tqdm(range(n_batches)):
            # Get batch
            batch_indices = np.random.choice(len(train_obs), batch_size, replace=False)
            obs_batch = train_obs[batch_indices]
            actions_batch = train_actions[batch_indices]
            
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
        
        # Log
        print(f"Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")
        logger.log_scalar(avg_train_loss, "train_loss", epoch)
        logger.log_scalar(avg_val_loss, "val_loss", epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(policy.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path), f"checkpoint_{epoch+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(save_path), 'learning_curves.png')
    plt.savefig(plot_path)
    print(f"Learning curves saved to {plot_path}")
    
    # Load best model for evaluation
    policy.load_state_dict(torch.load(save_path))
    
    return policy

def main():
    parser = argparse.ArgumentParser(description='Train a behavior cloning agent on SurRoL data')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the BC format data file')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate for optimization')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Fraction of data to use for validation')
    parser.add_argument('--no_goal', action='store_true',
                      help='Do not use goal information')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save model every this many epochs')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Path to save the final model
    save_path = os.path.join(args.save_dir, 'bc_policy.pt')
    
    # Train agent
    policy = train_bc_agent(
        data_path=args.data,
        save_path=save_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        use_goal=not args.no_goal,
        device=args.device,
        save_every=args.save_every,
    )
    
    print(f"Training complete! Best model saved to {save_path}")

if __name__ == '__main__':
    main()
