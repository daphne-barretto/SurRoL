"""
Train a behavior cloning policy with additional distance-based loss term
to explicitly optimize for distance reduction.
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

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure.logger import Logger

# Modify the EnhancedDictPolicy for distance-focused training
class DistanceFocusedPolicy(EnhancedDictPolicy):
    """
    Policy with additional distance-based loss term that
    explicitly optimizes for distance reduction
    """
    
    def update(self, obs, actions, distance_weight=1.0):
        """
        Update policy with BC loss and distance-based auxiliary loss
        
        Parameters
        ----------
        obs : np.ndarray
            Batch of observations
        actions : np.ndarray
            Batch of actions
        distance_weight : float
            Weight for the distance-based loss term
        """
        # Process observations
        processed_obs = self.process_observation(obs)
        
        # Convert to tensors
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
        actions_tensor = ptu.from_numpy(actions.astype(np.float32))
        
        # Get action distribution
        dist = self(obs_tensor)
        
        # Compute negative log likelihood loss (standard BC loss)
        log_probs = dist.log_prob(actions_tensor)
        bc_loss = -log_probs.sum(dim=-1).mean()
        
        # Compute distance-based loss if goal information is available
        distance_loss = torch.tensor(0.0, device=ptu.device)
        if self.use_goal and isinstance(obs[0], dict):
            for i in range(len(obs)):
                achieved_goal = ptu.from_numpy(obs[i]['achieved_goal'].astype(np.float32))
                desired_goal = ptu.from_numpy(obs[i]['desired_goal'].astype(np.float32))
                
                # Predicted action will move the end effector in a certain direction
                # We want this direction to align with the direction to the goal
                pred_action = dist.mean[i]
                
                # Direction to goal (simplified - this assumes actions directly relate to position)
                goal_direction = desired_goal - achieved_goal
                goal_direction = goal_direction / (goal_direction.norm() + 1e-8)  # Normalize
                
                # Direction of predicted action (first 3 components, assuming position control)
                action_direction = pred_action[:3]
                action_direction = action_direction / (action_direction.norm() + 1e-8)  # Normalize
                
                # Cosine similarity - we want to maximize this (minimize negative)
                cos_sim = (action_direction * goal_direction).sum()
                distance_loss = distance_loss - cos_sim
            
            distance_loss = distance_loss / len(obs)
        
        # Combined loss
        loss = bc_loss + distance_weight * distance_loss
        
        # Backpropagate and update with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'bc_loss': bc_loss.item(),
            'distance_loss': distance_loss.item()
        }

# Modified training function for distance-focused training
def train_distance_focused_bc(
    data_path,
    save_path,
    n_epochs=200,
    batch_size=64,
    learning_rate=5e-4,
    val_split=0.1,
    use_goal=True,
    goal_importance=2.0,
    distance_weight=1.0,
    n_layers=3,
    hidden_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_every=10,
    log_dir=None,
):
    """
    Train a behavior cloning agent with distance-focused loss
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
    
    # Create model
    policy = DistanceFocusedPolicy(
        ac_dim=5,  # Assuming 5D action space for NeedleReach
        ob_dim=ob_dim,
        use_goal=use_goal,
        goal_dim=goal_dim,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size,
        learning_rate=learning_rate
    )
    
    # Create logger
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"########################")
    print(f"Logging outputs to {log_dir}")
    print(f"########################")
    logger = Logger(log_dir)
    
    # TensorBoard writer
    try:
        import torch.utils.tensorboard as tensorboard
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = tensorboard.SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logs will be saved to {tensorboard_dir}")
        use_tensorboard = True
    except ImportError:
        print("TensorBoard not available, skipping TensorBoard logging")
        use_tensorboard = False
    
    # Training loop
    train_losses = []
    train_bc_losses = []
    train_distance_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Time tracking
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Display epoch progress
        print(f"Epoch {epoch+1}/{n_epochs} - Time elapsed: {time.time() - start_time:.1f}s")
        
        # Training
        policy.train()
        epoch_losses = []
        epoch_bc_losses = []
        epoch_distance_losses = []
        
        # Create batches
        n_batches = len(train_obs) // batch_size
        
        for i in tqdm(range(n_batches)):
            # Get batch
            batch_indices = np.random.choice(len(train_obs), batch_size, replace=False)
            obs_batch = train_obs[batch_indices]
            actions_batch = train_actions[batch_indices]
            
            # Update policy with distance-focused loss
            loss_dict = policy.update(obs_batch, actions_batch, distance_weight=distance_weight)
            
            epoch_losses.append(loss_dict['total_loss'])
            epoch_bc_losses.append(loss_dict['bc_loss'])
            epoch_distance_losses.append(loss_dict['distance_loss'])
        
        # Average training losses
        avg_train_loss = np.mean(epoch_losses)
        avg_train_bc_loss = np.mean(epoch_bc_losses)
        avg_train_distance_loss = np.mean(epoch_distance_losses)
        
        train_losses.append(avg_train_loss)
        train_bc_losses.append(avg_train_bc_loss)
        train_distance_losses.append(avg_train_distance_loss)
        
        # Validation
        policy.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_obs), batch_size):
                # Get batch
                obs_batch = val_obs[i:i+batch_size]
                actions_batch = val_actions[i:i+batch_size]
                
                # Compute validation loss
                processed_obs = policy.process_observation(obs_batch)
                obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
                actions_tensor = ptu.from_numpy(actions_batch.astype(np.float32))
                
                dist = policy(obs_tensor)
                val_loss = -dist.log_prob(actions_tensor).sum(dim=-1).mean().item()
                
                val_epoch_losses.append(val_loss)
        
        # Average validation loss
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        current_lr = policy.update_lr_scheduler(avg_val_loss)
        
        # Log
        print(f"Train loss: {avg_train_loss:.6f} (BC: {avg_train_bc_loss:.6f}, Dist: {avg_train_distance_loss:.6f})")
        print(f"Val loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        
        logger.log_scalar(avg_train_loss, "total_loss", epoch)
        logger.log_scalar(avg_train_bc_loss, "bc_loss", epoch)
        logger.log_scalar(avg_train_distance_loss, "distance_loss", epoch)
        logger.log_scalar(avg_val_loss, "val_loss", epoch)
        logger.log_scalar(current_lr, "learning_rate", epoch)
        
        # Log to TensorBoard
        if use_tensorboard:
            writer.add_scalar('train/total_loss', avg_train_loss, epoch)
            writer.add_scalar('train/bc_loss', avg_train_bc_loss, epoch)
            writer.add_scalar('train/distance_loss', avg_train_distance_loss, epoch)
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
            if patience_counter >= 20:  # Early stopping patience
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path), f"checkpoint_{epoch+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Plot learning curves
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot training components
    plt.subplot(2, 2, 2)
    plt.plot(train_bc_losses, label='BC Loss')
    plt.plot(train_distance_losses, label='Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    lrs = []
    for i, loss in enumerate(val_losses):
        lr = policy.optimizer.param_groups[0]['lr']
        policy.scheduler.step(loss)
        lrs.append(lr)
    
    plt.subplot(2, 2, 3)
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    # Display training configuration
    plt.subplot(2, 2, 4)
    plt.axis('off')
    config_text = f"""
    Training Configuration:
    
    Goal Importance: {goal_importance}
    Distance Weight: {distance_weight}
    Layers: {n_layers}
    Hidden Size: {hidden_size}
    Learning Rate: {learning_rate}
    Batch Size: {batch_size}
    Epochs: {epoch+1}
    
    Best Validation Loss: {best_val_loss:.6f}
    Training Time: {total_time:.2f}s
    """
    plt.text(0.1, 0.5, config_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(save_path), 'learning_curves.png')
    plt.savefig(plot_path)
    print(f"Learning curves saved to {plot_path}")
    
    # Close TensorBoard writer
    if use_tensorboard:
        writer.close()
    
    # Load best model for return
    policy.load_state_dict(torch.load(save_path))
    
    return policy

def main():
    parser = argparse.ArgumentParser(description='Train a distance-focused BC agent')
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
    parser.add_argument('--distance_weight', type=float, default=1.0,
                      help='Weight for distance-based loss term')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Size of hidden layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save model every this many epochs')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Path to save the final model
    save_path = os.path.join(args.save_dir, 'distance_focused_policy.pt')
    
    # Train agent
    policy = train_distance_focused_bc(
        data_path=args.data,
        save_path=save_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        use_goal=not args.no_goal,
        goal_importance=args.goal_importance,
        distance_weight=args.distance_weight,
        n_layers=args.layers,
        hidden_size=args.hidden_size,
        device=args.device,
        save_every=args.save_every,
    )
    
    print(f"Training complete! Best model saved to {save_path}")

if __name__ == '__main__':
    main()
