"""
Train a behavior cloning agent on SurRoL NeedleReach data.
This script assumes the data has been converted to BC format using convert_npz_to_bc.py.
"""

import os
import time
import torch
import numpy as np
import pickle
import argparse
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.logger import Logger

class ConvPolicy(nn.Module):
    """
    CNN policy for image-based observations.
    Specifically designed for SurRoL environments with image observations of shape (480, 640, 3).
    """
    def __init__(
        self,
        ac_dim,
        image_shape=(3, 480, 640),
        n_conv_layers=3,
        conv_channels=[16, 32, 64],
        conv_kernel_sizes=[7, 5, 3],
        conv_strides=[4, 2, 2],
        n_fc_layers=2,
        fc_size=64,
        learning_rate=1e-4,
    ):
        super().__init__()
        
        self.ac_dim = ac_dim
        self.image_shape = image_shape
        self.learning_rate = learning_rate
        
        # Build convolutional layers
        conv_layers = []
        in_channels = image_shape[0]  # First dimension is channels
        
        for i in range(n_conv_layers):
            out_channels = conv_channels[i]
            kernel_size = conv_kernel_sizes[i]
            stride = conv_strides[i]
            
            conv_layers.append(nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=kernel_size//2
            ))
            conv_layers.append(nn.ReLU())
            
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            dummy_output = self.conv_net(dummy_input)
            flattened_size = np.prod(dummy_output.shape[1:])
        
        # Build fully connected layers
        fc_layers = []
        
        # First FC layer
        fc_layers.append(nn.Linear(flattened_size, fc_size))
        fc_layers.append(nn.ReLU())
        
        # Hidden FC layers
        for i in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(fc_size, fc_size))
            fc_layers.append(nn.ReLU())
        
        # Output layer
        fc_layers.append(nn.Linear(fc_size, ac_dim))
        
        self.fc_net = nn.Sequential(*fc_layers)
        
        # Logstd parameter for continuous actions
        self.logstd = nn.Parameter(torch.zeros(ac_dim))
        
        # Move to device
        self.to(ptu.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate
        )
        
    def forward(self, obs):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (batch_size, C, H, W)
            
        Returns
        -------
        mean : torch.Tensor
            Mean of the action distribution
        """
        # Pass through convolutional layers
        conv_out = self.conv_net(obs)
        
        # Flatten
        flattened = conv_out.reshape(conv_out.shape[0], -1)
        
        # Pass through fully connected layers
        mean = self.fc_net(flattened)
        
        return mean
    
    def get_action(self, obs):
        """
        Get action for a given observation.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation array of shape (H, W, C) or (batch_size, H, W, C)
            
        Returns
        -------
        action : np.ndarray
            Action array
        """
        # Handle single observation
        if len(obs.shape) == 3:
            # Add batch dimension: (H, W, C) -> (1, H, W, C)
            obs = obs[None]
            
        # Convert from (B, H, W, C) to (B, C, H, W)
        obs = np.transpose(obs, (0, 3, 1, 2))
        
        # Convert to tensor
        obs = ptu.from_numpy(obs.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        
        # Get mean action
        with torch.no_grad():
            mean = self(obs)
            
        # Convert to numpy
        action = ptu.to_numpy(mean)
        
        if action.shape[0] == 1:
            # Remove batch dimension for single observation
            action = action[0]
            
        return action
    
    def update(self, obs, actions):
        """
        Update the policy using supervised learning.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation array of shape (batch_size, H, W, C)
        actions : np.ndarray
            Action array of shape (batch_size, ac_dim)
            
        Returns
        -------
        loss : float
            Training loss
        """
        # Convert from (B, H, W, C) to (B, C, H, W)
        obs = np.transpose(obs, (0, 3, 1, 2))
        
        # Convert to tensors
        obs = ptu.from_numpy(obs.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        actions = ptu.from_numpy(actions.astype(np.float32))
        
        # Forward pass
        pred_actions = self(obs)
        
        # Compute loss (MSE)
        loss = torch.mean((pred_actions - actions) ** 2)
        
        # Backprop and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def train_bc_agent(
    data_path,
    save_path,
    n_epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    val_split=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ac_dim=5,
    conv_channels=[32, 64, 128],
    save_every=10,
):
    """
    Train a behavior cloning agent on SurRoL data.
    
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
    device : str
        Device to use for training
    ac_dim : int
        Action dimension
    conv_channels : list
        Number of channels in each convolutional layer
    save_every : int
        Save model every this many epochs
    """
    # Set device
    ptu.init_gpu(use_gpu=device=='cuda')
    
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
    print(f"Observations shape: {all_obs.shape}")
    print(f"Actions shape: {all_actions.shape}")
    
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
    
    # Create model
    policy = ConvPolicy(
        ac_dim=ac_dim,
        image_shape=(3, 480, 640),
        conv_channels=conv_channels,
        learning_rate=learning_rate,
    )
    
    # Create logger
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    log_path = os.path.join(os.path.dirname(save_path), 'logs')
    os.makedirs(log_path, exist_ok=True)
    logger = Logger(log_path)
    
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
                obs_batch_t = np.transpose(obs_batch, (0, 3, 1, 2))
                obs_tensor = ptu.from_numpy(obs_batch_t.astype(np.float32) / 255.0)
                actions_tensor = ptu.from_numpy(actions_batch.astype(np.float32))
                
                pred_actions = policy(obs_tensor)
                loss = torch.mean((pred_actions - actions_tensor) ** 2).item()
                
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
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate for optimization')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Fraction of data to use for validation')
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
        device=args.device,
        save_every=args.save_every,
    )
    
    print(f"Training complete! Best model saved to {save_path}")

if __name__ == '__main__':
    main()