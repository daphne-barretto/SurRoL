import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy

class EnhancedDictPolicy(BasePolicy, nn.Module):
    """
    Enhanced MLP policy for dictionary observations with improved architecture
    """
    def __init__(
        self,
        ac_dim,
        ob_dim=None,
        use_goal=True,
        n_layers=3,  # Increased from 2
        size=128,    # Increased from 64
        learning_rate=5e-4,  # Slightly higher learning rate
        weight_decay=1e-5,   # Added weight decay for regularization
        dropout=0.1,         # Added dropout for regularization
        activation='relu',   # Configurable activation function
        goal_importance=2.0, # Weight goal information more heavily
        **kwargs
    ):
        super().__init__()
        
        self.ac_dim = ac_dim
        self.use_goal = use_goal
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.goal_importance = goal_importance
        
        # Determine input dimension
        if ob_dim is None:
            raise ValueError("ob_dim must be provided")
            
        # Input dimension with weighted goal importance
        self.ob_dim = ob_dim
        if use_goal:
            self.goal_dim = kwargs.get('goal_dim', 3)
            self.input_dim = ob_dim + int(self.goal_dim * 2) 
        else:
            self.input_dim = ob_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build MLP with layer normalization and dropout
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, size),
            nn.LayerNorm(size),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            layer = nn.Sequential(
                nn.Linear(size, size),
                nn.LayerNorm(size),
                self.activation,
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(size, ac_dim)
        
        # Learnable log standard deviation
        self.logstd = nn.Parameter(torch.zeros(ac_dim))
        
        # Move to device
        self.to(ptu.device)
        
        # Set up optimizer with weight decay
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6
        )
    
    def process_observation(self, obs_dict):
        """Process dictionary observation into a flat vector with goal weighting"""
        if isinstance(obs_dict, np.ndarray) and obs_dict.dtype == np.dtype('O'):
            # Handle numpy array of objects (dictionaries)
            processed = []
            for i in range(len(obs_dict)):
                processed.append(self.process_observation(obs_dict[i]))
            return np.array(processed)
        
        # Process a single dictionary
        if isinstance(obs_dict, dict):
            if self.use_goal:
                # Apply goal importance weighting
                obs_part = obs_dict['observation']
                achieved_goal = obs_dict['achieved_goal']
                desired_goal = obs_dict['desired_goal']
                
                # Optionally scale the goal parts by importance factor
                if self.goal_importance != 1.0:
                    achieved_goal = achieved_goal * self.goal_importance
                    desired_goal = desired_goal * self.goal_importance
                
                return np.concatenate([obs_part, achieved_goal, desired_goal])
            else:
                return obs_dict['observation']
        
        # Already processed or not a dictionary
        return obs_dict
    
    def forward(self, obs):
        """Forward pass with residual connections"""
        # Process observation if needed
        if isinstance(obs, np.ndarray) and obs.dtype == np.dtype('O'):
            obs = self.process_observation(obs)
        
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs.astype(np.float32))
        
        # Input layer
        x = self.input_layer(obs)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Output layer for mean
        mean = self.output_layer(x)
        
        # Action distribution
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, obs):
        """Get action with exploration noise annealing"""
        # Handle single observation vs batch
        is_single = not isinstance(obs, np.ndarray) or len(obs.shape) == 0
        
        if is_single:
            processed_obs = self.process_observation(obs)
            processed_obs = np.array([processed_obs])
        else:
            processed_obs = self.process_observation(obs)
        
        # Convert to tensor
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
        
        # Get action distribution
        with torch.no_grad():
            dist = self(obs_tensor)
            
            # During evaluation, we can either:
            # 1. Take the mean action (deterministic)
            # action = dist.mean
            
            # 2. Sample from the distribution (stochastic)
            action = dist.sample()
        
        # Convert to numpy
        action = ptu.to_numpy(action)
        
        # Remove batch dimension for single observation
        if is_single:
            action = action[0]
        
        return action
    
    def update(self, obs, actions):
        """Update policy with loss scaling and gradient clipping"""
        # Process observations
        processed_obs = self.process_observation(obs)
        
        # Convert to tensors
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
        actions_tensor = ptu.from_numpy(actions.astype(np.float32))
        
        # Get action distribution
        dist = self(obs_tensor)
        
        # Compute negative log likelihood loss
        log_probs = dist.log_prob(actions_tensor)
        loss = -log_probs.sum(dim=-1).mean()
        
        # Backpropagate and update with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
        
    def update_lr_scheduler(self, val_loss):
        """Update learning rate scheduler based on validation loss"""
        self.scheduler.step(val_loss)
        return self.optimizer.param_groups[0]['lr']
