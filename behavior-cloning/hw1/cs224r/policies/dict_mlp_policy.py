import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy

class DictMLPPolicy(BasePolicy, nn.Module):
    """
    MLP policy for dictionary observations with 'observation', 'achieved_goal', and 'desired_goal'
    """
    def __init__(
        self,
        ac_dim,
        ob_dim=None,  # Will be determined from data
        use_goal=True,  # Whether to use goal information
        n_layers=2,
        size=64,
        learning_rate=1e-4,
        **kwargs
    ):
        super().__init__()
        
        self.ac_dim = ac_dim
        self.use_goal = use_goal
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        
        # Determine input dimension from first observation if not provided
        if ob_dim is None:
            raise ValueError("ob_dim must be provided for DictMLPPolicy")
            
        # Input dimension is observation + (achieved_goal + desired_goal) if using goals
        self.ob_dim = ob_dim
        if use_goal:
            self.goal_dim = kwargs.get('goal_dim', 3)  # Default goal dimension is 3
            self.input_dim = ob_dim + self.goal_dim * 2  # observation + achieved_goal + desired_goal
        else:
            self.input_dim = ob_dim  # Just observation
        
        # Build MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(size, ac_dim))
        
        self.net = nn.Sequential(*layers)
        self.logstd = nn.Parameter(torch.zeros(ac_dim))
        
        # Move to device
        self.to(ptu.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def process_observation(self, obs_dict):
        """
        Process dictionary observation into a flat vector
        
        :param obs_dict: Dictionary with 'observation', 'achieved_goal', 'desired_goal'
        :return: Flat vector of observation (+ goals if use_goal=True)
        """
        if isinstance(obs_dict, np.ndarray) and obs_dict.dtype == np.dtype('O'):
            # Handle numpy array of objects (dictionaries)
            processed = []
            for i in range(len(obs_dict)):
                processed.append(self.process_observation(obs_dict[i]))
            return np.array(processed)
        
        # Process a single dictionary
        if isinstance(obs_dict, dict):
            if self.use_goal:
                return np.concatenate([
                    obs_dict['observation'], 
                    obs_dict['achieved_goal'], 
                    obs_dict['desired_goal']
                ])
            else:
                return obs_dict['observation']
        
        # Already processed or not a dictionary
        return obs_dict
    
    def forward(self, obs):
        """
        Forward pass through the network
        
        :param obs: Dictionary observation or processed observation
        :return: Action distribution
        """
        # Process observation if needed
        if isinstance(obs, np.ndarray) and obs.dtype == np.dtype('O'):
            obs = self.process_observation(obs)
        
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs.astype(np.float32))
        
        # Get action mean
        mean = self.net(obs)
        
        # Action distribution
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, obs):
        """
        Get action for a given observation
        
        :param obs: Dictionary observation or processed observation
        :return: Action
        """
        # Handle single observation vs batch
        is_single = not isinstance(obs, np.ndarray) or len(obs.shape) == 0
        
        if is_single:
            # Single observation, process and add batch dimension
            processed_obs = self.process_observation(obs)
            processed_obs = np.array([processed_obs])
        else:
            # Batch of observations
            processed_obs = self.process_observation(obs)
        
        # Convert to tensor
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
        
        # Get mean action
        with torch.no_grad():
            dist = self(obs_tensor)
            action = dist.mean
        
        # Convert to numpy
        action = ptu.to_numpy(action)
        
        # Remove batch dimension for single observation
        if is_single:
            action = action[0]
        
        return action
    
    def update(self, obs, actions):
        """
        Update policy parameters
        
        :param obs: Dictionary observations
        :param actions: Actions
        :return: Loss value
        """
        # Process observations
        processed_obs = self.process_observation(obs)
        
        # Convert to tensors
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
        actions_tensor = ptu.from_numpy(actions.astype(np.float32))
        
        # Get action distribution
        dist = self(obs_tensor)
        
        # Compute negative log likelihood loss
        loss = -dist.log_prob(actions_tensor).sum(dim=-1).mean()
        
        # Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
