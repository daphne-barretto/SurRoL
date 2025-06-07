"""
Image-based policy using convolutional neural networks for behavior cloning.
This extends the MLP_policy to handle image observations.

Implements:
    1. get_action
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy


class CNNPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines a CNN for supervised learning which maps image observations to actions

    Attributes
    ----------
    conv_layers: nn.Sequential
        A sequence of convolutional layers to process images
    fc_layers: nn.Sequential
        Fully connected layers that map from CNN features to actions
    mean_net: nn.Sequential
        Combined network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,  # For compatibility, but will use image_shape instead
                 image_shape=(3, 480, 640),  # (channels, height, width)
                 n_conv_layers=3,
                 conv_channels=[16, 32, 64],
                 conv_kernel_sizes=[5, 3, 3],
                 conv_strides=[2, 2, 2],
                 n_fc_layers=2,
                 fc_size=64,
                 learning_rate=1e-4,
                 training=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # Initialize variables for environment
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.image_shape = image_shape
        self.n_conv_layers = n_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.n_fc_layers = n_fc_layers
        self.fc_size = fc_size
        self.learning_rate = learning_rate
        self.training = training

        # Build convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = np.prod(dummy_output.shape[1:])
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers(flattened_size)
        
        # Combine into mean_net
        self.mean_net = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            self.fc_layers
        )
        
        self.mean_net.to(ptu.device)
        
        # Create logstd parameter for continuous actions
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def _build_conv_layers(self):
        """
        Builds the convolutional layers of the network
        """
        conv_layers = []
        in_channels = self.image_shape[0]  # First dimension is channels
        
        for i in range(self.n_conv_layers):
            out_channels = self.conv_channels[i]
            kernel_size = self.conv_kernel_sizes[i]
            stride = self.conv_strides[i]
            
            conv_layers.append(nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=kernel_size//2
            ))
            conv_layers.append(nn.ReLU())
            
            in_channels = out_channels
        
        return nn.Sequential(*conv_layers)
    
    def _build_fc_layers(self, input_size):
        """
        Builds the fully connected layers of the network
        """
        fc_layers = []
        
        # Input layer
        fc_layers.append(nn.Linear(input_size, self.fc_size))
        fc_layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(self.n_fc_layers - 1):
            fc_layers.append(nn.Linear(self.fc_size, self.fc_size))
            fc_layers.append(nn.ReLU())
        
        # Output layer
        fc_layers.append(nn.Linear(self.fc_size, self.ac_dim))
        
        return nn.Sequential(*fc_layers)

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save the model
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) == 3:  # Single image: (H, W, C)
            # Add batch dimension and convert to (B, C, H, W)
            observation = np.transpose(obs, (2, 0, 1))[None, ...]
        elif len(obs.shape) == 4:  # Batch of images: (B, H, W, C)
            # Convert to (B, C, H, W)
            observation = np.transpose(obs, (0, 3, 1, 2))
        else:
            # Handle non-image observations for compatibility
            observation = obs
            if len(obs.shape) == 1:
                observation = obs[None]
                
        # Convert to tensor
        observation = ptu.from_numpy(observation.astype(np.float32))
        
        # Forward pass and convert to numpy
        action = self(observation)
        return ptu.to_numpy(action)

    def forward(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # Check if observation is already in the right format for convolution
        if len(observation.shape) == 4 and observation.shape[1] == self.image_shape[0]:
            # Already in (B, C, H, W) format
            pass
        elif len(observation.shape) == 3 and observation.shape[0] == self.image_shape[0]:
            # Single image in (C, H, W) format, add batch dimension
            observation = observation.unsqueeze(0)
        elif len(observation.shape) == 3 and observation.shape[2] == self.image_shape[0]:
            # Single image in (H, W, C) format, convert to (B, C, H, W)
            observation = observation.permute(2, 0, 1).unsqueeze(0)
        elif len(observation.shape) == 4 and observation.shape[3] == self.image_shape[0]:
            # Batch of images in (B, H, W, C) format, convert to (B, C, H, W)
            observation = observation.permute(0, 3, 1, 2)
        
        # Get mean from the network
        mean = self.mean_net(observation)
        
        if self.training:
            # During training, we need the distribution for computing the loss
            std = torch.exp(self.logstd)
            action_distribution = distributions.Normal(mean, std)
            return action_distribution
        else:
            # During testing, just return the mean action
            return mean

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # Convert numpy arrays to tensors
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        
        # Get action distribution from policy
        action_distribution = self(observations)
        
        # Compute negative log likelihood loss
        loss = -action_distribution.log_prob(actions).sum(dim=-1).mean()
        
        # Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }