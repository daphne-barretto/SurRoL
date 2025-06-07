"""
Custom data preparation module for image-based observations.
This handles the format mentioned in your messages: (episodes, steps, <shape of thing>)
"""

import os
import numpy as np
import torch
import pickle
from cs224r.infrastructure import pytorch_util as ptu

class ImageDataProcessor:
    """
    A processor for image-based demonstration data.
    Converts data from the format (episodes, steps, <shape>) to the format
    expected by the BC agent.
    
    Methods
    -------
    process_data:
        Processes the data into observations and actions
    load_data:
        Loads data from a file
    convert_to_trajectories:
        Converts processed data to trajectories format for the BC agent
    """
    def __init__(self, use_images=True):
        """
        Initialize the processor
        
        Parameters
        ----------
        use_images : bool
            Whether to use image observations (True) or traditional state-based
            observations (False)
        """
        self.use_images = use_images
    
    def load_data(self, file_path):
        """
        Load data from a file
        
        Parameters
        ----------
        file_path : str
            Path to the data file
            
        Returns
        -------
        data : dict
            Dictionary containing the data
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def process_data(self, data):
        """
        Process the data into observations and actions
        
        Parameters
        ----------
        data : dict
            Dictionary containing the data with keys:
            - 'observations': image observations (episodes, steps, H, W, C)
            - 'traditional_obs': state-based observations
            - 'actions': actions
            - 'rewards': rewards
            
        Returns
        -------
        obs : np.ndarray
            Processed observations
        acts : np.ndarray
            Processed actions
        """
        # Choose between image and traditional observations
        if self.use_images:
            # Image observations: (episodes, steps, H, W, C)
            raw_obs = data['observations']
            # If needed, preprocess the images (resize, normalize, etc.)
            obs = self.preprocess_images(raw_obs)
        else:
            # Traditional observations: (episodes, steps, obs_dim)
            obs = data['traditional_obs']
        
        # Get actions
        acts = data['actions']
        
        # Flatten the episode dimension for both observations and actions
        # From (episodes, steps, *) to (episodes*steps, *)
        obs_dim = obs.shape[2:]
        act_dim = acts.shape[2:]
        
        # Reshape
        obs = obs.reshape(-1, *obs_dim)
        acts = acts.reshape(-1, *act_dim)
        
        return obs, acts
    
    def preprocess_images(self, images):
        """
        Preprocess images
        
        Parameters
        ----------
        images : np.ndarray
            Raw images of shape (episodes, steps, H, W, C)
            
        Returns
        -------
        processed_images : np.ndarray
            Processed images
        """
        # Normalize pixel values to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Additional preprocessing can be added here:
        # - Resize
        # - Crop
        # - Color transformations
        # - Data augmentation
        
        return images
    
    def convert_to_trajectories(self, data):
        """
        Convert the data to the trajectories format expected by the BC agent
        
        Parameters
        ----------
        data : dict
            Dictionary containing the data
            
        Returns
        -------
        paths : list
            List of dictionaries, each representing a trajectory
        """
        # Extract data
        observations = data['observations'] if self.use_images else data['traditional_obs']
        actions = data['actions']
        rewards = data['rewards']
        
        # Number of episodes
        num_episodes = observations.shape[0]
        
        paths = []
        for i in range(num_episodes):
            # Get data for this episode
            episode_obs = observations[i]
            episode_acts = actions[i]
            episode_rewards = rewards[i]
            
            # Calculate steps in this episode (excluding initial state if needed)
            # Note: Observations may include initial state (steps+1) while actions are just steps
            num_steps = len(episode_acts)
            
            # Create path dictionary
            path = {
                'observation': episode_obs[:num_steps],  # Match length with actions
                'action': episode_acts,
                'reward': episode_rewards[:num_steps],  # Match length with actions
                'next_observation': episode_obs[1:num_steps+1],  # Next states
                'terminal': np.zeros(num_steps, dtype=bool)  # All False except last one
            }
            
            # Set last step as terminal
            if num_steps > 0:
                path['terminal'][-1] = True
                
            paths.append(path)
            
        return paths

def load_and_process_data(data_path, use_images=True):
    """
    Helper function to load and process data
    
    Parameters
    ----------
    data_path : str
        Path to the data file
    use_images : bool
        Whether to use image observations
        
    Returns
    -------
    paths : list
        List of trajectories
    """
    processor = ImageDataProcessor(use_images=use_images)
    data = processor.load_data(data_path)
    paths = processor.convert_to_trajectories(data)
    return paths