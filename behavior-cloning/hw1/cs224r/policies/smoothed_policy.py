import torch
import numpy as np
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

class SmoothedPolicy(EnhancedDictPolicy):
    """Policy wrapper that applies action smoothing"""
    
    def __init__(
        self,
        base_policy_path,
        smoothing_factor=0.7,  # Higher = more smoothing
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Load base policy
        self.load_state_dict(torch.load(base_policy_path, map_location=ptu.device))
        
        # Smoothing parameters
        self.smoothing_factor = smoothing_factor
        self.last_action = None
    
    def get_action(self, obs):
        """Get smoothed action by interpolating with previous action"""
        # Get raw action from policy
        raw_action = super().get_action(obs)
        
        # Apply exponential smoothing if we have a previous action
        if self.last_action is not None:
            smoothed_action = self.smoothing_factor * self.last_action + (1 - self.smoothing_factor) * raw_action
        else:
            smoothed_action = raw_action
        
        # Update last action
        self.last_action = smoothed_action
        
        return smoothed_action
    
    def reset(self):
        """Reset the smoothing state"""
        self.last_action = None
