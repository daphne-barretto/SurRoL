import torch
import numpy as np
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

class PDControllerPolicy(EnhancedDictPolicy):
    """
    Policy wrapper that applies PD (Proportional-Derivative) control
    to smooth out the actions and create more stable trajectories
    """
    
    def __init__(
        self,
        base_policy_path,
        p_gain=0.8,     # Proportional gain
        d_gain=0.2,     # Derivative gain
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Load base policy
        self.load_state_dict(torch.load(base_policy_path, map_location=ptu.device))
        
        # PD controller parameters
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.last_action = None
        self.last_error = None
    
    def get_action(self, obs):
        """
        Get PD-controlled action for smoother, more stable trajectories
        """
        # Get target action from policy
        target_action = super().get_action(obs)
        
        # First time, just return the target
        if self.last_action is None:
            self.last_action = target_action
            self.last_error = np.zeros_like(target_action)
            return target_action
        
        # Calculate error (difference between target and current)
        error = target_action - self.last_action
        
        # Calculate error derivative
        error_derivative = error - self.last_error
        
        # PD control formula: output = P * error + D * error_derivative
        pd_output = self.p_gain * error + self.d_gain * error_derivative
        
        # Add PD output to last action for smoother transition
        new_action = self.last_action + pd_output
        
        # Update state
        self.last_action = new_action
        self.last_error = error
        
        return new_action
    
    def reset(self):
        """Reset the controller state"""
        self.last_action = None
        self.last_error = None
