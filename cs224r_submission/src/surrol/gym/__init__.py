"""
SurRoL Gymnasium Environment Registration

This module handles the registration of SurRoL environments with gymnasium.
It includes a compatibility wrapper for the old gym-based environments.
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import importlib
import importlib.util
import os

# Import gymnasium and register function
import gymnasium as gym
from gymnasium.envs.registration import register


def convert_gym_space_to_gymnasium(space):
    """Convert gym.spaces to gymnasium.spaces"""
    import gym as old_gym
    
    if isinstance(space, old_gym.spaces.Box):
        return gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, old_gym.spaces.Dict):
        converted_spaces = {}
        for key, subspace in space.spaces.items():
            converted_spaces[key] = convert_gym_space_to_gymnasium(subspace)
        return gym.spaces.Dict(converted_spaces)
    elif isinstance(space, old_gym.spaces.Discrete):
        return gym.spaces.Discrete(space.n)
    elif isinstance(space, old_gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(space.nvec)
    else:
        # For other space types, try to create a generic Box space
        import numpy as np
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=space.shape if hasattr(space, 'shape') else (1,),
            dtype=np.float32
        )


class SurRoLGymnasiumWrapper(gym.Env):
    """Gymnasium compatibility wrapper for SurRoL environments"""
    
    def __init__(self, env_class, **kwargs):
        # Initialize the original environment
        self._env = env_class(**kwargs)
        
        # Convert action space to gymnasium format
        self.action_space = convert_gym_space_to_gymnasium(self._env.action_space)
        
        # Copy metadata
        self.metadata = getattr(self._env, 'metadata', {})
        
        # Set a temporary observation space (we'll update it after first reset)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,), dtype=float),
            'achieved_goal': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=float),
            'desired_goal': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=float),
        })
        self._first_reset = True
    
    def _update_observation_space_from_obs(self, obs):
        """Update observation space based on actual observation structure"""
        if isinstance(obs, dict):
            spaces = {}
            for key, value in obs.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    spaces[key] = gym.spaces.Box(
                        low=-float('inf'),
                        high=float('inf'), 
                        shape=value.shape,
                        dtype=value.dtype
                    )
                else:
                    # Scalar value
                    spaces[key] = gym.spaces.Box(
                        low=-float('inf'),
                        high=float('inf'),
                        shape=(1,),
                        dtype=float
                    )
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            # Non-dict observation
            self.observation_space = gym.spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=obs.shape,
                dtype=obs.dtype
            )
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # Convert to gymnasium format (add truncated)
        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        
        # Update observation space based on actual observation on first reset
        if self._first_reset:
            self._update_observation_space_from_obs(obs)
            self._first_reset = False
        
        return obs, {}
    
    def render(self, mode='human'):
        return self._env.render(mode)
    
    def close(self):
        return self._env.close()
    
    def seed(self, seed=None):
        return self._env.seed(seed)


def create_env_factory(entry_point):
    """Create a factory function for environment registration"""
    if ':' in entry_point:
        # Standard module:class format
        module_path, class_name = entry_point.split(':')
        
        def env_factory(**kwargs):
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)
            return SurRoLGymnasiumWrapper(env_class, **kwargs)
            
    else:
        # File path format (for hyphenated files)
        def env_factory(**kwargs):
            # Handle different path formats
            if entry_point.startswith('/'):
                # Absolute path 
                full_path = entry_point
            elif entry_point.startswith('SurRoL/'):
                # Relative path from project root - use absolute path
                import os
                # __file__ is in SurRoL/surrol/gym/__init__.py, so go up 3 levels to get to project root
                current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                full_path = os.path.join(current_dir, entry_point)
            else:
                # Just filename - assume it's in tasks directory
                full_path = os.path.join('surrol', 'tasks', entry_point)
            
            spec = importlib.util.spec_from_file_location(
                f"surrol_env", full_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            env_class = module.PegTransfer
            return SurRoLGymnasiumWrapper(env_class, **kwargs)
    
    return env_factory


# Register all SurRoL environments with gymnasium compatibility
environments = [
    # Basic environments
    ('PegTransfer-v0', 'surrol.tasks.peg_transfer:PegTransfer'),
    ('PegTransferTargetBlock-v0', 'surrol.tasks.peg_transfer_target_block:PegTransfer'),
    ('PegTransferTargetBlockTargetPeg-v0', 'surrol.tasks.peg_transfer_target_block_and_target_peg:PegTransfer'),
    ('PegTransferOneHotTargetPeg-v0', 'surrol.tasks.peg_transfer_onehot_and_target_peg:PegTransfer'),
    ('PegTransferFourTuple-v0', 'surrol.tasks.peg_transfer_four_tuple:PegTransfer'),
    
    # Colored environments
    ('PegTransferColor-v0', 'surrol.tasks.peg_transfer_with_all_blocks_colored:PegTransfer'),
    ('PegTransferColorTargetBlock-v0', 'surrol.tasks.peg_transfer_with_all_blocks_colored_target_block:PegTransfer'),
    ('PegTransferColorTargetBlockTargetPeg-v0', 'surrol.tasks.peg_transfer_with_all_blocks_colored_target_block_peg:PegTransfer'),
    ('PegTransferColorOneHotTargetPeg-v0', 'surrol.tasks.peg_transfer_with_all_blocks_colored_one_hot_target_peg:PegTransfer'),
    
    # Target block only
    ('PegTransferTBO-v0', 'surrol.tasks.peg_transfer_with_only_target_block:PegTransfer'),
    
    # Two blocks with target block only (NEW)
    ('PegTransferTwoBlocksTargetBlockOnly-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_only_target_block.py'),
    ('PegTransferTwoBlocksTargetBlockOnlyNoColor-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_only_target_block-no_color.py'),
    
    # Two blocks environments - colored versions (existing)
    ('PegTransferTwoBlocksOneHot-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks_colored-no_obs.py'),
    ('PegTransferTwoBlocksFourTuple-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks_colored-no_obs_four_tuple.py'),
    ('PegTransferTwoBlocksOneHotTargetPeg-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks_colored-onehottargetpeg.py'),
    ('PegTransferTwoBlocksTargetBlockTargetPeg-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks_colored-targetblocktargetpeg.py'),
    
    # Two blocks environments - NO COLOR versions (new)
    ('PegTransferTwoBlocksNoColor-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs.py'),
    ('PegTransferTwoBlocksNoColorOneHot-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_onehot.py'),
    ('PegTransferTwoBlocksNoColorTargetBlock-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_target_block.py'),
    ('PegTransferTwoBlocksNoColorTargetPeg-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_target_peg.py'),
    ('PegTransferTwoBlocksNoColorTargetBlockPeg-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_target_block_peg.py'),
    ('PegTransferTwoBlocksNoColorOneHotTargetPeg-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_one_hot_target_peg.py'),
    ('PegTransferTwoBlocksNoColorFourTuple-v0', 'SurRoL/surrol/tasks/peg_transfer-two_blocks-with_all_blocks-no_obs_four_tuple.py'),
    
    # Other task environments
    ('NeedleReach-v0', 'surrol.tasks.needle_reach:NeedleReach'),
    ('GauzeRetrieve-v0', 'surrol.tasks.gauze_retrieve:GauzeRetrieve'),
    ('NeedlePick-v0', 'surrol.tasks.needle_pick:NeedlePick'),
]

# Register all environments
print(f"Starting registration of {len(environments)} environments...")
for env_id, entry_point in environments:
    try:
        print(f"Registering {env_id} with entry_point: {entry_point}")
        register(
            id=env_id,
            entry_point=create_env_factory(entry_point),
            max_episode_steps=50,
        )
        print(f"✅ Registered: {env_id}")
    except Exception as e:
        print(f"❌ Failed to register {env_id}: {e}")
        import traceback
        traceback.print_exc()

print(f"✅ SurRoL environments registration complete")