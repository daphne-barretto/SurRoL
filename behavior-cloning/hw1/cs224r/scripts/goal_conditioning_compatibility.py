"""
🎯 Goal Conditioning Compatibility Layer

This module provides a unified interface for all goal conditioning formats,
ensuring compatibility across training, evaluation, and visualization scripts.

GOAL CONDITIONING FORMATS EXPLAINED:
=====================================

1. 'none' - No goal conditioning (baseline)
   - Just uses raw observations
   - Good baseline to compare against
   - Useful when task doesn't need goal specification

2. 'one_hot' - Block color as one-hot vector [1,0,0,0] for red, etc.
   - Categorical representation of block identity
   - Efficient for discrete color choices
   - Standard ML approach for categorical data

3. 'target_block' - Position of target block (3D coordinates)
   - Spatial representation of what to manipulate
   - Helps robot understand "what" to pick up
   - Good for position-based reasoning

4. 'target_block_and_target_peg' - Target block position + target peg position
   - Complete spatial specification: "pick this, place there"
   - Most information-rich format
   - Best for complex manipulation tasks

5. 'one_hot_and_target_peg' - Block color + target peg position
   - Hybrid: categorical "what" + spatial "where"
   - Good balance of efficiency and information
   - Separates object identity from goal location

6. 'four_tuple' - Block color as RGBA tuple (1.0, 0.0, 0.0, 1.0) for red
   - Continuous color representation
   - Mimics visual processing
   - Can handle color variations/interpolations

7. 'color_language' - Block color as text string "red", "green", etc.
   - Natural language representation
   - Good for language-conditioned policies
   - Human-interpretable format

Each format has different advantages:
- Categorical (one_hot): Efficient, discrete
- Spatial (target_block): Position-aware
- Hybrid (combinations): Multi-modal information
- Continuous (four_tuple): Smooth representation
- Language (color_language): Human-readable
"""

import numpy as np
import torch
import os
import pickle
import json
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

# Suppress verbose logs
logging.basicConfig(level=logging.WARNING)

class GoalConditioningManager:
    """
    Unified manager for all goal conditioning formats
    """
    
    # Define all supported formats
    SUPPORTED_FORMATS = [
        'none',
        'one_hot', 
        'target_block',
        'target_block_and_target_peg',
        'one_hot_and_target_peg', 
        'four_tuple',
        'color_language'
    ]
    
    # Color mappings for different formats
    COLOR_MAPPINGS = {
        'one_hot': {
            'red': [1.0, 0.0, 0.0, 0.0],
            'green': [0.0, 1.0, 0.0, 0.0],
            'blue': [0.0, 0.0, 1.0, 0.0],
            'yellow': [0.0, 0.0, 0.0, 1.0]
        },
        'four_tuple': {
            'red': [1.0, 0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0],
            'yellow': [1.0, 1.0, 0.0, 1.0]
        },
        'color_language': {
            'red': 'red',
            'green': 'green', 
            'blue': 'blue',
            'yellow': 'yellow'
        }
    }
    
    @classmethod
    def get_conditioning_dimension(cls, condition_type: str) -> int:
        """Get the dimension added by each conditioning type"""
        dimension_map = {
            'none': 0,
            'one_hot': 4,  # 4 colors
            'target_block': 3,  # 3D position
            'target_block_and_target_peg': 6,  # 3D + 3D positions
            'one_hot_and_target_peg': 7,  # 4 + 3 dimensions
            'four_tuple': 4,  # RGBA values
            'color_language': 1  # Will be embedded/encoded
        }
        return dimension_map.get(condition_type, 0)
    
    @classmethod
    def apply_conditioning(cls, observation: np.ndarray, obs_dict: Dict[str, Any], 
                          condition_type: str) -> np.ndarray:
        """
        Apply goal conditioning to observation based on format
        
        Args:
            observation: Raw observation array
            obs_dict: Full observation dictionary with all fields
            condition_type: Type of conditioning to apply
            
        Returns:
            Conditioned observation array
        """
        if condition_type == 'none':
            return observation
            
        # Get conditioning components
        block_encoding = obs_dict.get('block_encoding', np.zeros(4))
        achieved_goal = obs_dict.get('achieved_goal', np.zeros(3))
        desired_goal = obs_dict.get('desired_goal', np.zeros(3))
        
        if condition_type == 'one_hot':
            return np.hstack([observation, block_encoding])
            
        elif condition_type == 'target_block':
            return np.hstack([observation, achieved_goal])
            
        elif condition_type == 'target_block_and_target_peg':
            return np.hstack([observation, achieved_goal, desired_goal])
            
        elif condition_type == 'one_hot_and_target_peg':
            return np.hstack([observation, block_encoding, desired_goal])
            
        elif condition_type == 'four_tuple':
            # Convert one-hot to RGBA
            color_idx = np.argmax(block_encoding)
            color_names = ['red', 'green', 'blue', 'yellow']
            color_name = color_names[color_idx]
            rgba = cls.COLOR_MAPPINGS['four_tuple'][color_name]
            return np.hstack([observation, rgba])
            
        elif condition_type == 'color_language':
            # For now, just use one-hot encoding (language would need special handling)
            # In practice, this would be embedded into a vector
            return np.hstack([observation, block_encoding])
            
        else:
            raise ValueError(f"Unsupported condition type: {condition_type}")
    
    @classmethod
    def calculate_total_obs_dim(cls, base_obs_dim: int, condition_type: str, 
                               use_goal: bool = True) -> int:
        """Calculate total observation dimension after conditioning"""
        if not use_goal or condition_type == 'none':
            return base_obs_dim
        return base_obs_dim + cls.get_conditioning_dimension(condition_type)
    
    @classmethod
    def validate_conditioning_compatibility(cls, data_path: str, condition_type: str) -> Dict[str, Any]:
        """
        Validate that data supports the specified conditioning type
        """
        import numpy as np
        
        # Load data
        data = np.load(data_path, allow_pickle=True)
        obs_data = data['obs']
        
        # Check first observation
        sample_obs = obs_data[0][0]
        
        validation_result = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'sample_obs_keys': list(sample_obs.keys()) if isinstance(sample_obs, dict) else [],
            'condition_type': condition_type
        }
        
        # Check required fields for each condition type
        required_fields = {
            'one_hot': ['block_encoding'],
            'target_block': ['achieved_goal'],
            'target_block_and_target_peg': ['achieved_goal', 'desired_goal'],
            'one_hot_and_target_peg': ['block_encoding', 'desired_goal'],
            'four_tuple': ['block_encoding'],
            'color_language': ['block_encoding']
        }
        
        if condition_type in required_fields:
            for field in required_fields[condition_type]:
                if field not in sample_obs:
                    validation_result['compatible'] = False
                    validation_result['issues'].append(f"Missing required field: {field}")
        
        # Check dimensions
        if isinstance(sample_obs, dict):
            obs_array = sample_obs.get('observation', np.array([]))
            validation_result['base_obs_dim'] = len(obs_array) if obs_array.size > 0 else 0
            validation_result['conditioning_dim'] = cls.get_conditioning_dimension(condition_type)
            validation_result['total_obs_dim'] = cls.calculate_total_obs_dim(
                validation_result['base_obs_dim'], condition_type, True
            )
        
        return validation_result
    
    @classmethod
    def create_model_config(cls, condition_type: str, base_obs_dim: int, 
                           action_dim: int = 5, use_goal: bool = True, **kwargs) -> Dict[str, Any]:
        """Create model configuration for the specified conditioning type"""
        
        total_obs_dim = cls.calculate_total_obs_dim(base_obs_dim, condition_type, use_goal)
        goal_dim = cls.get_conditioning_dimension(condition_type) if use_goal else 0
        
        config = {
            'obs_dim': total_obs_dim,
            'action_dim': action_dim,
            'condition_type': condition_type,
            'use_goal': use_goal,
            'goal_dim': goal_dim,
            'base_obs_dim': base_obs_dim,
            'conditioning_dim': goal_dim,
            **kwargs
        }
        
        return config

class UniversalModelLoader:
    """
    Universal model loader that can handle all goal conditioning formats
    """
    
    @staticmethod
    def load_model_with_auto_config(model_path: str, fallback_config: Optional[Dict] = None):
        """
        Load model with automatic configuration detection
        """
        try:
            # Try to load with embedded config
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                model_state = checkpoint.get('model_state_dict', checkpoint)
            else:
                # Try separate config file
                config_path = model_path.replace('.pt', '.pkl').replace('final_model', 'model_config')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        config = pickle.load(f)
                    model_state = checkpoint
                else:
                    # Use fallback config
                    if fallback_config is None:
                        raise ValueError("No config found and no fallback provided")
                    config = fallback_config
                    model_state = checkpoint
            
            # Ensure config has all required fields
            config.setdefault('condition_type', 'unknown')
            config.setdefault('use_goal', True)
            config.setdefault('goal_dim', 3)
            config.setdefault('action_dim', 5)
            
            # Fix obs_dim if it seems incorrect for the condition type
            condition_type = config.get('condition_type', 'unknown')
            current_obs_dim = config.get('obs_dim', 19)
            
            # Calculate what the obs_dim should be
            if condition_type != 'none' and condition_type != 'unknown':
                base_obs_dim = 19  # Current environment base observation dimension
                conditioning_dim = GoalConditioningManager.get_conditioning_dimension(condition_type)
                expected_obs_dim = base_obs_dim + conditioning_dim
                
                if current_obs_dim != expected_obs_dim:
                    print(f"⚠️  Model obs_dim ({current_obs_dim}) doesn't match expected ({expected_obs_dim}) for {condition_type}")
                    print(f"   This suggests the model was trained with incorrect dimensions")
                    print(f"   Using model's obs_dim ({current_obs_dim}) for compatibility")
                    
                    # Keep the model's obs_dim but note the issue
                    config['expected_obs_dim'] = expected_obs_dim
                    config['dimension_mismatch'] = True
            else:
                config.setdefault('obs_dim', 19)
            
            return model_state, config
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    @staticmethod
    def create_policy_from_config(config: Dict[str, Any]):
        """Create policy instance from configuration"""
        from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
        
        policy = EnhancedDictPolicy(
            ac_dim=config.get('action_dim', 5),
            ob_dim=config.get('obs_dim', 19),
            use_goal=config.get('use_goal', True),
            goal_dim=config.get('goal_dim', 3),
            goal_importance=config.get('goal_importance', 2.0),
            n_layers=config.get('n_layers', 3),
            size=config.get('size', 128),
            learning_rate=config.get('learning_rate', 5e-4)
        )
        
        return policy

class UniversalEvaluator:
    """
    Universal evaluator that works with all goal conditioning formats
    """
    
    @staticmethod
    def evaluate_policy_universal(model_path: str, condition_type: str = None, 
                                 num_episodes: int = 10, env_name: str = "PegTransfer-v0",
                                 max_steps: int = 100, save_video: bool = False,
                                 video_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate policy with automatic format detection and compatibility
        """
        # Load model
        model_state, config = UniversalModelLoader.load_model_with_auto_config(model_path)
        if model_state is None:
            return {'error': 'Failed to load model'}
        
        # Override condition type if specified
        if condition_type is not None:
            config['condition_type'] = condition_type
        
        # Create policy
        policy = UniversalModelLoader.create_policy_from_config(config)
        policy.load_state_dict(model_state)
        policy.eval()
        
        # Run evaluation
        import gymnasium as gym
        import surrol.gym
        
        env = gym.make(env_name, render_mode="rgb_array" if save_video else None)
        
        results = {
            'model_path': model_path,
            'condition_type': config['condition_type'],
            'config': config,
            'episodes': [],
            'success_rate': 0.0,
            'mean_return': 0.0,
            'compatibility_check': True
        }
        
        success_count = 0
        returns = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_return = 0
            frames = []
            
            for step in range(max_steps):
                try:
                    # Apply conditioning based on format using universal system
                    if isinstance(obs, dict) and config['condition_type'] != 'none':
                        conditioned_obs = GoalConditioningManager.apply_conditioning(
                            obs['observation'], obs, config['condition_type']
                        )
                        
                        # Handle dimension mismatch for models trained incorrectly
                        model_obs_dim = config.get('obs_dim')
                        expected_obs_dim = config.get('expected_obs_dim', model_obs_dim)
                        
                        if config.get('dimension_mismatch', False):
                            # Model was trained with wrong dimensions, adjust conditioning
                            if conditioned_obs.shape[0] > model_obs_dim:
                                # Truncate to what the model expects
                                conditioned_obs = conditioned_obs[:model_obs_dim]
                                if episode == 0 and step == 0:
                                    print(f"🔧 Truncating conditioned obs from {expected_obs_dim}D to {model_obs_dim}D for compatibility")
                            elif conditioned_obs.shape[0] < model_obs_dim:
                                # Pad to what the model expects
                                padding = np.zeros(model_obs_dim - conditioned_obs.shape[0])
                                conditioned_obs = np.hstack([conditioned_obs, padding])
                                if episode == 0 and step == 0:
                                    print(f"🔧 Padding conditioned obs from {conditioned_obs.shape[0]}D to {model_obs_dim}D for compatibility")
                        
                        # Create conditioned observation dict for policy
                        eval_obs = obs.copy()
                        eval_obs['observation'] = conditioned_obs
                        
                        # Debug dimension info
                        if episode == 0 and step == 0:
                            print(f"🔍 Debug: Original obs shape: {obs['observation'].shape}")
                            print(f"🔍 Debug: Conditioned obs shape: {conditioned_obs.shape}")
                            print(f"🔍 Debug: Model expects obs_dim: {config.get('obs_dim')}")
                            if config.get('dimension_mismatch', False):
                                print(f"🔍 Debug: Expected obs_dim for format: {expected_obs_dim}")
                    else:
                        eval_obs = obs
                        
                        # Debug for none case
                        if episode == 0 and step == 0:
                            if isinstance(obs, dict):
                                print(f"🔍 Debug: Raw obs shape: {obs['observation'].shape}")
                                print(f"🔍 Debug: Model expects obs_dim: {config.get('obs_dim')}")
                                
                                # Check for dimension mismatch in none case
                                if obs['observation'].shape[0] != config.get('obs_dim', 0):
                                    print(f"⚠️  Dimension mismatch in 'none' mode!")
                                    expected_dim = config.get('obs_dim', obs['observation'].shape[0])
                                    
                                    if obs['observation'].shape[0] > expected_dim:
                                        print(f"🔧 Truncating observation to {expected_dim} dimensions")
                                        eval_obs = obs.copy()
                                        eval_obs['observation'] = obs['observation'][:expected_dim]
                                    elif obs['observation'].shape[0] < expected_dim:
                                        print(f"🔧 Padding observation to {expected_dim} dimensions")
                                        padding = np.zeros(expected_dim - obs['observation'].shape[0])
                                        eval_obs = obs.copy()
                                        eval_obs['observation'] = np.hstack([obs['observation'], padding])
                    
                    action = policy.get_action(eval_obs)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_return += reward
                    
                    if save_video:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    
                    if done or truncated:
                        if info.get('is_success', False):
                            success_count += 1
                        break
                        
                except Exception as e:
                    print(f"❌ Error in episode {episode}, step {step}: {e}")
                    results['compatibility_check'] = False
                    break
            
            returns.append(episode_return)
            
            episode_result = {
                'episode': episode,
                'return': episode_return,
                'success': info.get('is_success', False),
                'steps': step + 1
            }
            results['episodes'].append(episode_result)
            
            # Save video if requested
            if save_video and frames and video_dir:
                os.makedirs(video_dir, exist_ok=True)
                from .bc_visualization_tools import save_gif_from_frames
                gif_path = os.path.join(video_dir, f'episode_{episode}_{config["condition_type"]}.gif')
                save_gif_from_frames(frames, gif_path)
        
        results['success_rate'] = success_count / num_episodes
        results['mean_return'] = np.mean(returns)
        
        env.close()
        return results

def print_format_explanation():
    """Print detailed explanation of all goal conditioning formats"""
    print("\n" + "="*80)
    print("🎯 GOAL CONDITIONING FORMATS COMPREHENSIVE GUIDE")
    print("="*80)
    
    formats = {
        'none': {
            'description': 'No goal conditioning (baseline)',
            'use_case': 'Baseline comparison, simple tasks',
            'advantages': 'Simple, no extra complexity',
            'disadvantages': 'Cannot handle multi-goal tasks',
            'example_dim': '+0 dimensions'
        },
        'one_hot': {
            'description': 'Block color as one-hot vector',
            'use_case': 'Categorical color selection',
            'advantages': 'Efficient, discrete representation',
            'disadvantages': 'No spatial information',
            'example_dim': '+4 dimensions (4 colors)'
        },
        'target_block': {
            'description': 'Position of target block (3D coords)',
            'use_case': 'Spatial manipulation tasks',
            'advantages': 'Spatial awareness, position-based',
            'disadvantages': 'No object identity info',
            'example_dim': '+3 dimensions (x,y,z)'
        },
        'target_block_and_target_peg': {
            'description': 'Target block + target peg positions',
            'use_case': 'Complete pick-and-place tasks',
            'advantages': 'Full spatial specification',
            'disadvantages': 'High dimensional, complex',
            'example_dim': '+6 dimensions (3D+3D)'
        },
        'one_hot_and_target_peg': {
            'description': 'Block color + target peg position',
            'use_case': 'Hybrid categorical + spatial',
            'advantages': 'Balanced information content',
            'disadvantages': 'Still somewhat complex',
            'example_dim': '+7 dimensions (4+3)'
        },
        'four_tuple': {
            'description': 'Block color as RGBA tuple',
            'use_case': 'Continuous color representation',
            'advantages': 'Smooth color space, visual',
            'disadvantages': 'Redundant alpha channel',
            'example_dim': '+4 dimensions (R,G,B,A)'
        },
        'color_language': {
            'description': 'Block color as text string',
            'use_case': 'Language-conditioned policies',
            'advantages': 'Human-interpretable, natural',
            'disadvantages': 'Requires text processing',
            'example_dim': '+4 dimensions (embedded)'
        }
    }
    
    for format_name, details in formats.items():
        print(f"\n📋 {format_name.upper()}")
        print(f"   Description: {details['description']}")
        print(f"   Use Case: {details['use_case']}")
        print(f"   Advantages: {details['advantages']}")
        print(f"   Disadvantages: {details['disadvantages']}")
        print(f"   Dimensions: {details['example_dim']}")
    
    print("\n" + "="*80)
    print("💡 CHOOSING THE RIGHT FORMAT:")
    print("="*80)
    print("• For simple tasks: 'none' (baseline)")
    print("• For color-based tasks: 'one_hot' or 'four_tuple'")
    print("• For spatial tasks: 'target_block'")
    print("• For complex manipulation: 'target_block_and_target_peg'")
    print("• For balanced approach: 'one_hot_and_target_peg'")
    print("• For language models: 'color_language'")
    print("="*80)

if __name__ == '__main__':
    print_format_explanation() 