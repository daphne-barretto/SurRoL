"""
Universal BC Training System for PegTransfer

This system automatically detects data format and applies correct conditioning
for training and evaluation across all data types:
- 2 blocks vs 4 blocks
- With/without colors
- Different conditioning methods (one-hot, target block, etc.)
"""

import os
import sys

# CRITICAL: Add our SurRoL directory to Python path (before SurRol-elsa)
# This ensures we use the correct SurRoL with 2-block environments
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
surrol_path = os.path.join(project_root, 'SurRoL')
if surrol_path not in sys.path:
    sys.path.insert(0, surrol_path)

import torch
import numpy as np
import pickle
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gymnasium as gym
import surrol.gym
from typing import Dict, Tuple, List, Any, Optional
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
try:
    import seaborn as sns
    # Set up plotting style
    plt.style.use('default')  # Use default instead of seaborn-v0_8
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    print("⚠️  Seaborn not available, using matplotlib defaults")
    plt.style.use('default')
    HAS_SEABORN = False

from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

class DataType(Enum):
    """Enumeration of all possible data types"""
    ALL_BLOCKS_2 = "all_blocks_2"  # 2 blocks, robot + all block positions
    ALL_BLOCKS_4 = "all_blocks_4"  # 4 blocks, robot + all block positions  
    ALL_BLOCKS_COLORED_2 = "all_blocks_colored_2"  # 2 blocks + colors
    ALL_BLOCKS_COLORED_4 = "all_blocks_colored_4"  # 4 blocks + colors
    ONLY_TARGET_BLOCK_4 = "only_target_block_4"  # 4 blocks, robot + target block only
    ONLY_TARGET_BLOCK_2 = "only_target_block_2"  # 2 blocks, robot + target block only
    ONLY_TARGET_BLOCK_2_NO_COLOR = "only_target_block_2_no_color"  # 2 blocks, robot + target block only, no color

class ConditioningType(Enum):
    """Enumeration of conditioning methods"""
    NONE = "none"
    ONE_HOT = "one_hot"
    TARGET_BLOCK = "target_block"
    TARGET_PEG = "target_peg"
    TARGET_BLOCK_AND_PEG = "target_block_and_peg"
    ONE_HOT_AND_TARGET_PEG = "one_hot_and_target_peg"
    RGBA_COLOR = "rgba_color"

@dataclass
class DataSpec:
    """Specification for different data formats"""
    data_type: DataType
    conditioning_type: ConditioningType
    num_blocks: int
    has_colors: bool
    robot_state_dim: int
    base_obs_dim: int  # Before conditioning
    final_obs_dim: int  # After conditioning
    block_encoding_dim: Optional[int] = None
    
    @property
    def conditioning_dim(self) -> int:
        return self.final_obs_dim - self.base_obs_dim

# Define environment mappings for different data types and conditioning
ENVIRONMENT_MAPPINGS = {
    # 2-block environments - use the correct 2-block environments
    (DataType.ALL_BLOCKS_2, ConditioningType.NONE): "PegTransferTwoBlocksNoColor-v0",  # NEW: proper 2-block base env
    (DataType.ALL_BLOCKS_2, ConditioningType.ONE_HOT): "PegTransferTwoBlocksNoColorOneHot-v0",  # NEW: 2-block one-hot only
    (DataType.ALL_BLOCKS_2, ConditioningType.TARGET_BLOCK): "PegTransferTwoBlocksNoColorTargetBlock-v0",  # NEW: 2-block target block
    (DataType.ALL_BLOCKS_2, ConditioningType.TARGET_PEG): "PegTransferTwoBlocksNoColorTargetPeg-v0",  # NEW: 2-block target peg
    (DataType.ALL_BLOCKS_2, ConditioningType.TARGET_BLOCK_AND_PEG): "PegTransferTwoBlocksNoColorTargetBlockPeg-v0",  # NEW: 2-block both
    (DataType.ALL_BLOCKS_2, ConditioningType.ONE_HOT_AND_TARGET_PEG): "PegTransferTwoBlocksNoColorOneHotTargetPeg-v0",  # NEW: 2-block one-hot + target peg
    (DataType.ALL_BLOCKS_2, ConditioningType.RGBA_COLOR): "PegTransferTwoBlocksNoColorFourTuple-v0",  # NEW: 2-block four tuple
    
    # 2-block colored environments - use 2-block environments  
    (DataType.ALL_BLOCKS_COLORED_2, ConditioningType.NONE): "PegTransferTwoBlocksOneHot-v0",  # Use 2-block env as base
    (DataType.ALL_BLOCKS_COLORED_2, ConditioningType.ONE_HOT): "PegTransferTwoBlocksOneHot-v0",  # 2-block env
    (DataType.ALL_BLOCKS_COLORED_2, ConditioningType.TARGET_BLOCK): "PegTransferTargetBlock-v0",
    (DataType.ALL_BLOCKS_COLORED_2, ConditioningType.RGBA_COLOR): "PegTransferTwoBlocksFourTuple-v0", 
    (DataType.ALL_BLOCKS_COLORED_2, ConditioningType.ONE_HOT_AND_TARGET_PEG): "PegTransferTwoBlocksOneHotTargetPeg-v0",  # CORRECT 2-block env!
    
    # 4-block environments
    (DataType.ALL_BLOCKS_4, ConditioningType.NONE): "PegTransfer-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.ONE_HOT): "PegTransfer-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.TARGET_BLOCK): "PegTransferTargetBlock-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.TARGET_PEG): "PegTransfer-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.TARGET_BLOCK_AND_PEG): "PegTransferTargetBlockTargetPeg-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.ONE_HOT_AND_TARGET_PEG): "PegTransferOneHotTargetPeg-v0",
    (DataType.ALL_BLOCKS_4, ConditioningType.RGBA_COLOR): "PegTransferFourTuple-v0",
    
    # 4-block colored environments
    (DataType.ALL_BLOCKS_COLORED_4, ConditioningType.NONE): "PegTransferColor-v0",
    (DataType.ALL_BLOCKS_COLORED_4, ConditioningType.TARGET_BLOCK): "PegTransferColorTargetBlock-v0",
    (DataType.ALL_BLOCKS_COLORED_4, ConditioningType.TARGET_BLOCK_AND_PEG): "PegTransferColorTargetBlockTargetPeg-v0",
    (DataType.ALL_BLOCKS_COLORED_4, ConditioningType.ONE_HOT_AND_TARGET_PEG): "PegTransferColorOneHotTargetPeg-v0",
    (DataType.ALL_BLOCKS_COLORED_4, ConditioningType.RGBA_COLOR): "PegTransferFourTuple-v0",
    
    # Target block only environments  
    (DataType.ONLY_TARGET_BLOCK_4, ConditioningType.NONE): "PegTransferTBO-v0",
    (DataType.ONLY_TARGET_BLOCK_2, ConditioningType.NONE): "PegTransferTwoBlocksTargetBlockOnly-v0",
    (DataType.ONLY_TARGET_BLOCK_2_NO_COLOR, ConditioningType.NONE): "PegTransferTwoBlocksTargetBlockOnlyNoColor-v0",
}

def get_environment_name(data_spec: DataSpec) -> str:
    """Get the correct environment name for the given data specification"""
    env_key = (data_spec.data_type, data_spec.conditioning_type)
    
    if env_key in ENVIRONMENT_MAPPINGS:
        env_name = ENVIRONMENT_MAPPINGS[env_key]
        print(f"🎮 Using environment: {env_name}")
        return env_name
    else:
        # Fallback to default environment with warning
        fallback_env = "PegTransfer-v0"
        print(f"⚠️  No specific environment found for {data_spec.data_type.value} + {data_spec.conditioning_type.value}")
        print(f"   Using fallback: {fallback_env}")
        print(f"   This may cause dimension mismatches!")
        return fallback_env

# Define all data specifications
DATA_SPECIFICATIONS = {
    # Raw data specifications (before post-processing)
    ("goal_conditioned_demos_with_all_blocks_2_blocks", ConditioningType.NONE): DataSpec(
        data_type=DataType.ALL_BLOCKS_2,
        conditioning_type=ConditioningType.NONE,
        num_blocks=2,
        has_colors=False,
        robot_state_dim=7,  # Typical robot state (pos, quat, jaw)
        base_obs_dim=13,    # robot(7) + target_block_pos(3) + target_block_rel_pos(3)
        final_obs_dim=13
    ),
    ("goal_conditioned_demos_with_all_blocks", ConditioningType.NONE): DataSpec(
        data_type=DataType.ALL_BLOCKS_4,
        conditioning_type=ConditioningType.NONE,
        num_blocks=4,
        has_colors=False,
        robot_state_dim=7,
        base_obs_dim=31,    # robot(7) + 4_blocks*(pos(3)+rel_pos(3)) = 7+24 = 31
        final_obs_dim=31
    ),
    ("goal_conditioned_demos_with_all_blocks_colored_2_blocks", ConditioningType.NONE): DataSpec(
        data_type=DataType.ALL_BLOCKS_COLORED_2,
        conditioning_type=ConditioningType.NONE,
        num_blocks=2,
        has_colors=True,
        robot_state_dim=7,
        base_obs_dim=18,    # robot(7) + target_block_pos(3) + target_block_rel_pos(3) + rgba(4) + goal(3) = 20, but might be different
        final_obs_dim=18
    ),
    ("goal_conditioned_demos_with_all_blocks_colored", ConditioningType.NONE): DataSpec(
        data_type=DataType.ALL_BLOCKS_COLORED_4,
        conditioning_type=ConditioningType.NONE,
        num_blocks=4,
        has_colors=True,
        robot_state_dim=7,
        base_obs_dim=47,    # robot(7) + 4_blocks*(pos(3)+rel_pos(3)+rgba(4)) = 7+40 = 47
        final_obs_dim=47
    ),
    ("goal_conditioned_demos_with_only_target_block", ConditioningType.NONE): DataSpec(
        data_type=DataType.ONLY_TARGET_BLOCK_4,
        conditioning_type=ConditioningType.NONE,
        num_blocks=4,
        has_colors=False,
        robot_state_dim=7,
        base_obs_dim=13,    # robot(7) + target_block_pos(3) + target_block_rel_pos(3)
        final_obs_dim=13
    ),
    ("goal_conditioned_demos_with_only_target_block_2_blocks", ConditioningType.NONE): DataSpec(
        data_type=DataType.ONLY_TARGET_BLOCK_2,
        conditioning_type=ConditioningType.NONE,
        num_blocks=2,
        has_colors=True,
        robot_state_dim=7,
        base_obs_dim=17,    # robot(7) + target_block_pos(3) + target_block_rel_pos(3) + target_block_color(4) = 17
        final_obs_dim=17
    ),
    ("goal_conditioned_demos_with_only_target_block_2_blocks_no_color", ConditioningType.NONE): DataSpec(
        data_type=DataType.ONLY_TARGET_BLOCK_2_NO_COLOR,
        conditioning_type=ConditioningType.NONE,
        num_blocks=2,
        has_colors=False,
        robot_state_dim=7,
        base_obs_dim=13,    # robot(7) + target_block_pos(3) + target_block_rel_pos(3) = 13 (no color)
        final_obs_dim=13
    ),
}

class UniversalDataDetector:
    """Detects data type and conditioning from filename and data structure"""
    
    @staticmethod
    def detect_data_spec(data_path: str, base_type: str = None, conditioning_type: str = None) -> DataSpec:
        """Detect complete data specification from file"""
        print(f"🔍 DETECTING DATA SPECIFICATION")
        print("="*50)
        
        filename = os.path.basename(data_path)
        print(f"   • File: {filename}")
        
        # Load and analyze data
        data = np.load(data_path, allow_pickle=True)
        obs_data = data['obs']
        sample_obs = obs_data[0][0]
        
        # Analyze actual observation dimension
        if isinstance(sample_obs, dict):
            actual_obs_dim = len(sample_obs['observation'])
        else:
            actual_obs_dim = len(sample_obs)
        
        print(f"   • Actual observation dimension: {actual_obs_dim}")
        
        # Determine base data type
        if base_type:
            print(f"   • Using provided base type: {base_type}")
            base_type_key = base_type
        else:
            try:
                base_type_key = UniversalDataDetector._detect_base_type(filename)
            except ValueError:
                # If detection fails, ask user for input
                base_type_key = UniversalDataDetector._ask_user_for_base_type()
        
        # Determine conditioning type
        if conditioning_type:
            print(f"   • Using provided conditioning type: {conditioning_type}")
            conditioning_enum = ConditioningType(conditioning_type)
        else:
            try:
                conditioning_enum = UniversalDataDetector._detect_conditioning_type(filename)
            except ValueError:
                conditioning_enum = UniversalDataDetector._ask_user_for_conditioning_type()
        
        # Get base specification
        spec_key = (base_type_key, ConditioningType.NONE)
        if spec_key not in DATA_SPECIFICATIONS:
            raise ValueError(f"Unknown base data type: {base_type_key}")
        
        base_spec = DATA_SPECIFICATIONS[spec_key]
        
        # Calculate conditioning specification
        final_spec = UniversalDataDetector._calculate_conditioned_spec(
            base_spec, conditioning_enum, actual_obs_dim
        )
        
        print(f"   • Base type: {base_spec.data_type.value}")
        print(f"   • Conditioning: {conditioning_enum.value}")
        print(f"   • Blocks: {final_spec.num_blocks}")
        print(f"   • Has colors: {final_spec.has_colors}")
        print(f"   • Base obs dim: {final_spec.base_obs_dim}")
        print(f"   • Final obs dim: {final_spec.final_obs_dim}")
        print(f"   • Conditioning dim: {final_spec.conditioning_dim}")
        
        return final_spec
    
    @staticmethod
    def _detect_base_type(filename: str) -> str:
        """Detect base data type from filename"""
        # Remove conditioning suffixes to get base type
        base_filename = filename.replace('_onehot', '').replace('_targetblock', '').replace('_targetpeg', '').replace('_fourtuple', '').replace('.npz', '')
        
        if 'all_blocks_colored_2_blocks' in base_filename:
            return 'goal_conditioned_demos_with_all_blocks_colored_2_blocks'
        elif 'all_blocks_colored' in base_filename:
            return 'goal_conditioned_demos_with_all_blocks_colored'
        elif 'all_blocks_2_blocks' in base_filename:
            return 'goal_conditioned_demos_with_all_blocks_2_blocks'
        elif 'all_blocks' in base_filename:
            return 'goal_conditioned_demos_with_all_blocks'
        elif 'only_target_block_2_blocks_no_color' in base_filename or 'TargetBlockOnlyNoColor' in base_filename:
            return 'goal_conditioned_demos_with_only_target_block_2_blocks_no_color'
        elif 'only_target_block_2_blocks' in base_filename:
            return 'goal_conditioned_demos_with_only_target_block_2_blocks'
        elif 'only_target_block' in base_filename:
            return 'goal_conditioned_demos_with_only_target_block'
        else:
            raise ValueError(f"Cannot detect base type from filename: {filename}")
    
    @staticmethod
    def _ask_user_for_base_type() -> str:
        """Ask user to specify base data type when auto-detection fails"""
        print("\n❓ Could not auto-detect base data type from filename.")
        print("Please select the base data type:")
        print("1. all_blocks_2_blocks (2 blocks, robot + all block positions)")
        print("2. all_blocks_4_blocks (4 blocks, robot + all block positions)")
        print("3. all_blocks_colored_2_blocks (2 blocks + colors)")
        print("4. all_blocks_colored_4_blocks (4 blocks + colors)")
        print("5. only_target_block_4_blocks (4 blocks, robot + target block only)")
        print("6. only_target_block_2_blocks (2 blocks, robot + target block only)")
        print("7. only_target_block_2_blocks_no_color (2 blocks, robot + target block only, no color)")
        
        type_mapping = {
            '1': 'goal_conditioned_demos_with_all_blocks_2_blocks',
            '2': 'goal_conditioned_demos_with_all_blocks',
            '3': 'goal_conditioned_demos_with_all_blocks_colored_2_blocks',
            '4': 'goal_conditioned_demos_with_all_blocks_colored',
            '5': 'goal_conditioned_demos_with_only_target_block',
            '6': 'goal_conditioned_demos_with_only_target_block_2_blocks',
            '7': 'goal_conditioned_demos_with_only_target_block_2_blocks_no_color'
        }
        
        while True:
            choice = input("Enter choice (1-7): ").strip()
            if choice in type_mapping:
                selected_type = type_mapping[choice]
                print(f"   ✅ Selected: {selected_type}")
                return selected_type
            else:
                print("   ❌ Invalid choice. Please enter 1-7.")
    
    @staticmethod
    def _detect_conditioning_type(filename: str) -> ConditioningType:
        """Detect conditioning type from filename"""
        if '_onehottargetpeg' in filename:
            return ConditioningType.ONE_HOT_AND_TARGET_PEG
        elif '_targetblocktargetpeg' in filename:
            return ConditioningType.TARGET_BLOCK_AND_PEG
        elif '_onehot' in filename:
            return ConditioningType.ONE_HOT
        elif '_targetblock' in filename:
            return ConditioningType.TARGET_BLOCK
        elif '_targetpeg' in filename:
            return ConditioningType.TARGET_PEG
        elif '_fourtuple' in filename:
            return ConditioningType.RGBA_COLOR
        else:
            return ConditioningType.NONE
    
    @staticmethod
    def _ask_user_for_conditioning_type() -> ConditioningType:
        """Ask user to specify conditioning type when auto-detection fails"""
        print("\n❓ Could not auto-detect conditioning type from filename.")
        print("Please select the conditioning type:")
        print("1. none (no conditioning)")
        print("2. one_hot (one-hot block encoding)")
        print("3. target_block (target block position)")
        print("4. target_peg (target peg position)")
        print("5. target_block_and_peg (both target block and peg positions)")
        print("6. one_hot_and_target_peg (one-hot + target peg)")
        print("7. rgba_color (RGBA color encoding)")
        
        type_mapping = {
            '1': ConditioningType.NONE,
            '2': ConditioningType.ONE_HOT,
            '3': ConditioningType.TARGET_BLOCK,
            '4': ConditioningType.TARGET_PEG,
            '5': ConditioningType.TARGET_BLOCK_AND_PEG,
            '6': ConditioningType.ONE_HOT_AND_TARGET_PEG,
            '7': ConditioningType.RGBA_COLOR
        }
        
        while True:
            choice = input("Enter choice (1-7): ").strip()
            if choice in type_mapping:
                selected_type = type_mapping[choice]
                print(f"   ✅ Selected: {selected_type.value}")
                return selected_type
            else:
                print("   ❌ Invalid choice. Please enter 1-7.")
    
    @staticmethod
    def _calculate_conditioned_spec(base_spec: DataSpec, conditioning_type: ConditioningType, 
                                  actual_obs_dim: int) -> DataSpec:
        """Calculate final specification with conditioning"""
        
        # Calculate expected conditioning dimensions
        conditioning_dim = 0
        block_encoding_dim = None
        
        if conditioning_type == ConditioningType.ONE_HOT:
            block_encoding_dim = base_spec.num_blocks
            conditioning_dim = block_encoding_dim
        elif conditioning_type == ConditioningType.TARGET_BLOCK:
            conditioning_dim = 3  # 3D position
        elif conditioning_type == ConditioningType.TARGET_PEG:
            conditioning_dim = 3  # 3D position
        elif conditioning_type == ConditioningType.TARGET_BLOCK_AND_PEG:
            conditioning_dim = 6  # 3D + 3D
        elif conditioning_type == ConditioningType.ONE_HOT_AND_TARGET_PEG:
            block_encoding_dim = base_spec.num_blocks
            conditioning_dim = block_encoding_dim + 3
        elif conditioning_type == ConditioningType.RGBA_COLOR:
            conditioning_dim = 4  # RGBA
        
        expected_final_dim = base_spec.base_obs_dim + conditioning_dim
        
        # Verify against actual dimension
        if actual_obs_dim != expected_final_dim:
            print(f"   ⚠️  Dimension mismatch: expected {expected_final_dim}, got {actual_obs_dim}")
            print(f"      Using actual dimension: {actual_obs_dim}")
            final_obs_dim = actual_obs_dim
        else:
            final_obs_dim = expected_final_dim
        
        return DataSpec(
            data_type=base_spec.data_type,
            conditioning_type=conditioning_type,
            num_blocks=base_spec.num_blocks,
            has_colors=base_spec.has_colors,
            robot_state_dim=base_spec.robot_state_dim,
            base_obs_dim=base_spec.base_obs_dim,
            final_obs_dim=final_obs_dim,
            block_encoding_dim=block_encoding_dim
        )

class UniversalObservationProcessor:
    """Processes observations for both training and evaluation based on data specification"""
    
    def __init__(self, data_spec: DataSpec):
        self.spec = data_spec
        self.is_postprocessed = data_spec.conditioning_type != ConditioningType.NONE
        
        print(f"🔧 UNIVERSAL OBSERVATION PROCESSOR")
        print(f"   • Data type: {data_spec.data_type.value}")
        print(f"   • Conditioning: {data_spec.conditioning_type.value}")
        print(f"   • Post-processed: {self.is_postprocessed}")
        print(f"   • Input dim: {data_spec.final_obs_dim}")
    
    def process_training_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Process observation for training"""
        if self.is_postprocessed:
            # For post-processed data, conditioning is already applied
            return obs['observation']
        else:
            # For raw data, no additional processing needed
            return obs['observation']
    
    def process_evaluation_observation(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process environment observation for evaluation"""
        
        env_obs_dim = len(env_obs['observation'])
        target_dim = self.spec.final_obs_dim
        
        # print(f"      🔧 Processing evaluation obs: {env_obs_dim} -> {target_dim} dims")
        
        # Handle dimension mismatches
        if env_obs_dim == target_dim:
            # Perfect match - use as is
            processed_obs = env_obs['observation']
            # print(f"      ✅ Perfect dimension match: {env_obs_dim}")
            
        elif env_obs_dim == 19 and target_dim in [13, 15]:
            # 4-block environment (19 dims) -> 2-block training data (13-15 dims)
            print(f"      🔧 Converting 4-block env ({env_obs_dim}) to 2-block format ({target_dim})")
            
            # Extract: robot_state(7) + first_block_info(6) = 13 dims
            robot_state = env_obs['observation'][:7]
            
            # Use the first block from 4-block environment 
            # Assuming 4-block env structure: robot(7) + 4*block_info(3) = 19
            # Take first block position as target block
            first_block_pos = env_obs['observation'][7:10]  # First block position
            first_block_rel_pos = first_block_pos - robot_state[:3]  # Relative position
            
            # Create base 2-block observation (13 dims)
            processed_obs = np.concatenate([robot_state, first_block_pos, first_block_rel_pos])
            
            # Add conditioning if needed
            if target_dim == 15:  # Need 2-dim one-hot encoding
                one_hot = np.zeros(2)
                one_hot[0] = 1.0  # Default to first block
                processed_obs = np.concatenate([processed_obs, one_hot])
            
        elif env_obs_dim == 19 and target_dim > 19:
            # Environment smaller than training data - pad with zeros
            print(f"      🔧 Padding env obs from {env_obs_dim} to {target_dim}")
            padding = np.zeros(target_dim - env_obs_dim)
            processed_obs = np.concatenate([env_obs['observation'], padding])
            
        elif env_obs_dim > target_dim:
            # Environment larger than training data - truncate
            print(f"      🔧 Truncating env obs from {env_obs_dim} to {target_dim}")
            processed_obs = env_obs['observation'][:target_dim]
            
        else:
            # General padding case
            print(f"      🔧 Padding env obs from {env_obs_dim} to {target_dim}")
            padding = np.zeros(target_dim - env_obs_dim)
            processed_obs = np.concatenate([env_obs['observation'], padding])
        
        # print(f"      ✅ Final processed obs shape: {len(processed_obs)}")
        
        return {
            'observation': processed_obs,
            'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
            'desired_goal': env_obs.get('desired_goal', np.zeros(3))
        }

class TrainingTracker:
    """Comprehensive training tracking and visualization"""
    
    def __init__(self, save_dir: str, data_spec: 'DataSpec'):
        self.save_dir = save_dir
        self.data_spec = data_spec
        
        # Training metrics
        self.training_logs = {
            'epochs': [],
            'train_losses': [],
            'eval_epochs': [],
            'success_rates': [],
            'mean_returns': [],
            'std_returns': [],
            'mean_episode_lengths': [],
            'best_success_rate': 0.0,
            'timestamps': [],
            'eval_timestamps': [],
            'individual_episodes': []  # Store all episode results
        }
        
        # Create plots directory
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"📊 TRAINING TRACKER INITIALIZED")
        print(f"   • Plots will be saved to: {self.plots_dir}")
        print(f"   • Data type: {data_spec.data_type.value}")
        print(f"   • Conditioning: {data_spec.conditioning_type.value}")
    
    def log_training_step(self, epoch: int, loss: float):
        """Log training step metrics"""
        self.training_logs['epochs'].append(epoch)
        self.training_logs['train_losses'].append(loss)
        self.training_logs['timestamps'].append(datetime.now())
    
    def log_evaluation(self, epoch: int, eval_results: Dict[str, Any]):
        """Log evaluation metrics"""
        self.training_logs['eval_epochs'].append(epoch)
        self.training_logs['success_rates'].append(eval_results['success_rate'])
        self.training_logs['mean_returns'].append(eval_results['mean_return'])
        self.training_logs['std_returns'].append(eval_results['std_return'])
        self.training_logs['mean_episode_lengths'].append(eval_results['mean_episode_length'])
        self.training_logs['eval_timestamps'].append(datetime.now())
        
        # Store individual episode results
        for episode in eval_results.get('episodes', []):
            episode_log = {
                'epoch': epoch,
                'episode_id': episode['episode'],
                'success': episode['success'],
                'return': episode['return'],
                'length': episode['length'],
                'timestamp': datetime.now()
            }
            self.training_logs['individual_episodes'].append(episode_log)
        
        # Update best success rate
        if eval_results['success_rate'] > self.training_logs['best_success_rate']:
            self.training_logs['best_success_rate'] = eval_results['success_rate']
    
    def plot_training_curves(self, show_plots: bool = False):
        """Create comprehensive training curve plots"""
        if not self.training_logs['epochs']:
            print("⚠️  No training data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress: {self.data_spec.data_type.value} + {self.data_spec.conditioning_type.value}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        if self.training_logs['train_losses']:
            axes[0, 0].plot(self.training_logs['epochs'], self.training_logs['train_losses'], 
                           'b-', alpha=0.7, linewidth=2, label='Training Loss')
            axes[0, 0].set_title('Training Loss Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Add smoothed trend line
            if len(self.training_logs['train_losses']) > 10:
                window = min(10, len(self.training_logs['train_losses']) // 4)
                smoothed = np.convolve(self.training_logs['train_losses'], 
                                     np.ones(window)/window, mode='valid')
                smoothed_epochs = self.training_logs['epochs'][window-1:]
                axes[0, 0].plot(smoothed_epochs, smoothed, 'r--', 
                               alpha=0.8, linewidth=2, label=f'Smoothed (window={window})')
                axes[0, 0].legend()
        
        # Plot 2: Success Rate
        if self.training_logs['eval_epochs']:
            axes[0, 1].plot(self.training_logs['eval_epochs'], 
                           [rate * 100 for rate in self.training_logs['success_rates']], 
                           'g-o', linewidth=2, markersize=6, label='Success Rate')
            axes[0, 1].axhline(y=self.training_logs['best_success_rate'] * 100, 
                              color='r', linestyle='--', alpha=0.7, 
                              label=f'Best: {self.training_logs["best_success_rate"]*100:.1f}%')
            axes[0, 1].set_title('Success Rate Over Time', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_ylim(0, 105)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Plot 3: Episode Returns
        if self.training_logs['eval_epochs']:
            axes[1, 0].errorbar(self.training_logs['eval_epochs'], 
                               self.training_logs['mean_returns'],
                               yerr=self.training_logs['std_returns'],
                               fmt='o-', linewidth=2, markersize=6, capsize=5,
                               label='Mean Return ± Std')
            axes[1, 0].set_title('Episode Returns Over Time', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Plot 4: Episode Lengths
        if self.training_logs['eval_epochs']:
            axes[1, 1].plot(self.training_logs['eval_epochs'], 
                           self.training_logs['mean_episode_lengths'], 
                           'purple', marker='s', linewidth=2, markersize=6, label='Mean Episode Length')
            axes[1, 1].set_title('Episode Length Over Time', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_episode_distribution(self, show_plots: bool = False):
        """Plot distribution of episode outcomes"""
        if not self.training_logs['individual_episodes']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Episode Analysis', fontsize=16, fontweight='bold')
        
        # Convert to DataFrame for easier plotting
        import pandas as pd
        episodes_df = pd.DataFrame(self.training_logs['individual_episodes'])
        
        # Plot 1: Success rate by epoch
        success_by_epoch = episodes_df.groupby('epoch')['success'].agg(['mean', 'count']).reset_index()
        axes[0].bar(success_by_epoch['epoch'], success_by_epoch['mean'] * 100, 
                   alpha=0.7, color='green', label='Success Rate')
        axes[0].set_title('Success Rate by Evaluation Epoch', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Return distribution by success
        successful_returns = episodes_df[episodes_df['success'] == True]['return']
        failed_returns = episodes_df[episodes_df['success'] == False]['return']
        
        if len(successful_returns) > 0:
            axes[1].hist(successful_returns, alpha=0.7, color='green', 
                        label=f'Successful Episodes (n={len(successful_returns)})', bins=20)
        if len(failed_returns) > 0:
            axes[1].hist(failed_returns, alpha=0.7, color='red', 
                        label=f'Failed Episodes (n={len(failed_returns)})', bins=20)
        
        axes[1].set_title('Return Distribution by Outcome', fontweight='bold')
        axes[1].set_xlabel('Episode Return')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'episode_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Episode analysis saved to: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_detailed_logs(self):
        """Save detailed training logs to JSON and CSV"""
        # Save complete logs as JSON
        logs_path = os.path.join(self.save_dir, 'detailed_training_logs.json')
        
        # Convert datetime objects and numpy types to strings/native Python types for JSON serialization
        serializable_logs = {}
        for key, value in self.training_logs.items():
            if key in ['timestamps', 'eval_timestamps']:
                serializable_logs[key] = [t.isoformat() for t in value]
            elif key == 'individual_episodes':
                serializable_logs[key] = []
                for episode in value:
                    ep_copy = episode.copy()
                    ep_copy['timestamp'] = episode['timestamp'].isoformat()
                    # Convert numpy types to Python native types
                    for ep_key, ep_value in ep_copy.items():
                        if hasattr(ep_value, 'item'):  # numpy scalar
                            ep_copy[ep_key] = ep_value.item()
                        elif isinstance(ep_value, np.ndarray):  # numpy array
                            ep_copy[ep_key] = ep_value.tolist()
                    serializable_logs[key].append(ep_copy)
            else:
                # Convert numpy arrays/scalars to Python lists/floats
                if isinstance(value, list):
                    serializable_logs[key] = []
                    for item in value:
                        if hasattr(item, 'item'):  # numpy scalar
                            serializable_logs[key].append(item.item())
                        elif isinstance(item, np.ndarray):  # numpy array
                            serializable_logs[key].append(item.tolist())
                        else:
                            serializable_logs[key].append(item)
                elif hasattr(value, 'item'):  # numpy scalar
                    serializable_logs[key] = value.item()
                elif isinstance(value, np.ndarray):  # numpy array
                    serializable_logs[key] = value.tolist()
                else:
                    serializable_logs[key] = value
        
        with open(logs_path, 'w') as f:
            json.dump(serializable_logs, f, indent=2)
        print(f"💾 Detailed logs saved to: {logs_path}")
        
        # Save summary statistics
        summary = self.get_training_summary()
        summary_path = os.path.join(self.save_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize_helper)
        print(f"📋 Training summary saved to: {summary_path}")
        
        # Save episode-level data as CSV
        if self.training_logs['individual_episodes']:
            import pandas as pd
            episodes_df = pd.DataFrame(self.training_logs['individual_episodes'])
            episodes_df['timestamp'] = episodes_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv_path = os.path.join(self.save_dir, 'episode_data.csv')
            episodes_df.to_csv(csv_path, index=False)
            print(f"📊 Episode data saved to: {csv_path}")
    
    def _json_serialize_helper(self, obj):
        """Helper function to serialize numpy types for JSON"""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):  # numpy array
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary statistics"""
        summary = {
            'data_specification': {
                'data_type': self.data_spec.data_type.value,
                'conditioning_type': self.data_spec.conditioning_type.value,
                'num_blocks': self.data_spec.num_blocks,
                'has_colors': self.data_spec.has_colors,
                'final_obs_dim': self.data_spec.final_obs_dim
            },
            'training_metrics': {
                'total_epochs': len(self.training_logs['epochs']),
                'total_evaluations': len(self.training_logs['eval_epochs']),
                'total_episodes_evaluated': len(self.training_logs['individual_episodes']),
                'best_success_rate': self.training_logs['best_success_rate'],
                'final_success_rate': self.training_logs['success_rates'][-1] if self.training_logs['success_rates'] else 0.0,
                'training_duration_minutes': 0  # Will be calculated when training ends
            },
            'performance_statistics': {},
            'learning_curve_analysis': {}
        }
        
        # Calculate training duration if we have timestamps
        if len(self.training_logs['timestamps']) > 1:
            start_time = self.training_logs['timestamps'][0]
            end_time = self.training_logs['timestamps'][-1]
            duration = (end_time - start_time).total_seconds() / 60
            summary['training_metrics']['training_duration_minutes'] = duration
        
        # Performance statistics
        if self.training_logs['success_rates']:
            summary['performance_statistics'] = {
                'success_rate_mean': np.mean(self.training_logs['success_rates']),
                'success_rate_std': np.std(self.training_logs['success_rates']),
                'success_rate_max': np.max(self.training_logs['success_rates']),
                'success_rate_min': np.min(self.training_logs['success_rates']),
                'return_mean': np.mean(self.training_logs['mean_returns']),
                'return_std': np.mean(self.training_logs['std_returns']),
                'episode_length_mean': np.mean(self.training_logs['mean_episode_lengths'])
            }
        
        # Learning curve analysis
        if len(self.training_logs['train_losses']) > 1:
            losses = np.array(self.training_logs['train_losses'])
            summary['learning_curve_analysis'] = {
                'loss_improvement': losses[0] - losses[-1],
                'loss_improvement_percent': ((losses[0] - losses[-1]) / abs(losses[0])) * 100 if losses[0] != 0 else 0,
                'loss_volatility': np.std(losses),
                'converged': self._check_convergence(losses)
            }
        
        return summary
    
    def _check_convergence(self, losses: np.ndarray, window: int = 20, threshold: float = 0.01) -> bool:
        """Check if training has converged based on loss stability"""
        if len(losses) < window * 2:
            return False
        
        recent_losses = losses[-window:]
        recent_std = np.std(recent_losses)
        recent_mean = np.mean(recent_losses)
        
        # Consider converged if relative standard deviation is below threshold
        if recent_mean != 0:
            relative_std = recent_std / abs(recent_mean)
            return relative_std < threshold
        
        return False
    
    def print_progress_summary(self, epoch: int, total_epochs: int):
        """Print a comprehensive progress summary"""
        if not self.training_logs['eval_epochs']:
            return
        
        print(f"\n🔄 TRAINING PROGRESS SUMMARY (Epoch {epoch}/{total_epochs})")
        print("="*60)
        
        # Current metrics
        current_success = self.training_logs['success_rates'][-1] * 100
        current_return = self.training_logs['mean_returns'][-1]
        best_success = self.training_logs['best_success_rate'] * 100
        
        print(f"   🎯 Current Success Rate: {current_success:6.1f}%")
        print(f"   🏆 Best Success Rate:    {best_success:6.1f}%")
        print(f"   📈 Current Return:       {current_return:7.2f}")
        
        # Progress indicators
        if len(self.training_logs['success_rates']) > 1:
            prev_success = self.training_logs['success_rates'][-2] * 100
            success_change = current_success - prev_success
            trend = "📈" if success_change > 0 else "📉" if success_change < 0 else "➡️"
            print(f"   {trend} Success Change:      {success_change:+6.1f}%")
        
        # Training efficiency
        if len(self.training_logs['timestamps']) > 1:
            start_time = self.training_logs['timestamps'][0]
            current_time = self.training_logs['timestamps'][-1]
            elapsed = (current_time - start_time).total_seconds() / 60
            print(f"   ⏱️  Elapsed Time:        {elapsed:6.1f} minutes")
        
        # Loss trend
        if len(self.training_logs['train_losses']) > 10:
            recent_losses = self.training_logs['train_losses'][-10:]
            loss_trend = np.mean(np.diff(recent_losses))
            trend_symbol = "📉" if loss_trend < 0 else "📈" if loss_trend > 0 else "➡️"
            print(f"   {trend_symbol} Loss Trend:          {'Decreasing' if loss_trend < 0 else 'Increasing' if loss_trend > 0 else 'Stable'}")
        
        print("="*60)

class UniversalBCTrainer:
    """Universal BC trainer that handles all data types and conditioning methods"""
    
    def __init__(self, data_path: str, save_dir: str, base_type: str = None, conditioning_type: str = None):
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Detect data specification
        self.data_spec = UniversalDataDetector.detect_data_spec(data_path, base_type, conditioning_type)
        self.processor = UniversalObservationProcessor(self.data_spec)
        
        # Load data
        self.data = np.load(data_path, allow_pickle=True)
        self.obs_data = self.data['obs']
        self.acs_data = self.data['acs']
        
        print(f"\n🎯 UNIVERSAL BC TRAINER INITIALIZED")
        print(f"Episodes: {len(self.obs_data)}")
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data"""
        print(f"\n📊 PREPARING TRAINING DATA")
        
        train_obs = []
        train_actions = []
        
        for episode_idx in range(len(self.obs_data)):
            episode_obs = self.obs_data[episode_idx]
            episode_acs = self.acs_data[episode_idx]
            
            for t in range(len(episode_obs) - 1):
                obs = episode_obs[t]
                action = episode_acs[t]
                
                processed_obs = self.processor.process_training_observation(obs)
                
                train_obs.append(processed_obs)
                train_actions.append(action)
        
        train_obs = np.array(train_obs)
        train_actions = np.array(train_actions)
        
        print(f"   • Transitions: {len(train_obs):,}")
        print(f"   • Observation shape: {train_obs.shape}")
        print(f"   • Action shape: {train_actions.shape}")
        
        return train_obs, train_actions
    
    def create_policy(self, input_dim: int, action_dim: int) -> EnhancedDictPolicy:
        """Create policy with correct architecture"""
        policy = EnhancedDictPolicy(
            ac_dim=action_dim,
            ob_dim=input_dim,
            use_goal=False,
            goal_dim=0,
            goal_importance=1.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        print(f"\n🧠 POLICY ARCHITECTURE:")
        print(f"   • Input dim: {input_dim}")
        print(f"   • Action dim: {action_dim}")
        print(f"   • Data type: {self.data_spec.data_type.value}")
        print(f"   • Conditioning: {self.data_spec.conditioning_type.value}")
        
        return policy
    
    def evaluate_policy(self, policy, num_episodes: int = 5, max_steps: int = 100) -> Dict[str, Any]:
        """Evaluate policy using the correct environment"""
        try:
            os.environ['PYBULLET_EGL'] = '0'
            
            # Get the correct environment for this data specification
            env_name = get_environment_name(self.data_spec)
            env = gym.make(env_name)
            
            episodes = []
            episode_returns = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(num_episodes):
                env_obs, _ = env.reset()
                episode_return = 0
                episode_length = 0
                
                # Debug first episode
                if episode == 0:
                    print(f"      🔧 Debug - Environment: {env_name}")
                    print(f"      🔧 Debug - Raw env obs dim: {env_obs['observation'].shape}")
                    processed_debug = self.processor.process_evaluation_observation(env_obs)
                    print(f"      🔧 Debug - Processed obs dim: {processed_debug['observation'].shape}")
                    print(f"      🔧 Debug - Expected dim: {self.data_spec.final_obs_dim}")
                
                for step in range(max_steps):
                    processed_obs_dict = self.processor.process_evaluation_observation(env_obs)
                    action = policy.get_action(processed_obs_dict)
                    env_obs, reward, done, truncated, info = env.step(action)
                    episode_return += reward
                    episode_length += 1
                    
                    if done or truncated:
                        break
                
                success = info.get('is_success', False)
                if success:
                    success_count += 1
                
                episodes.append({
                    'episode': episode + 1,
                    'success': success,
                    'return': episode_return,
                    'length': episode_length
                })
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
            
            env.close()
            
            return {
                'environment_name': env_name,
                'episodes': episodes,
                'success_count': success_count,
                'total_episodes': num_episodes,
                'success_rate': success_count / num_episodes,
                'mean_return': np.mean(episode_returns),
                'std_return': np.std(episode_returns),
                'mean_episode_length': np.mean(episode_lengths)
            }
            
        except Exception as e:
            print(f"   ⚠️  Evaluation error: {str(e)}")
            return {
                'environment_name': 'unknown',
                'episodes': [],
                'success_count': 0,
                'total_episodes': num_episodes,
                'success_rate': 0.0,
                'mean_return': -100.0,
                'std_return': 0.0,
                'mean_episode_length': max_steps
            }
    
    def train(self, epochs: int = 100, batch_size: int = 64, eval_interval: int = 10, 
              eval_episodes: int = 5) -> Dict[str, Any]:
        """Train the BC policy"""
        print(f"\n🚀 TRAINING BC POLICY")
        print("="*60)
        
        # Initialize training tracker
        tracker = TrainingTracker(self.save_dir, self.data_spec)
        
        # Prepare data
        train_obs, train_actions = self.prepare_training_data()
        
        # Split train/val
        val_split = 0.1
        n_samples = len(train_obs)
        indices = np.random.permutation(n_samples)
        n_val = int(n_samples * val_split)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train, X_val = train_obs[train_indices], train_obs[val_indices]
        y_train, y_val = train_actions[train_indices], train_actions[val_indices]
        
        # Create policy
        input_dim = X_train.shape[1]
        action_dim = y_train.shape[1]
        policy = self.create_policy(input_dim, action_dim)
        
        # Training variables
        best_success_rate = 0
        train_losses = []
        eval_results_history = []
        
        print(f"\n📈 TRAINING PROGRESS:")
        print("="*60)
        
        for epoch in range(epochs):
            # Training phase
            policy.train()
            epoch_losses = []
            
            n_batches = len(X_train) // batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                # Create observation batch
                obs_batch = []
                for i in range(start_idx, end_idx):
                    obs_dict = {
                        'observation': X_train[i],
                        'achieved_goal': np.zeros(3),
                        'desired_goal': np.zeros(3)
                    }
                    obs_batch.append(obs_dict)
                
                obs_batch = np.array(obs_batch, dtype=object)
                action_batch = y_train[start_idx:end_idx]
                
                try:
                    loss = policy.update(obs_batch, action_batch)
                    epoch_losses.append(loss)
                except Exception as e:
                    print(f"   ❌ Training error: {e}")
                    continue
            
            # Log training metrics
            avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            train_losses.append(avg_train_loss)
            tracker.log_training_step(epoch, avg_train_loss)
            
            # Evaluation phase
            eval_results = None
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                print(f"\n   🎯 Evaluating at epoch {epoch+1}...")
                eval_results = self.evaluate_policy(policy, eval_episodes)
                eval_results_history.append((epoch, eval_results))
                
                # Log evaluation metrics
                tracker.log_evaluation(epoch, eval_results)
                
                print(f"   📈 Success: {eval_results['success_rate']:6.1%} | "
                      f"Return: {eval_results['mean_return']:7.2f} ± {eval_results['std_return']:5.2f}")
                
                # Save best model
                if eval_results['success_rate'] >= best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    self.save_model(policy, epoch, eval_results['success_rate'], input_dim, action_dim)
                    print(f"   💾 New best model saved! Success: {eval_results['success_rate']:.1%}")
                
                # Print progress summary every few evaluations
                if len(tracker.training_logs['eval_epochs']) >= 2:
                    tracker.print_progress_summary(epoch, epochs)
            
            # Progress update
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {avg_train_loss:.4f} | Best Success: {best_success_rate:.1%}")
        
        # Final visualization and logging
        print(f"\n📊 GENERATING TRAINING REPORTS...")
        tracker.plot_training_curves()
        tracker.plot_episode_distribution()
        tracker.save_detailed_logs()
        
        # Save results
        results = {
            'data_spec': {
                'data_type': self.data_spec.data_type.value,
                'conditioning_type': self.data_spec.conditioning_type.value,
                'num_blocks': self.data_spec.num_blocks,
                'has_colors': self.data_spec.has_colors,
                'final_obs_dim': self.data_spec.final_obs_dim
            },
            'training': {
                'train_losses': train_losses,
                'eval_results_history': eval_results_history
            },
            'best_success_rate': best_success_rate,
            'final_epoch': epochs,
            'training_summary': tracker.get_training_summary()
        }
        
        self.save_results(results)
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"   • Data type: {self.data_spec.data_type.value}")
        print(f"   • Conditioning: {self.data_spec.conditioning_type.value}")
        print(f"   • Best success rate: {best_success_rate:.1%}")
        
        return results
    
    def save_model(self, policy, epoch: int, success_rate: float, input_dim: int, action_dim: int):
        """Save model with configuration"""
        model_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(policy.state_dict(), model_path)
        
        config = {
            'data_spec': {
                'data_type': self.data_spec.data_type.value,
                'conditioning_type': self.data_spec.conditioning_type.value,
                'num_blocks': self.data_spec.num_blocks,
                'has_colors': self.data_spec.has_colors,
                'base_obs_dim': self.data_spec.base_obs_dim,
                'final_obs_dim': self.data_spec.final_obs_dim,
                'block_encoding_dim': self.data_spec.block_encoding_dim
            },
            'model_config': {
                'input_dim': input_dim,
                'action_dim': action_dim,
                'ob_dim': input_dim,
                'use_goal': False,
                'goal_dim': 0
            },
            'training_info': {
                'epoch': epoch,
                'success_rate': success_rate,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results"""
        results_path = os.path.join(self.save_dir, 'results.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

class UniversalBCEvaluator:
    """Universal BC evaluator that handles all data types"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Reconstruct data specification
        spec_data = self.config['data_spec']
        self.data_spec = DataSpec(
            data_type=DataType(spec_data['data_type']),
            conditioning_type=ConditioningType(spec_data['conditioning_type']),
            num_blocks=spec_data['num_blocks'],
            has_colors=spec_data['has_colors'],
            robot_state_dim=7,  # Standard
            base_obs_dim=spec_data['base_obs_dim'],
            final_obs_dim=spec_data['final_obs_dim'],
            block_encoding_dim=spec_data.get('block_encoding_dim')
        )
        
        self.processor = UniversalObservationProcessor(self.data_spec)
        
        print(f"🔍 UNIVERSAL BC EVALUATOR")
        print(f"Data type: {self.data_spec.data_type.value}")
        print(f"Conditioning: {self.data_spec.conditioning_type.value}")
        
        # Load model
        self.policy = self.load_model()
    
    def load_model(self) -> EnhancedDictPolicy:
        """Load the trained model"""
        model_path = os.path.join(self.model_dir, 'best_model.pt')
        
        model_config = self.config['model_config']
        
        policy = EnhancedDictPolicy(
            ac_dim=model_config['action_dim'],
            ob_dim=model_config['ob_dim'],
            use_goal=model_config['use_goal'],
            goal_dim=model_config['goal_dim'],
            goal_importance=1.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
        policy.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   • Input dim: {model_config['input_dim']}")
        print(f"   • Action dim: {model_config['action_dim']}")
        
        return policy
    
    def evaluate(self, num_episodes: int = 20, max_steps: int = 100, 
                verbose: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation"""
        
        if verbose:
            print(f"\n🎯 EVALUATING {self.data_spec.conditioning_type.value.upper()} MODEL")
            print("="*60)
        
        # Suppress PyBullet output
        os.environ['PYBULLET_EGL'] = '0'
        
        try:
            # Get the correct environment for this data specification
            env_name = get_environment_name(self.data_spec)
            env = gym.make(env_name)
            
            results = {
                'environment_name': env_name,
                'data_spec': {
                    'data_type': self.data_spec.data_type.value,
                    'conditioning_type': self.data_spec.conditioning_type.value,
                    'num_blocks': self.data_spec.num_blocks,
                    'has_colors': self.data_spec.has_colors,
                    'final_obs_dim': self.data_spec.final_obs_dim
                },
                'episodes': [],
                'success_count': 0,
                'total_episodes': num_episodes,
                'success_rate': 0.0,
                'mean_return': 0.0,
                'mean_episode_length': 0.0,
                'episode_returns': [],
                'episode_lengths': [],
                'success_episodes': [],
                'failed_episodes': [],
                'evaluation_errors': []
            }
            
            for episode in range(num_episodes):
                try:
                    env_obs, _ = env.reset()
                    episode_return = 0
                    episode_length = 0
                    
                    # Debug first episode
                    if episode == 0 and verbose:
                        print(f"   🔧 Debug Episode 1:")
                        print(f"      • Environment: {env_name}")
                        print(f"      • Raw env obs shape: {env_obs['observation'].shape}")
                        print(f"      • Training data type: {self.data_spec.data_type.value}")
                        print(f"      • Training conditioning: {self.data_spec.conditioning_type.value}")
                        print(f"      • Expected final dim: {self.data_spec.final_obs_dim}")
                        processed_debug = self.processor.process_evaluation_observation(env_obs)
                        print(f"      • Processed obs shape: {processed_debug['observation'].shape}")
                        
                        # Check for perfect match
                        # if len(env_obs['observation']) == self.data_spec.final_obs_dim:
                        #     print(f"      ✅ Perfect dimension match!")
                        # else:
                        #     print(f"      ⚠️  Dimension mismatch handled by truncation/padding")
                    
                    for step in range(max_steps):
                        # Process observation for policy
                        processed_obs_dict = self.processor.process_evaluation_observation(env_obs)
                        
                        # Get action from policy
                        action = self.policy.get_action(processed_obs_dict)
                        
                        # Take step
                        env_obs, reward, done, truncated, info = env.step(action)
                        episode_return += reward
                        episode_length += 1
                        
                        # Stop immediately when episode terminates
                        if done or truncated:
                            break
                    
                    # Record episode results
                    success = info.get('is_success', False)
                    episode_result = {
                        'episode': episode + 1,
                        'success': success,
                        'return': episode_return,
                        'length': episode_length,
                        'final_info': dict(info)
                    }
                    
                    results['episodes'].append(episode_result)
                    results['episode_returns'].append(episode_return)
                    results['episode_lengths'].append(episode_length)
                    
                    if success:
                        results['success_count'] += 1
                        results['success_episodes'].append(episode_result)
                    else:
                        results['failed_episodes'].append(episode_result)
                    
                    # Print progress
                    if verbose:
                        status = "✅" if success else "❌"
                        print(f"   Episode {episode+1:2d}: {status} Return: {episode_return:6.1f}, Steps: {episode_length:2d}")
                
                except Exception as e:
                    error_msg = f"Episode {episode+1} failed: {str(e)[:100]}"
                    results['evaluation_errors'].append(error_msg)
                    if verbose:
                        print(f"   ❌ {error_msg}")
            
            env.close()
            
            # Calculate final statistics
            if results['episode_returns']:
                results['success_rate'] = results['success_count'] / num_episodes
                results['mean_return'] = np.mean(results['episode_returns'])
                results['mean_episode_length'] = np.mean(results['episode_lengths'])
                results['std_return'] = np.std(results['episode_returns'])
                
                # Success-specific statistics
                if results['success_episodes']:
                    success_returns = [ep['return'] for ep in results['success_episodes']]
                    success_lengths = [ep['length'] for ep in results['success_episodes']]
                    results['success_mean_return'] = np.mean(success_returns)
                    results['success_mean_length'] = np.mean(success_lengths)
                else:
                    results['success_mean_return'] = 0.0
                    results['success_mean_length'] = 0.0
                
                # Failure-specific statistics
                if results['failed_episodes']:
                    failed_returns = [ep['return'] for ep in results['failed_episodes']]
                    failed_lengths = [ep['length'] for ep in results['failed_episodes']]
                    results['failed_mean_return'] = np.mean(failed_returns)
                    results['failed_mean_length'] = np.mean(failed_lengths)
                else:
                    results['failed_mean_return'] = 0.0
                    results['failed_mean_length'] = 0.0
            else:
                results['success_rate'] = 0.0
                results['mean_return'] = 0.0
                results['mean_episode_length'] = 0.0
                results['std_return'] = 0.0
            
            if verbose:
                print(f"\n📊 EVALUATION RESULTS:")
                print(f"   • Success Rate: {results['success_rate']:.1%} ({results['success_count']}/{num_episodes})")
                print(f"   • Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
                print(f"   • Mean Episode Length: {results['mean_episode_length']:.1f}")
                
                if results['success_episodes']:
                    print(f"   • Success Stats: Return={results['success_mean_return']:.2f}, Length={results['success_mean_length']:.1f}")
                if results['failed_episodes']:
                    print(f"   • Failure Stats: Return={results['failed_mean_return']:.2f}, Length={results['failed_mean_length']:.1f}")
                
                print(f"   • Evaluation Errors: {len(results['evaluation_errors'])}")
            
            return results
            
        except Exception as e:
            if verbose:
                print(f"❌ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
            return {
                'data_spec': {
                    'conditioning_type': self.data_spec.conditioning_type.value,
                    'data_type': self.data_spec.data_type.value
                },
                'error': str(e),
                'success_rate': 0.0,
                'mean_return': 0.0
            }
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to file"""
        if output_path is None:
            output_path = os.path.join(self.model_dir, 'evaluation_results.json')
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Universal BC Training and Evaluation System for PegTransfer')
    
    # Main operation modes
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], 
                       required=True, help='Operation mode')
    
    # Training arguments
    parser.add_argument('--data_path', type=str, help='Path to data file (.npz) for single training')
    parser.add_argument('--save_dir', type=str, help='Directory to save training results')
    
    # Data specification arguments (to avoid interactive prompts)
    parser.add_argument('--base_type', type=str, 
                       choices=['goal_conditioned_demos_with_all_blocks_2_blocks',
                               'goal_conditioned_demos_with_all_blocks',
                               'goal_conditioned_demos_with_all_blocks_colored_2_blocks',
                               'goal_conditioned_demos_with_all_blocks_colored',
                               'goal_conditioned_demos_with_only_target_block',
                               'goal_conditioned_demos_with_only_target_block_2_blocks',
                               'goal_conditioned_demos_with_only_target_block_2_blocks_no_color'],
                       help='Base data type (avoids interactive prompt)')
    parser.add_argument('--conditioning_type', type=str,
                       choices=['none', 'one_hot', 'target_block', 'target_peg', 
                               'target_block_and_peg', 'one_hot_and_target_peg', 'rgba_color'],
                       help='Conditioning type (avoids interactive prompt)')
    
    # Evaluation arguments
    parser.add_argument('--model_dir', type=str, help='Directory containing trained model for single evaluation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval during training')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Episodes for evaluation during training')
    
    # Evaluation parameters
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes for final evaluation')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_path or not args.save_dir:
            print("❌ Error: --data_path and --save_dir required for train mode")
            return
        
        print(f"🎯 TRAINING SINGLE MODEL")
        print(f"Data: {args.data_path}")
        print(f"Save Dir: {args.save_dir}")
        
        trainer = UniversalBCTrainer(args.data_path, args.save_dir, args.base_type, args.conditioning_type)
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes
        )
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   • Data Type: {trainer.data_spec.data_type.value}")
        print(f"   • Conditioning: {trainer.data_spec.conditioning_type.value}")
        print(f"   • Best Success Rate: {results['best_success_rate']:.1%}")
    
    elif args.mode == 'evaluate':
        if not args.model_dir:
            print("❌ Error: --model_dir required for evaluate mode")
            return
        
        print(f"🎯 EVALUATING SINGLE MODEL")
        print(f"Model Dir: {args.model_dir}")
        
        evaluator = UniversalBCEvaluator(args.model_dir)
        results = evaluator.evaluate(num_episodes=args.num_episodes, max_steps=args.max_steps)
        evaluator.save_evaluation_results(results)
        
        print(f"\n✅ EVALUATION COMPLETED!")
    
    else:
        print("❌ Invalid mode. Use --help for usage information.")

if __name__ == '__main__':
    import sys
    main()