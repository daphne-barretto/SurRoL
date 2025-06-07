"""
Debug BC Trainer - Identifies and fixes dimension mismatches

This version adds extensive debugging to identify exactly what's happening
with observation dimensions during training and evaluation.
"""

import os
import torch
import numpy as np
import pickle
import json
import argparse
import matplotlib.pyplot as plt
import gymnasium as gym
import surrol.gym
from typing import Dict, Tuple, List, Any
import time
from datetime import datetime

from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

class DataTypeDetector:
    """Detects the type of data and its characteristics with detailed debugging"""
    
    @staticmethod
    def detect_data_type(data_path: str) -> Dict[str, Any]:
        """Detect data type and characteristics from file with extensive debugging"""
        print(f"🔍 DETECTING DATA TYPE")
        print("="*40)
        
        # Load data
        data = np.load(data_path, allow_pickle=True)
        obs_data = data['obs']
        sample_obs = obs_data[0][0]
        
        print(f"📊 DATA ANALYSIS:")
        print(f"   • File: {os.path.basename(data_path)}")
        print(f"   • Total episodes: {len(obs_data)}")
        print(f"   • Sample episode length: {len(obs_data[0])}")
        
        # Analyze filename for post-processing indicators
        filename = os.path.basename(data_path)
        is_postprocessed = any(suffix in filename for suffix in ['_onehot', '_targetblock', '_fourtuple'])
        
        if is_postprocessed:
            conditioning_type = DataTypeDetector._detect_conditioning_type(filename)
        else:
            conditioning_type = 'none'
        
        print(f"   • Is post-processed: {is_postprocessed}")
        print(f"   • Conditioning type: {conditioning_type}")
        
        # Detailed observation analysis
        print(f"\n🔬 OBSERVATION STRUCTURE:")
        if isinstance(sample_obs, dict):
            print(f"   • Observation type: Dictionary")
            print(f"   • Keys: {list(sample_obs.keys())}")
            
            for key, value in sample_obs.items():
                print(f"     - {key}: shape {np.array(value).shape}, dtype {np.array(value).dtype}")
                if key == 'observation':
                    print(f"       First 5 values: {np.array(value)[:5]}")
        else:
            print(f"   • Observation type: {type(sample_obs)}")
            print(f"   • Shape: {np.array(sample_obs).shape}")
        
        # Analyze observation components
        obs_analysis = {
            'filename': filename,
            'is_postprocessed': is_postprocessed,
            'conditioning_type': conditioning_type,
            'total_episodes': len(obs_data),
        }
        
        if isinstance(sample_obs, dict):
            obs_analysis['observation_keys'] = list(sample_obs.keys())
            obs_analysis['base_obs_dim'] = len(sample_obs['observation'])
            
            # Check for standard goal conditioning fields
            obs_analysis['has_achieved_goal'] = 'achieved_goal' in sample_obs
            obs_analysis['has_desired_goal'] = 'desired_goal' in sample_obs
            obs_analysis['has_block_encoding'] = 'block_encoding' in sample_obs
            
            if obs_analysis['has_achieved_goal']:
                obs_analysis['goal_dim'] = len(sample_obs['achieved_goal'])
                print(f"   • Achieved goal dim: {obs_analysis['goal_dim']}")
            if obs_analysis['has_desired_goal']:
                obs_analysis['desired_goal_dim'] = len(sample_obs['desired_goal'])
                print(f"   • Desired goal dim: {obs_analysis['desired_goal_dim']}")
            if obs_analysis['has_block_encoding']:
                obs_analysis['block_encoding_dim'] = len(sample_obs['block_encoding'])
                obs_analysis['num_blocks'] = len(sample_obs['block_encoding'])
                print(f"   • Block encoding dim: {obs_analysis['block_encoding_dim']}")
                print(f"   • Block encoding values: {sample_obs['block_encoding']}")
        
        # Detect base data type
        base_data_type = DataTypeDetector._detect_base_data_type(obs_analysis)
        obs_analysis['base_data_type'] = base_data_type
        
        print(f"\n✅ DETECTION RESULTS:")
        print(f"   • Base data type: {base_data_type}")
        print(f"   • Base observation dim: {obs_analysis['base_obs_dim']}")
        if 'num_blocks' in obs_analysis:
            print(f"   • Number of blocks: {obs_analysis['num_blocks']}")
        
        return obs_analysis
    
    @staticmethod
    def _detect_conditioning_type(filename: str) -> str:
        """Detect conditioning type from filename"""
        if '_onehottargetpeg' in filename:
            return 'one_hot_and_target_peg'
        elif '_targetblocktargetpeg' in filename:
            return 'target_block_and_target_peg'
        elif '_onehot' in filename:
            return 'one_hot'
        elif '_targetblock' in filename:
            return 'target_block'
        elif '_fourtuple' in filename:
            return 'four_tuple'
        else:
            return 'none'
    
    @staticmethod
    def _detect_base_data_type(obs_analysis: Dict) -> str:
        """Detect base data type from observation analysis"""
        base_dim = obs_analysis['base_obs_dim']
        num_blocks = obs_analysis.get('num_blocks', 0)
        
        if num_blocks == 4:
            if base_dim > 20:
                return 'all_blocks_colored_4'
            elif base_dim > 15:
                return 'all_blocks_4'
            else:
                return 'only_target_block_4'
        elif num_blocks == 2:
            if base_dim > 15:
                return 'all_blocks_colored_2'
            else:
                return 'all_blocks_2'
        else:
            if base_dim > 20:
                return 'all_blocks_colored_4'
            elif base_dim > 15:
                return 'all_blocks_4'
            else:
                return 'all_blocks_2'

class DebugObservationProcessor:
    """Observation processor with extensive debugging"""
    
    def __init__(self, data_analysis: Dict[str, Any]):
        self.data_analysis = data_analysis
        self.conditioning_type = data_analysis['conditioning_type']
        self.is_postprocessed = data_analysis['is_postprocessed']
        self.base_obs_dim = data_analysis['base_obs_dim']
        
        print(f"\n🔧 OBSERVATION PROCESSOR")
        print(f"   • Conditioning: {self.conditioning_type}")
        print(f"   • Post-processed: {self.is_postprocessed}")
        print(f"   • Base obs dim: {self.base_obs_dim}")
    
    def process_observation_for_training(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Process observation for training with debugging"""
        if self.is_postprocessed:
            # For post-processed data, the conditioning is already applied
            processed_obs = obs['observation']
            print(f"   🔧 Post-processed training obs: {processed_obs.shape}")
            return processed_obs
        else:
            # For raw data, apply conditioning
            processed_obs = self._apply_conditioning(obs)
            print(f"   🔧 Raw training obs processed: {processed_obs.shape}")
            return processed_obs
    
    def process_observation_for_evaluation(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process environment observation for evaluation with debugging"""
        print(f"   🔧 Environment obs keys: {list(env_obs.keys())}")
        print(f"   🔧 Environment obs['observation'] shape: {env_obs['observation'].shape}")
        
        if self.is_postprocessed:
            # For post-processed training data, we need to apply the same processing to env obs
            processed_obs = self._apply_conditioning_to_env_obs(env_obs)
            print(f"   🔧 Post-processed eval obs: {processed_obs.shape}")
        else:
            # For raw training data, use base observation
            processed_obs = env_obs['observation'][:self.base_obs_dim]
            print(f"   🔧 Raw eval obs: {processed_obs.shape}")
        
        result = {
            'observation': processed_obs,
            'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
            'desired_goal': env_obs.get('desired_goal', np.zeros(3))
        }
        
        print(f"   🔧 Final eval obs dict: observation={result['observation'].shape}")
        return result
    
    def _apply_conditioning(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply conditioning to raw observation"""
        base_obs = obs['observation']
        
        if self.conditioning_type == 'none':
            return base_obs
        elif self.conditioning_type == 'one_hot' and 'block_encoding' in obs:
            result = np.concatenate([base_obs, obs['block_encoding']])
            print(f"     • Applied one_hot: {base_obs.shape} + {obs['block_encoding'].shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'target_block' and 'achieved_goal' in obs:
            result = np.concatenate([base_obs, obs['achieved_goal']])
            print(f"     • Applied target_block: {base_obs.shape} + {obs['achieved_goal'].shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'target_block_and_target_peg':
            if 'achieved_goal' in obs and 'desired_goal' in obs:
                result = np.concatenate([base_obs, obs['achieved_goal'], obs['desired_goal']])
                print(f"     • Applied target_block_and_target_peg: {base_obs.shape} + {obs['achieved_goal'].shape} + {obs['desired_goal'].shape} = {result.shape}")
                return result
        elif self.conditioning_type == 'one_hot_and_target_peg':
            if 'block_encoding' in obs and 'desired_goal' in obs:
                result = np.concatenate([base_obs, obs['block_encoding'], obs['desired_goal']])
                print(f"     • Applied one_hot_and_target_peg: {base_obs.shape} + {obs['block_encoding'].shape} + {obs['desired_goal'].shape} = {result.shape}")
                return result
        elif self.conditioning_type == 'four_tuple' and 'block_encoding' in obs:
            result = np.concatenate([base_obs, obs['block_encoding']])
            print(f"     • Applied four_tuple: {base_obs.shape} + {obs['block_encoding'].shape} = {result.shape}")
            return result
        
        print(f"     • No conditioning applied, returning base_obs: {base_obs.shape}")
        return base_obs
    
    def _apply_conditioning_to_env_obs(self, env_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply conditioning to environment observation to match training format"""
        base_obs = env_obs['observation'][:self.base_obs_dim]
        print(f"     • Base env obs: {base_obs.shape}")
        
        if self.conditioning_type == 'none':
            return base_obs
        elif self.conditioning_type == 'one_hot':
            # Create block encoding matching training data
            block_encoding_dim = self.data_analysis.get('block_encoding_dim', 2)
            block_encoding = np.zeros(block_encoding_dim)
            block_encoding[0] = 1  # Default to first block
            result = np.concatenate([base_obs, block_encoding])
            print(f"     • Applied one_hot to env: {base_obs.shape} + {block_encoding.shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'target_block':
            achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
            result = np.concatenate([base_obs, achieved_goal])
            print(f"     • Applied target_block to env: {base_obs.shape} + {achieved_goal.shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'target_block_and_target_peg':
            achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
            desired_goal = env_obs.get('desired_goal', np.zeros(3))
            result = np.concatenate([base_obs, achieved_goal, desired_goal])
            print(f"     • Applied target_block_and_target_peg to env: {base_obs.shape} + {achieved_goal.shape} + {desired_goal.shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'one_hot_and_target_peg':
            block_encoding_dim = self.data_analysis.get('block_encoding_dim', 2)
            block_encoding = np.zeros(block_encoding_dim)
            block_encoding[0] = 1
            desired_goal = env_obs.get('desired_goal', np.zeros(3))
            result = np.concatenate([base_obs, block_encoding, desired_goal])
            print(f"     • Applied one_hot_and_target_peg to env: {base_obs.shape} + {block_encoding.shape} + {desired_goal.shape} = {result.shape}")
            return result
        elif self.conditioning_type == 'four_tuple':
            # Create RGBA encoding
            rgba_encoding = np.array([1.0, 0.0, 0.0, 1.0])  # Default red
            result = np.concatenate([base_obs, rgba_encoding])
            print(f"     • Applied four_tuple to env: {base_obs.shape} + {rgba_encoding.shape} = {result.shape}")
            return result
        
        return base_obs

class DebugBCTrainer:
    """BC trainer with extensive debugging for dimension mismatches"""
    
    def __init__(self, data_path: str, save_dir: str):
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Detect data type and characteristics
        self.data_analysis = DataTypeDetector.detect_data_type(data_path)
        self.processor = DebugObservationProcessor(self.data_analysis)
        
        # Load actual data
        self.data = np.load(data_path, allow_pickle=True)
        self.obs_data = self.data['obs']
        self.acs_data = self.data['acs']
        
        print(f"\n🎯 DEBUG BC TRAINER")
        print(f"Data type: {self.data_analysis['base_data_type']}")
        print(f"Conditioning: {self.data_analysis['conditioning_type']}")
        print(f"Episodes: {len(self.obs_data)}")
    
    def analyze_training_data_dimensions(self) -> Dict[str, Any]:
        """Analyze dimensions of training data"""
        print(f"\n🔍 ANALYZING TRAINING DATA DIMENSIONS")
        print("="*50)
        
        sample_obs_dims = []
        sample_action_dims = []
        
        # Analyze first few episodes
        for episode_idx in range(min(3, len(self.obs_data))):
            episode_obs = self.obs_data[episode_idx]
            episode_acs = self.acs_data[episode_idx]
            
            print(f"\n   Episode {episode_idx + 1}:")
            print(f"   • Length: {len(episode_obs)}")
            
            # Check first few observations
            for t in range(min(3, len(episode_obs) - 1)):
                obs = episode_obs[t]
                action = episode_acs[t]
                
                processed_obs = self.processor.process_observation_for_training(obs)
                
                sample_obs_dims.append(len(processed_obs))
                sample_action_dims.append(len(action))
                
                if episode_idx == 0 and t == 0:
                    print(f"   • Raw obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'not dict'}")
                    if isinstance(obs, dict):
                        for key, value in obs.items():
                            print(f"     - {key}: {np.array(value).shape}")
                    print(f"   • Processed obs dim: {len(processed_obs)}")
                    print(f"   • Action dim: {len(action)}")
        
        obs_dim_stats = {
            'min': min(sample_obs_dims),
            'max': max(sample_obs_dims),
            'unique_dims': list(set(sample_obs_dims)),
            'most_common': max(set(sample_obs_dims), key=sample_obs_dims.count)
        }
        
        action_dim_stats = {
            'min': min(sample_action_dims),
            'max': max(sample_action_dims),
            'unique_dims': list(set(sample_action_dims)),
            'most_common': max(set(sample_action_dims), key=sample_action_dims.count)
        }
        
        print(f"\n📊 DIMENSION STATISTICS:")
        print(f"   • Observation dims: {obs_dim_stats}")
        print(f"   • Action dims: {action_dim_stats}")
        
        return {
            'obs_dims': obs_dim_stats,
            'action_dims': action_dim_stats
        }
    
    def create_policy(self, input_dim: int, action_dim: int) -> EnhancedDictPolicy:
        """Create policy with correct architecture and debugging"""
        print(f"\n🧠 CREATING POLICY")
        print("="*30)
        
        # Determine if we should use goal processing
        use_goal = (self.data_analysis['conditioning_type'] != 'none' and 
                   not self.data_analysis['is_postprocessed'])
        
        if self.data_analysis['is_postprocessed']:
            # Post-processed data: conditioning is in the observation
            ob_dim = input_dim  # Use the actual processed input dimension
            goal_dim = 0
            use_goal = False
            print(f"   • Post-processed data detected")
            print(f"   • Policy will expect obs dim: {ob_dim}")
        else:
            # Raw data: let policy handle goal processing if needed
            ob_dim = self.data_analysis['base_obs_dim']
            goal_dim = self.data_analysis.get('goal_dim', 3) if use_goal else 0
            print(f"   • Raw data detected")
            print(f"   • Policy will expect base obs dim: {ob_dim}")
            print(f"   • Goal processing: {use_goal}")
        
        policy = EnhancedDictPolicy(
            ac_dim=action_dim,
            ob_dim=ob_dim,
            use_goal=use_goal,
            goal_dim=goal_dim,
            goal_importance=2.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        print(f"   ✅ Policy created:")
        print(f"      - Expected obs dim: {ob_dim}")
        print(f"      - Action dim: {action_dim}")
        print(f"      - Use goal: {use_goal}")
        print(f"      - Goal dim: {goal_dim}")
        
        return policy
    
    def debug_evaluate_policy(self, policy, num_episodes: int = 2) -> Dict[str, Any]:
        """Evaluate policy with extensive debugging"""
        print(f"\n🔍 DEBUG EVALUATION")
        print("="*30)
        
        try:
            os.environ['PYBULLET_EGL'] = '0'
            env = gym.make("PegTransfer-v0")
            
            for episode in range(num_episodes):
                print(f"\n   Episode {episode + 1}:")
                env_obs, _ = env.reset()
                
                print(f"   • Environment reset")
                print(f"   • Raw env obs keys: {list(env_obs.keys())}")
                print(f"   • Raw env obs['observation'] shape: {env_obs['observation'].shape}")
                
                # Process observation
                try:
                    processed_obs_dict = self.processor.process_observation_for_evaluation(env_obs)
                    print(f"   • Processed obs dict keys: {list(processed_obs_dict.keys())}")
                    print(f"   • Processed obs shape: {processed_obs_dict['observation'].shape}")
                    
                    # Try to get action
                    try:
                        action = policy.get_action(processed_obs_dict)
                        print(f"   • Action shape: {action.shape}")
                        print(f"   ✅ Success! No dimension mismatch")
                        break
                    except Exception as action_error:
                        print(f"   ❌ Action error: {action_error}")
                        print(f"   • This suggests a dimension mismatch between processed obs and policy")
                        break
                        
                except Exception as process_error:
                    print(f"   ❌ Processing error: {process_error}")
                    break
            
            env.close()
            
            return {'debug_completed': True}
            
        except Exception as e:
            print(f"   ❌ Debug evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'debug_completed': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Debug BC Training for PegTransfer')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file (.npz)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--debug_only', action='store_true', help='Only run debugging, no training')
    
    args = parser.parse_args()
    
    print(f"🔍 DEBUG BC TRAINING")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"Save Dir: {args.save_dir}")
    
    try:
        # Create trainer
        trainer = DebugBCTrainer(args.data_path, args.save_dir)
        
        # Analyze training data dimensions
        dim_analysis = trainer.analyze_training_data_dimensions()
        
        # Create policy with detected dimensions
        obs_dim = dim_analysis['obs_dims']['most_common']
        action_dim = dim_analysis['action_dims']['most_common']
        
        policy = trainer.create_policy(obs_dim, action_dim)
        
        # Debug evaluation
        debug_results = trainer.debug_evaluate_policy(policy)
        
        if not args.debug_only and debug_results.get('debug_completed', False):
            print(f"\n✅ Debug successful! Dimensions are compatible.")
            print(f"You can now run full training with:")
            print(f"python universal_bc_trainer.py --data_path {args.data_path} --save_dir {args.save_dir}")
        else:
            print(f"\n❌ Debug revealed dimension issues. Check the output above.")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()