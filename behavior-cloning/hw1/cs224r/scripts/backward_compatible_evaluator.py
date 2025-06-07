"""
Backward Compatible BC Evaluator for PegTransfer

This script can evaluate both old and new model formats:
- Old format: direct .pt files with separate config files
- New format: directories with config.json and best_model.pt

It automatically detects the format and adapts accordingly.
"""

import os
import torch
import numpy as np
import json
import argparse
import gymnasium as gym
import surrol.gym
from typing import Dict, List, Tuple, Any

from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

class BackwardCompatibleEvaluator:
    """Evaluator that handles both old and new model formats"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Detect if this is a file or directory
        if os.path.isfile(model_path):
            # Old format: direct .pt file
            self.model_file = model_path
            self.model_dir = os.path.dirname(model_path)
            self.is_old_format = True
            print(f"🔍 DETECTED OLD FORMAT MODEL FILE")
        elif os.path.isdir(model_path):
            # New format: directory with config.json and best_model.pt
            self.model_dir = model_path
            self.model_file = os.path.join(model_path, 'best_model.pt')
            self.is_old_format = False
            print(f"🔍 DETECTED NEW FORMAT MODEL DIRECTORY")
        else:
            raise ValueError(f"Path not found: {model_path}")
        
        # Load configuration
        self.config = self.load_config()
        
        # Extract conditioning info
        if self.is_old_format:
            # For old format, try to infer from filename
            self.conditioning_type = self.infer_conditioning_from_filename()
            self.is_postprocessed = False  # Old models were typically not post-processed
            self.data_analysis = {
                'conditioning_type': self.conditioning_type,
                'is_postprocessed': self.is_postprocessed,
                'base_obs_dim': 14,  # Default for PegTransfer
                'base_data_type': 'inferred'
            }
        else:
            # New format has full data analysis
            self.data_analysis = self.config['data_analysis']
            self.conditioning_type = self.data_analysis['conditioning_type']
            self.is_postprocessed = self.data_analysis['is_postprocessed']
        
        print(f"Model file: {self.model_file}")
        print(f"Conditioning: {self.conditioning_type}")
        print(f"Old format: {self.is_old_format}")
        
        # Load model
        self.policy = self.load_model()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with backward compatibility"""
        if self.is_old_format:
            # Try to find config in various places
            config_paths = [
                os.path.join(self.model_dir, 'config.json'),
                os.path.join(self.model_dir, 'config.pkl'),
                self.model_file.replace('.pt', '_config.json'),
            ]
            
            config = None
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        print(f"✅ Found config: {config_path}")
                        break
                    except:
                        continue
            
            if config is None:
                # Create minimal config for old format
                print("⚠️  No config found, creating minimal config")
                config = {
                    'action_dim': 6,  # Default for PegTransfer
                    'policy_config': {
                        'ob_dim': 14,  # Default base obs dim
                        'use_goal': False,
                        'goal_dim': 0
                    }
                }
            
            return config
        else:
            # New format
            config_path = os.path.join(self.model_dir, 'config.json')
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                return json.load(f)
    
    def infer_conditioning_from_filename(self) -> str:
        """Infer conditioning type from filename for old format"""
        filename = os.path.basename(self.model_file).lower()
        
        if 'onehot' in filename:
            return 'one_hot'
        elif 'targetblock' in filename:
            return 'target_block'
        elif 'fourtuple' in filename:
            return 'four_tuple'
        elif 'none' in filename:
            return 'none'
        else:
            # Default assumption
            return 'none'
    
    def load_model(self) -> EnhancedDictPolicy:
        """Load the trained model with correct architecture"""
        if not os.path.exists(self.model_file):
            raise ValueError(f"Model file not found: {self.model_file}")
        
        # Get policy configuration
        if 'policy_config' in self.config:
            policy_config = self.config['policy_config']
        else:
            # Create default policy config for old format
            policy_config = {
                'ob_dim': 14,
                'use_goal': self.conditioning_type != 'none',
                'goal_dim': 3 if self.conditioning_type != 'none' else 0
            }
        
        action_dim = self.config.get('action_dim', 6)
        
        policy = EnhancedDictPolicy(
            ac_dim=action_dim,
            ob_dim=policy_config['ob_dim'],
            use_goal=policy_config['use_goal'],
            goal_dim=policy_config['goal_dim'],
            goal_importance=2.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        # Load state dict
        try:
            state_dict = torch.load(self.model_file, map_location='cpu')
            policy.load_state_dict(state_dict)
            policy.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print(f"   • Policy input dim: {policy_config['ob_dim']}")
        print(f"   • Policy use goal: {policy_config['use_goal']}")
        print(f"   • Action dim: {action_dim}")
        
        return policy
    
    def process_environment_observation(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process environment observation based on conditioning type"""
        
        # Get base observation dimension
        base_obs_dim = self.data_analysis.get('base_obs_dim', 14)
        base_obs = env_obs['observation'][:base_obs_dim]
        
        if self.is_postprocessed:
            # For post-processed data, apply conditioning
            if self.conditioning_type == 'one_hot':
                block_encoding = np.zeros(4)  # Assume 4 blocks
                block_encoding[0] = 1
                processed_obs = np.concatenate([base_obs, block_encoding])
            elif self.conditioning_type == 'target_block':
                achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
                processed_obs = np.concatenate([base_obs, achieved_goal])
            elif self.conditioning_type == 'target_block_and_target_peg':
                achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
                desired_goal = env_obs.get('desired_goal', np.zeros(3))
                processed_obs = np.concatenate([base_obs, achieved_goal, desired_goal])
            else:
                processed_obs = base_obs
            
            return {
                'observation': processed_obs,
                'achieved_goal': np.zeros(3),
                'desired_goal': np.zeros(3)
            }
        else:
            # For raw data or old format, use base observation
            return {
                'observation': base_obs,
                'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
                'desired_goal': env_obs.get('desired_goal', np.zeros(3))
            }
    
    def evaluate(self, num_episodes: int = 20, max_steps: int = 100, 
                verbose: bool = True) -> Dict[str, Any]:
        """Evaluate the model with proper trajectory termination"""
        
        if verbose:
            print(f"\n🎯 EVALUATING {self.conditioning_type.upper()} MODEL")
            print("="*60)
        
        # Suppress PyBullet output
        os.environ['PYBULLET_EGL'] = '0'
        
        try:
            env = gym.make("PegTransfer-v0")
            
            results = {
                'conditioning_type': self.conditioning_type,
                'is_old_format': self.is_old_format,
                'model_path': self.model_path,
                'episodes': [],
                'success_count': 0,
                'total_episodes': num_episodes,
                'success_rate': 0.0,
                'mean_return': 0.0,
                'episode_returns': [],
                'episode_lengths': [],
                'evaluation_errors': []
            }
            
            for episode in range(num_episodes):
                try:
                    env_obs, _ = env.reset()
                    episode_return = 0
                    episode_length = 0
                    
                    for step in range(max_steps):
                        # Process observation for policy
                        processed_obs_dict = self.process_environment_observation(env_obs)
                        
                        # Get action from policy
                        action = self.policy.get_action(processed_obs_dict)
                        
                        # Take step
                        env_obs, reward, done, truncated, info = env.step(action)
                        episode_return += reward
                        episode_length += 1
                        
                        # CRITICAL: Stop immediately when episode terminates
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
                results['std_return'] = np.std(results['episode_returns'])
                results['mean_episode_length'] = np.mean(results['episode_lengths'])
            else:
                results['success_rate'] = 0.0
                results['mean_return'] = 0.0
                results['std_return'] = 0.0
                results['mean_episode_length'] = 0.0
            
            if verbose:
                print(f"\n📊 EVALUATION RESULTS:")
                print(f"   • Success Rate: {results['success_rate']:.1%} ({results['success_count']}/{num_episodes})")
                print(f"   • Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
                print(f"   • Mean Episode Length: {results['mean_episode_length']:.1f}")
                print(f"   • Evaluation Errors: {len(results['evaluation_errors'])}")
            
            return results
            
        except Exception as e:
            if verbose:
                print(f"❌ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
            return {
                'conditioning_type': self.conditioning_type,
                'error': str(e),
                'success_rate': 0.0,
                'mean_return': 0.0
            }
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to file"""
        if output_path is None:
            if self.is_old_format:
                output_path = self.model_file.replace('.pt', '_evaluation_results.json')
            else:
                output_path = os.path.join(self.model_dir, 'evaluation_results.json')
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Backward Compatible BC Evaluation for PegTransfer')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model file (.pt) or model directory')
    parser.add_argument('--num_episodes', type=int, default=20, 
                       help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=100, 
                       help='Maximum steps per episode')
    parser.add_argument('--output_file', type=str, 
                       help='Output file for results')
    parser.add_argument('--quiet', action='store_true', 
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator (handles both old and new formats)
        evaluator = BackwardCompatibleEvaluator(args.model_path)
        
        # Evaluate
        results = evaluator.evaluate(
            num_episodes=args.num_episodes, 
            max_steps=args.max_steps, 
            verbose=not args.quiet
        )
        
        # Save results
        if args.output_file:
            evaluator.save_evaluation_results(results, args.output_file)
        else:
            evaluator.save_evaluation_results(results)
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()