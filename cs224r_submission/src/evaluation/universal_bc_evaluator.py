"""
Fixed Universal BC Evaluator for PegTransfer

This evaluator works with the fixed trainer and correctly handles
dimension mismatches for post-processed data.

Key fix: For post-processed training data, don't apply conditioning during evaluation.
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

class FixedUniversalBCEvaluator:
    """Fixed evaluator compatible with the new trainer format"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract data analysis and conditioning info
        self.data_analysis = self.config['data_analysis']
        self.conditioning_type = self.data_analysis['conditioning_type']
        self.is_postprocessed = self.data_analysis['is_postprocessed']
        self.training_obs_dim = self.data_analysis['training_obs_dim']
        
        print(f"🔍 FIXED UNIVERSAL BC EVALUATOR")
        print(f"Data Type: {self.data_analysis['base_data_type']}")
        print(f"Conditioning: {self.conditioning_type}")
        print(f"Post-processed: {self.is_postprocessed}")
        print(f"Training obs dim: {self.training_obs_dim}")
        
        # Load model
        self.policy = self.load_model()
    
    def load_model(self) -> EnhancedDictPolicy:
        """Load the trained model with correct architecture"""
        model_path = os.path.join(self.model_dir, 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Create policy with same architecture as training
        policy_config = self.config['policy_config']
        
        policy = EnhancedDictPolicy(
            ac_dim=self.config['action_dim'],
            ob_dim=policy_config['ob_dim'],
            use_goal=policy_config['use_goal'],
            goal_dim=policy_config['goal_dim'],
            goal_importance=2.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
        policy.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   • Model expects input dim: {policy_config['ob_dim']}")
        print(f"   • Use goal: {policy_config['use_goal']}")
        print(f"   • Trained on data with conditioning: {self.conditioning_type}")
        
        return policy
    
    def process_environment_observation(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process environment observation with FIXED dimension handling"""
        
        if self.is_postprocessed:
            # CRITICAL FIX: For post-processed training data, 
            # DON'T apply conditioning during evaluation!
            # Just use the first N dimensions that match the training data
            processed_obs = env_obs['observation'][:self.training_obs_dim]
            
            # If the environment observation is shorter than expected, pad with zeros
            if len(processed_obs) < self.training_obs_dim:
                padding = np.zeros(self.training_obs_dim - len(processed_obs))
                processed_obs = np.concatenate([processed_obs, padding])
            
            return {
                'observation': processed_obs,
                'achieved_goal': np.zeros(3),
                'desired_goal': np.zeros(3)
            }
        else:
            # For raw training data, apply conditioning to match training format
            original_base_dim = self.data_analysis.get('original_base_dim', self.training_obs_dim)
            base_obs = env_obs['observation'][:original_base_dim]
            
            if self.conditioning_type == 'none':
                processed_obs = base_obs
            elif self.conditioning_type == 'one_hot':
                block_encoding = np.zeros(self.data_analysis.get('block_encoding_dim', 2))
                block_encoding[0] = 1  # Default to first block
                processed_obs = np.concatenate([base_obs, block_encoding])
            elif self.conditioning_type == 'target_block':
                achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
                processed_obs = np.concatenate([base_obs, achieved_goal])
            elif self.conditioning_type == 'target_block_and_target_peg':
                achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
                desired_goal = env_obs.get('desired_goal', np.zeros(3))
                processed_obs = np.concatenate([base_obs, achieved_goal, desired_goal])
            elif self.conditioning_type == 'one_hot_and_target_peg':
                block_encoding = np.zeros(self.data_analysis.get('block_encoding_dim', 2))
                block_encoding[0] = 1  # Default
                desired_goal = env_obs.get('desired_goal', np.zeros(3))
                processed_obs = np.concatenate([base_obs, block_encoding, desired_goal])
            elif self.conditioning_type == 'four_tuple':
                rgba_encoding = np.array([1.0, 0.0, 0.0, 1.0])  # Default red
                processed_obs = np.concatenate([base_obs, rgba_encoding])
            else:
                processed_obs = base_obs
            
            return {
                'observation': processed_obs,
                'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
                'desired_goal': env_obs.get('desired_goal', np.zeros(3))
            }
    
    def evaluate(self, num_episodes: int = 20, max_steps: int = 100, 
                verbose: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation with proper trajectory termination and fixed dimensions"""
        
        if verbose:
            print(f"\n🎯 EVALUATING {self.conditioning_type.upper()} MODEL")
            print("="*60)
        
        # Suppress PyBullet output
        os.environ['PYBULLET_EGL'] = '0'
        
        try:
            env = gym.make("PegTransfer-v0")
            
            results = {
                'conditioning_type': self.conditioning_type,
                'is_postprocessed': self.is_postprocessed,
                'training_obs_dim': self.training_obs_dim,
                'data_analysis': self.data_analysis,
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
                        print(f"      • Raw env obs shape: {env_obs['observation'].shape}")
                        processed_debug = self.process_environment_observation(env_obs)
                        print(f"      • Processed obs shape: {processed_debug['observation'].shape}")
                        print(f"      • Expected by model: {self.training_obs_dim}")
                    
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
                'conditioning_type': self.conditioning_type,
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

def compare_trained_models(model_dirs: List[str], num_episodes: int = 10):
    """Compare multiple trained models"""
    print(f"\n📊 COMPARING TRAINED BC MODELS")
    print("="*80)
    
    comparison_results = {}
    
    for model_dir in model_dirs:
        try:
            evaluator = FixedUniversalBCEvaluator(model_dir)
            results = evaluator.evaluate(num_episodes=num_episodes, verbose=False)
            
            conditioning_type = results['conditioning_type']
            comparison_results[conditioning_type] = results
            
            model_name = os.path.basename(model_dir)
            print(f"   {model_name:30} | {conditioning_type:15} | Success: {results['success_rate']:6.1%} | Return: {results['mean_return']:7.2f}")
            
        except Exception as e:
            model_name = os.path.basename(model_dir)
            print(f"   {model_name:30} | ERROR: {str(e)[:40]}...")
            comparison_results[model_name] = {'error': str(e), 'success_rate': 0.0, 'mean_return': 0.0}
    
    return comparison_results

def batch_evaluate_experiments_directory(experiments_dir: str, num_episodes: int = 10):
    """Evaluate all models in an experiments directory"""
    print(f"\n🔍 BATCH EVALUATION")
    print("="*80)
    print(f"Directory: {experiments_dir}")
    
    if not os.path.exists(experiments_dir):
        print(f"❌ Directory not found: {experiments_dir}")
        return {}
    
    # Find all model directories (those with config.json and best_model.pt)
    model_dirs = []
    
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, 'config.json')
            model_path = os.path.join(item_path, 'best_model.pt')
            
            if os.path.exists(config_path) and os.path.exists(model_path):
                model_dirs.append(item_path)
    
    if not model_dirs:
        print(f"❌ No valid models found in {experiments_dir}")
        print("   Looking for directories with 'config.json' and 'best_model.pt'")
        return {}
    
    print(f"Found {len(model_dirs)} models to evaluate")
    
    # Evaluate all models
    results = compare_trained_models(model_dirs, num_episodes)
    
    # Save batch results
    batch_results_path = os.path.join(experiments_dir, 'batch_evaluation_results.json')
    with open(batch_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Batch results saved to: {batch_results_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Fixed Universal BC Evaluation for PegTransfer')
    parser.add_argument('--model_dir', type=str, help='Directory containing trained model')
    parser.add_argument('--experiments_dir', type=str, help='Directory containing multiple experiments for batch evaluation')
    parser.add_argument('--compare_dirs', type=str, nargs='+', help='Multiple model directories to compare')
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--output_file', type=str, help='Output file for results')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if args.experiments_dir:
        # Batch evaluation
        results = batch_evaluate_experiments_directory(args.experiments_dir, args.num_episodes)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Batch results saved to: {args.output_file}")
        
    elif args.compare_dirs:
        # Compare specific model directories
        results = compare_trained_models(args.compare_dirs, args.num_episodes)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Comparison results saved to: {args.output_file}")
    
    elif args.model_dir:
        # Single model evaluation
        evaluator = FixedUniversalBCEvaluator(args.model_dir)
        results = evaluator.evaluate(num_episodes=args.num_episodes, max_steps=args.max_steps, verbose=not args.quiet)
        
        if args.output_file:
            evaluator.save_evaluation_results(results, args.output_file)
        else:
            evaluator.save_evaluation_results(results)
    
    else:
        print("❌ Error: Must specify either:")
        print("   • --model_dir for single evaluation")
        print("   • --experiments_dir for batch evaluation")
        print("   • --compare_dirs for comparison")
        print("\nExamples:")
        print("   python fixed_universal_bc_evaluator.py --model_dir /path/to/model")
        print("   python fixed_universal_bc_evaluator.py --experiments_dir /path/to/experiments")
        print("   python fixed_universal_bc_evaluator.py --compare_dirs /path/to/model1 /path/to/model2")
        return

if __name__ == '__main__':
    main()