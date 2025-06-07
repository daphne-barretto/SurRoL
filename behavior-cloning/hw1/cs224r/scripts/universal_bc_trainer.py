"""
Fixed Universal BC Training System for PegTransfer

This script fixes the dimension mismatch issue by correctly handling
post-processed data during evaluation.

Key fix: For post-processed training data, don't apply conditioning during evaluation
since the conditioning is already baked into the training observations.
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
    """Detects the type of data and its characteristics"""
    
    @staticmethod
    def detect_data_type(data_path: str) -> Dict[str, Any]:
        """Detect data type and characteristics from file"""
        print(f"🔍 DETECTING DATA TYPE")
        print("="*40)
        
        # Load data
        data = np.load(data_path, allow_pickle=True)
        obs_data = data['obs']
        sample_obs = obs_data[0][0]
        
        # Analyze filename for post-processing indicators
        filename = os.path.basename(data_path)
        is_postprocessed = any(suffix in filename for suffix in ['_onehot', '_targetblock', '_fourtuple'])
        
        if is_postprocessed:
            conditioning_type = DataTypeDetector._detect_conditioning_type(filename)
        else:
            conditioning_type = 'none'
        
        # Analyze observation structure
        obs_analysis = {
            'filename': filename,
            'is_postprocessed': is_postprocessed,
            'conditioning_type': conditioning_type,
            'total_episodes': len(obs_data),
        }
        
        # Analyze observation components
        if isinstance(sample_obs, dict):
            obs_analysis['observation_keys'] = list(sample_obs.keys())
            obs_analysis['base_obs_dim'] = len(sample_obs['observation'])
            
            # Check for standard goal conditioning fields
            obs_analysis['has_achieved_goal'] = 'achieved_goal' in sample_obs
            obs_analysis['has_desired_goal'] = 'desired_goal' in sample_obs
            obs_analysis['has_block_encoding'] = 'block_encoding' in sample_obs
            
            if obs_analysis['has_achieved_goal']:
                obs_analysis['goal_dim'] = len(sample_obs['achieved_goal'])
            if obs_analysis['has_block_encoding']:
                obs_analysis['block_encoding_dim'] = len(sample_obs['block_encoding'])
                obs_analysis['num_blocks'] = len(sample_obs['block_encoding'])
        
        # For post-processed data, the 'base_obs_dim' is actually the FULL observation
        # including the conditioning information
        if is_postprocessed:
            obs_analysis['training_obs_dim'] = obs_analysis['base_obs_dim']
            # Estimate the original base dimension before conditioning was applied
            if conditioning_type == 'one_hot':
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim'] - obs_analysis.get('block_encoding_dim', 2)
            elif conditioning_type == 'target_block':
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim'] - 3
            elif conditioning_type == 'target_block_and_target_peg':
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim'] - 6
            elif conditioning_type == 'one_hot_and_target_peg':
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim'] - obs_analysis.get('block_encoding_dim', 2) - 3
            elif conditioning_type == 'four_tuple':
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim'] - 4
            else:
                obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim']
        else:
            obs_analysis['training_obs_dim'] = obs_analysis['base_obs_dim']
            obs_analysis['original_base_dim'] = obs_analysis['base_obs_dim']
        
        # Detect base data type
        base_data_type = DataTypeDetector._detect_base_data_type(obs_analysis)
        obs_analysis['base_data_type'] = base_data_type
        
        print(f"   • File: {filename}")
        print(f"   • Base data type: {base_data_type}")
        print(f"   • Is post-processed: {is_postprocessed}")
        print(f"   • Conditioning type: {conditioning_type}")
        print(f"   • Training obs dim: {obs_analysis['training_obs_dim']}")
        if is_postprocessed:
            print(f"   • Original base dim: {obs_analysis['original_base_dim']}")
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
        base_dim = obs_analysis.get('original_base_dim', obs_analysis['base_obs_dim'])
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

class FixedObservationProcessor:
    """Fixed observation processor that handles post-processed data correctly"""
    
    def __init__(self, data_analysis: Dict[str, Any]):
        self.data_analysis = data_analysis
        self.conditioning_type = data_analysis['conditioning_type']
        self.is_postprocessed = data_analysis['is_postprocessed']
        self.training_obs_dim = data_analysis['training_obs_dim']
        self.original_base_dim = data_analysis.get('original_base_dim', data_analysis['training_obs_dim'])
        
        print(f"🔧 FIXED OBSERVATION PROCESSOR")
        print(f"   • Conditioning: {self.conditioning_type}")
        print(f"   • Post-processed: {self.is_postprocessed}")
        print(f"   • Training obs dim: {self.training_obs_dim}")
        print(f"   • Original base dim: {self.original_base_dim}")
    
    def process_observation_for_training(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Process observation for training"""
        if self.is_postprocessed:
            # For post-processed data, conditioning is already applied
            return obs['observation']
        else:
            # For raw data, apply conditioning
            return self._apply_conditioning(obs)
    
    def process_observation_for_evaluation(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process environment observation for evaluation - FIXED VERSION"""
        
        if self.is_postprocessed:
            # CRITICAL FIX: For post-processed training data, 
            # DON'T apply conditioning during evaluation!
            # Just use the base observation that matches the training dimension
            processed_obs = env_obs['observation'][:self.training_obs_dim]
            
            print(f"   🔧 Post-processed mode: using first {self.training_obs_dim} dims from env obs")
        else:
            # For raw training data, apply conditioning to match training format
            processed_obs = self._apply_conditioning_to_env_obs(env_obs)
            print(f"   🔧 Raw mode: applied conditioning, result dim: {len(processed_obs)}")
        
        return {
            'observation': processed_obs,
            'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
            'desired_goal': env_obs.get('desired_goal', np.zeros(3))
        }
    
    def _apply_conditioning(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply conditioning to raw observation"""
        base_obs = obs['observation']
        
        if self.conditioning_type == 'none':
            return base_obs
        elif self.conditioning_type == 'one_hot' and 'block_encoding' in obs:
            return np.concatenate([base_obs, obs['block_encoding']])
        elif self.conditioning_type == 'target_block' and 'achieved_goal' in obs:
            return np.concatenate([base_obs, obs['achieved_goal']])
        elif self.conditioning_type == 'target_block_and_target_peg':
            if 'achieved_goal' in obs and 'desired_goal' in obs:
                return np.concatenate([base_obs, obs['achieved_goal'], obs['desired_goal']])
        elif self.conditioning_type == 'one_hot_and_target_peg':
            if 'block_encoding' in obs and 'desired_goal' in obs:
                return np.concatenate([base_obs, obs['block_encoding'], obs['desired_goal']])
        elif self.conditioning_type == 'four_tuple' and 'block_encoding' in obs:
            return np.concatenate([base_obs, obs['block_encoding']])
        
        return base_obs
    
    def _apply_conditioning_to_env_obs(self, env_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply conditioning to environment observation for raw training data"""
        base_obs = env_obs['observation'][:self.original_base_dim]
        
        if self.conditioning_type == 'none':
            return base_obs
        elif self.conditioning_type == 'one_hot':
            block_encoding = np.zeros(self.data_analysis.get('block_encoding_dim', 2))
            block_encoding[0] = 1
            return np.concatenate([base_obs, block_encoding])
        elif self.conditioning_type == 'target_block':
            achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
            return np.concatenate([base_obs, achieved_goal])
        elif self.conditioning_type == 'target_block_and_target_peg':
            achieved_goal = env_obs.get('achieved_goal', np.zeros(3))
            desired_goal = env_obs.get('desired_goal', np.zeros(3))
            return np.concatenate([base_obs, achieved_goal, desired_goal])
        elif self.conditioning_type == 'one_hot_and_target_peg':
            block_encoding = np.zeros(self.data_analysis.get('block_encoding_dim', 2))
            block_encoding[0] = 1
            desired_goal = env_obs.get('desired_goal', np.zeros(3))
            return np.concatenate([base_obs, block_encoding, desired_goal])
        elif self.conditioning_type == 'four_tuple':
            rgba_encoding = np.array([1.0, 0.0, 0.0, 1.0])
            return np.concatenate([base_obs, rgba_encoding])
        
        return base_obs

class TrainingLogger:
    """Enhanced training logger with detailed metrics tracking"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.start_time = time.time()
        
        # Training metrics
        self.train_losses = []
        self.eval_success_rates = []
        self.eval_mean_returns = []
        self.eval_std_returns = []
        self.eval_episodes_data = []
        self.eval_epochs = []
        
        # Timing
        self.epoch_times = []
        
        print(f"📊 TRAINING LOGGER INITIALIZED")
    
    def log_epoch(self, epoch: int, train_loss: float):
        """Log training metrics for an epoch"""
        self.train_losses.append(train_loss)
        
        # Log timing
        current_time = time.time()
        if len(self.epoch_times) == 0:
            epoch_duration = current_time - self.start_time
        else:
            epoch_duration = current_time - sum(self.epoch_times) - self.start_time
        self.epoch_times.append(epoch_duration)
    
    def log_evaluation(self, epoch: int, eval_results: Dict[str, Any]):
        """Log evaluation results"""
        self.eval_epochs.append(epoch)
        self.eval_success_rates.append(eval_results['success_rate'])
        self.eval_mean_returns.append(eval_results['mean_return'])
        self.eval_std_returns.append(eval_results.get('std_return', 0.0))
        self.eval_episodes_data.append(eval_results['episodes'])
        
        print(f"   📈 Epoch {epoch:3d} | Success: {eval_results['success_rate']:6.1%} | "
              f"Return: {eval_results['mean_return']:7.2f} ± {eval_results.get('std_return', 0):5.2f}")
    
    def save_training_curves(self):
        """Save training curves as plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_losses, 'b-', label='Training Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        if self.eval_epochs:
            axes[0, 1].plot(self.eval_epochs, self.eval_success_rates, 'g-o', label='Success Rate', markersize=4)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Evaluation Success Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Mean returns with error bars
        if self.eval_epochs:
            axes[1, 0].errorbar(self.eval_epochs, self.eval_mean_returns, 
                              yerr=self.eval_std_returns, fmt='ro-', 
                              label='Mean Return', markersize=4, capsize=3)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Mean Return')
            axes[1, 0].set_title('Evaluation Returns')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning curve (success rate vs time)
        if self.eval_epochs and len(self.epoch_times) > 0:
            eval_times_cumulative = [sum(self.epoch_times[:i]) for i in self.eval_epochs]
            axes[1, 1].plot(eval_times_cumulative, self.eval_success_rates, 'purple', marker='s', markersize=4)
            axes[1, 1].set_xlabel('Training Time (seconds)')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_title('Learning Curve')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Training curves saved: {plot_path}")
    
    def save_detailed_logs(self):
        """Save detailed training logs"""
        logs = {
            'training': {
                'train_losses': self.train_losses,
                'epoch_times': self.epoch_times,
                'total_training_time': sum(self.epoch_times)
            },
            'evaluation': {
                'eval_epochs': self.eval_epochs,
                'success_rates': self.eval_success_rates,
                'mean_returns': self.eval_mean_returns,
                'std_returns': self.eval_std_returns,
                'episodes_data': self.eval_episodes_data
            },
            'summary': {
                'total_epochs': len(self.train_losses),
                'best_success_rate': max(self.eval_success_rates) if self.eval_success_rates else 0.0,
                'best_success_epoch': self.eval_epochs[np.argmax(self.eval_success_rates)] if self.eval_success_rates else -1,
                'final_success_rate': self.eval_success_rates[-1] if self.eval_success_rates else 0.0,
                'final_mean_return': self.eval_mean_returns[-1] if self.eval_mean_returns else 0.0
            }
        }
        
        logs_path = os.path.join(self.save_dir, 'training_logs.json')
        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
        
        print(f"   💾 Detailed logs saved: {logs_path}")

class FixedUniversalBCTrainer:
    """Fixed Universal BC trainer"""
    
    def __init__(self, data_path: str, save_dir: str):
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(save_dir)
        
        # Detect data type and characteristics
        self.data_analysis = DataTypeDetector.detect_data_type(data_path)
        self.processor = FixedObservationProcessor(self.data_analysis)
        
        # Load actual data
        self.data = np.load(data_path, allow_pickle=True)
        self.obs_data = self.data['obs']
        self.acs_data = self.data['acs']
        
        print(f"\n🎯 FIXED UNIVERSAL BC TRAINER")
        print(f"Data type: {self.data_analysis['base_data_type']}")
        print(f"Conditioning: {self.data_analysis['conditioning_type']}")
        print(f"Episodes: {len(self.obs_data)}")
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data compatible with detected format"""
        print(f"\n📊 PREPARING TRAINING DATA")
        
        train_obs = []
        train_actions = []
        
        for episode_idx in range(len(self.obs_data)):
            episode_obs = self.obs_data[episode_idx]
            episode_acs = self.acs_data[episode_idx]
            
            for t in range(len(episode_obs) - 1):
                obs = episode_obs[t]
                action = episode_acs[t]
                
                processed_obs = self.processor.process_observation_for_training(obs)
                
                train_obs.append(processed_obs)
                train_actions.append(action)
        
        train_obs = np.array(train_obs)
        train_actions = np.array(train_actions)
        
        print(f"   • Processed {len(train_obs):,} transitions")
        print(f"   • Observation shape: {train_obs.shape}")
        print(f"   • Action shape: {train_actions.shape}")
        
        return train_obs, train_actions
    
    def create_policy(self, input_dim: int, action_dim: int) -> EnhancedDictPolicy:
        """Create policy with correct architecture"""
        # For both post-processed and raw data, use the actual training input dimension
        policy = EnhancedDictPolicy(
            ac_dim=action_dim,
            ob_dim=input_dim,  # Use the actual processed input dimension
            use_goal=False,    # Disable goal processing for simplicity
            goal_dim=0,
            goal_importance=1.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        print(f"\n🧠 POLICY ARCHITECTURE:")
        print(f"   • Input dim: {input_dim}")
        print(f"   • Action dim: {action_dim}")
        print(f"   • Conditioning: {self.data_analysis['conditioning_type']}")
        
        return policy
    
    def evaluate_policy(self, policy, num_episodes: int = 5, max_steps: int = 100) -> Dict[str, Any]:
        """Evaluate policy with proper termination and fixed dimensions"""
        try:
            os.environ['PYBULLET_EGL'] = '0'
            env = gym.make("PegTransfer-v0")
            
            episodes = []
            episode_returns = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(num_episodes):
                env_obs, _ = env.reset()
                episode_return = 0
                episode_length = 0
                
                for step in range(max_steps):
                    processed_obs_dict = self.processor.process_observation_for_evaluation(env_obs)
                    action = policy.get_action(processed_obs_dict)
                    env_obs, reward, done, truncated, info = env.step(action)
                    episode_return += reward
                    episode_length += 1
                    
                    # Stop immediately when task is successful
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
                'episodes': episodes,
                'success_count': success_count,
                'total_episodes': num_episodes,
                'success_rate': success_count / num_episodes,
                'mean_return': np.mean(episode_returns),
                'std_return': np.std(episode_returns),
                'mean_episode_length': np.mean(episode_lengths)
            }
            
        except Exception as e:
            print(f"   ⚠️  Evaluation error: {str(e)[:50]}...")
            return {
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
        """Train the BC policy with enhanced logging"""
        print(f"\n🚀 TRAINING BC POLICY")
        print("="*60)
        
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
                
                # Create observation batch (simple format for fixed policy)
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
            self.logger.log_epoch(epoch, avg_train_loss)
            
            # Evaluation phase
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                print(f"\n   🎯 Evaluating at epoch {epoch+1}...")
                eval_results = self.evaluate_policy(policy, eval_episodes)
                self.logger.log_evaluation(epoch, eval_results)
                
                # Save best model
                if eval_results['success_rate'] >= best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    self.save_model(policy, epoch, eval_results['success_rate'], input_dim, action_dim)
                    print(f"   💾 New best model saved! Success: {eval_results['success_rate']:.1%}")
            
            # Progress update
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {avg_train_loss:.4f} | Best Success: {best_success_rate:.1%}")
        
        # Save training artifacts
        self.logger.save_training_curves()
        self.logger.save_detailed_logs()
        
        # Save final results
        results = {
            'data_analysis': self.data_analysis,
            'best_success_rate': best_success_rate,
            'final_epoch': epochs
        }
        
        self.save_results(results)
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"   • Best success rate: {best_success_rate:.1%}")
        print(f"   • Total training time: {sum(self.logger.epoch_times):.1f}s")
        
        return results
    
    def save_model(self, policy, epoch: int, success_rate: float, input_dim: int, action_dim: int):
        """Save model with configuration"""
        model_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(policy.state_dict(), model_path)
        
        config = {
            'data_analysis': self.data_analysis,
            'input_dim': input_dim,
            'action_dim': action_dim,
            'epoch': epoch,
            'success_rate': success_rate,
            'policy_config': {
                'ob_dim': input_dim,
                'use_goal': False,
                'goal_dim': 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results"""
        results_path = os.path.join(self.save_dir, 'results.json')
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Fixed Universal BC Training for PegTransfer')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file (.npz)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    print(f"🎯 FIXED UNIVERSAL BC TRAINING")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"Save Dir: {args.save_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Eval Interval: {args.eval_interval}")
    print(f"Eval Episodes: {args.eval_episodes}")
    
    try:
        # Create trainer (automatically detects data type and fixes dimensions)
        trainer = FixedUniversalBCTrainer(args.data_path, args.save_dir)
        
        # Train with enhanced logging and fixed dimensions
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes
        )
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   • Data Type: {trainer.data_analysis['base_data_type']}")
        print(f"   • Conditioning: {trainer.data_analysis['conditioning_type']}")
        print(f"   • Best Success Rate: {results['best_success_rate']:.1%}")
        print(f"   • Training artifacts saved to: {args.save_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()