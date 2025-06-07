"""
train_goal_conditioned_bc.py: 

Enhanced training script for goal-conditioned behavior cloning on PegTransfer tasks.
Supports different conditioning methods and handles npz data format.
"""

import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import datetime
import torch.utils.tensorboard as tensorboard
import json
from contextlib import redirect_stdout, redirect_stderr

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure.logger import Logger

def load_npz_data(npz_path):
    """
    Load demonstration data from npz file format.
    Returns paths in the format expected by BC training.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract data arrays
    obs_data = data['obs']  # Array of episodes, each episode is array of obs dicts
    acs_data = data['acs']  # Array of episodes, each episode is array of actions
    info_data = data['info']  # Array of episodes, each episode is array of info dicts
    
    # Convert to paths format expected by BC
    paths = []
    
    for episode_idx in range(len(obs_data)):
        episode_obs = obs_data[episode_idx]
        episode_acs = acs_data[episode_idx]
        episode_info = info_data[episode_idx]
        
        # Skip last observation since we don't have an action for it
        path_obs = []
        path_acs = []
        
        for t in range(len(episode_obs) - 1):  # -1 because last obs has no action
            path_obs.append(episode_obs[t])
            path_acs.append(episode_acs[t])
        
        if len(path_obs) > 0:  # Only add non-empty paths
            path = {
                'observation': np.array(path_obs, dtype=object),
                'action': np.array(path_acs)
            }
            paths.append(path)
    
    print(f"   • Loaded {len(obs_data)} episodes → {len(paths)} trajectories")
    
    return paths

def analyze_conditioning_data(paths, condition_type="one_hot"):
    """Analyze the conditioning data to understand its structure"""
    sample_obs = paths[0]['observation'][0]
    
    obs_dim = sample_obs['observation'].shape[0]
    
    print(f"   • Raw observation dimension: {obs_dim}")
    
    if 'block_encoding' in sample_obs:
        block_enc_dim = sample_obs['block_encoding'].shape[0]
        print(f"   • Block encoding dimension: {block_enc_dim}")
    
    if 'achieved_goal' in sample_obs:
        achieved_goal_dim = sample_obs['achieved_goal'].shape[0]
        print(f"   • Achieved goal dimension: {achieved_goal_dim}")
        
    if 'desired_goal' in sample_obs:
        desired_goal_dim = sample_obs['desired_goal'].shape[0]
        print(f"   • Desired goal dimension: {desired_goal_dim}")
    
    return False  # Always return False to indicate we're using raw data

def evaluate_during_training(policy, env_name, num_episodes=5, max_steps=100, training_obs_dim=None):
    """Evaluate policy during training to track success rate"""
    try:
        import gymnasium as gym
        import surrol.gym
        import os
        import sys
        import pybullet as p
        
        # Completely suppress ALL PyBullet output
        os.environ['PYBULLET_EGL'] = '0'
        
        # Redirect ALL output to suppress PyBullet logs
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # Completely silence PyBullet
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            # Connect PyBullet in DIRECT mode (no GUI, minimal output)
            p.connect(p.DIRECT)
            
            env = gym.make(env_name)
            
            success_count = 0
            returns = []
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_return = 0
                
                for step in range(max_steps):
                    try:
                        # Handle observation format mismatch between training and environment
                        if isinstance(obs, dict) and 'observation' in obs:
                            if hasattr(policy, 'use_goal') and not policy.use_goal:
                                # For baseline (no goal), use only first training_obs_dim dimensions
                                obs_dim = training_obs_dim if training_obs_dim is not None else 13
                                processed_obs = {
                                    'observation': obs['observation'][:obs_dim],
                                    'achieved_goal': obs.get('achieved_goal', np.zeros(3)),
                                    'desired_goal': obs.get('desired_goal', np.zeros(3))
                                }
                            else:
                                # For goal-conditioned, extract dimensions to match training data
                                obs_dim = training_obs_dim if training_obs_dim is not None else len(obs['observation'])
                                processed_obs = {
                                    'observation': obs['observation'][:obs_dim],
                                    'achieved_goal': obs.get('achieved_goal', np.zeros(3)),
                                    'desired_goal': obs.get('desired_goal', np.zeros(3))
                                }
                        else:
                            processed_obs = obs
                        
                        action = policy.get_action(processed_obs)
                        obs, reward, done, truncated, info = env.step(action)
                        episode_return += reward
                        
                        if done or truncated:
                            if 'is_success' in info and info['is_success']:
                                success_count += 1
                            break
                    except Exception as e:
                        # Restore output temporarily for error reporting
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        print(f"Error during episode step: {e}")
                        sys.stdout = open(os.devnull, 'w')
                        sys.stderr = open(os.devnull, 'w')
                        break
                
                returns.append(episode_return)
            
            env.close()
            
        finally:
            # Always restore output
            if sys.stdout != old_stdout:
                sys.stdout.close()
            if sys.stderr != old_stderr:
                sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Disconnect PyBullet cleanly
            try:
                p.disconnect()
            except:
                pass
        
        success_rate = success_count / num_episodes
        mean_return = np.mean(returns)
        
        return success_rate, mean_return
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0, 0.0

def train_goal_conditioned_bc(
    data_path,
    save_path,
    condition_type="one_hot",
    n_epochs=200,
    batch_size=64,
    learning_rate=5e-4,
    val_split=0.1,
    use_goal=True,
    goal_importance=2.0,
    n_layers=3,
    hidden_size=128,
    use_data_augmentation=False,
    noise_level=0.0,
    curriculum_learning=False,  # Disable curriculum by default for baseline
    curriculum_phases=3,
    early_stopping_patience=20,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ac_dim=5,
    save_every=10,
    log_dir=None,
    eval_interval=10,
    eval_episodes=5,
    eval_env="PegTransfer-v0",
):
    """
    Train goal-conditioned behavior cloning agent
    """
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    ptu.init_gpu(use_gpu=device=='cuda')
    
    print("=" * 60)
    print("🚀 BEHAVIOR CLONING TRAINING")
    print("=" * 60)
    print(f"📊 Condition Type: {condition_type}")
    print(f"🎯 Use Goal: {use_goal}")
    print(f"💻 Device: {ptu.device}")
    print()

    # Load data from npz file
    print("📁 LOADING DATA...")
    paths = load_npz_data(data_path)
    
    # Analyze the data structure
    is_postprocessed = analyze_conditioning_data(paths, condition_type)
    
    # Combine all data
    all_obs = []
    all_actions = []
    
    for path in paths:
        all_obs.append(path['observation'])
        all_actions.append(path['action'])
    
    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    # Split into train and validation sets
    n_samples = len(all_obs)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_obs = all_obs[train_indices]
    train_actions = all_actions[train_indices]
    val_obs = all_obs[val_indices]
    val_actions = all_actions[val_indices]
    
    # Determine observation dimension from the processed data
    sample_obs = all_obs[0]
    raw_obs_dim = len(sample_obs['observation'])
    
    # Calculate goal dimension based on whether we're using goals
    goal_dim = 0
    if use_goal and 'achieved_goal' in sample_obs:
        # goal_dim should be the dimension of ONE goal vector 
        # (the policy will multiply by 2 for achieved + desired)
        goal_dim = len(sample_obs['achieved_goal'])
    
    # For goal conditioning, test what the actual processing produces
    if use_goal:
        # Create a temporary policy with the same configuration as final policy
        temp_policy = EnhancedDictPolicy(
            ac_dim=ac_dim,
            ob_dim=raw_obs_dim,
            use_goal=use_goal,
            goal_dim=goal_dim,  # This is the dimension of ONE goal vector
            goal_importance=goal_importance,
            n_layers=n_layers,
            size=hidden_size,
            learning_rate=learning_rate
        )
        
        # Get the actual processed observation dimension
        processed_sample = temp_policy.process_observation(sample_obs)
        actual_ob_dim = len(processed_sample)
        model_use_goal = True
        model_goal_dim = goal_dim
    else:
        # For baseline (no goal), we know the dimension is just the raw observation
        actual_ob_dim = raw_obs_dim
        model_use_goal = False
        model_goal_dim = 0
    
    print()
    print("📊 DATA SUMMARY:")
    print(f"   • Total samples: {len(all_obs):,}")
    print(f"   • Training samples: {len(train_obs):,}")
    print(f"   • Validation samples: {len(val_obs):,}")
    print(f"   • Raw observation dimension: {raw_obs_dim}")
    print(f"   • Processed observation dimension: {actual_ob_dim}")
    print(f"   • Goal dimension: {goal_dim}")
    print(f"   • Action dimension: {ac_dim}")
    print(f"   • Use goal: {use_goal}")
    print()

    # Create logger
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create TensorBoard writer
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    # Create model with correct dimensions
    print("🧠 CREATING MODEL...")
    policy = EnhancedDictPolicy(
        ac_dim=ac_dim,
        ob_dim=raw_obs_dim,  # Use raw observation dimension for the policy
        use_goal=model_use_goal,
        goal_dim=model_goal_dim,
        goal_importance=goal_importance,
        n_layers=n_layers,
        size=hidden_size,
        learning_rate=learning_rate
    )
    print(f"   • Network: {n_layers} layers x {hidden_size} units")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Model input dimension: {actual_ob_dim}")
    print(f"   • Model expects goal processing: {model_use_goal}")
    print()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    success_rates = []
    eval_returns = []
    eval_epochs = []
    
    print("🏋️ STARTING TRAINING...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training
        policy.train()
        epoch_losses = []
        n_batches = len(train_obs) // batch_size
        
        for i in range(n_batches):
            batch_indices = np.random.choice(len(train_obs), batch_size, replace=False)
            obs_batch = train_obs[batch_indices]
            actions_batch = train_actions[batch_indices]
            
            try:
                loss = policy.update(obs_batch, actions_batch)
                epoch_losses.append(loss)
            except Exception as e:
                print(f"❌ Error in training batch {i}: {e}")
                continue
        
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation
        policy.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_obs), batch_size):
                obs_batch = val_obs[i:i+batch_size]
                actions_batch = val_actions[i:i+batch_size]
                
                try:
                    processed_obs = policy.process_observation(obs_batch)
                    obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32))
                    actions_tensor = ptu.from_numpy(actions_batch.astype(np.float32))
                    
                    dist = policy(obs_tensor)
                    loss = -dist.log_prob(actions_tensor).sum(dim=-1).mean().item()
                    val_epoch_losses.append(loss)
                except Exception as e:
                    print(f"❌ Error in validation batch {i//batch_size}: {e}")
                    continue
        
        avg_val_loss = np.mean(val_epoch_losses) if val_epoch_losses else float('inf')
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        try:
            current_lr = policy.update_lr_scheduler(avg_val_loss)
        except:
            current_lr = learning_rate
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(policy.state_dict(), save_path)
            patience_counter = 0
            best_marker = "⭐"
        else:
            patience_counter += 1
            best_marker = "  "
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Evaluation
        eval_marker = ""
        if epoch % eval_interval == 0 or epoch == n_epochs - 1:
            success_rate, mean_return = evaluate_during_training(
                policy, eval_env, num_episodes=eval_episodes, max_steps=100, training_obs_dim=raw_obs_dim
            )
            success_rates.append(success_rate)
            eval_returns.append(mean_return)
            eval_epochs.append(epoch)
            eval_marker = f" | 🎯 Success: {success_rate:.1%} | 📈 Return: {mean_return:.1f}"
            
            writer.add_scalar('evaluation/success_rate', success_rate, epoch)
            writer.add_scalar('evaluation/mean_return', mean_return, epoch)
        
        # Logging
        print(f"{best_marker} Epoch {epoch+1:3d}/{n_epochs} | 📉 Train: {avg_train_loss:.4f} | 📊 Val: {avg_val_loss:.4f} | 🔄 LR: {current_lr:.1e}{eval_marker}")
        
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('validation/loss', avg_val_loss, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path), f"checkpoint_{epoch+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ Training completed in {total_time:.1f} seconds")
    print()
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves - {condition_type}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(eval_epochs, success_rates, 'g-o')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.title('Success Rate During Training')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(eval_epochs, eval_returns, 'm-o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Return')
    plt.title('Evaluation Returns')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    if len(val_losses) > 1:
        improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        plt.plot(improvement)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss Improvement (%)')
        plt.title('Learning Progress')
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(save_path), f'training_curves_{condition_type}.png')
    plt.savefig(plot_path)
    
    # Load best model
    policy.load_state_dict(torch.load(save_path))
    
    # Save configuration
    model_config = {
        'ac_dim': ac_dim,
        'ob_dim': raw_obs_dim,  # Use raw obs dim like the model
        'use_goal': model_use_goal,  # Use the actual model setting
        'goal_dim': model_goal_dim,  # Use the actual model setting
        'goal_importance': goal_importance,
        'n_layers': n_layers,
        'size': hidden_size,
        'learning_rate': learning_rate,
        'condition_type': condition_type,
        'data_path': data_path
    }
    config_path = os.path.join(os.path.dirname(save_path), f'model_config_{condition_type}.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    
    print("💾 SAVING RESULTS...")
    print(f"   • Training curves: {plot_path}")
    print(f"   • Model config: {config_path}")
    
    writer.close()
    
    # Return the policy and config for main() to use
    return policy, {
        'obs_dim': raw_obs_dim,  # Use raw obs dim to match model config
        'action_dim': ac_dim,
        'condition_type': condition_type,
        'use_goal': model_use_goal,  # Use actual model setting
        'goal_dim': model_goal_dim,  # Use actual model setting
        'train_losses': train_losses,
        'val_losses': val_losses,
        'success_rates': success_rates,
        'eval_returns': eval_returns,
        'eval_epochs': eval_epochs
    }

def main():
    parser = argparse.ArgumentParser(description='Train goal-conditioned behavior cloning agent')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the npz data file')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the trained model')
    parser.add_argument('--condition_type', type=str, 
                      choices=['one_hot', 'target_block', 'target_block_and_target_peg', 
                              'one_hot_and_target_peg', 'four_tuple', 'color_language', 'none'],
                      default='one_hot',
                      help='Type of goal conditioning (baseline uses none)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Validation split fraction')
    parser.add_argument('--no_goal', action='store_true',
                      help='Disable goal processing (for baseline)')
    parser.add_argument('--goal_importance', type=float, default=2.0,
                      help='Weight for goal information')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden layer size')
    parser.add_argument('--no_augmentation', action='store_true', default=True,
                      help='Data augmentation disabled by default')
    parser.add_argument('--noise_level', type=float, default=0.02,
                      help='Noise level for data augmentation')
    parser.add_argument('--no_curriculum', action='store_true', default=True,
                      help='Disable curriculum learning (disabled by default)')
    parser.add_argument('--patience', type=int, default=20,
                      help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint frequency')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Evaluation frequency')
    parser.add_argument('--eval_episodes', type=int, default=5,
                      help='Number of evaluation episodes')
    parser.add_argument('--eval_env', type=str, default="PegTransfer-v0",
                      help='Environment for evaluation')
    parser.add_argument('--no_timestamp', action='store_true',
                      help='Do not add timestamp to save directory')
    
    args = parser.parse_args()
    
    # Add timestamp and condition type to save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_timestamp:
        args.save_dir = f"{args.save_dir}_{args.condition_type}_{timestamp}"
    else:
        args.save_dir = f"{args.save_dir}_{args.condition_type}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_path = os.path.join(args.save_dir, f'goal_conditioned_bc_policy_{args.condition_type}.pt')
    
    try:
        policy, config = train_goal_conditioned_bc(
            data_path=args.data,
            save_path=save_path,
            condition_type=args.condition_type,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split,
            use_goal=not args.no_goal,
            goal_importance=args.goal_importance,
            n_layers=args.layers,
            hidden_size=args.hidden_size,
            use_data_augmentation=not args.no_augmentation,
            noise_level=args.noise_level,
            curriculum_learning=not args.no_curriculum,
            early_stopping_patience=args.patience,
            device=args.device,
            save_every=args.save_every,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_env=args.eval_env,
        )
        
        # Save training results
        final_save_path = os.path.join(args.save_dir, f'final_model_{args.condition_type}.pt')
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': {
                'obs_dim': config['obs_dim'],
                'action_dim': config['action_dim'],
                'condition_type': config['condition_type'],
                'use_goal': config['use_goal']
            },
            'train_losses': config['train_losses'],
            'eval_results': {
                'success_rates': config['success_rates'],
                'eval_returns': config['eval_returns'],
                'eval_epochs': config['eval_epochs']
            }
        }, final_save_path)
        
        # Save training log
        log_path = os.path.join(args.save_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump({
                'train_losses': config['train_losses'],
                'val_losses': config['val_losses'],
                'eval_results': {
                    'success_rates': config['success_rates'],
                    'eval_returns': config['eval_returns'],
                    'eval_epochs': config['eval_epochs']
                },
                'config': {
                    'obs_dim': config['obs_dim'],
                    'action_dim': config['action_dim'],
                    'condition_type': config['condition_type'],
                    'use_goal': config['use_goal'],
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'lr': args.lr
                }
            }, f, indent=2)
        
        print(f"   • Final model: {final_save_path}")
        print(f"   • Training log: {log_path}")
        print()
        
        # Final evaluation
        print("🎯 FINAL EVALUATION...")
        final_success, final_return = evaluate_during_training(policy, args.eval_env, 10, 100, config['obs_dim'])
        print(f"   • Success rate: {final_success:.1%}")
        print(f"   • Mean return: {final_return:.1f}")
        print()
        
        print("=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()