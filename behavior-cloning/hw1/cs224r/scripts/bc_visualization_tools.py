"""
BC Training Visualization and Debugging Tools

Tools to visualize training progress, evaluate policies, and create videos
for goal-conditioned behavior cloning experiments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import gymnasium as gym
import surrol.gym
from tqdm import tqdm
import cv2
from collections import defaultdict
import argparse
from PIL import Image

# Import your policy class and the new compatibility layer
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu
from goal_conditioning_compatibility import (
    GoalConditioningManager, 
    UniversalModelLoader, 
    print_format_explanation
)

def load_model_and_config(model_path):
    """Load trained model and its configuration using universal loader"""
    model_state, config = UniversalModelLoader.load_model_with_auto_config(model_path)
    
    if model_state is None:
        return None, None
    
    # Create policy using universal loader
    policy = UniversalModelLoader.create_policy_from_config(config)
    policy.load_state_dict(model_state)
    policy.eval()
    
    return policy, config

def save_gif_from_frames(frames, gif_path, fps=10, duration_per_frame=0.1):
    """Save frames as GIF"""
    if not frames:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    
    # Convert frames to PIL Images and resize for smaller file size
    pil_frames = []
    
    for frame in frames[::2]:  # Take every 2nd frame to reduce file size
        # Resize frame to reduce file size
        height, width = frame.shape[:2]
        new_width = min(400, width)  # Max width 400px
        new_height = int(height * new_width / width)
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        pil_frames.append(img)
    
    # Save as GIF
    if pil_frames:
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration_per_frame * 1000),  # Convert to milliseconds
            loop=0
        )

def evaluate_policy_detailed(policy, config, env_name="PegTransfer-v0", num_episodes=10, max_steps=100, 
                           save_video=False, video_dir=None):
    """
    Detailed evaluation of a policy with trajectory analysis
    UPDATED: Now compatible with ALL goal conditioning formats
    """
    env = gym.make(env_name, render_mode="rgb_array" if save_video else None)
    
    # Get condition type from config
    condition_type = config.get('condition_type', 'unknown')
    
    results = {
        'episodes': [],
        'success_rate': 0.0,
        'mean_return': 0.0,
        'mean_episode_length': 0.0,
        'trajectories': [],
        'debug_info': [],
        'condition_type': condition_type,
        'config': config
    }
    
    success_count = 0
    returns = []
    episode_lengths = []
    
    print(f"🎯 Evaluating policy with condition type: {condition_type}")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'success': False,
            'episode_length': 0,
            'final_distance': 0.0,
            'action_errors': 0,
            'total_reward': 0.0
        }
        
        frames = []
        episode_return = 0
        action_error_count = 0
        
        # Debug: Print initial observation structure
        if episode == 0:
            print(f"🔍 Debug Info for Episode {episode + 1}:")
            print(f"   • Condition Type: {condition_type}")
            print(f"   • Obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
            if isinstance(obs, dict):
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"   • {key}: shape {value.shape}, mean {np.mean(value):.3f}")
        
        for step in range(max_steps):
            # Store trajectory data
            trajectory['observations'].append(obs.copy())
            
            # Get action from policy
            try:
                # Apply conditioning based on format using universal system
                if isinstance(obs, dict) and condition_type != 'none':
                    conditioned_obs = GoalConditioningManager.apply_conditioning(
                        obs['observation'], obs, condition_type
                    )
                    # Create conditioned observation dict for policy
                    eval_obs = obs.copy()
                    eval_obs['observation'] = conditioned_obs
                else:
                    eval_obs = obs
                
                action = policy.get_action(eval_obs)
                trajectory['actions'].append(action.copy())
                
                # Debug: Check action range
                if episode == 0 and step < 5:
                    print(f"   • Step {step}: action = {action[:3]}... (range: [{np.min(action):.3f}, {np.max(action):.3f}])")
                    if condition_type != 'none':
                        print(f"   • Conditioned obs shape: {conditioned_obs.shape}")
                
            except Exception as e:
                print(f"   ❌ Error getting action at step {step}: {e}")
                action = env.action_space.sample()  # Fallback to random action
                trajectory['actions'].append(action)
                action_error_count += 1
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            trajectory['rewards'].append(reward)
            episode_return += reward
            
            # Debug: Print rewards and info
            if episode == 0 and step < 5:
                print(f"   • Step {step}: reward = {reward:.3f}, info = {info}")
            
            # Save frame for video
            if save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            
            if done or truncated:
                trajectory['success'] = info.get('is_success', False)
                trajectory['episode_length'] = step + 1
                trajectory['final_distance'] = info.get('distance_to_goal', float('inf'))
                trajectory['action_errors'] = action_error_count
                trajectory['total_reward'] = episode_return
                
                if trajectory['success']:
                    success_count += 1
                
                # Debug final info
                print(f"   📊 Episode {episode + 1} ended:")
                print(f"      • Success: {trajectory['success']}")
                print(f"      • Length: {trajectory['episode_length']} steps")
                print(f"      • Total reward: {episode_return:.2f}")
                print(f"      • Action errors: {action_error_count}")
                print(f"      • Final distance: {trajectory['final_distance']}")
                print(f"      • Done reason: {'done' if done else 'truncated'}")
                
                break
        
        returns.append(episode_return)
        episode_lengths.append(trajectory['episode_length'])
        results['trajectories'].append(trajectory)
        
        # Save GIF for this episode
        if save_video and frames and video_dir:
            gif_path = os.path.join(video_dir, f"episode_{episode+1}_{condition_type}_{'success' if trajectory['success'] else 'fail'}.gif")
            save_gif_from_frames(frames, gif_path)
            print(f"   🎬 GIF saved: {gif_path}")
    
    env.close()
    
    # Calculate statistics
    results['success_rate'] = success_count / num_episodes
    results['mean_return'] = np.mean(returns)
    results['mean_episode_length'] = np.mean(episode_lengths)
    results['episodes'] = list(zip(returns, episode_lengths, [t['success'] for t in results['trajectories']]))
    
    # Add debug summary
    total_action_errors = sum(t['action_errors'] for t in results['trajectories'])
    results['debug_info'] = {
        'total_action_errors': total_action_errors,
        'avg_action_errors_per_episode': total_action_errors / num_episodes,
        'episodes_with_errors': sum(1 for t in results['trajectories'] if t['action_errors'] > 0),
        'condition_type': condition_type,
        'compatibility_check': action_error_count == 0  # No errors means good compatibility
    }
    
    return results

def plot_training_comparison(experiment_dirs, save_path=None):
    """
    Plot comparison of multiple training runs
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Goal-Conditioned BC Training Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, exp_dir in enumerate(experiment_dirs):
        # Load training log
        log_path = os.path.join(exp_dir, 'training_log.json')
        if not os.path.exists(log_path):
            print(f"No training log found in {exp_dir}")
            continue
        
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        condition_type = data['config'].get('condition_type', f'exp_{i}')
        color = colors[i % len(colors)]
        
        # Plot training loss
        axes[0, 0].plot(data['train_losses'], label=f'{condition_type} (train)', 
                       color=color, linestyle='-', alpha=0.7)
        axes[0, 0].plot(data['val_losses'], label=f'{condition_type} (val)', 
                       color=color, linestyle='--', alpha=0.7)
        
        # Plot success rate
        eval_results = data['eval_results']
        if eval_results['success_rates']:
            axes[0, 1].plot(eval_results['eval_epochs'], 
                           [sr * 100 for sr in eval_results['success_rates']], 
                           label=condition_type, color=color, marker='o', markersize=4)
        
        # Plot evaluation returns
        if eval_results['eval_returns']:
            axes[0, 2].plot(eval_results['eval_epochs'], eval_results['eval_returns'],
                           label=condition_type, color=color, marker='s', markersize=4)
        
        # Plot final success rate as bar
        final_success = eval_results['success_rates'][-1] * 100 if eval_results['success_rates'] else 0
        axes[1, 0].bar(i, final_success, color=color, alpha=0.7, label=condition_type)
        
        # Plot learning curve smoothness (loss improvement)
        if len(data['val_losses']) > 1:
            improvement = [(data['val_losses'][0] - loss) / abs(data['val_losses'][0]) * 100 
                          for loss in data['val_losses']]
            axes[1, 1].plot(improvement, label=condition_type, color=color, alpha=0.7)
    
    # Set titles and labels
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Success Rate During Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].set_title('Evaluation Returns')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Mean Return')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].set_title('Final Success Rate Comparison')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_xlabel('Experiment')
    
    axes[1, 1].set_title('Learning Progress (Val Loss Improvement)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def analyze_trajectory_patterns(trajectories):
    """
    Analyze patterns in successful vs failed trajectories
    """
    successful = [t for t in trajectories if t['success']]
    failed = [t for t in trajectories if not t['success']]
    
    analysis = {
        'success_stats': {
            'count': len(successful),
            'avg_length': np.mean([t['episode_length'] for t in successful]) if successful else 0,
            'avg_return': np.mean([sum(t['rewards']) for t in successful]) if successful else 0
        },
        'failure_stats': {
            'count': len(failed),
            'avg_length': np.mean([t['episode_length'] for t in failed]) if failed else 0,
            'avg_return': np.mean([sum(t['rewards']) for t in failed]) if failed else 0
        }
    }
    
    return analysis

def create_evaluation_report(model_path, output_dir, num_episodes=20):
    """
    Create comprehensive evaluation report for a trained model
    """
    print(f"Creating evaluation report for {model_path}")
    
    # Load model
    policy, config = load_model_and_config(model_path)
    if policy is None:
        print("Failed to load model")
        return
    
    print(f"🤖 Model Configuration:")
    print(f"   • Observation dim: {config.get('obs_dim', 'unknown')}")
    print(f"   • Action dim: {config.get('action_dim', 'unknown')}")
    print(f"   • Use goal: {config.get('use_goal', 'unknown')}")
    print(f"   • Condition type: {config.get('condition_type', 'unknown')}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate policy
    print("Running detailed evaluation...")
    results = evaluate_policy_detailed(
        policy, 
        config, 
        num_episodes=num_episodes, 
        save_video=True,
        video_dir=os.path.join(output_dir, 'videos')
    )
    
    # Analyze trajectories
    trajectory_analysis = analyze_trajectory_patterns(results['trajectories'])
    
    # Print debug summary
    print(f"\n🔍 DEBUG SUMMARY:")
    print(f"   • Total action errors: {results['debug_info']['total_action_errors']}")
    print(f"   • Avg action errors per episode: {results['debug_info']['avg_action_errors_per_episode']:.1f}")
    print(f"   • Episodes with errors: {results['debug_info']['episodes_with_errors']}/{num_episodes}")
    
    # Analyze failure patterns
    failed_trajectories = [t for t in results['trajectories'] if not t['success']]
    if failed_trajectories:
        print(f"\n❌ FAILURE ANALYSIS:")
        avg_failed_length = np.mean([t['episode_length'] for t in failed_trajectories])
        avg_failed_reward = np.mean([t['total_reward'] for t in failed_trajectories])
        print(f"   • Failed episodes: {len(failed_trajectories)}/{num_episodes}")
        print(f"   • Avg failed episode length: {avg_failed_length:.1f} steps")
        print(f"   • Avg failed episode reward: {avg_failed_reward:.2f}")
        
        # Check if episodes are timing out
        max_length_episodes = sum(1 for t in failed_trajectories if t['episode_length'] >= 50)
        print(f"   • Episodes that timed out (≥50 steps): {max_length_episodes}")
        
        # Check reward patterns
        very_low_reward_episodes = sum(1 for t in failed_trajectories if t['total_reward'] <= -40)
        print(f"   • Episodes with very low rewards (≤-40): {very_low_reward_episodes}")
    
    # Create report with debug info
    report = {
        'model_path': model_path,
        'model_config': config,
        'evaluation_results': {
            'success_rate': results['success_rate'],
            'mean_return': results['mean_return'],
            'mean_episode_length': results['mean_episode_length'],
            'num_episodes': num_episodes
        },
        'debug_info': results['debug_info'],
        'trajectory_analysis': trajectory_analysis,
        'episode_details': results['episodes']
    }
    
    # Save report (with proper JSON serialization)
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    try:
        # Convert numpy types to native Python types for JSON serialization
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
        
        report_serializable = convert_numpy(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save JSON report: {e}")
    
    # Create plots
    plot_trajectory_analysis(results, os.path.join(output_dir, 'trajectory_analysis.png'))
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   • Success rate: {results['success_rate']:.1%}")
    print(f"   • Mean return: {results['mean_return']:.2f}")
    print(f"   • Mean episode length: {results['mean_episode_length']:.1f}")
    print(f"\n📁 Results saved to: {output_dir}")
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if results['success_rate'] == 0:
        print("   ❌ 0% success rate indicates major issues:")
        if results['debug_info']['total_action_errors'] > 0:
            print("      • Dimension mismatch errors detected - check model architecture")
        if np.mean([t['total_reward'] for t in results['trajectories']]) <= -40:
            print("      • Very low rewards suggest model hasn't learned the task")
            print("      • Consider training for more epochs or checking data quality")
        if np.mean([t['episode_length'] for t in results['trajectories']]) >= 45:
            print("      • Episodes timing out - model may be taking random actions")
            print("      • Check goal conditioning and action space bounds")
    
    return report

def plot_trajectory_analysis(results, save_path):
    """Plot detailed trajectory analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    trajectories = results['trajectories']
    
    # Episode lengths
    lengths = [t['episode_length'] for t in trajectories]
    colors = ['green' if t['success'] else 'red' for t in trajectories]
    
    axes[0, 0].scatter(range(len(lengths)), lengths, c=colors, alpha=0.7)
    axes[0, 0].set_title('Episode Lengths (Green=Success, Red=Failure)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Steps')
    axes[0, 0].grid(True)
    
    # Returns
    returns = [sum(t['rewards']) for t in trajectories]
    axes[0, 1].scatter(range(len(returns)), returns, c=colors, alpha=0.7)
    axes[0, 1].set_title('Episode Returns')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].grid(True)
    
    # Success rate over time
    success_rate_rolling = []
    window = 5
    for i in range(len(trajectories)):
        start_idx = max(0, i - window + 1)
        window_trajectories = trajectories[start_idx:i+1]
        success_count = sum(1 for t in window_trajectories if t['success'])
        success_rate_rolling.append(success_count / len(window_trajectories))
    
    axes[1, 0].plot(success_rate_rolling, 'b-', alpha=0.7)
    axes[1, 0].set_title(f'Rolling Success Rate (window={window})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, 1)
    
    # Final distances
    distances = [t.get('final_distance', float('inf')) for t in trajectories]
    finite_distances = [d for d in distances if d != float('inf')]
    
    if finite_distances:
        axes[1, 1].hist(finite_distances, bins=10, alpha=0.7, color='blue')
        axes[1, 1].set_title('Final Distance to Goal Distribution')
        axes[1, 1].set_xlabel('Distance')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No distance data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def debug_goal_conditioning(model_path, data_path, num_samples=10):
    """
    Debug goal conditioning by examining how the model processes different goals
    """
    print(f"Debugging goal conditioning for {model_path}")
    
    # Load model
    policy, config = load_model_and_config(model_path)
    if policy is None:
        print("Failed to load model")
        return
    
    # Load some data samples
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs'][0]  # First episode
    
    print(f"Model configuration:")
    print(f"  Use goal: {config.get('use_goal', 'unknown')}")
    print(f"  Goal dim: {config.get('goal_dim', 'unknown')}")
    print(f"  Condition type: {config.get('condition_type', 'unknown')}")
    
    # Test goal processing
    for i in range(min(num_samples, len(obs_data))):
        obs = obs_data[i]
        print(f"\nSample {i+1}:")
        print(f"  Raw obs shape: {obs['observation'].shape}")
        
        if 'achieved_goal' in obs:
            print(f"  Achieved goal: {obs['achieved_goal']}")
        if 'desired_goal' in obs:
            print(f"  Desired goal: {obs['desired_goal']}")
        
        # Process observation through policy
        try:
            processed_obs = policy.process_observation(obs)
            print(f"  Processed obs shape: {processed_obs.shape}")
            
            # Get action
            action = policy.get_action(obs)
            print(f"  Predicted action: {action}")
            
        except Exception as e:
            print(f"  Error processing: {e}")

def main():
    parser = argparse.ArgumentParser(description='Universal BC Visualization and Debugging Tools - Works with ALL goal conditioning formats')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['evaluate', 'compare', 'debug', 'explain'],
                      help='Mode: evaluate single model, compare multiple, debug, or explain formats')
    parser.add_argument('--model_path', type=str, help='Path to model file')
    parser.add_argument('--experiment_dirs', type=str, nargs='+', 
                      help='List of experiment directories for comparison')
    parser.add_argument('--data_path', type=str, help='Path to data file for debugging')
    parser.add_argument('--output_dir', type=str, default='./evaluation_output',
                      help='Output directory for results')
    parser.add_argument('--num_episodes', type=int, default=20,
                      help='Number of episodes for evaluation')
    parser.add_argument('--condition_type', type=str, 
                      choices=GoalConditioningManager.SUPPORTED_FORMATS,
                      help='Override condition type for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'explain':
        print_format_explanation()
        return
    
    if args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path required for evaluate mode")
            return
        
        # Load model to get/override config
        policy, config = load_model_and_config(args.model_path)
        if policy is None:
            print("❌ Failed to load model")
            return
        
        # Override condition type if specified
        if args.condition_type:
            print(f"🔄 Overriding condition type: {config.get('condition_type')} → {args.condition_type}")
            config['condition_type'] = args.condition_type
        
        print(f"🎯 Using condition type: {config['condition_type']}")
        
        create_evaluation_report(
            args.model_path, 
            args.output_dir, 
            args.num_episodes
        )
    
    elif args.mode == 'compare':
        if not args.experiment_dirs:
            print("Error: --experiment_dirs required for compare mode")
            return
        
        print("📊 Comparing training runs...")
        plot_training_comparison(
            args.experiment_dirs,
            os.path.join(args.output_dir, 'training_comparison.png')
        )
    
    elif args.mode == 'debug':
        if not args.model_path or not args.data_path:
            print("Error: --model_path and --data_path required for debug mode")
            return
        
        print("🔍 Debugging goal conditioning...")
        debug_goal_conditioning(args.model_path, args.data_path)

if __name__ == '__main__':
    main()