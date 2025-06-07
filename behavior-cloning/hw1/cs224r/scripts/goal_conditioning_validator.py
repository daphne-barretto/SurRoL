"""
Goal Conditioning Validation Script

Validates that goal conditioning is working correctly by checking:
1. Data preprocessing
2. Model input dimensions
3. Goal information flow
4. Action prediction differences for different goals
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

def validate_data_preprocessing(data_path, condition_type):
    """
    Validate that data preprocessing worked correctly
    """
    print(f"🔍 VALIDATING DATA PREPROCESSING: {condition_type}")
    print("=" * 60)
    
    data = np.load(data_path, allow_pickle=True)
    
    # Check structure
    print(f"📁 Data file structure:")
    for key in data.keys():
        print(f"   • {key}: {type(data[key])}, shape: {data[key].shape}")
    
    # Analyze first episode
    first_episode_obs = data['obs'][0]
    first_episode_acs = data['acs'][0]
    
    print(f"\n📊 First episode analysis:")
    print(f"   • Episode length: {len(first_episode_obs)} observations")
    print(f"   • Actions shape: {first_episode_acs.shape}")
    
    # Check observation structure
    sample_obs = first_episode_obs[0]
    print(f"\n🎯 Sample observation structure:")
    for key, value in sample_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: shape {value.shape}, dtype {value.dtype}")
            if key == 'observation':
                print(f"     Raw observation: {value[:5]}...")  # First 5 elements
        else:
            print(f"   • {key}: {value}")
    
    # Check if goal conditioning data is present
    conditioning_keys = {
        'one_hot': ['block_encoding'],
        'target_block': ['target_block_pos'],
        'target_block_and_target_peg': ['target_block_pos', 'target_peg_pos'],
        'one_hot_and_target_peg': ['block_encoding', 'target_peg_pos'],
        'four_tuple': ['four_tuple_encoding']
    }
    
    expected_keys = conditioning_keys.get(condition_type, [])
    
    print(f"\n🎯 Goal conditioning validation for '{condition_type}':")
    for key in expected_keys:
        if key in sample_obs:
            print(f"   ✅ {key}: shape {sample_obs[key].shape}")
            print(f"      Sample values: {sample_obs[key]}")
        else:
            print(f"   ❌ {key}: MISSING!")
    
    # Check goal consistency across episode
    print(f"\n🔄 Goal consistency check (first 5 timesteps):")
    for t in range(min(5, len(first_episode_obs))):
        obs = first_episode_obs[t]
        if 'desired_goal' in obs:
            print(f"   t={t}: desired_goal = {obs['desired_goal']}")
        if 'block_encoding' in obs:
            print(f"   t={t}: block_encoding = {obs['block_encoding']}")
    
    return sample_obs

def validate_model_setup(sample_obs, condition_type, use_goal=True):
    """
    Validate that the model is set up correctly for goal conditioning
    """
    print(f"\n🧠 VALIDATING MODEL SETUP")
    print("=" * 60)
    
    # Extract dimensions
    raw_obs_dim = len(sample_obs['observation'])
    goal_dim = len(sample_obs.get('achieved_goal', []))
    
    print(f"📏 Dimensions:")
    print(f"   • Raw observation: {raw_obs_dim}")
    print(f"   • Goal dimension: {goal_dim}")
    print(f"   • Use goal: {use_goal}")
    
    # Create policy
    policy = EnhancedDictPolicy(
        ac_dim=5,
        ob_dim=raw_obs_dim,
        use_goal=use_goal,
        goal_dim=goal_dim,
        goal_importance=2.0,
        n_layers=3,
        size=128,
        learning_rate=5e-4
    )
    
    print(f"\n🏗️  Model architecture:")
    print(f"   • Input expects: {policy.ob_dim} dimensions")
    print(f"   • Goal processing: {policy.use_goal}")
    print(f"   • Expected goal dim: {policy.goal_dim}")
    
    # Test observation processing
    print(f"\n🔄 Testing observation processing:")
    try:
        processed_obs = policy.process_observation(sample_obs)
        print(f"   ✅ Processed observation shape: {processed_obs.shape}")
        print(f"   ✅ Model input dimension: {len(processed_obs)}")
        
        # Test forward pass
        obs_tensor = ptu.from_numpy(processed_obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = policy(obs_tensor)
            action = dist.sample()
            print(f"   ✅ Action sampling works: {action.shape}")
            
    except Exception as e:
        print(f"   ❌ Error in processing: {e}")
        return None
    
    return policy

def test_goal_conditioning_effect(policy, sample_obs, condition_type):
    """
    Test if different goals produce different actions
    """
    print(f"\n🎯 TESTING GOAL CONDITIONING EFFECT")
    print("=" * 60)
    
    if not policy.use_goal:
        print("   ℹ️  Model doesn't use goals - skipping test")
        return
    
    # Create modified observations with different goals
    test_observations = []
    goal_descriptions = []
    
    if condition_type == 'one_hot':
        # Test different block encodings
        for block_id in range(2):  # Assuming 2 blocks
            modified_obs = sample_obs.copy()
            one_hot = np.zeros(2)
            one_hot[block_id] = 1
            modified_obs['block_encoding'] = one_hot
            test_observations.append(modified_obs)
            goal_descriptions.append(f"Block {block_id}")
    
    elif condition_type == 'target_block':
        # Test different target positions (simulate different blocks)
        positions = [
            np.array([0.1, 0.1, 0.1]),
            np.array([0.2, 0.2, 0.2]),
            np.array([0.3, 0.3, 0.3])
        ]
        for i, pos in enumerate(positions):
            modified_obs = sample_obs.copy()
            modified_obs['target_block_pos'] = pos
            test_observations.append(modified_obs)
            goal_descriptions.append(f"Position {i}: {pos}")
    
    else:
        # For other methods, just use the original observation
        test_observations = [sample_obs]
        goal_descriptions = ["Original goal"]
    
    # Test action prediction for each goal
    print(f"   Testing {len(test_observations)} different goals:")
    actions = []
    
    for i, (obs, desc) in enumerate(zip(test_observations, goal_descriptions)):
        try:
            action = policy.get_action(obs)
            actions.append(action)
            print(f"   • {desc}: action = {action[:3]}...")  # First 3 action dims
        except Exception as e:
            print(f"   ❌ Error with {desc}: {e}")
            actions.append(None)
    
    # Check if actions are different
    if len(actions) > 1 and all(a is not None for a in actions):
        action_diffs = []
        for i in range(1, len(actions)):
            diff = np.linalg.norm(actions[i] - actions[0])
            action_diffs.append(diff)
            print(f"   • Action difference {i}: {diff:.4f}")
        
        avg_diff = np.mean(action_diffs)
        if avg_diff > 0.01:  # Threshold for "different" actions
            print(f"   ✅ Goal conditioning working! Average difference: {avg_diff:.4f}")
        else:
            print(f"   ⚠️  Actions very similar. Average difference: {avg_diff:.4f}")
            print(f"       This might indicate goal conditioning isn't working properly.")
    
    return actions

def validate_training_data_quality(data_path, max_episodes=10):
    """
    Validate the quality of training data
    """
    print(f"\n📊 VALIDATING TRAINING DATA QUALITY")
    print("=" * 60)
    
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs']
    acs_data = data['acs']
    
    print(f"   • Total episodes: {len(obs_data)}")
    
    # Check episode lengths
    episode_lengths = [len(episode) for episode in obs_data]
    print(f"   • Episode length stats:")
    print(f"     - Mean: {np.mean(episode_lengths):.1f}")
    print(f"     - Min: {np.min(episode_lengths)}")
    print(f"     - Max: {np.max(episode_lengths)}")
    print(f"     - Std: {np.std(episode_lengths):.1f}")
    
    # Check action diversity
    all_actions = []
    for episode_acs in acs_data[:max_episodes]:
        if len(episode_acs) > 0:  # Make sure episode has actions
            all_actions.extend(episode_acs)
    
    if len(all_actions) == 0:
        print(f"\n   • Action statistics: No actions found in first {max_episodes} episodes")
        return {
            'episode_lengths': episode_lengths,
            'action_stats': {'mean': 0, 'std': 0},
            'num_unique_goals': 0
        }
    
    try:
        all_actions = np.array(all_actions)
        if all_actions.ndim == 1:
            # If it's 1D, try to reshape or handle appropriately
            print(f"   • Warning: Actions appear to be 1D, shape: {all_actions.shape}")
        
        print(f"\n   • Action statistics (first {max_episodes} episodes):")
        print(f"     - Total actions: {len(all_actions)}")
        print(f"     - Action shape: {all_actions.shape}")
        
        if all_actions.ndim >= 2 and all_actions.shape[1] > 0:
            action_mean = np.mean(all_actions, axis=0)
            action_std = np.std(all_actions, axis=0)
            print(f"     - Action mean: {action_mean}")
            print(f"     - Action std: {action_std}")
        else:
            # Handle 1D case
            action_mean = np.mean(all_actions)
            action_std = np.std(all_actions)
            print(f"     - Action mean: {action_mean}")
            print(f"     - Action std: {action_std}")
    
    except Exception as e:
        print(f"   • Error computing action statistics: {e}")
        action_mean = 0
        action_std = 0
    
    # Check goal diversity
    goal_types = set()
    for episode_obs in obs_data[:max_episodes]:
        for obs in episode_obs:
            if isinstance(obs, dict) and 'desired_goal' in obs:
                goal_tuple = tuple(obs['desired_goal'])
                goal_types.add(goal_tuple)
    
    print(f"\n   • Goal diversity:")
    print(f"     - Unique goals: {len(goal_types)}")
    if len(goal_types) <= 10:
        for i, goal in enumerate(list(goal_types)[:5]):
            print(f"     - Goal {i+1}: {goal}")
    
    return {
        'episode_lengths': episode_lengths,
        'action_stats': {
            'mean': action_mean,
            'std': action_std
        },
        'num_unique_goals': len(goal_types)
    }

def create_validation_report(data_path, condition_type, use_goal=True):
    """
    Create comprehensive validation report
    """
    print(f"\n🎯 GOAL CONDITIONING VALIDATION REPORT")
    print(f"Condition Type: {condition_type}")
    print(f"Use Goal: {use_goal}")
    print("=" * 80)
    
    # 1. Validate data preprocessing
    sample_obs = validate_data_preprocessing(data_path, condition_type)
    
    # 2. Validate model setup
    policy = validate_model_setup(sample_obs, condition_type, use_goal)
    
    if policy is None:
        print("\n❌ VALIDATION FAILED: Model setup error")
        return
    
    # 3. Test goal conditioning effect
    if use_goal:
        test_goal_conditioning_effect(policy, sample_obs, condition_type)
    
    # 4. Validate training data quality
    data_quality = validate_training_data_quality(data_path)
    
    # 5. Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Check for common issues
    issues = []
    warnings = []
    
    if not use_goal:
        warnings.append("Model configured to NOT use goals (baseline mode)")
    
    if condition_type != 'none' and use_goal:
        conditioning_keys = {
            'one_hot': 'block_encoding',
            'target_block': 'target_block_pos',
            'target_block_and_target_peg': ['target_block_pos', 'target_peg_pos'],
            'one_hot_and_target_peg': ['block_encoding', 'target_peg_pos']
        }
        
        expected = conditioning_keys.get(condition_type, [])
        if isinstance(expected, str):
            expected = [expected]
        
        for key in expected:
            if key not in sample_obs:
                issues.append(f"Missing conditioning key: {key}")
    
    if data_quality['num_unique_goals'] < 2:
        warnings.append("Very few unique goals found - may not need goal conditioning")
    
    # Print results
    if not issues and not warnings:
        print("   ✅ All validations passed!")
    else:
        if issues:
            print("   ❌ ISSUES FOUND:")
            for issue in issues:
                print(f"      • {issue}")
        
        if warnings:
            print("   ⚠️  WARNINGS:")
            for warning in warnings:
                print(f"      • {warning}")
    
    print(f"\n   📊 Data Summary:")
    print(f"      • Episodes analyzed: {len(data_quality['episode_lengths'])}")
    print(f"      • Unique goals: {data_quality['num_unique_goals']}")
    print(f"      • Average episode length: {np.mean(data_quality['episode_lengths']):.1f}")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'data_quality': data_quality,
        'validation_passed': len(issues) == 0
    }

def quick_training_check(experiment_dir):
    """
    Quick check of training progress
    """
    print(f"\n🔍 QUICK TRAINING CHECK: {experiment_dir}")
    print("=" * 60)
    
    # Check if training log exists
    log_path = os.path.join(experiment_dir, 'training_log.json')
    if not os.path.exists(log_path):
        print("   ❌ No training log found")
        return
    
    import json
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    eval_results = data.get('eval_results', {})
    
    print(f"   📊 Training Configuration:")
    print(f"      • Condition type: {config.get('condition_type', 'unknown')}")
    print(f"      • Use goal: {config.get('use_goal', 'unknown')}")
    print(f"      • Epochs completed: {len(train_losses)}")
    print(f"      • Obs dim: {config.get('obs_dim', 'unknown')}")
    
    if train_losses:
        print(f"\n   📈 Training Progress:")
        print(f"      • Initial train loss: {train_losses[0]:.4f}")
        print(f"      • Final train loss: {train_losses[-1]:.4f}")
        print(f"      • Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    if val_losses:
        print(f"      • Final val loss: {val_losses[-1]:.4f}")
    
    success_rates = eval_results.get('success_rates', [])
    if success_rates:
        print(f"\n   🎯 Evaluation Results:")
        print(f"      • Final success rate: {success_rates[-1]:.1%}")
        print(f"      • Best success rate: {max(success_rates):.1%}")
        
        if success_rates[-1] == 0:
            print("      ⚠️  SUCCESS RATE IS 0% - POTENTIAL ISSUES:")
            print("         - Goal conditioning may not be working")
            print("         - Model architecture mismatch")
            print("         - Data preprocessing problems")
            print("         - Environment evaluation issues")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate goal conditioning setup')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the npz data file')
    parser.add_argument('--condition_type', type=str, required=True,
                      choices=['one_hot', 'target_block', 'target_block_and_target_peg', 
                              'one_hot_and_target_peg', 'four_tuple', 'none'],
                      help='Type of goal conditioning')
    parser.add_argument('--use_goal', action='store_true', default=True,
                      help='Whether model should use goals')
    parser.add_argument('--no_goal', action='store_true',
                      help='Disable goal usage (for baseline)')
    parser.add_argument('--experiment_dir', type=str,
                      help='Check training progress in experiment directory')
    
    args = parser.parse_args()
    
    if args.no_goal:
        args.use_goal = False
    
    # Validate goal conditioning setup
    report = create_validation_report(args.data_path, args.condition_type, args.use_goal)
    
    # Check training progress if directory provided
    if args.experiment_dir:
        quick_training_check(args.experiment_dir)
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    if report['validation_passed']:
        print("   ✅ Setup looks good! Training should work correctly.")
    else:
        print("   ❌ Issues found. Please fix before training:")
        for issue in report['issues']:
            print(f"      • {issue}")
    
    if args.condition_type != 'none' and not args.use_goal:
        print("   ⚠️  You're using goal conditioning data but model won't use goals!")
        print("      Remove --no_goal flag or use condition_type='none'")
    
    if args.condition_type == 'none' and args.use_goal:
        print("   ⚠️  You're using raw data but model expects goals!")
        print("      Add --no_goal flag or use processed data")

if __name__ == '__main__':
    main()