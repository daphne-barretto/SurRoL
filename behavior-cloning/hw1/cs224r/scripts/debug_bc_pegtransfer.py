"""
Comprehensive BC Debugging Script for PegTransfer

This script systematically debugs the BC training issues by:
1. Analyzing data quality and format
2. Testing observation processing
3. Comparing training vs evaluation data formats
4. Identifying dimension mismatches
5. Testing the trained model step-by-step
"""

import os
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
import gymnasium as gym
import surrol.gym
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

def analyze_demo_data_quality(data_path):
    """Analyze the quality of demonstration data"""
    print(f"🔍 ANALYZING DEMONSTRATION DATA QUALITY")
    print("="*60)
    
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs']
    acs_data = data['acs']
    info_data = data['info']
    
    # Check success rates in demos
    successful_episodes = 0
    episode_lengths = []
    action_magnitudes = []
    
    for i in range(len(info_data)):
        episode_info = info_data[i]
        episode_acs = acs_data[i]
        episode_obs = obs_data[i]
        
        episode_lengths.append(len(episode_obs))
        
        # Check if episode was successful
        if len(episode_info) > 0:
            final_info = episode_info[-1]
            if isinstance(final_info, dict) and final_info.get('is_success', False):
                successful_episodes += 1
        
        # Analyze action magnitudes
        for action in episode_acs:
            action_magnitudes.append(np.linalg.norm(action))
    
    success_rate = successful_episodes / len(info_data)
    
    print(f"📊 DEMO QUALITY METRICS:")
    print(f"   • Success rate: {success_rate:.1%} ({successful_episodes}/{len(info_data)})")
    print(f"   • Episode lengths: mean={np.mean(episode_lengths):.1f}, std={np.std(episode_lengths):.1f}")
    print(f"   • Action magnitudes: mean={np.mean(action_magnitudes):.3f}, std={np.std(action_magnitudes):.3f}")
    
    if success_rate < 0.5:
        print(f"   ⚠️  WARNING: Low success rate in demos! BC needs high-quality demonstrations.")
    
    # Analyze observation consistency
    sample_obs = obs_data[0][0]
    print(f"\n📏 OBSERVATION STRUCTURE:")
    for key, value in sample_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: shape {value.shape}, range [{np.min(value):.3f}, {np.max(value):.3f}]")
    
    return {
        'success_rate': success_rate,
        'mean_episode_length': np.mean(episode_lengths),
        'mean_action_magnitude': np.mean(action_magnitudes)
    }

def test_observation_processing(data_path, condition_type='none'):
    """Test if observation processing works correctly"""
    print(f"\n🔧 TESTING OBSERVATION PROCESSING")
    print("="*60)
    
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs']
    
    # Get sample observations
    sample_obs = obs_data[0][0]
    
    print(f"📥 INPUT OBSERVATION:")
    for key, value in sample_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: {value}")
    
    # Test different processing methods
    def apply_conditioning(obs, cond_type):
        if cond_type == 'none':
            return obs['observation']
        elif cond_type == 'one_hot' and 'block_encoding' in obs:
            return np.concatenate([obs['observation'], obs['block_encoding']])
        elif cond_type == 'target_block' and 'achieved_goal' in obs:
            return np.concatenate([obs['observation'], obs['achieved_goal']])
        # Add other methods as needed
        return obs['observation']
    
    processed_obs = apply_conditioning(sample_obs, condition_type)
    
    print(f"\n📤 PROCESSED OBSERVATION ({condition_type}):")
    print(f"   • Shape: {processed_obs.shape}")
    print(f"   • Values: {processed_obs}")
    print(f"   • Range: [{np.min(processed_obs):.3f}, {np.max(processed_obs):.3f}]")
    
    return processed_obs

def test_model_loading_and_inference(model_path, data_path):
    """Test if the trained model can be loaded and run inference"""
    print(f"\n🧠 TESTING MODEL LOADING AND INFERENCE")
    print("="*60)
    
    try:
        # Load model configuration
        config_path = model_path.replace('.pt', '.pkl').replace('final_model', 'model_config')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            print(f"✅ Model config loaded: {config}")
        else:
            print(f"⚠️  No config file found, using defaults")
            config = {'ac_dim': 5, 'ob_dim': 13, 'use_goal': False, 'goal_dim': 0}
        
        # Create and load model
        policy = EnhancedDictPolicy(
            ac_dim=config.get('ac_dim', 5),
            ob_dim=config.get('ob_dim', 13),
            use_goal=config.get('use_goal', False),
            goal_dim=config.get('goal_dim', 0),
            goal_importance=2.0,
            n_layers=3,
            size=128,
            learning_rate=1e-3
        )
        
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
        policy.eval()
        print(f"✅ Model loaded successfully")
        
        # Test inference on sample data
        data = np.load(data_path, allow_pickle=True)
        sample_obs = data['obs'][0][0]
        
        # Create observation in expected format
        obs_dict = {
            'observation': sample_obs['observation'],
            'achieved_goal': sample_obs.get('achieved_goal', np.zeros(3)),
            'desired_goal': sample_obs.get('desired_goal', np.zeros(3))
        }
        
        action = policy.get_action(obs_dict)
        print(f"✅ Inference test passed")
        print(f"   • Input obs shape: {sample_obs['observation'].shape}")
        print(f"   • Output action: {action}")
        print(f"   • Action magnitude: {np.linalg.norm(action):.3f}")
        
        return policy, True
        
    except Exception as e:
        print(f"❌ Model loading/inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_environment_interaction(policy, num_steps=10):
    """Test if the model can interact with the environment"""
    print(f"\n🌍 TESTING ENVIRONMENT INTERACTION")
    print("="*60)
    
    try:
        # Suppress PyBullet output
        import os
        os.environ['PYBULLET_EGL'] = '0'
        
        env = gym.make("PegTransfer-v0")
        obs, _ = env.reset()
        
        print(f"✅ Environment created and reset")
        print(f"📥 Initial environment observation:")
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"   • {key}: shape {value.shape}, sample {value[:3] if len(value) > 3 else value}")
        
        returns = []
        actions_taken = []
        
        for step in range(num_steps):
            try:
                # Create observation dict for policy
                if isinstance(obs, dict):
                    obs_dict = {
                        'observation': obs['observation'],
                        'achieved_goal': obs.get('achieved_goal', np.zeros(3)),
                        'desired_goal': obs.get('desired_goal', np.zeros(3))
                    }
                else:
                    obs_dict = {'observation': obs, 'achieved_goal': np.zeros(3), 'desired_goal': np.zeros(3)}
                
                action = policy.get_action(obs_dict)
                actions_taken.append(action.copy())
                
                obs, reward, done, truncated, info = env.step(action)
                returns.append(reward)
                
                print(f"   Step {step+1}: action={action[:3]}..., reward={reward:.3f}, done={done}")
                
                if done or truncated:
                    print(f"   Episode ended: success={info.get('is_success', False)}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Step {step+1} failed: {e}")
                break
        
        env.close()
        
        total_return = sum(returns)
        action_magnitude = np.mean([np.linalg.norm(a) for a in actions_taken])
        
        print(f"\n📊 INTERACTION RESULTS:")
        print(f"   • Steps completed: {len(returns)}")
        print(f"   • Total return: {total_return:.3f}")
        print(f"   • Mean action magnitude: {action_magnitude:.3f}")
        print(f"   • Rewards: {returns}")
        
        # Diagnose issues
        if total_return <= -40:
            print(f"   ⚠️  Very negative return suggests immediate failure or penalties")
        if action_magnitude < 0.001:
            print(f"   ⚠️  Very small actions suggest model isn't learning meaningful policies")
        if all(r == returns[0] for r in returns):
            print(f"   ⚠️  Constant rewards suggest environment issues or immediate termination")
        
        return total_return, action_magnitude
        
    except Exception as e:
        print(f"❌ Environment interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return -100, 0

def compare_training_vs_evaluation_format(data_path):
    """Compare training data format with environment format"""
    print(f"\n🔄 COMPARING TRAINING VS EVALUATION FORMATS")
    print("="*60)
    
    # Load training data
    data = np.load(data_path, allow_pickle=True)
    train_obs = data['obs'][0][0]
    
    print(f"📚 TRAINING DATA FORMAT:")
    for key, value in train_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: shape {value.shape}, dtype {value.dtype}")
    
    # Get environment observation
    try:
        import os
        os.environ['PYBULLET_EGL'] = '0'
        
        env = gym.make("PegTransfer-v0")
        env_obs, _ = env.reset()
        
        print(f"\n🌍 ENVIRONMENT FORMAT:")
        if isinstance(env_obs, dict):
            for key, value in env_obs.items():
                if isinstance(value, np.ndarray):
                    print(f"   • {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Check for mismatches
        print(f"\n🔍 FORMAT COMPARISON:")
        mismatches = []
        
        for key in train_obs.keys():
            if key not in env_obs:
                mismatches.append(f"'{key}' in training but not in environment")
            elif isinstance(train_obs[key], np.ndarray) and isinstance(env_obs[key], np.ndarray):
                if train_obs[key].shape != env_obs[key].shape:
                    mismatches.append(f"'{key}' shape mismatch: train={train_obs[key].shape} vs env={env_obs[key].shape}")
        
        for key in env_obs.keys():
            if key not in train_obs:
                mismatches.append(f"'{key}' in environment but not in training")
        
        if mismatches:
            print(f"   ❌ MISMATCHES FOUND:")
            for mismatch in mismatches:
                print(f"      • {mismatch}")
        else:
            print(f"   ✅ Formats match!")
        
        env.close()
        return len(mismatches) == 0
        
    except Exception as e:
        print(f"❌ Environment comparison failed: {e}")
        return False

def run_full_diagnosis(data_path, model_path):
    """Run complete diagnosis of BC training issues"""
    print(f"🚨 COMPREHENSIVE BC DIAGNOSIS")
    print("="*80)
    print(f"Data: {data_path}")
    print(f"Model: {model_path}")
    
    issues_found = []
    
    # 1. Analyze demo quality
    demo_stats = analyze_demo_data_quality(data_path)
    if demo_stats['success_rate'] < 0.7:
        issues_found.append(f"Low demo success rate: {demo_stats['success_rate']:.1%}")
    
    # 2. Test observation processing
    test_observation_processing(data_path, 'none')
    
    # 3. Test model loading
    policy, model_ok = test_model_loading_and_inference(model_path, data_path)
    if not model_ok:
        issues_found.append("Model loading/inference failed")
        return issues_found
    
    # 4. Test environment interaction
    total_return, action_mag = test_environment_interaction(policy)
    if total_return <= -40:
        issues_found.append(f"Very negative environment return: {total_return}")
    if action_mag < 0.001:
        issues_found.append(f"Very small action magnitudes: {action_mag:.6f}")
    
    # 5. Compare formats
    formats_match = compare_training_vs_evaluation_format(data_path)
    if not formats_match:
        issues_found.append("Training vs evaluation format mismatch")
    
    # Summary
    print(f"\n🎯 DIAGNOSIS SUMMARY")
    print("="*80)
    if issues_found:
        print(f"❌ ISSUES FOUND ({len(issues_found)}):")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print(f"✅ No obvious issues found - problem may be more subtle")
    
    return issues_found

def suggest_fixes(issues):
    """Suggest fixes based on identified issues"""
    print(f"\n💡 SUGGESTED FIXES")
    print("="*80)
    
    if "Low demo success rate" in str(issues):
        print(f"1. 📊 IMPROVE DEMONSTRATION QUALITY:")
        print(f"   • Generate new demos with higher success rate")
        print(f"   • Filter existing demos to keep only successful episodes")
        print(f"   • Check oracle policy implementation")
    
    if "Very negative environment return" in str(issues):
        print(f"2. 🌍 FIX ENVIRONMENT INTERACTION:")
        print(f"   • Check action space bounds and scaling")
        print(f"   • Verify reward function isn't giving excessive penalties")
        print(f"   • Test if actions are in correct format/range")
    
    if "Very small action magnitudes" in str(issues):
        print(f"3. 🎯 FIX ACTION SCALING:")
        print(f"   • Normalize actions during training")
        print(f"   • Check if action space needs different scaling")
        print(f"   • Verify policy output activation function")
    
    if "format mismatch" in str(issues):
        print(f"4. 🔧 FIX DATA FORMAT:")
        print(f"   • Ensure training and evaluation use same observation format")
        print(f"   • Check dimension compatibility")
        print(f"   • Verify goal conditioning is applied consistently")
    
    print(f"\n🚀 IMMEDIATE NEXT STEPS:")
    print(f"   1. Run this diagnosis script to identify specific issues")
    print(f"   2. Fix the highest priority issue first")
    print(f"   3. Test with a simple baseline (fewer epochs, simpler setup)")
    print(f"   4. Gradually add complexity once baseline works")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug BC training issues')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Run full diagnosis
    issues = run_full_diagnosis(args.data_path, args.model_path)
    
    # Suggest fixes
    suggest_fixes(issues)

if __name__ == '__main__':
    main()