"""
Fix Model Loading Issues

This script fixes the model loading problem by:
1. Correctly handling the nested model state dict
2. Testing the fixed model on the environment
3. Providing a working evaluation
"""

import os
import torch
import numpy as np
import pickle
import gymnasium as gym
import surrol.gym
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy
from cs224r.infrastructure import pytorch_util as ptu

def load_model_correctly(model_path):
    """Load model with correct handling of nested state dict"""
    print(f"🔧 FIXING MODEL LOADING")
    print("="*40)
    
    # Load the saved file
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ Checkpoint loaded")
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Extract the actual model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        print(f"✅ Found nested model_state_dict")
    else:
        model_state_dict = checkpoint
        config = {}
        print(f"✅ Using direct state dict")
    
    print(f"   Model state dict keys: {list(model_state_dict.keys())[:5]}...")
    
    # Create policy with correct config
    policy = EnhancedDictPolicy(
        ac_dim=config.get('action_dim', 5),
        ob_dim=config.get('obs_dim', 13),
        use_goal=config.get('use_goal', False),
        goal_dim=config.get('goal_dim', 0),
        goal_importance=2.0,
        n_layers=3,
        size=128,
        learning_rate=1e-3
    )
    
    # Load the correct state dict
    policy.load_state_dict(model_state_dict)
    policy.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Config: {config}")
    
    return policy, config

def test_fixed_model(policy, num_episodes=5):
    """Test the correctly loaded model"""
    print(f"\n🧪 TESTING FIXED MODEL")
    print("="*40)
    
    # Suppress PyBullet output
    os.environ['PYBULLET_EGL'] = '0'
    
    env = gym.make("PegTransfer-v0")
    
    results = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        for step in range(100):  # Max 100 steps
            # Create observation dict for policy
            obs_dict = {
                'observation': obs['observation'],
                'achieved_goal': obs.get('achieved_goal', np.zeros(3)),
                'desired_goal': obs.get('desired_goal', np.zeros(3))
            }
            
            # Get action from policy
            action = policy.get_action(obs_dict)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        success = info.get('is_success', False)
        results.append({
            'episode': episode + 1,
            'return': episode_return,
            'length': episode_length,
            'success': success
        })
        
        status = "✅" if success else "❌"
        print(f"   Episode {episode+1}: {status} Return={episode_return:.2f}, Steps={episode_length}, Success={success}")
    
    env.close()
    
    # Calculate stats
    success_rate = sum(1 for r in results if r['success']) / len(results)
    mean_return = np.mean([r['return'] for r in results])
    
    print(f"\n📊 RESULTS:")
    print(f"   • Success Rate: {success_rate:.1%}")
    print(f"   • Mean Return: {mean_return:.2f}")
    
    return results, success_rate, mean_return

def compare_with_random_policy():
    """Compare with a random policy baseline"""
    print(f"\n🎲 COMPARING WITH RANDOM POLICY")
    print("="*40)
    
    os.environ['PYBULLET_EGL'] = '0'
    env = gym.make("PegTransfer-v0")
    
    random_returns = []
    
    for episode in range(3):
        obs, _ = env.reset()
        episode_return = 0
        
        for step in range(50):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            
            if done or truncated:
                break
        
        random_returns.append(episode_return)
        success = info.get('is_success', False)
        print(f"   Random Episode {episode+1}: Return={episode_return:.2f}, Success={success}")
    
    env.close()
    
    mean_random_return = np.mean(random_returns)
    print(f"   Mean Random Return: {mean_random_return:.2f}")
    
    return mean_random_return

def analyze_action_quality(policy, data_path):
    """Analyze if the policy produces reasonable actions"""
    print(f"\n🎯 ANALYZING ACTION QUALITY")
    print("="*40)
    
    # Load demo data to compare
    data = np.load(data_path, allow_pickle=True)
    demo_actions = []
    for episode_acs in data['acs']:
        demo_actions.extend(episode_acs)
    demo_actions = np.array(demo_actions)
    
    print(f"📊 Demo action stats:")
    for i in range(demo_actions.shape[1]):
        mean_val = np.mean(demo_actions[:, i])
        std_val = np.std(demo_actions[:, i])
        print(f"   Dim {i}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Generate policy actions for comparison
    obs_data = data['obs']
    policy_actions = []
    
    for episode_idx in range(min(10, len(obs_data))):  # First 10 episodes
        episode = obs_data[episode_idx]
        for obs in episode[:10]:  # First 10 steps
            obs_dict = {
                'observation': obs['observation'],
                'achieved_goal': obs.get('achieved_goal', np.zeros(3)),
                'desired_goal': obs.get('desired_goal', np.zeros(3))
            }
            action = policy.get_action(obs_dict)
            policy_actions.append(action)
    
    policy_actions = np.array(policy_actions)
    
    print(f"\n🤖 Policy action stats:")
    for i in range(policy_actions.shape[1]):
        mean_val = np.mean(policy_actions[:, i])
        std_val = np.std(policy_actions[:, i])
        print(f"   Dim {i}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Check if they're similar
    print(f"\n🔍 Action comparison:")
    for i in range(min(demo_actions.shape[1], policy_actions.shape[1])):
        demo_mean = np.mean(demo_actions[:, i])
        policy_mean = np.mean(policy_actions[:, i])
        diff = abs(demo_mean - policy_mean)
        print(f"   Dim {i}: Demo={demo_mean:.3f}, Policy={policy_mean:.3f}, Diff={diff:.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix and test BC model loading')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return
    
    try:
        # Fix model loading
        policy, config = load_model_correctly(args.model_path)
        
        # Test the fixed model
        results, success_rate, mean_return = test_fixed_model(policy)
        
        # Compare with random
        random_return = compare_with_random_policy()
        
        # Analyze actions
        analyze_action_quality(policy, args.data_path)
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("="*60)
        
        if success_rate > 0:
            print(f"✅ SUCCESS! Model is working with {success_rate:.1%} success rate")
        elif mean_return > random_return:
            print(f"🟡 PARTIAL SUCCESS: Model performs better than random")
            print(f"   Model return: {mean_return:.2f}")
            print(f"   Random return: {random_return:.2f}")
        else:
            print(f"❌ STILL NOT WORKING: Model performs similar to random")
            print(f"   This suggests training issues beyond model loading")
        
        if success_rate == 0 and mean_return <= -40:
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. The model loading is now fixed")
            print(f"   2. But the model still gets 0% success")
            print(f"   3. This suggests the training process has other issues")
            print(f"   4. Try training for more epochs or with different hyperparameters")
            print(f"   5. The demonstrations are perfect (100% success) so the data is good")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()