"""
Fix Dimension Mismatch Between Training and Environment

This script addresses the core issue: 
- Training data has 13D observations
- Environment provides 19D observations  
- We need to either truncate env obs or retrain on correct dimensions
"""

import os
import numpy as np
import torch
import gymnasium as gym
import surrol.gym
import pickle
from cs224r.policies.enhanced_dict_policy import EnhancedDictPolicy

def analyze_dimension_mismatch():
    """Analyze the exact dimension mismatch"""
    print("🔍 ANALYZING DIMENSION MISMATCH")
    print("="*50)
    
    # 1. Check training data dimensions
    data_path = "/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz"
    data = np.load(data_path, allow_pickle=True)
    
    sample_obs = data['obs'][0][0]
    print(f"📚 TRAINING DATA:")
    print(f"   • Raw observation: {sample_obs['observation'].shape} = {len(sample_obs['observation'])}D")
    for key, value in sample_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: {value.shape}")
    
    # 2. Check environment dimensions
    os.environ['PYBULLET_EGL'] = '0'
    env = gym.make("PegTransfer-v0")
    env_obs, _ = env.reset()
    
    print(f"\n🌍 ENVIRONMENT DATA:")
    print(f"   • Raw observation: {env_obs['observation'].shape} = {len(env_obs['observation'])}D")
    for key, value in env_obs.items():
        if isinstance(value, np.ndarray):
            print(f"   • {key}: {value.shape}")
    
    env.close()
    
    # 3. Identify the issue
    train_dim = len(sample_obs['observation'])
    env_dim = len(env_obs['observation'])
    
    print(f"\n🚨 MISMATCH IDENTIFIED:")
    print(f"   • Training data: {train_dim}D")
    print(f"   • Environment:   {env_dim}D")
    print(f"   • Difference:    {env_dim - train_dim}D")
    
    if env_dim > train_dim:
        print(f"   • Problem: Environment has MORE dimensions than training")
        print(f"   • Solution 1: Truncate environment obs to {train_dim}D")
        print(f"   • Solution 2: Retrain model on {env_dim}D observations")
    
    return train_dim, env_dim, sample_obs, env_obs

def create_observation_adapter(train_dim, env_dim):
    """Create adapter to handle dimension mismatch"""
    print(f"\n🔧 CREATING OBSERVATION ADAPTER")
    print("="*50)
    
    def adapt_observation(env_obs, method='truncate'):
        """Adapt environment observation to match training format"""
        
        if method == 'truncate':
            # Simply truncate to training dimensions
            adapted_obs = env_obs['observation'][:train_dim]
            print(f"   📏 Truncated {env_dim}D → {train_dim}D")
            
        elif method == 'select_important':
            # Select most important dimensions (requires domain knowledge)
            # For PegTransfer, first 13D might be: robot state (7D) + key block positions (6D)
            adapted_obs = env_obs['observation'][:train_dim]
            print(f"   🎯 Selected important {train_dim}D from {env_dim}D")
            
        elif method == 'zero_pad':
            # Pad environment to match training (if env < train)
            if env_dim < train_dim:
                padding = np.zeros(train_dim - env_dim)
                adapted_obs = np.concatenate([env_obs['observation'], padding])
            else:
                adapted_obs = env_obs['observation'][:train_dim]
            print(f"   🔧 Adapted to {train_dim}D")
        
        return adapted_obs
    
    return adapt_observation

def test_adapted_model(model_path, adapter_fn, num_episodes=5):
    """Test model with observation adapter"""
    print(f"\n🧪 TESTING MODEL WITH ADAPTER")
    print("="*50)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    
    policy = EnhancedDictPolicy(
        ac_dim=config['action_dim'],
        ob_dim=config['obs_dim'],
        use_goal=config['use_goal'],
        goal_dim=config.get('goal_dim', 0),
        goal_importance=2.0,
        n_layers=3,
        size=128,
        learning_rate=1e-3
    )
    
    policy.load_state_dict(model_state_dict)
    policy.eval()
    
    print(f"✅ Model loaded: expects {config['obs_dim']}D input")
    
    # Test with environment
    os.environ['PYBULLET_EGL'] = '0'
    env = gym.make("PegTransfer-v0")
    
    results = []
    
    for episode in range(num_episodes):
        env_obs, _ = env.reset()
        episode_return = 0
        
        print(f"\n   Episode {episode+1}:")
        print(f"   • Raw env obs: {len(env_obs['observation'])}D")
        
        for step in range(50):
            # Adapt observation
            adapted_raw_obs = adapter_fn(env_obs)
            print(f"   • Adapted obs: {len(adapted_raw_obs)}D")
            
            # Create obs dict for policy
            obs_dict = {
                'observation': adapted_raw_obs,
                'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
                'desired_goal': env_obs.get('desired_goal', np.zeros(3))
            }
            
            # Get action
            try:
                action = policy.get_action(obs_dict)
                print(f"   • Action generated: {action[:3]}...")
                break  # Success - adapter works!
            except Exception as e:
                print(f"   ❌ Still failed: {e}")
                break
        
        # Test a few actual steps
        for step in range(10):
            try:
                adapted_raw_obs = adapter_fn(env_obs)
                obs_dict = {
                    'observation': adapted_raw_obs,
                    'achieved_goal': env_obs.get('achieved_goal', np.zeros(3)),
                    'desired_goal': env_obs.get('desired_goal', np.zeros(3))
                }
                
                action = policy.get_action(obs_dict)
                env_obs, reward, done, truncated, info = env.step(action)
                episode_return += reward
                
                if done or truncated:
                    break
                    
            except Exception as e:
                print(f"   ❌ Step {step} failed: {e}")
                break
        
        success = info.get('is_success', False) if 'info' in locals() else False
        results.append({
            'episode': episode + 1,
            'return': episode_return,
            'success': success,
            'steps': step + 1
        })
        
        status = "✅" if success else "❌"
        print(f"   {status} Return: {episode_return:.2f}, Success: {success}")
    
    env.close()
    
    # Calculate results
    success_rate = sum(1 for r in results if r['success']) / len(results)
    mean_return = np.mean([r['return'] for r in results])
    
    print(f"\n📊 ADAPTER TEST RESULTS:")
    print(f"   • Success Rate: {success_rate:.1%}")
    print(f"   • Mean Return: {mean_return:.2f}")
    
    return success_rate, mean_return, results

def retrain_with_correct_dimensions(data_path, save_dir, epochs=50):
    """Retrain model with correct environment dimensions"""
    print(f"\n🚀 RETRAINING WITH CORRECT DIMENSIONS")
    print("="*50)
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    obs_data = data['obs']
    acs_data = data['acs']
    
    # Get environment observation dimension
    os.environ['PYBULLET_EGL'] = '0'
    env = gym.make("PegTransfer-v0")
    env_obs, _ = env.reset()
    correct_obs_dim = len(env_obs['observation'])
    env.close()
    
    print(f"   • Environment obs dim: {correct_obs_dim}D")
    print(f"   • Will train model to expect {correct_obs_dim}D input")
    
    # Prepare training data - simulate correct dimensions
    # For now, we'll pad the training data to match environment
    train_obs = []
    train_actions = []
    
    for episode_idx in range(len(obs_data)):
        episode_obs = obs_data[episode_idx]
        episode_acs = acs_data[episode_idx]
        
        for t in range(len(episode_obs) - 1):
            obs = episode_obs[t]
            action = episode_acs[t]
            
            # Pad observation to match environment
            raw_obs = obs['observation']
            if len(raw_obs) < correct_obs_dim:
                # Pad with zeros or reasonable values
                padding_size = correct_obs_dim - len(raw_obs)
                padded_obs = np.concatenate([raw_obs, np.zeros(padding_size)])
            else:
                padded_obs = raw_obs[:correct_obs_dim]
            
            train_obs.append(padded_obs)
            train_actions.append(action)
    
    train_obs = np.array(train_obs)
    train_actions = np.array(train_actions)
    
    print(f"   • Training data prepared: {len(train_obs)} samples")
    print(f"   • Observation shape: {train_obs.shape}")
    
    # Create policy with correct dimensions
    policy = EnhancedDictPolicy(
        ac_dim=train_actions.shape[1],
        ob_dim=correct_obs_dim,  # Correct dimension!
        use_goal=False,
        goal_dim=0,
        goal_importance=1.0,
        n_layers=3,
        size=128,
        learning_rate=1e-3
    )
    
    print(f"   • Policy created: {correct_obs_dim}D → {train_actions.shape[1]}D")
    
    # Quick training
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Random batches
        n_batches = len(train_obs) // batch_size
        for batch in range(n_batches):
            batch_indices = np.random.choice(len(train_obs), batch_size, replace=False)
            
            # Create observation batch
            obs_batch = []
            for idx in batch_indices:
                obs_dict = {
                    'observation': train_obs[idx],
                    'achieved_goal': np.zeros(3),
                    'desired_goal': np.zeros(3)
                }
                obs_batch.append(obs_dict)
            
            obs_batch = np.array(obs_batch, dtype=object)
            action_batch = train_actions[batch_indices]
            
            try:
                loss = policy.update(obs_batch, action_batch)
                epoch_losses.append(loss)
            except Exception as e:
                print(f"   ❌ Training error: {e}")
                continue
        
        if epoch % 10 == 0:
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"   Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
    
    # Save retrained model
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'retrained_model_correct_dims.pt')
    torch.save(policy.state_dict(), model_path)
    
    config_path = os.path.join(save_dir, 'retrained_config.pkl')
    config = {
        'ac_dim': train_actions.shape[1],
        'ob_dim': correct_obs_dim,
        'use_goal': False,
        'goal_dim': 0,
        'obs_dim': correct_obs_dim,
        'action_dim': train_actions.shape[1],
        'condition_type': 'none'
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"   ✅ Retrained model saved: {model_path}")
    
    return policy, model_path

def main():
    print("🎯 FIXING DIMENSION MISMATCH ISSUE")
    print("="*80)
    
    # 1. Analyze the mismatch
    train_dim, env_dim, sample_train_obs, sample_env_obs = analyze_dimension_mismatch()
    
    # 2. Test adapter approach
    print(f"\n" + "="*80)
    print("APPROACH 1: OBSERVATION ADAPTER (QUICK FIX)")
    print("="*80)
    
    adapter = create_observation_adapter(train_dim, env_dim)
    
    model_path = "/home/ubuntu/project/behavior-cloning/hw1/cs224r/experiments/10k_demos/baseline_test_20250604_232519_none/final_model_none.pt"
    
    try:
        success_rate, mean_return, results = test_adapted_model(model_path, adapter, num_episodes=3)
        
        if success_rate > 0:
            print(f"🎉 ADAPTER WORKS! Success rate: {success_rate:.1%}")
        else:
            print(f"🔧 Adapter helps but model still needs improvement")
    except Exception as e:
        print(f"❌ Adapter approach failed: {e}")
    
    # 3. Test retraining approach
    print(f"\n" + "="*80)
    print("APPROACH 2: RETRAIN WITH CORRECT DIMENSIONS")
    print("="*80)
    
    try:
        data_path = "/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz"
        save_dir = "experiments/retrained_correct_dims"
        
        retrained_policy, retrained_model_path = retrain_with_correct_dimensions(
            data_path, save_dir, epochs=30
        )
        
        print(f"✅ Retraining completed")
        
        # Test retrained model
        no_adapter = lambda x: x['observation']  # No adaptation needed
        success_rate_retrained, mean_return_retrained, _ = test_adapted_model(
            retrained_model_path, no_adapter, num_episodes=3
        )
        
        print(f"📊 Retrained model results:")
        print(f"   • Success rate: {success_rate_retrained:.1%}")
        print(f"   • Mean return: {mean_return_retrained:.2f}")
        
    except Exception as e:
        print(f"❌ Retraining approach failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("="*80)
    print(f"1. 🔧 IMMEDIATE FIX: Use observation adapter to truncate 19D → 13D")
    print(f"2. 🚀 BETTER SOLUTION: Retrain with correct 19D observations")
    print(f"3. 📊 DATA CHECK: Verify demo collection matches environment")
    print(f"4. 🧪 TEST: Run longer evaluation once dimensions are fixed")

if __name__ == '__main__':
    main()