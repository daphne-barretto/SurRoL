"""
Simple test to verify data loading and basic BC training works
"""

import numpy as np
import torch
import torch.nn as nn
import os

def test_data_loading():
    """Test loading the data file"""
    data_path = "/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz"
    
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    print(f"Available keys: {list(data.keys())}")
    
    # Load using correct structure
    obs_episodes = data['obs']  # (1000, 51) 
    action_episodes = data['acs']  # (1000, 50, 5)
    
    print(f"Obs episodes shape: {obs_episodes.shape}")
    print(f"Action episodes shape: {action_episodes.shape}")
    
    # Process into flat arrays
    all_observations = []
    all_actions = []
    
    for episode_idx in range(len(obs_episodes)):
        obs_episode = obs_episodes[episode_idx]  # Array of 51 dicts
        action_episode = action_episodes[episode_idx]  # (50, 5)
        
        # Skip the last observation since we have 51 obs but 50 actions
        for step in range(len(action_episode)):
            obs_dict = obs_episode[step]
            action = action_episode[step]
            
            all_observations.append(obs_dict['observation'])
            all_actions.append(action)
    
    # Convert to numpy arrays
    observations = np.array(all_observations)
    actions = np.array(all_actions)
    
    print(f"Flattened observations shape: {observations.shape}")
    print(f"Flattened actions shape: {actions.shape}")
    
    return observations, actions

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def main():
    print("=== Simple BC Test ===")
    
    # Load data
    try:
        observations, actions = test_data_loading()
        print("✓ Data loading successful")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Create model
    try:
        obs_dim = observations.shape[1]
        action_dim = actions.shape[1]
        
        print(f"Creating model with obs_dim={obs_dim}, action_dim={action_dim}")
        policy = SimplePolicy(obs_dim, action_dim)
        print("✓ Model creation successful")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    # Test training step
    try:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        # Single batch test
        batch_size = 32
        indices = np.random.choice(len(observations), batch_size, replace=False)
        obs_batch = torch.FloatTensor(observations[indices])
        actions_batch = torch.FloatTensor(actions[indices])
        
        # Forward pass
        predicted_actions = policy(obs_batch)
        loss = criterion(predicted_actions, actions_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful, loss: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test saving
    try:
        save_dir = "./test_save"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_model.pt")
        
        torch.save({
            'model_state_dict': policy.state_dict(),
            'obs_dim': obs_dim,
            'action_dim': action_dim
        }, save_path)
        
        print(f"✓ Model saving successful: {save_path}")
        
        # Verify file exists
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"✓ Saved file exists, size: {size} bytes")
        else:
            print("✗ Saved file not found")
            
    except Exception as e:
        print(f"✗ Model saving failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=== All tests passed! ===")

if __name__ == "__main__":
    main()