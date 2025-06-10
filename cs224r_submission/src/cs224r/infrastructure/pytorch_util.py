"""
PyTorch utilities for CS224R
"""

import torch
import torch.nn as nn
import numpy as np

# GPU/CPU handling
device = None

def init_gpu(use_gpu=True, gpu_id=0):
    """Initialize GPU usage"""
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def get_device():
    """Get current device"""
    global device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def from_numpy(data):
    """Convert numpy array to tensor"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(get_device())
    return data

def to_numpy(tensor):
    """Convert tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

# Network building utilities
def build_mlp(input_size, output_size, hidden_sizes, activation=nn.ReLU):
    """Build a multi-layer perceptron"""
    layers = []
    sizes = [input_size] + hidden_sizes + [output_size]
    
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # No activation after last layer
            layers.append(activation())
    
    return nn.Sequential(*layers)

# Weight initialization
def weight_init(m):
    """Initialize weights for neural network layers"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Training utilities
def soft_update_params(net, target_net, tau):
    """Soft update of target network parameters"""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def hard_update_params(net, target_net):
    """Hard update of target network parameters"""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)

# Initialize device on import
init_gpu() 