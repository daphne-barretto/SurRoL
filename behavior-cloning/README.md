# Advanced Behavior Cloning Framework

This repository contains an advanced behavior cloning implementation that extends beyond the basic CS224R homework framework to include additional features and integrations with SurRoL for surgical robotics tasks.

## Repository Structure

```
behavior-cloning/
│
└── hw1/
    ├── cs224r/
    │   ├── agents/           # Agent implementations
    │   ├── data/             # Training data
    │   ├── infrastructure/   # Core utilities
    │   ├── models/           # Trained model checkpoints and logs
    │   ├── policies/         # Policy network implementations
    │   ├── scripts/          # Training and evaluation scripts
    │   └── videos/           # Rollout videos
    │
    ├── installation.md       # Installation instructions
    ├── requirements.txt      # Dependencies
    ├── setup.py              # Package setup
    └── README.md             # Original homework instructions
```

## Key Components

### Policy Implementations

- **MLP_policy.py**: Basic MLP policy for standard environments
- **CNN_policy.py**: Convolutional policy for image-based observations
- **dict_mlp_policy.py**: Policy for dictionary observations
- **enhanced_dict_policy.py**: Advanced policy for dictionaries with goal-conditioned features
- **smoothed_policy.py**: Policy with action smoothing
- **pd_controller_policy.py**: PD controller-based policy

### Training Scripts

- **run_hw1.py**: Original homework script for basic behavior cloning and DAgger
- **train_enhanced_bc.py**: Enhanced BC with curriculum learning, data augmentation and early stopping
- **train_bc_surrol.py**: BC training specifically for SurRoL environments
- **train_dict_bc.py**: BC training for dictionary observation spaces
- **train_distance_focused_bc.py**: BC with distance-based reward shaping

### Evaluation Scripts

- **evaluate_enhanced_bc.py**: Evaluate enhanced BC models
- **evaluate_bc_surrol.py**: Evaluate models on SurRoL environments
- **evaluate_dict_bc.py**: Evaluate dictionary-based models
- **evaluate_smoothed_policy.py**: Evaluate policies with action smoothing
- **evaluate_pd_policy.py**: Evaluate PD controller policies

### Utility Scripts

- **convert_npz_to_bc.py**: Convert SurRoL data format (.npz) to behavior cloning format (.pkl)

## SurRoL Integration

This project integrates with [SurRoL](https://github.com/med-air/SurRoL), an open-source reinforcement learning platform for surgical robot learning. The integration enables behavior cloning on surgical robotics tasks.

### SurRoL Features

- dVRK compatible robots
- Gym-style API for reinforcement learning
- Surgical-related tasks
- Physics simulation via PyBullet

### SurRoL Changes

The integration includes:
- Custom CNN policy for SurRoL's image-based observations
- Data conversion utilities to transform SurRoL demonstrations to BC format
- Task-specific hyperparameters for surgical tasks like NeedleReach

## Usage Guide

### Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n cs224r python=3.11 -y
   conda activate cs224r
   ```

2. Install PyTorch with appropriate CUDA support:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. Install remaining requirements:
   ```bash
   cd behavior-cloning/hw1
   pip install -r requirements.txt
   ```

4. Install the package:
   ```bash
   pip install -e .
   ```

### Training Models

#### Enhanced Behavior Cloning

```bash
python -m cs224r.scripts.train_enhanced_bc \
    --data cs224r/data/needle_reach_bc_data_1000_fixed.pkl \
    --save_dir cs224r/models/needle_reach_high_goal_weight_1000 \
    --epochs 150 \
    --goal_importance 5.0 \
    --hidden_size 256
```

Parameters:
- `--data`: Path to training data (pickle format)
- `--save_dir`: Directory to save model and logs
- `--epochs`: Number of training epochs
- `--goal_importance`: Weight for goal-conditioned learning (higher values emphasize goal achievement)
- `--hidden_size`: Size of hidden layers
- `--learning_rate`: Learning rate (default: 5e-4)
- `--batch_size`: Batch size (default: 64)
- `--no_data_augmentation`: Disable data augmentation
- `--no_curriculum`: Disable curriculum learning

### Evaluating Models

#### Enhanced BC Evaluation

For evaluating a specific model with visualization:

```bash
python -m cs224r.scripts.evaluate_enhanced_bc \
    --model cs224r/models/needle_reach_high_goal_weight_1000_20250519_225221/enhanced_bc_policy.pt \
    --env NeedleReach-v0 \
    --episodes 10 \
    --video_path cs224r/videos/needle_reach_high_goal_weight_1000_20250519_225221 \
    --goal_importance 5.0 \
    --hidden_size 256 \
    --layers 3
```

Parameters:
- `--model` or `--model_path`: Path to the trained model file (.pt)
- `--env`: Name of the environment to test in (e.g., "NeedleReach-v0")
- `--episodes`: Number of evaluation episodes to run
- `--video_path`: Directory to save evaluation videos and trajectory visualizations
- `--goal_importance`: Weight for goal information (should match training value)
- `--hidden_size`: Size of hidden layers (should match training architecture)
- `--layers`: Number of hidden layers in the policy network
- `--no_video`: Disable video recording
- `--no_goal`: Disable goal conditioning

The evaluation script runs the specified number of episodes and reports metrics including:
- Success rate (episodes where goal was reached)
- Mean episode return and length
- Distance metrics (initial and final distances to goal)
- Progress towards the goal

When video recording is enabled, the script creates:
- Video recordings of each episode
- Trajectory visualizations showing the path of the end effector
- A summary plot with statistics across all episodes

### Converting SurRoL Data

```bash
python -m cs224r.scripts.convert_npz_to_bc \
    --input surrol_demos.npz \
    --output cs224r/data/surrol_needle_reach.pkl
```

## Performance Tips

1. **GPU Acceleration**: Use GPU for training when available:
   ```bash
   # Check if GPU is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Curriculum Learning**: For complex tasks, enable curriculum learning to start with easier examples.

3. **Data Augmentation**: Use data augmentation to improve robustness and generalization.

4. **Early Stopping**: The enhanced trainer includes early stopping to prevent overfitting.

5. **Hyperparameter Tuning**: Key hyperparameters to adjust:
   - Learning rate
   - Network depth/width
   - Goal importance (for goal-conditioned tasks)
   - Batch size

## Extending the Framework

To add a new environment or task:
1. Create training data in pickle format
2. Choose an appropriate policy architecture
3. Adjust hyperparameters based on task complexity
4. Train and evaluate using the provided scripts

For image-based tasks, use the CNN policy; for state-based tasks, use MLP or dict-based policies depending on observation format.