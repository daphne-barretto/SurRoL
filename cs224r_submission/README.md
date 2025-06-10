# CS224R Final Project: Goal-Conditioned Learning for Surgical Robotics (PART WITH BEHAVIOR CLONING)

**Project**: Goal-Conditioned Reinforcement Learning for Surgical Robotics Manipulation  
**Course**: CS224R - Deep Reinforcement Learning
**Task**: Two-block peg transfer with goal conditioning comparison using behavior cloning

## 🎯 Project Overview

This project investigates different goal conditioning strategies for behavior cloning in surgical robotics. We compare 4 approaches across spatial vs semantic representations and with/without color information using the SurRoL surgical robotics simulation environment.

**⚠️ IMPORTANT SETUP NOTE:**
This submission is **SELF-CONTAINED** and includes a minimal SurRoL package (~1MB) with all necessary environment definitions. The scripts automatically set up the correct PYTHONPATH to use the local SurRoL package. No external SurRoL installation is required. All scripts have been tested and verified to work correctly.

### Key Research Questions
1. How do spatial versus semantic goal representations affect learning performance?
2. What is the impact of color information encoding on goal-conditioned behavior cloning?
3. How does training performance relate to evaluation performance in surgical robotics tasks?

### Main Findings
- **Best Method**: Spatial conditioning without color (10.0% evaluation success rate)
- **Spatial > Semantic**: Spatial outperforms semantic by 38.7% in evaluation (7.2% vs 5.2%)
- **Color Impact**: Negative for both spatial (-5.7%) and semantic (-5.0%) approaches
- **Generalization Gap**: Training performance is not predictive of evaluation performance
- **Overfitting**: Semantic with color shows severe overfitting (25.0% training → 2.7% evaluation)

## 📁 Project Structure

```
cs224r_submission/
├── README.md                    # This file
├── scripts/                     # Executable scripts
│   ├── train.sh                # Training script
│   ├── evaluate.sh             # Evaluation script
│   ├── analyze.sh              # Analysis script
│   └── run_all.sh              # Complete pipeline
├── src/                         # Source code
│   ├── training/               # Training modules
│   │   ├── universal_bc_system.py
│   │   └── universal_bc_trainer.py
│   ├── evaluation/             # Evaluation modules
│   │   └── universal_bc_evaluator.py
│   ├── analysis/               # Analysis scripts
│   │   ├── corrected_100ep_analysis.py
│   │   ├── focused_analysis_report.py
│   │   └── corrected_results_table.py
│   └── tasks/                  # SurRoL task definitions
├── data/                       # Data and experiments
│   ├── demo/                   # Demonstration data (11 .npz files, ~400MB)
│   ├── experiments/            # Trained models (12 experiments)
│   └── logs/                   # Training logs
└── results/                    # Generated results
    ├── plots/                  # Visualizations
    └── tables/                 # Data tables
```

## 🚀 Quick Start

### Prerequisites
1. **Conda Environment**: Create and activate the `gcrl` environment with required dependencies:
   ```bash
   conda create -n gcrl python=3.8
   conda activate gcrl
   pip install torch torchvision numpy pandas matplotlib seaborn
   # Install SurRoL and other dependencies as needed
   ```

2. **SurRoL Environment**: Ensure SurRoL surgical robotics simulator is properly installed

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Quick test run (30-60 minutes)
./scripts/run_all.sh --quick

# Full experimental pipeline (4-8 hours)
./scripts/run_all.sh
```

### Option 2: Run Individual Components

#### Training
```bash
# Train specific method
./scripts/train.sh --method spatial_no_color --demos 5000 --epochs 50 --runs 3

# Train all methods
for method in spatial_no_color spatial_with_color semantic_no_color semantic_with_color; do
    ./scripts/train.sh --method $method --demos 5000 --runs 3
done
```

#### Evaluation
```bash
# Evaluate specific method
./scripts/evaluate.sh --method spatial_no_color --episodes 100

# Evaluate all trained models
./scripts/evaluate.sh --method all --episodes 100
```

#### Analysis
```bash
# Run all analysis types
./scripts/analyze.sh --type all --format both

# Run specific analysis
./scripts/analyze.sh --type corrected --format pdf
```

## 🎯 Goal Conditioning Methods

### 1. Spatial without Color (`spatial_no_color`)
- **Representation**: Direct (x, y, z) coordinates for target block and target peg
- **Dimensionality**: 18D observation space
- **Task File**: `peg_transfer-two_blocks-with_all_blocks-no_obs_target_block_peg.py`
- **Demo Data**: `data_PegTransferTwoBlocksNoColorTargetBlockPeg-v0_random_5000_2025-06-06_03-32-20.npz`

### 2. Spatial with Color (`spatial_with_color`)
- **Representation**: Spatial coordinates + RGBA color encoding
- **Dimensionality**: 22D observation space  
- **Task File**: `peg_transfer-two_blocks-with_all_blocks_colored-targetblocktargetpeg.py`
- **Demo Data**: `data_PegTransferTwoBlocksTargetBlockTargetPeg-v0_5000_2025-06-08_03-51-06.npz`

### 3. Semantic without Color (`semantic_no_color`)
- **Representation**: One-hot categorical encoding + target peg coordinates
- **Dimensionality**: 18D observation space
- **Task File**: `peg_transfer-two_blocks-with_all_blocks-no_obs_one_hot_target_peg.py`
- **Demo Data**: `data_PegTransferTwoBlocksNoColorOneHotTargetPeg-v0_random_5000_2025-06-06_03-31-40.npz`

### 4. Semantic with Color (`semantic_with_color`)
- **Representation**: Categorical + spatial + RGBA encoding
- **Dimensionality**: 22D observation space
- **Task File**: `peg_transfer-two_blocks-with_all_blocks_colored-onehottargetpeg.py`
- **Demo Data**: `data_PegTransferTwoBlocksOneHotTargetPeg-v0_random_5000_2025-06-06_11-52-13.npz`

## 📊 Results Summary

### Training vs Evaluation Performance

| Method | Training Mean | Evaluation Mean | Performance Gap | Best Evaluation Run |
|--------|---------------|-----------------|------------------|-------------------|
| Spatial without Color | 18.3% ± 12.5% | **10.0% ± 1.6%** | +8.3% | 12.0% |
| Spatial with Color | 15.0% ± 4.1% | 4.3% ± 1.7% | +10.7% | 6.0% |
| Semantic without Color | 18.3% ± 2.4% | 7.7% ± 4.1% | +10.7% | 13.0% |
| Semantic with Color | **25.0% ± 10.8%** | 2.7% ± 2.4% | +22.3% | 6.0% |

### Key Insights
1. **Training ≠ Evaluation**: Best training method (semantic with color) performs worst in evaluation
2. **Overfitting**: Large performance gaps indicate overfitting, especially for semantic with color
3. **Simplicity Wins**: Simpler representations (spatial without color) generalize better
4. **Color Hurts**: Color information consistently reduces evaluation performance

## 📈 Generated Results

### Plots
- `comprehensive_100ep_corrected_analysis.png/pdf` - Main 6-panel analysis
- `detailed_performance_heatmap.png` - Training vs evaluation heatmap
- `corrected_comprehensive_comparison.png/pdf` - Method comparison plots
- `focused_training_analysis.png/pdf` - Training-focused analysis

### Tables
- `corrected_comprehensive_results.csv` - Complete results table
- `comprehensive_training_vs_evaluation_comparison.csv` - Detailed comparison
- `training_performance_summary.csv` - Training statistics
- `method_comparison_table.csv` - Method comparisons

## 📊 Data Included

### Demonstration Data (11 files, ~400MB total)
The submission includes complete demonstration datasets for all goal conditioning approaches:

- **Spatial Methods**: NoColor and Colored target block + target peg demonstrations
- **Semantic Methods**: OneHot and OneHotTargetPeg demonstrations  
- **Additional Variants**: Various conditioning approaches for comprehensive comparison
- **Episode Counts**: 5,000 demonstrations per main method (some include 10-episode test sets)

### Trained Models (12 experiments)
- **4 Methods × 3 Runs Each**: Complete experimental matrix
- **Model Files**: Trained neural network weights and configurations
- **Training Logs**: Detailed training performance and learning curves
- **Evaluation Results**: 100-episode evaluation data for each model

## 🔧 Script Usage Details

### Training Script (`./scripts/train.sh`)
```bash
Options:
  --method    Goal conditioning method (spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color)
  --demos     Number of demonstration episodes (default: 5000)
  --epochs    Training epochs (default: 50)
  --runs      Number of training runs (default: 3)

Examples:
  ./scripts/train.sh --method spatial_no_color --demos 5000 --runs 3
  ./scripts/train.sh --method semantic_with_color --epochs 100
```

### Evaluation Script (`./scripts/evaluate.sh`)
```bash
Options:
  --method     Method to evaluate (spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color, all)
  --episodes   Number of evaluation episodes per model (default: 100)
  --verbose    Enable verbose output

Examples:
  ./scripts/evaluate.sh --method spatial_no_color --episodes 100
  ./scripts/evaluate.sh --method all --episodes 50
```

### Analysis Script (`./scripts/analyze.sh`)
```bash
Options:
  --type      Analysis type (corrected, focused, comparison, all)
  --format    Output format (png, pdf, both)

Analysis Types:
  corrected    - Corrected 100-episode analysis with training vs evaluation
  focused      - Focused training analysis with statistical insights
  comparison   - Comprehensive comparison plots and tables
  all          - Run all analysis types

Examples:
  ./scripts/analyze.sh --type corrected --format pdf
  ./scripts/analyze.sh --type all --format both
```

## 🧪 Experimental Design

### Training Configuration
- **Demonstrations**: 5,000 episodes per method (included in submission)
- **Training Runs**: 3 independent runs per method
- **Total Models**: 12 (4 methods × 3 runs)
- **Training Epochs**: 50 (adjustable)

### Evaluation Protocol
- **Episodes**: 100 episodes per model
- **Total Evaluations**: 1,200 episodes
- **Metrics**: Success rate, episode patterns
- **Environment**: SurRoL two-block peg transfer task

### Statistical Analysis
- **Comparisons**: Training vs evaluation performance
- **Insights**: Generalization gaps, overfitting detection
- **Visualizations**: Heatmaps, distribution plots, individual run tracking

## 📝 Important Notes

### Environment Dependencies
- **SurRoL**: Surgical robotics simulation environment
- **Conda Environment**: `gcrl` with PyTorch, pandas, matplotlib
- **Python**: 3.8+ recommended
- **CUDA**: Optional for GPU acceleration

### Submission Size
- **Total Size**: ~408MB (reasonable for submission)
- **Data**: ~400MB (demonstration datasets)
- **Code & Results**: ~8MB (scripts, models, analysis)

### File Organization
- All scripts are executable and include help documentation (`--help`)
- Results are automatically organized into plots/, tables/, and reports/
- Temporary files are cleaned up automatically
- Source code is modular and well-documented

### Innovation
- **Semantic vs Spatial**: Novel comparison of goal representation strategies
- **Overfitting Detection**: Identified severe overfitting in complex representations
- **Color Analysis**: Demonstrated that color information is detrimental despite intuition
- **Generalization Insights**: Showed that simpler representations generalize better

---
