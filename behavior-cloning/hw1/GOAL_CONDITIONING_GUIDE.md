# 🎯 Complete Goal Conditioning Guide

## Overview

This guide explains all goal conditioning formats supported by your behavior cloning system, why they exist, and how to use them effectively.

## 🎯 Goal Conditioning Formats

### 1. **`none`** - No Goal Conditioning (Baseline)
- **Description**: Just uses raw observations without any goal information
- **Use Case**: Baseline comparison, simple tasks that don't require goal specification
- **Advantages**: Simple, no extra complexity, good baseline
- **Disadvantages**: Cannot handle multi-goal tasks
- **Dimensions**: +0 dimensions (19D total)
- **When to use**: For comparing against goal-conditioned approaches

### 2. **`one_hot`** - Block Color as One-Hot Vector
- **Description**: Block color encoded as [1,0,0,0] for red, [0,1,0,0] for green, etc.
- **Use Case**: Categorical color selection tasks
- **Advantages**: Efficient discrete representation, standard ML approach
- **Disadvantages**: No spatial information about block location
- **Dimensions**: +4 dimensions (23D total: 19D obs + 4D color)
- **When to use**: When the task is primarily about selecting the right colored block

### 3. **`target_block`** - Target Block Position
- **Description**: 3D coordinates of the target block to manipulate
- **Use Case**: Spatial manipulation tasks where position matters
- **Advantages**: Spatial awareness, position-based reasoning
- **Disadvantages**: No object identity information
- **Dimensions**: +3 dimensions (22D total: 19D obs + 3D position)
- **When to use**: When the robot needs to know "where" to pick up

### 4. **`target_block_and_target_peg`** - Complete Spatial Specification
- **Description**: Position of target block + position of target peg
- **Use Case**: Complete pick-and-place tasks
- **Advantages**: Full spatial specification ("pick this, place there")
- **Disadvantages**: High dimensional, more complex
- **Dimensions**: +6 dimensions (25D total: 19D obs + 3D block + 3D peg)
- **When to use**: For complex manipulation requiring both pickup and placement locations

### 5. **`one_hot_and_target_peg`** - Hybrid Approach
- **Description**: Block color (one-hot) + target peg position
- **Use Case**: Hybrid categorical + spatial tasks
- **Advantages**: Balanced information content, separates "what" from "where"
- **Disadvantages**: Still somewhat complex
- **Dimensions**: +7 dimensions (26D total: 19D obs + 4D color + 3D peg)
- **When to use**: When you need both object identity and goal location

### 6. **`four_tuple`** - RGBA Color Representation
- **Description**: Block color as RGBA tuple (1.0, 0.0, 0.0, 1.0) for red
- **Use Case**: Continuous color representation
- **Advantages**: Smooth color space, mimics visual processing
- **Disadvantages**: Redundant alpha channel
- **Dimensions**: +4 dimensions (23D total: 19D obs + 4D RGBA)
- **When to use**: For continuous color spaces or visual-based policies

### 7. **`color_language`** - Natural Language
- **Description**: Block color as text string "red", "green", etc.
- **Use Case**: Language-conditioned policies
- **Advantages**: Human-interpretable, natural language interface
- **Disadvantages**: Requires text processing/embedding
- **Dimensions**: +4 dimensions (embedded, 23D total)
- **When to use**: For language-conditioned or instruction-following policies

## 🛠️ How to Use Each Format

### Training
```bash
# Baseline (no goal conditioning)
python cs224r/scripts/train_goal_conditioned_bc.py \
    --data data.npz \
    --save_dir output_baseline \
    --condition_type none \
    --no_goal

# One-hot color conditioning
python cs224r/scripts/train_goal_conditioned_bc.py \
    --data data.npz \
    --save_dir output_one_hot \
    --condition_type one_hot

# Spatial conditioning
python cs224r/scripts/train_goal_conditioned_bc.py \
    --data data.npz \
    --save_dir output_spatial \
    --condition_type target_block_and_target_peg

# Hybrid conditioning
python cs224r/scripts/train_goal_conditioned_bc.py \
    --data data.npz \
    --save_dir output_hybrid \
    --condition_type one_hot_and_target_peg
```

### Evaluation
```bash
# Universal evaluator (auto-detects format)
python cs224r/scripts/universal_bc_evaluator.py \
    --model_path output_one_hot/final_model_one_hot.pt \
    --num_episodes 10 \
    --save_video

# Override format if needed
python cs224r/scripts/universal_bc_evaluator.py \
    --model_path model.pt \
    --condition_type target_block \
    --num_episodes 10

# Batch evaluation
python cs224r/scripts/universal_bc_evaluator.py \
    --batch_eval ./models/ \
    --num_episodes 5 \
    --output_json results.json
```

### Visualization
```bash
# Create detailed evaluation report
python cs224r/scripts/bc_visualization_tools.py \
    --mode evaluate \
    --model_path model.pt \
    --num_episodes 20 \
    --output_dir evaluation_results

# Compare multiple training runs
python cs224r/scripts/bc_visualization_tools.py \
    --mode compare \
    --experiment_dirs output_one_hot/ output_baseline/ output_spatial/ \
    --output_dir comparison_results

# Debug goal conditioning
python cs224r/scripts/bc_visualization_tools.py \
    --mode debug \
    --model_path model.pt \
    --data_path data.npz
```

## 🔍 Validation and Testing

### Validate Data Compatibility
```bash
# Check which formats your data supports
python cs224r/scripts/universal_bc_evaluator.py \
    --validate_data data.npz

# Test all formats comprehensively
python cs224r/scripts/test_all_formats.py \
    --data_path data.npz

# Quick validation only
python cs224r/scripts/test_all_formats.py \
    --quick_test
```

### Format Explanation
```bash
# Get detailed explanation of all formats
python cs224r/scripts/goal_conditioning_compatibility.py
python cs224r/scripts/universal_bc_evaluator.py --explain_formats
python cs224r/scripts/bc_visualization_tools.py --mode explain
```

## 🎯 Choosing the Right Format

| Task Type | Recommended Format | Reason |
|-----------|-------------------|---------|
| Simple baseline | `none` | No goal complexity needed |
| Color-based selection | `one_hot` or `four_tuple` | Efficient categorical representation |
| Position-based tasks | `target_block` | Spatial awareness |
| Full pick-and-place | `target_block_and_target_peg` | Complete spatial info |
| Balanced approach | `one_hot_and_target_peg` | Good compromise |
| Language interfaces | `color_language` | Human-readable |

## 🔧 Compatibility Features

### Automatic Dimension Fixing
The universal evaluator automatically handles dimension mismatches:
- Detects when models were trained with incorrect dimensions
- Automatically truncates or pads observations for compatibility
- Provides clear warnings about dimension issues

### Universal Model Loading
- Automatically detects goal conditioning format from model
- Handles both embedded configs and separate config files
- Provides fallback configurations for older models

### Cross-Format Evaluation
- Can override the detected format for testing
- Handles format conversion on-the-fly
- Provides compatibility checks

## 📊 Performance Analysis

### Success Metrics
- **Success Rate**: Percentage of episodes that achieve the goal
- **Mean Return**: Average cumulative reward per episode
- **Episode Length**: Average number of steps to completion
- **Compatibility Check**: Whether the model runs without errors

### Debugging Features
- Detailed episode-by-episode breakdown
- Action error counting and analysis
- Dimension mismatch detection and fixing
- Trajectory pattern analysis

## 🎬 Video Generation

All evaluation scripts can generate videos:
- **Format**: GIF files for easy viewing
- **Naming**: Includes episode number, format type, and success/failure
- **Content**: Shows robot behavior with goal conditioning applied
- **Analysis**: Automatic failure pattern detection

## 🚨 Common Issues and Solutions

### Dimension Mismatches
**Problem**: Model expects different observation dimensions than environment provides
**Solution**: Universal evaluator automatically detects and fixes these

### Format Incompatibility
**Problem**: Data doesn't contain required fields for a format
**Solution**: Use validation tools to check compatibility before training

### Poor Performance
**Problem**: 0% success rate despite no errors
**Solution**: 
- Check if model was trained for enough epochs
- Verify goal conditioning is working correctly
- Use visualization tools to debug behavior

### Legacy Models
**Problem**: Old models don't have proper configuration
**Solution**: Universal loader provides fallback configurations

## 📁 File Structure

```
cs224r/scripts/
├── goal_conditioning_compatibility.py    # Core compatibility layer
├── universal_bc_evaluator.py            # Universal evaluation
├── bc_visualization_tools.py            # Visualization and analysis
├── test_all_formats.py                  # Comprehensive testing
└── train_goal_conditioned_bc.py         # Training script
```

## 🎉 Quick Start

1. **Explain formats**: `python cs224r/scripts/goal_conditioning_compatibility.py`
2. **Validate data**: `python cs224r/scripts/universal_bc_evaluator.py --validate_data data.npz`
3. **Train model**: `python cs224r/scripts/train_goal_conditioned_bc.py --data data.npz --save_dir output --condition_type one_hot`
4. **Evaluate**: `python cs224r/scripts/universal_bc_evaluator.py --model_path output/final_model.pt --save_video`
5. **Visualize**: `python cs224r/scripts/bc_visualization_tools.py --mode evaluate --model_path output/final_model.pt`

This system provides complete compatibility across all goal conditioning formats with automatic error detection and fixing! 