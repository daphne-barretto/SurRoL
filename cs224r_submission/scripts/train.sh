#!/bin/bash

# CS224R Training Script
# Trains goal-conditioned behavior cloning models for surgical robotics

set -e  # Exit on any error

echo "🚀 CS224R GOAL-CONDITIONED BEHAVIOR CLONING TRAINING"
echo "===================================================="

# Check if conda environment exists
if ! conda env list | grep -q "gcrl"; then
    echo "❌ Error: 'gcrl' conda environment not found"
    echo "Please create the environment first with required dependencies"
    exit 1
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gcrl

# Set paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
DATA_DIR="$PROJECT_ROOT/data"

# Set PYTHONPATH to include local SurRoL (critical for environment registration)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Default parameters
METHOD="spatial_no_color"  # Options: spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color
DEMOS=5000
EPOCHS=50
RUNS=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --demos)
            DEMOS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --method    Goal conditioning method (spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color)"
            echo "  --demos     Number of demonstration episodes (default: 5000)"
            echo "  --epochs    Training epochs (default: 50)"
            echo "  --runs      Number of training runs (default: 3)"
            echo ""
            echo "Examples:"
            echo "  $0 --method spatial_no_color --demos 5000 --runs 3"
            echo "  $0 --method semantic_with_color --epochs 100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Map method to experiment configuration, data file, and BC system arguments
case $METHOD in
    "spatial_no_color")
        DATA_FILE="$DATA_DIR/demo/data_PegTransferTwoBlocksNoColorTargetBlockPeg-v0_random_5000_2025-06-06_03-32-20.npz"
        BASE_TYPE="goal_conditioned_demos_with_all_blocks_2_blocks"
        CONDITIONING_TYPE="target_block_and_peg"
        EXPERIMENT_PREFIX="bc_results_2block_targetblocktargetpeg"
        ;;
    "spatial_with_color")
        DATA_FILE="$DATA_DIR/demo/data_PegTransferTwoBlocksTargetBlockTargetPeg-v0_5000_2025-06-08_03-51-06.npz"
        BASE_TYPE="goal_conditioned_demos_with_all_blocks_colored_2_blocks"
        CONDITIONING_TYPE="target_block_and_peg"
        EXPERIMENT_PREFIX="bc_results_2block_targetblocktargetpeg_color"
        ;;
    "semantic_no_color")
        DATA_FILE="$DATA_DIR/demo/data_PegTransferTwoBlocksNoColorOneHotTargetPeg-v0_random_5000_2025-06-06_03-31-40.npz"
        BASE_TYPE="goal_conditioned_demos_with_all_blocks_2_blocks"
        CONDITIONING_TYPE="one_hot_and_target_peg"
        EXPERIMENT_PREFIX="bc_results_2block_onehottargetpeg"
        ;;
    "semantic_with_color")
        DATA_FILE="$DATA_DIR/demo/data_PegTransferTwoBlocksOneHotTargetPeg-v0_random_5000_2025-06-06_11-52-13.npz"
        BASE_TYPE="goal_conditioned_demos_with_all_blocks_colored_2_blocks"
        CONDITIONING_TYPE="one_hot_and_target_peg"
        EXPERIMENT_PREFIX="bc_results_2block_onehottargetpeg_color"
        ;;
    *)
        echo "❌ Error: Unknown method '$METHOD'"
        echo "Valid methods: spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color"
        exit 1
        ;;
esac

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    echo "Please ensure the demo data is available in the data/demo directory"
    exit 1
fi

echo "🎯 Training Configuration:"
echo "   Method: $METHOD"
echo "   Data File: $(basename $DATA_FILE)"
echo "   Base Type: $BASE_TYPE"
echo "   Conditioning: $CONDITIONING_TYPE"
echo "   Demos: $DEMOS"
echo "   Epochs: $EPOCHS"
echo "   Runs: $RUNS"
echo ""

# Create output directory
OUTPUT_DIR="$DATA_DIR/experiments"
mkdir -p "$OUTPUT_DIR"

# Run training for specified number of runs
for run in $(seq 1 $RUNS); do
    # Convert demos to proper k notation (e.g., 5000 -> 5k)
    if [ $DEMOS -ge 1000 ]; then
        DEMOS_K=$((DEMOS / 1000))k
    else
        DEMOS_K=$DEMOS
    fi
    
    if [ $run -eq 1 ]; then
        EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${DEMOS_K}"
    else
        EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${DEMOS_K}_${run}"
    fi
    
    # Create experiment directory
    SAVE_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"
    mkdir -p "$SAVE_DIR"
    
    echo "🏃 Starting training run $run/$RUNS: $EXPERIMENT_NAME"
    
    # Run training with the working universal BC system
    cd "$PROJECT_ROOT"
    python "$SRC_DIR/training/universal_bc_system.py" \
        --mode train \
        --data_path "$DATA_FILE" \
        --save_dir "$SAVE_DIR" \
        --base_type "$BASE_TYPE" \
        --conditioning_type "$CONDITIONING_TYPE" \
        --epochs "$EPOCHS" \
        --batch_size 64 \
        --eval_interval 10 \
        --eval_episodes 20
    
    echo "✅ Completed training run $run/$RUNS"
done

echo ""
echo "🎉 Training completed successfully!"
echo "📁 Results saved in: $OUTPUT_DIR"
echo "🔍 To evaluate models, run: ./scripts/evaluate.sh --method $METHOD" 