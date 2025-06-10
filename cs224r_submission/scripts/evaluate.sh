#!/bin/bash

# CS224R Evaluation Script
# Evaluates trained goal-conditioned behavior cloning models

set -e  # Exit on any error

echo "🔍 CS224R MODEL EVALUATION"
echo "=========================="

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
RESULTS_DIR="$PROJECT_ROOT/results"

# Set PYTHONPATH to include local SurRoL (critical for environment registration)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Default parameters
METHOD="all"  # Options: spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color, all
EPISODES=100
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --method     Goal conditioning method to evaluate (spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color, all)"
            echo "  --episodes   Number of evaluation episodes per model (default: 100)"
            echo "  --verbose    Enable verbose output"
            echo ""
            echo "Examples:"
            echo "  $0 --method spatial_no_color --episodes 100"
            echo "  $0 --method all --episodes 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to evaluate a specific method
evaluate_method() {
    local method_name=$1
    local experiment_pattern=$2
    
    echo "🎯 Evaluating $method_name models..."
    
    # Find all experiment directories for this method
    local experiments=($(find "$DATA_DIR/experiments" -maxdepth 1 -type d -name "$experiment_pattern*"))
    
    if [ ${#experiments[@]} -eq 0 ]; then
        echo "⚠️  No experiments found for pattern: $experiment_pattern"
        return
    fi
    
    for exp_dir in "${experiments[@]}"; do
        local exp_name=$(basename "$exp_dir")
        echo "🏃 Evaluating: $exp_name"
        
        # Check if model exists
        if [ ! -f "$exp_dir/best_model.pt" ]; then
            echo "⚠️  No trained model found in $exp_dir"
            continue
        fi
        
        # Run evaluation using the working universal BC system
        cd "$PROJECT_ROOT"
        python "$SRC_DIR/training/universal_bc_system.py" \
            --mode evaluate \
            --model_dir "$exp_dir" \
            --num_episodes "$EPISODES" \
            --max_steps 100
        
        echo "✅ Completed evaluation: $exp_name"
    done
}

# Create output directory
mkdir -p "$RESULTS_DIR/evaluation"

echo "🎯 Evaluation Configuration:"
echo "   Method: $METHOD"
echo "   Episodes per model: $EPISODES"
echo "   Verbose: $VERBOSE"
echo ""

# Evaluate based on method selection
case $METHOD in
    "spatial_no_color")
        evaluate_method "Spatial without Color" "bc_results_2block_targetblocktargetpeg"
        ;;
    "spatial_with_color")
        evaluate_method "Spatial with Color" "bc_results_2block_targetblocktargetpeg_color"
        ;;
    "semantic_no_color")
        evaluate_method "Semantic without Color" "bc_results_2block_onehottargetpeg"
        ;;
    "semantic_with_color")
        evaluate_method "Semantic with Color" "bc_results_2block_onehottargetpeg_color"
        ;;
    "all")
        echo "📊 Evaluating all methods..."
        evaluate_method "Spatial without Color" "bc_results_2block_targetblocktargetpeg"
        evaluate_method "Spatial with Color" "bc_results_2block_targetblocktargetpeg_color"
        evaluate_method "Semantic without Color" "bc_results_2block_onehottargetpeg"
        evaluate_method "Semantic with Color" "bc_results_2block_onehottargetpeg_color"
        ;;
    *)
        echo "❌ Error: Unknown method '$METHOD'"
        echo "Valid methods: spatial_no_color, spatial_with_color, semantic_no_color, semantic_with_color, all"
        exit 1
        ;;
esac

echo ""
echo "🎉 Evaluation completed successfully!"
echo "📁 Results saved in model directories"
echo "📊 To run analysis, use: ./scripts/analyze.sh" 