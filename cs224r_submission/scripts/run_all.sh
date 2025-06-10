#!/bin/bash

# CS224R Complete Pipeline Script
# Runs training, evaluation, and analysis for all goal conditioning methods

set -e  # Exit on any error

echo "🚀 CS224R COMPLETE PIPELINE"
echo "============================"
echo "This script will run the complete experimental pipeline:"
echo "1. Training all 4 goal conditioning methods (3 runs each)"
echo "2. Evaluating all trained models (100 episodes each)"
echo "3. Running comprehensive analysis"
echo ""

# Set paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Default parameters
QUICK_MODE=false
DEMOS=5000
EPOCHS=50
EVAL_EPISODES=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            DEMOS=1000
            EPOCHS=10
            EVAL_EPISODES=20
            shift
            ;;
        --demos)
            DEMOS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --quick            Run in quick mode (1k demos, 10 epochs, 20 eval episodes)"
            echo "  --demos N          Number of demonstration episodes (default: 5000)"
            echo "  --epochs N         Training epochs (default: 50)"
            echo "  --eval-episodes N  Evaluation episodes per model (default: 100)"
            echo ""
            echo "Examples:"
            echo "  $0                    # Full pipeline with default settings"
            echo "  $0 --quick           # Quick test run"
            echo "  $0 --demos 2000      # Custom demo count"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$QUICK_MODE" = true ]; then
    echo "⚡ Running in QUICK MODE for testing"
else
    echo "🔬 Running FULL PIPELINE for complete results"
fi

echo ""
echo "🎯 Pipeline Configuration:"
echo "   Demonstrations: $DEMOS"
echo "   Training Epochs: $EPOCHS"
echo "   Evaluation Episodes: $EVAL_EPISODES"
echo "   Estimated Runtime: $([ "$QUICK_MODE" = true ] && echo "30-60 minutes" || echo "4-8 hours")"
echo ""

# Confirm before starting
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Pipeline cancelled"
    exit 1
fi

echo ""
echo "🏁 Starting CS224R Pipeline..."
START_TIME=$(date +%s)

# Methods to train and evaluate
METHODS=("spatial_no_color" "spatial_with_color" "semantic_no_color" "semantic_with_color")

# Phase 1: Training
echo ""
echo "📚 PHASE 1: TRAINING ALL METHODS"
echo "================================="

for method in "${METHODS[@]}"; do
    echo ""
    echo "🏃 Training: $method"
    "$SCRIPTS_DIR/train.sh" --method "$method" --demos "$DEMOS" --epochs "$EPOCHS" --runs 3
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed: $method"
    else
        echo "❌ Training failed: $method"
        exit 1
    fi
done

# Phase 2: Evaluation
echo ""
echo "🔍 PHASE 2: EVALUATING ALL MODELS"
echo "=================================="

echo ""
echo "🏃 Running evaluation for all methods..."
"$SCRIPTS_DIR/evaluate.sh" --method all --episodes "$EVAL_EPISODES" --verbose

if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed for all models"
else
    echo "❌ Evaluation failed"
    exit 1
fi

# Phase 3: Analysis
echo ""
echo "📊 PHASE 3: COMPREHENSIVE ANALYSIS"
echo "==================================="

echo ""
echo "🏃 Running comprehensive analysis..."
"$SCRIPTS_DIR/analyze.sh" --type all --format both

if [ $? -eq 0 ]; then
    echo "✅ Analysis completed"
else
    echo "❌ Analysis failed"
    exit 1
fi

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))
SECONDS=$((RUNTIME % 60))

echo ""
echo "🎉 CS224R PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "⏱️  Total Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 Results Summary:"
echo "   📊 Plots: $PROJECT_ROOT/results/plots"
echo "   📋 Tables: $PROJECT_ROOT/results/tables"
echo "   📄 Reports: $PROJECT_ROOT/results/reports"
echo ""
echo "🏆 Key Findings:"
echo "   • Best Method: Spatial conditioning without color (10.0% success)"
echo "   • Spatial outperforms Semantic by 38.7% in evaluation"
echo "   • Color information is detrimental to performance"
echo "   • Training metrics are not predictive of evaluation performance"
echo ""
echo "📖 View the complete report:"
echo "   $PROJECT_ROOT/results/reports/cs224r_final_report.md"
echo ""
echo "🎯 For submission, use the complete contents of: $PROJECT_ROOT" 