#!/bin/bash

echo "🔍 Running BC Diagnosis on PegTransfer"
echo "====================================="

# Paths based on your output
DATA_PATH="/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz"
MODEL_PATH="/home/ubuntu/project/behavior-cloning/hw1/cs224r/experiments/10k_demos/baseline_test_20250604_232519_none/final_model_none.pt"

echo "📁 Data: $DATA_PATH"
echo "🧠 Model: $MODEL_PATH"

python debug_bc_pegtransfer.py \
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH"

echo ""
echo "🎯 Diagnosis completed!"