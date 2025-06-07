#!/bin/bash

DATA_DIR="/home/ubuntu/project/SurRoL/surrol/data/two_blocks"
BASE_FILE="data_PegTransfer-v0_random_1000_2025-06-01_10-17-43"
SAVE_BASE="/home/ubuntu/project/behavior-cloning/hw1/cs224r/experiments/1k_demos"

# List of conditioning methods to train
METHODS=("onehot" "targetblock" "targetblocktargetpeg" "onehottargetpeg" "fourtuple")

for method in "${METHODS[@]}"; do
    DATA_FILE="${DATA_DIR}/${BASE_FILE}_${method}.npz"
    SAVE_DIR="${SAVE_BASE}/${method}"
    
    if [ -f "$DATA_FILE" ]; then
        echo "🚀 Training $method conditioning..."
        python universal_bc_trainer.py \
            --data_path "$DATA_FILE" \
            --save_dir "$SAVE_DIR" \
            --epochs 100 \
            --eval_interval 5 \
            --eval_episodes 10
        echo "✅ Completed $method"
        echo ""
    else
        echo "❌ Data file not found: $DATA_FILE"
    fi
done

echo "🎉 All training completed!"
