# Data Usage Guide for BC Goal Conditioning Experiments

## Data Pipeline Overview

Your data follows this pipeline:
1. **Raw Data** (from GDrive) → 2. **Post-processing** (`data_postprocessing.py`) → 3. **BC Training**

## Available Raw Data Files

### 1. `goal_conditioned_demos_with_all_blocks.npz` (4 blocks)
- **Content**: Robot state + positions of all 4 blocks
- **Use Case**: 4-block PegTransfer with full spatial information
- **Observation**: Robot (7D) + All block positions (~12-15D) = ~19-22D total
- **Post-process for**: All conditioning methods

### 2. `goal_conditioned_demos_with_all_blocks_2_blocks.npz` (2 blocks)  
- **Content**: Robot state + positions of both blocks
- **Use Case**: 2-block PegTransfer (simpler task)
- **Observation**: Robot (7D) + Both block positions (~6-8D) = ~13-15D total
- **Post-process for**: All conditioning methods

### 3. `goal_conditioned_demos_with_all_blocks_colored.npz` (4 blocks + colors)
- **Content**: Robot state + all block positions + block colors
- **Use Case**: 4-block PegTransfer with color information for richer conditioning
- **Observation**: Robot (7D) + All block positions (~12-15D) + Colors (~4-8D) = ~23-30D total
- **Post-process for**: All conditioning methods + color-based conditioning

### 4. `goal_conditioned_demos_with_all_blocks_colored_2_blocks.npz` (2 blocks + colors)
- **Content**: Robot state + both block positions + block colors
- **Use Case**: 2-block PegTransfer with color information
- **Observation**: Robot (7D) + Both block positions (~6-8D) + Colors (~2-4D) = ~15-19D total
- **Post-process for**: All conditioning methods + color-based conditioning

### 5. `goal_conditioned_demos_with_only_target_block.npz` (4 blocks, minimal)
- **Content**: Robot state + only target block position
- **Use Case**: Minimal information baseline for 4-block task
- **Observation**: Robot (7D) + Target block only (~3D) = ~10D total
- **Post-process for**: NO post-processing needed (already minimal)

## Post-Processed Data Files

After running `data_postprocessing.py`, you get files with conditioning features added:

### Conditioning Types Available:

#### `*_onehot.npz`
- **Added**: One-hot block encoding
- **Dimension increase**: +4D (4-block) or +2D (2-block)
- **Use for**: Block identity conditioning ("pick the red block")

#### `*_targetblock.npz`
- **Added**: Target block position (3D coordinates)
- **Dimension increase**: +3D
- **Use for**: Spatial conditioning ("pick block at position X")

#### `*_targetblocktargetpeg.npz`
- **Added**: Target block position + target peg position
- **Dimension increase**: +6D (3D + 3D)
- **Use for**: Full spatial conditioning ("pick block at X, place at Y")

#### `*_onehottargetpeg.npz`
- **Added**: One-hot encoding + target peg position
- **Dimension increase**: +4D/+2D + 3D = +7D/+5D
- **Use for**: Hybrid conditioning ("pick red block, place at Y")

#### `*_fourtuple.npz`
- **Added**: RGBA color encoding
- **Dimension increase**: +4D
- **Use for**: Color-based conditioning experiments

## Which Files to Use for Which Tasks

### For 2-Block PegTransfer (Recommended for Development)

**Start with**: `goal_conditioned_demos_with_all_blocks_2_blocks.npz`

**Experiments to run**:
1. **Baseline**: Use raw file directly → `updated_universal_bc_trainer.py`
2. **One-hot**: Post-process → `*_2_blocks_onehot.npz` → BC training
3. **Spatial**: Post-process → `*_2_blocks_targetblock.npz` → BC training
4. **Full spatial**: Post-process → `*_2_blocks_targetblocktargetpeg.npz` → BC training
5. **Hybrid**: Post-process → `*_2_blocks_onehottargetpeg.npz` → BC training

### For 4-Block PegTransfer (Full Complexity)

**Start with**: `goal_conditioned_demos_with_all_blocks.npz`

**Advanced experiments**: `goal_conditioned_demos_with_all_blocks_colored.npz`

### For Minimal Baseline

**Use**: `goal_conditioned_demos_with_only_target_block.npz` (NO post-processing)

## Step-by-Step Workflow

### Step 1: Choose Base Data
```bash
# For 2-block development
BASE_DATA="goal_conditioned_demos_with_all_blocks_2_blocks.npz"

# For 4-block full task
BASE_DATA="goal_conditioned_demos_with_all_blocks.npz"

# For colored blocks
BASE_DATA="goal_conditioned_demos_with_all_blocks_colored_2_blocks.npz"
```

### Step 2: Post-Process for Different Conditioning Methods
```bash
# One-hot encoding
python data_postprocessing.py $BASE_DATA \
    --output_path ${BASE_DATA%%.npz}_onehot.npz \
    --condition_type one_hot

# Target block position
python data_postprocessing.py $BASE_DATA \
    --output_path ${BASE_DATA%%.npz}_targetblock.npz \
    --condition_type target_block

# Full spatial conditioning
python data_postprocessing.py $BASE_DATA \
    --output_path ${BASE_DATA%%.npz}_targetblocktargetpeg.npz \
    --condition_type target_block_target_peg

# Hybrid conditioning
python data_postprocessing.py $BASE_DATA \
    --output_path ${BASE_DATA%%.npz}_onehottargetpeg.npz \
    --condition_type one_hot_target_peg
```

### Step 3: Train BC Models
```bash
# Train on each post-processed file
for CONDITIONED_FILE in *_onehot.npz *_targetblock.npz *_targetblocktargetpeg.npz *_onehottargetpeg.npz; do
    SAVE_DIR="experiments/bc_$(basename $CONDITIONED_FILE .npz)"
    
    python updated_universal_bc_trainer.py \
        --data_path $CONDITIONED_FILE \
        --save_dir $SAVE_DIR \
        --epochs 100
done
```

### Step 4: Evaluate and Compare
```bash
# The updated trainer automatically detects data type and conditioning
# No need to specify conditioning method manually
```

## Recommended Experiment Sequence

### Phase 1: Development (2-block task)
1. **Baseline**: `goal_conditioned_demos_with_all_blocks_2_blocks.npz` (raw)
2. **One-hot**: `goal_conditioned_demos_with_all_blocks_2_blocks_onehot.npz`
3. **Spatial**: `goal_conditioned_demos_with_all_blocks_2_blocks_targetblock.npz`
4. **Hybrid**: `goal_conditioned_demos_with_all_blocks_2_blocks_onehottargetpeg.npz`

### Phase 2: Full Task (4-block)
1. **All methods** on `goal_conditioned_demos_with_all_blocks.npz`
2. **Color experiments** on `goal_conditioned_demos_with_all_blocks_colored.npz`

### Phase 3: Ablation Studies
1. **Minimal baseline**: `goal_conditioned_demos_with_only_target_block.npz`
2. **Information comparison**: Compare observation richness effects

## Compatibility Matrix

| Data File | BC Trainer | Post-Processing | Use Case |
|-----------|------------|----------------|----------|
| `*_2_blocks.npz` | ✅ | ✅ All methods | Development |
| `*_4_blocks.npz` | ✅ | ✅ All methods | Full task |
| `*_colored.npz` | ✅ | ✅ + Color methods | Advanced |
| `*_only_target.npz` | ✅ | ❌ None needed | Minimal baseline |
| `*_onehot.npz` | ✅ | ❌ Already processed | Direct training |
| `*_targetblock.npz` | ✅ | ❌ Already processed | Direct training |

## Expected Performance Hierarchy

### 2-Block Task (Easier)
1. **Target block spatial**: 60-80% success
2. **Hybrid**: 50-70% success  
3. **One-hot**: 30-50% success
4. **Baseline**: 20-40% success

### 4-Block Task (Harder)
1. **Full spatial**: 40-60% success
2. **Target block**: 30-50% success
3. **Hybrid**: 25-45% success
4. **One-hot**: 15-30% success
5. **Minimal**: 10-20% success

## Error Prevention

The updated system prevents dimension mismatches by:

1. **Automatic Detection**: Analyzes file to determine data type and conditioning
2. **Flexible Architecture**: Adapts model architecture to match data
3. **Consistent Processing**: Uses same observation processing for training and evaluation
4. **Configuration Saving**: Saves exact configuration for reproducible evaluation

## Quick Start Commands

### Test the System (2-block baseline)
```bash
# Download 2-block data
DATA_PATH="goal_conditioned_demos_with_all_blocks_2_blocks.npz"

# Test baseline (no conditioning)
python updated_universal_bc_trainer.py \
    --data_path $DATA_PATH \
    --save_dir experiments/test_baseline \
    --epochs 20

# Test with one-hot conditioning
python data_postprocessing.py $DATA_PATH \
    --output_path goal_conditioned_demos_with_all_blocks_2_blocks_onehot.npz \
    --condition_type one_hot

python updated_universal_bc_trainer.py \
    --data_path goal_conditioned_demos_with_all_blocks_2_blocks_onehot.npz \
    --save_dir experiments/test_onehot \
    --epochs 20
```

### Full Comparison (All Methods)
```bash
# Create batch experiment script
python create_full_comparison_experiment.py \
    --base_data goal_conditioned_demos_with_all_blocks_2_blocks.npz \
    --output_dir experiments/full_comparison \
    --epochs 100
```

This approach ensures compatibility with your actual data pipeline and prevents the dimension mismatch issues that caused the original 0% success rate.