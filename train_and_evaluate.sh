#!/bin/bash

# Lichess DPO Training and Evaluation Pipeline
# This script trains a model using DPO with Lichess evaluations and compares performance before/after

set -e  # Exit on error

# Default parameters
BASE_MODEL="9M"
NUM_ITERATIONS=-1  # -1 = train until early stopping or data exhaustion
PAIRS_PER_ITERATION=10000
BATCH_SIZE=32
PAIRS_PER_UPDATE=1000  # Number of pairs to accumulate before training update
TRAIN_MINI_BATCH=128  # Mini-batch size for gradient accumulation (prevents OOM)
LEARNING_RATE=0.000002
GRADIENT_STEPS=-1
BETA=10.0  # Chess Q-values (~0.3) need much higher beta than LM log probs
TEMPERATURE=1.0
MAX_GRAD_NORM=1.0
UPDATE_REF_EVERY=1
MAX_KL_DIVERGENCE=0.5
ANCHOR_WEIGHT=1.0  # Q-value anchoring to prevent drift (0.5-2.0 recommended)
DYNAMIC_PAIRS=true  # Enable ONLY dynamic pairs mode (trains on intruders only)
DYNAMIC_PAIRS_MAX_PER_POS=-1  # Max dynamic pairs to generate per position (-1 = unlimited, use all intruders)
CHECKPOINT_EVERY=10000  # Checkpoints saved after each training update (pairs_per_update)
MAX_PAIRS=-1  # -1 = unlimited, otherwise max number of pairs to generate
LICHESS_DB_PATH="../data/lichess_db_eval.jsonl.zst"
NUM_PUZZLES=100
SKIP_BASELINE=false
SKIP_HYPERTEST=true
RESUME=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base_model=*)
      BASE_MODEL="${1#*=}"
      shift
      ;;
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --num_iterations=*)
      NUM_ITERATIONS="${1#*=}"
      shift
      ;;
    --num_iterations)
      NUM_ITERATIONS="$2"
      shift 2
      ;;
    --pairs_per_iteration=*)
      PAIRS_PER_ITERATION="${1#*=}"
      shift
      ;;
    --pairs_per_iteration)
      PAIRS_PER_ITERATION="$2"
      shift 2
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --pairs_per_update=*)
      PAIRS_PER_UPDATE="${1#*=}"
      shift
      ;;
    --pairs_per_update)
      PAIRS_PER_UPDATE="$2"
      shift 2
      ;;
    --learning_rate=*)
      LEARNING_RATE="${1#*=}"
      shift
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --gradient_steps=*)
      GRADIENT_STEPS="${1#*=}"
      shift
      ;;
    --gradient_steps)
      GRADIENT_STEPS="$2"
      shift 2
      ;;
    --beta=*)
      BETA="${1#*=}"
      shift
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --temperature=*)
      TEMPERATURE="${1#*=}"
      shift
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max_grad_norm=*)
      MAX_GRAD_NORM="${1#*=}"
      shift
      ;;
    --max_grad_norm)
      MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --update_ref_every=*)
      UPDATE_REF_EVERY="${1#*=}"
      shift
      ;;
    --update_ref_every)
      UPDATE_REF_EVERY="$2"
      shift 2
      ;;
    --max_kl_divergence=*)
      MAX_KL_DIVERGENCE="${1#*=}"
      shift
      ;;
    --max_kl_divergence)
      MAX_KL_DIVERGENCE="$2"
      shift 2
      ;;
    --lichess_db_path=*)
      LICHESS_DB_PATH="${1#*=}"
      shift
      ;;
    --lichess_db_path)
      LICHESS_DB_PATH="$2"
      shift 2
      ;;
    --checkpoint_every=*)
      CHECKPOINT_EVERY="${1#*=}"
      shift
      ;;
    --checkpoint_every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --max_pairs=*)
      MAX_PAIRS="${1#*=}"
      shift
      ;;
    --max_pairs)
      MAX_PAIRS="$2"
      shift 2
      ;;
    --anchor_weight=*)
      ANCHOR_WEIGHT="${1#*=}"
      shift
      ;;
    --anchor_weight)
      ANCHOR_WEIGHT="$2"
      shift 2
      ;;
    --dynamic_pairs)
      DYNAMIC_PAIRS=true
      shift
      ;;
    --no_dynamic_pairs)
      DYNAMIC_PAIRS=false
      shift
      ;;
    --dynamic_pairs_max_per_pos=*)
      DYNAMIC_PAIRS_MAX_PER_POS="${1#*=}"
      shift
      ;;
    --dynamic_pairs_max_per_pos)
      DYNAMIC_PAIRS_MAX_PER_POS="$2"
      shift 2
      ;;
    --skip_baseline)
      SKIP_BASELINE=true
      shift
      ;;
    --skip_hypertest)
      SKIP_HYPERTEST=true
      shift
      ;;
    --run_hypertest)
      SKIP_HYPERTEST=false
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Core Options:"
      echo "  --base_model MODEL              Base model (9M, 136M, 270M) [default: 9M]"
      echo "  --batch_size N                  Position batch size for pair generation [default: 32]"
      echo "  --pairs_per_update N            Pairs to accumulate before training update [default: 1000]"
      echo "  --learning_rate LR              Learning rate [default: 0.000002]"
      echo "  --beta BETA                     DPO KL penalty coefficient [default: 10.0]"
      echo "  --temperature TEMP              Policy extraction temperature [default: 1.0]"
      echo "  --max_grad_norm NORM            Maximum gradient norm [default: 1.0]"
      echo "  --max_kl_divergence KL          Early stop if KL > threshold [default: 0.5]"
      echo ""
      echo "Q-Value Anchoring (prevents drift):"
      echo "  --anchor_weight WEIGHT          Anchoring strength (0.0=off, 0.5-2.0 recommended) [default: 1.0]"
      echo ""
      echo "Dynamic Pair Generation (ONLY mode - trains on intruders only):"
      echo "  --dynamic_pairs                 Enable ONLY dynamic pairs mode [default: enabled]"
      echo "  --no_dynamic_pairs              Disable (use static pairs - legacy mode)"
      echo "  --dynamic_pairs_max_per_pos N   Max pairs per position (-1=unlimited) [default: -1]"
      echo ""
      echo "Training Control:"
      echo "  --checkpoint_every N            Save checkpoint every N pairs [default: 10000]"
      echo "  --max_pairs N                   Max pairs to generate (-1=unlimited) [default: -1]"
      echo "  --lichess_db_path PATH          Path to Lichess database"
      echo ""
      echo "Evaluation Options:"
      echo "  --skip_baseline                 Skip baseline evaluation (if already done)"
      echo "  --run_hypertest                 Run hyperparameter testing before training"
      echo ""
      echo "Other:"
      echo "  --help                          Show this help message"
      echo ""
      echo "Note: Training streams positions from Lichess database."
      echo "      Pairs are generated dynamically on-the-fly and used once."
      echo ""
      echo "Examples:"
      echo "  # Standard training with anchoring and dynamic pairs (recommended)"
      echo "  $0 --base_model=9M --max_pairs=1000000"
      echo ""
      echo "  # Conservative anchoring, no dynamic pairs"
      echo "  $0 --anchor_weight=0.5 --no_dynamic_pairs --max_pairs=500000"
      echo ""
      echo "  # Aggressive dynamic pair generation"
      echo "  $0 --dynamic_pairs_max_per_pos=5"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Setup environment
echo "=========================================="
echo "Lichess DPO Training and Evaluation Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
if [ "$MAX_PAIRS" -eq -1 ]; then
  echo "  Training: Unlimited pairs (trains until KL threshold or data exhaustion)"
else
  echo "  Training: Up to $MAX_PAIRS DPO pairs"
fi
echo ""
echo "Training Parameters:"
echo "  Batch Size (positions): $BATCH_SIZE"
echo "  Pairs Per Update: $PAIRS_PER_UPDATE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  DPO Beta: $BETA"
echo "  Temperature: $TEMPERATURE"
echo "  Max Gradient Norm: $MAX_GRAD_NORM"
echo "  Max KL Divergence: $MAX_KL_DIVERGENCE"
echo ""
echo "Q-Value Anchoring:"
echo "  Anchor Weight: $ANCHOR_WEIGHT (prevents Q-value drift)"
echo ""
echo "Dynamic Pair Generation:"
echo "  Enabled: $DYNAMIC_PAIRS"
if [ "$DYNAMIC_PAIRS" = true ]; then
  echo "  Mode: ONLY dynamic pairs (trains on intruders only)"
  if [ "$DYNAMIC_PAIRS_MAX_PER_POS" -le 0 ]; then
    echo "  Pairs per Position: Unlimited (all intruders)"
  else
    echo "  Max Pairs per Position: $DYNAMIC_PAIRS_MAX_PER_POS"
  fi
else
  echo "  Mode: Static pairs (legacy mode)"
fi
echo ""
echo "Checkpointing:"
echo "  Checkpoint Every: $CHECKPOINT_EVERY pairs"
echo "  Lichess DB Path: $LICHESS_DB_PATH"
echo ""
echo "Evaluation:"
echo "  Puzzles (baseline/final): $NUM_PUZZLES"
echo ""
echo "Note: Streaming training - positions used once, pairs generated on-the-fly"
echo "      Q-value anchoring + dynamic pairs prevent intruder moves from getting higher Q-values"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate searchless_chess

# Set PYTHONPATH
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects

# Navigate to src directory
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Step 1: Baseline evaluation (optional)
if [ "$SKIP_BASELINE" = false ]; then
  echo ""
  echo "=========================================="
  echo "Step 1: Baseline Evaluation"
  echo "=========================================="
  echo "Evaluating base model: $BASE_MODEL"
  echo ""

  python puzzles.py --agent=$BASE_MODEL --num_puzzles=$NUM_PUZZLES > ../data/${BASE_MODEL}_baseline_results.txt

  echo "Baseline evaluation complete!"
  echo "Results saved to: ../data/${BASE_MODEL}_baseline_results.txt"
else
  echo ""
  echo "Skipping baseline evaluation (--skip_baseline flag set)"
fi

# Step 2: Hyperparameter testing (optional)
if [ "$SKIP_HYPERTEST" = false ] && [ "$RESUME" = false ]; then
  echo ""
  echo "=========================================="
  echo "Step 2: Hyperparameter Testing"
  echo "=========================================="
  echo "Running quick tests to validate hyperparameters..."
  echo ""

  python test_hyperparams.py \
    --base_model=$BASE_MODEL \
    --num_pairs=500 \
    --num_batches=100 \
    --learning_rates=5e-7,1e-6,5e-6 \
    --betas=$BETA \
    --temperatures=$TEMPERATURE \
    --lichess_db_path=$LICHESS_DB_PATH

  echo ""
  echo "Hyperparameter testing complete!"
  echo "Review the results above and adjust parameters if needed."
  echo ""
  read -p "Continue with training? (y/n) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. Adjust hyperparameters and try again."
    exit 0
  fi
else
  echo ""
  echo "Skipping hyperparameter testing (--skip_hypertest flag set or resuming)"
fi

# Step 3: DPO training with Lichess data
echo ""
echo "=========================================="
echo "Step 3: DPO Training with Lichess Data"
echo "=========================================="
echo "Training ${BASE_MODEL} with Lichess evaluations..."
echo ""

# Build training command with Q-value anchoring and streaming dynamic pairs
TRAIN_CMD="python lichess_stream_train.py \
  --base_model=$BASE_MODEL \
  --batch_size=$BATCH_SIZE \
  --pairs_per_update=$PAIRS_PER_UPDATE \
  --train_mini_batch=$TRAIN_MINI_BATCH \
  --learning_rate=$LEARNING_RATE \
  --beta=$BETA \
  --temperature=$TEMPERATURE \
  --max_grad_norm=$MAX_GRAD_NORM \
  --max_kl_divergence=$MAX_KL_DIVERGENCE \
  --anchor_weight=$ANCHOR_WEIGHT \
  --dynamic_pairs_max_per_pos=$DYNAMIC_PAIRS_MAX_PER_POS \
  --checkpoint_every=$CHECKPOINT_EVERY \
  --lichess_db_path=$LICHESS_DB_PATH"

# Add max_positions if max_pairs is set (approximate: 1000 pairs ≈ 1000 positions if we get ~1 pair/position)
if [ "$MAX_PAIRS" -ne -1 ]; then
  # Estimate positions needed: max_pairs / avg_pairs_per_position
  # We generate up to 3 pairs per position, but many positions have 0 intruders
  # So roughly 1000 pairs ≈ 3000-5000 positions
  MAX_POSITIONS=$((MAX_PAIRS * 4))
  TRAIN_CMD="$TRAIN_CMD --max_positions=$MAX_POSITIONS"
fi

# Note: The streaming training script processes positions from Lichess database
# It generates dynamic pairs on-the-fly by finding intruder moves
# Each position is used once then discarded (pairs are ephemeral)
# Checkpoints are saved every CHECKPOINT_EVERY pairs and named by total pairs trained

# Execute training
eval $TRAIN_CMD

echo ""
echo "Training complete!"

# Find latest checkpoint
echo "Finding latest checkpoint..."
CHECKPOINT_DIR="../checkpoints/${BASE_MODEL}_lichess"

# Find the checkpoint with highest number (directories that are all digits)
LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR/*/ 2>/dev/null | grep -oP '\d+(?=/)' | sort -n | tail -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
  CHECKPOINT_NAME="${LATEST_CHECKPOINT}"
  echo "Latest checkpoint: $CHECKPOINT_NAME (${LATEST_CHECKPOINT} pairs trained)"
else
  echo "Warning: Could not find any checkpoints in $CHECKPOINT_DIR"
  CHECKPOINT_NAME=""
fi

# Step 4: Post-training evaluation
echo ""
echo "=========================================="
echo "Step 4: Post-Training Evaluation"
echo "=========================================="

# Determine which checkpoint to evaluate
if [ -n "$CHECKPOINT_NAME" ]; then
  # Map checkpoint number to the closest 100k checkpoint name
  CHECKPOINT_NUM=$CHECKPOINT_NAME
  if [ "$CHECKPOINT_NUM" -lt 150000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_100k"
  elif [ "$CHECKPOINT_NUM" -lt 250000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_200k"
  elif [ "$CHECKPOINT_NUM" -lt 350000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_300k"
  elif [ "$CHECKPOINT_NUM" -lt 450000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_400k"
  elif [ "$CHECKPOINT_NUM" -lt 550000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_500k"
  elif [ "$CHECKPOINT_NUM" -lt 650000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_600k"
  elif [ "$CHECKPOINT_NUM" -lt 750000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_700k"
  elif [ "$CHECKPOINT_NUM" -lt 850000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_800k"
  elif [ "$CHECKPOINT_NUM" -lt 950000 ]; then
    EVAL_AGENT="${BASE_MODEL}_lichess_900k"
  else
    EVAL_AGENT="${BASE_MODEL}_lichess_1000k"
  fi
  echo "Evaluating Lichess DPO model: $EVAL_AGENT (checkpoint: $CHECKPOINT_NAME)"
else
  EVAL_AGENT="${BASE_MODEL}_lichess_best"
  echo "Evaluating Lichess DPO model: $EVAL_AGENT"
fi
echo ""

python puzzles.py --agent=$EVAL_AGENT --num_puzzles=$NUM_PUZZLES > ../data/${BASE_MODEL}_lichess_results.txt

echo "Post-training evaluation complete!"
echo "Results saved to: ../data/${BASE_MODEL}_lichess_results.txt"

# Step 5: Comparison
echo ""
echo "=========================================="
echo "Step 5: Performance Comparison"
echo "=========================================="

if [ -n "$CHECKPOINT_NAME" ]; then
  echo "Evaluating checkpoint: $CHECKPOINT_NAME"
  # Note: evaluate_lichess.py may need to be updated to handle the new checkpoint naming
fi

# Step 6: Elo Comparison (optional)
echo ""
echo "=========================================="
echo "Step 6: Elo Comparison (Optional)"
echo "=========================================="
echo ""
echo "Run head-to-head games to calculate Elo difference:"
echo "  python compare_engines.py --engine1=$BASE_MODEL --engine2=$EVAL_AGENT --num_games=50"
echo ""
read -p "Run Elo comparison now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
  python compare_engines.py \
    --engine1=$BASE_MODEL \
    --engine2=$EVAL_AGENT \
    --num_games=50
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: ../checkpoints/${BASE_MODEL}_lichess/"
echo "Evaluation results saved to: ../data/"
echo ""
echo "To evaluate the trained model on puzzles:"
echo "  python puzzles.py --agent=${BASE_MODEL}_lichess_600k --num_puzzles=100"
echo ""
echo "To calculate Elo difference:"
echo "  python compare_engines.py --engine1=$BASE_MODEL --engine2=${BASE_MODEL}_lichess_600k --num_games=100"
echo ""
echo "Available checkpoint agents: ${BASE_MODEL}_lichess_100k, ${BASE_MODEL}_lichess_200k, ..., ${BASE_MODEL}_lichess_1000k"
echo "Checkpoints are named by number of pairs trained (e.g., 100000, 200000, 600000)."
echo "Training metrics include DPO loss, preference accuracy, and KL divergence."
