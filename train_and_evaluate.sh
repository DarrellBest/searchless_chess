#!/bin/bash

# Self-Play Training and Evaluation Pipeline
# This script trains a model using self-play and compares performance before/after

set -e  # Exit on error

# Default parameters
BASE_MODEL="9M"
NUM_ITERATIONS=10
GAMES_PER_ITERATION=1000
BATCH_SIZE=32
LEARNING_RATE=0.00001
GRADIENT_STEPS=50
STOCKFISH_TIME=0.1
STOCKFISH_DEPTH=20
EVAL_THRESHOLD=0.3
BETA=0.1
NUM_PUZZLES=100
SKIP_BASELINE=false
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
    --games_per_iteration=*)
      GAMES_PER_ITERATION="${1#*=}"
      shift
      ;;
    --games_per_iteration)
      GAMES_PER_ITERATION="$2"
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
    --stockfish_time=*)
      STOCKFISH_TIME="${1#*=}"
      shift
      ;;
    --stockfish_time)
      STOCKFISH_TIME="$2"
      shift 2
      ;;
    --stockfish_depth=*)
      STOCKFISH_DEPTH="${1#*=}"
      shift
      ;;
    --stockfish_depth)
      STOCKFISH_DEPTH="$2"
      shift 2
      ;;
    --eval_threshold=*)
      EVAL_THRESHOLD="${1#*=}"
      shift
      ;;
    --eval_threshold)
      EVAL_THRESHOLD="$2"
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
    --skip_baseline)
      SKIP_BASELINE=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --base_model MODEL         Base model (9M, 136M, 270M) [default: 9M]"
      echo "  --num_iterations N         Training iterations [default: 10]"
      echo "  --games_per_iteration N    Self-play games per iteration [default: 1000]"
      echo "  --batch_size N             Training batch size [default: 32]"
      echo "  --learning_rate LR         Learning rate [default: 0.00001]"
      echo "  --gradient_steps N         Gradient steps per iteration [default: 50]"
      echo "  --stockfish_time TIME      Stockfish time per position in seconds [default: 0.1]"
      echo "  --stockfish_depth N        Stockfish analysis depth [default: 20]"
      echo "  --eval_threshold PAWNS     Min eval difference for preference pairs [default: 0.3]"
      echo "  --beta BETA                DPO KL penalty coefficient [default: 0.1]"
      echo "  --skip_baseline            Skip baseline evaluation (if already done)"
      echo "  --resume                   Resume training from latest checkpoint"
      echo "  --help                     Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --base_model=9M --num_iterations=5"
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
echo "Self-Play Training and Evaluation Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Training Iterations: $NUM_ITERATIONS"
echo "  Games per Iteration: $GAMES_PER_ITERATION"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Gradient Steps: $GRADIENT_STEPS"
echo "  Stockfish Time: $STOCKFISH_TIME seconds"
echo "  Stockfish Depth: $STOCKFISH_DEPTH"
echo "  Eval Threshold: $EVAL_THRESHOLD pawns"
echo "  DPO Beta: $BETA"
echo "  Evaluation Puzzles (baseline/final): $NUM_PUZZLES"
echo "  Resume from checkpoint: $RESUME"
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

# Step 2: Self-play training
echo ""
echo "=========================================="
echo "Step 2: Self-Play Training"
echo "=========================================="
echo "Training ${BASE_MODEL} with self-play..."
echo ""

# Build training command
TRAIN_CMD="python selfplay_train.py \
  --base_model=$BASE_MODEL \
  --num_iterations=$NUM_ITERATIONS \
  --games_per_iteration=$GAMES_PER_ITERATION \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --gradient_steps_per_iteration=$GRADIENT_STEPS \
  --stockfish_time=$STOCKFISH_TIME \
  --stockfish_depth=$STOCKFISH_DEPTH \
  --eval_threshold=$EVAL_THRESHOLD \
  --beta=$BETA"

# Add resume flag if set
if [ "$RESUME" = true ]; then
  TRAIN_CMD="$TRAIN_CMD --resume"
fi

# Execute training
eval $TRAIN_CMD

echo ""
echo "Training complete!"

# Update constants.py to use the latest checkpoint
echo "Updating checkpoint iteration in constants.py..."
CHECKPOINT_DIR="../checkpoints/${BASE_MODEL}_selfplay"
LATEST_ITERATION=$(ls -d $CHECKPOINT_DIR/*/ 2>/dev/null | grep -o '[0-9]\+' | sort -n | tail -1)

if [ -n "$LATEST_ITERATION" ]; then
  echo "Latest checkpoint: iteration $LATEST_ITERATION"
  sed -i "s/'${BASE_MODEL}_selfplay': lambda: _build_selfplay_engine('${BASE_MODEL}', iteration=[0-9]\+)/'${BASE_MODEL}_selfplay': lambda: _build_selfplay_engine('${BASE_MODEL}', iteration=${LATEST_ITERATION})/g" engines/constants.py
  echo "Updated constants.py to use iteration $LATEST_ITERATION"
else
  echo "Warning: Could not find latest checkpoint"
fi

# Step 3: Post-training evaluation
echo ""
echo "=========================================="
echo "Step 3: Post-Training Evaluation"
echo "=========================================="
echo "Evaluating selfplay model: ${BASE_MODEL}_selfplay"
echo ""

python puzzles.py --agent=${BASE_MODEL}_selfplay --num_puzzles=$NUM_PUZZLES > ../data/${BASE_MODEL}_selfplay_results.txt

echo "Post-training evaluation complete!"
echo "Results saved to: ../data/${BASE_MODEL}_selfplay_results.txt"

# Step 4: Comparison
echo ""
echo "=========================================="
echo "Step 4: Performance Comparison"
echo "=========================================="

python evaluate_selfplay.py \
  --base_model=$BASE_MODEL \
  --iteration=$NUM_ITERATIONS \
  --num_puzzles=$NUM_PUZZLES

# Step 5: Elo Comparison (optional)
echo ""
echo "=========================================="
echo "Step 5: Elo Comparison (Optional)"
echo "=========================================="
echo ""
echo "Run head-to-head games to calculate Elo difference:"
echo "  python compare_engines.py --engine1=$BASE_MODEL --engine2=${BASE_MODEL}_selfplay --num_games=50"
echo ""
read -p "Run Elo comparison now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
  python compare_engines.py \
    --engine1=$BASE_MODEL \
    --engine2=${BASE_MODEL}_selfplay \
    --num_games=50
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: ../checkpoints/${BASE_MODEL}_selfplay/"
echo "Evaluation results saved to: ../data/"
echo ""
echo "To evaluate the trained model on puzzles:"
echo "  python puzzles.py --agent=${BASE_MODEL}_selfplay --num_puzzles=100"
echo ""
echo "To calculate Elo difference:"
echo "  python compare_engines.py --engine1=$BASE_MODEL --engine2=${BASE_MODEL}_selfplay --num_games=100"
echo ""
echo "Note: Puzzle evaluation is not performed during training for speed."
echo "      Training metrics include mistake rate and average eval margin instead."
