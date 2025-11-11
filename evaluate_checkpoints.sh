#!/bin/bash

# Checkpoint Evaluation Script
# Tests all Lichess DPO checkpoints against puzzles and outputs a comparison table

set -e  # Exit on error

# Configuration
BASE_MODEL="9M"
NUM_PUZZLES=100
CHECKPOINT_DIR="../checkpoints/${BASE_MODEL}_lichess"
RESULTS_DIR="../data/checkpoint_evals"

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
    --num_puzzles=*)
      NUM_PUZZLES="${1#*=}"
      shift
      ;;
    --num_puzzles)
      NUM_PUZZLES="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --base_model MODEL    Base model to evaluate (9M, 136M, 270M) [default: 9M]"
      echo "  --num_puzzles N       Number of puzzles to test [default: 100]"
      echo "  --help                Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --base_model=9M --num_puzzles=200"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "Checkpoint Evaluation Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Puzzles per model: $NUM_PUZZLES"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate searchless_chess

# Set PYTHONPATH
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects

# Navigate to src directory
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Create results directory (after cd to src)
mkdir -p "$RESULTS_DIR"

# Find all available checkpoints
echo "Scanning for checkpoints..."
CHECKPOINTS=($(ls -d $CHECKPOINT_DIR/*/ 2>/dev/null | grep -oP '\d+(?=/)' | sort -n))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "No checkpoints found in $CHECKPOINT_DIR"
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints: ${CHECKPOINTS[@]}"
echo ""

# Evaluate base model
echo "=========================================="
echo "Evaluating Base Model: $BASE_MODEL"
echo "=========================================="
BASE_RESULT_FILE="$RESULTS_DIR/${BASE_MODEL}_baseline.txt"
python puzzles.py --agent=$BASE_MODEL --num_puzzles=$NUM_PUZZLES > "$BASE_RESULT_FILE" 2>&1

# Analyze results with Python helper
BASE_ANALYSIS=$(python analyze_puzzle_results.py "$BASE_RESULT_FILE")

if [ $? -ne 0 ]; then
  echo "Error: Failed to analyze base model results"
  echo "$BASE_ANALYSIS"
  exit 1
fi

BASE_TOTAL=$(echo "$BASE_ANALYSIS" | grep "^TOTAL:" | cut -d: -f2)
BASE_CORRECT=$(echo "$BASE_ANALYSIS" | grep "^CORRECT:" | cut -d: -f2)
BASE_ACCURACY=$(echo "$BASE_ANALYSIS" | grep "^ACCURACY:" | cut -d: -f2)
BASE_PCT=$(awk "BEGIN {printf \"%.1f\", $BASE_ACCURACY*100}")

# Extract rating bucket stats
BASE_BUCKET_1=$(echo "$BASE_ANALYSIS" | grep "^BUCKET:<1500:")
BASE_BUCKET_2=$(echo "$BASE_ANALYSIS" | grep "^BUCKET:1500-2000:")
BASE_BUCKET_3=$(echo "$BASE_ANALYSIS" | grep "^BUCKET:2000-2500:")
BASE_BUCKET_4=$(echo "$BASE_ANALYSIS" | grep "^BUCKET:2500+:")

echo "Base model score: $BASE_CORRECT/$BASE_TOTAL ($BASE_PCT%)"
echo ""

# Arrays to store results
declare -a CHECKPOINT_NAMES
declare -a CHECKPOINT_SCORES
declare -a CHECKPOINT_PCTS
declare -a CHECKPOINT_DIFFS
declare -a CHECKPOINT_BUCKET_1
declare -a CHECKPOINT_BUCKET_2
declare -a CHECKPOINT_BUCKET_3
declare -a CHECKPOINT_BUCKET_4

# Evaluate each checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  # Map checkpoint number to agent name
  if [ "$CHECKPOINT" -lt 150000 ]; then
    AGENT="${BASE_MODEL}_lichess_100k"
    CHECKPOINT_LABEL="100k"
  elif [ "$CHECKPOINT" -lt 250000 ]; then
    AGENT="${BASE_MODEL}_lichess_200k"
    CHECKPOINT_LABEL="200k"
  elif [ "$CHECKPOINT" -lt 350000 ]; then
    AGENT="${BASE_MODEL}_lichess_300k"
    CHECKPOINT_LABEL="300k"
  elif [ "$CHECKPOINT" -lt 450000 ]; then
    AGENT="${BASE_MODEL}_lichess_400k"
    CHECKPOINT_LABEL="400k"
  elif [ "$CHECKPOINT" -lt 550000 ]; then
    AGENT="${BASE_MODEL}_lichess_500k"
    CHECKPOINT_LABEL="500k"
  elif [ "$CHECKPOINT" -lt 650000 ]; then
    AGENT="${BASE_MODEL}_lichess_600k"
    CHECKPOINT_LABEL="600k"
  elif [ "$CHECKPOINT" -lt 750000 ]; then
    AGENT="${BASE_MODEL}_lichess_700k"
    CHECKPOINT_LABEL="700k"
  elif [ "$CHECKPOINT" -lt 850000 ]; then
    AGENT="${BASE_MODEL}_lichess_800k"
    CHECKPOINT_LABEL="800k"
  elif [ "$CHECKPOINT" -lt 950000 ]; then
    AGENT="${BASE_MODEL}_lichess_900k"
    CHECKPOINT_LABEL="900k"
  else
    AGENT="${BASE_MODEL}_lichess_1000k"
    CHECKPOINT_LABEL="1000k"
  fi

  echo "Evaluating checkpoint: $CHECKPOINT ($AGENT)..."
  RESULT_FILE="$RESULTS_DIR/${AGENT}.txt"

  python puzzles.py --agent=$AGENT --num_puzzles=$NUM_PUZZLES > "$RESULT_FILE" 2>&1 || {
    echo "  Skipping (checkpoint not found or error)"
    continue
  }

  # Analyze results with Python helper
  ANALYSIS=$(python analyze_puzzle_results.py "$RESULT_FILE")

  if [ $? -ne 0 ]; then
    echo "  Skipping (analysis failed)"
    continue
  fi

  TOTAL=$(echo "$ANALYSIS" | grep "^TOTAL:" | cut -d: -f2)
  CORRECT=$(echo "$ANALYSIS" | grep "^CORRECT:" | cut -d: -f2)
  ACCURACY=$(echo "$ANALYSIS" | grep "^ACCURACY:" | cut -d: -f2)

  if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ]; then
    echo "  Skipping (no puzzle results found)"
    continue
  fi

  SCORE="${CORRECT}/${TOTAL}"
  PCT=$(awk "BEGIN {printf \"%.1f\", $ACCURACY*100}")
  DIFF=$(awk "BEGIN {printf \"%+.1f\", $ACCURACY*100 - $BASE_PCT}")

  # Extract rating bucket stats
  BUCKET_1=$(echo "$ANALYSIS" | grep "^BUCKET:<1500:")
  BUCKET_2=$(echo "$ANALYSIS" | grep "^BUCKET:1500-2000:")
  BUCKET_3=$(echo "$ANALYSIS" | grep "^BUCKET:2000-2500:")
  BUCKET_4=$(echo "$ANALYSIS" | grep "^BUCKET:2500+:")

  echo "  Score: $SCORE ($PCT%, $DIFF%)"

  CHECKPOINT_NAMES+=("$CHECKPOINT_LABEL")
  CHECKPOINT_SCORES+=("$SCORE")
  CHECKPOINT_PCTS+=("$PCT")
  CHECKPOINT_DIFFS+=("$DIFF")
  CHECKPOINT_BUCKET_1+=("$BUCKET_1")
  CHECKPOINT_BUCKET_2+=("$BUCKET_2")
  CHECKPOINT_BUCKET_3+=("$BUCKET_3")
  CHECKPOINT_BUCKET_4+=("$BUCKET_4")
done

echo ""
echo "=========================================="
echo "Results Summary - Overall Performance"
echo "=========================================="
echo ""

# Print table header
printf "%-15s | %-12s | %-10s | %-10s\n" "Checkpoint" "Score" "Accuracy" "vs Base"
printf "%-15s-+-%-12s-+-%-10s-+-%-10s\n" "---------------" "------------" "----------" "----------"

# Print base model row
printf "%-15s | %-12s | %-10s | %-10s\n" "Base ($BASE_MODEL)" "$BASE_CORRECT/$BASE_TOTAL" "${BASE_PCT}%" "baseline"

# Print checkpoint rows
for i in "${!CHECKPOINT_NAMES[@]}"; do
  printf "%-15s | %-12s | %-10s | %-10s\n" \
    "${CHECKPOINT_NAMES[$i]}" \
    "${CHECKPOINT_SCORES[$i]}" \
    "${CHECKPOINT_PCTS[$i]}%" \
    "${CHECKPOINT_DIFFS[$i]}%"
done

echo ""
echo "=========================================="
echo "Performance by Rating Bucket"
echo "=========================================="
echo ""

# Helper function to extract accuracy from bucket string
get_bucket_accuracy() {
  echo "$1" | cut -d: -f4
}

# Helper function to extract score from bucket string
get_bucket_score() {
  echo "$1" | cut -d: -f3
}

# Print rating bucket table for each bucket
for BUCKET_NAME in "<1500" "1500-2000" "2000-2500" "2500+"; do
  echo "Rating Bucket: $BUCKET_NAME"
  printf "%-15s | %-12s | %-10s\n" "Checkpoint" "Score" "Accuracy"
  printf "%-15s-+-%-12s-+-%-10s\n" "---------------" "------------" "----------"

  # Get base model bucket stats
  if [ "$BUCKET_NAME" = "<1500" ]; then
    BASE_BUCKET_VAR="$BASE_BUCKET_1"
  elif [ "$BUCKET_NAME" = "1500-2000" ]; then
    BASE_BUCKET_VAR="$BASE_BUCKET_2"
  elif [ "$BUCKET_NAME" = "2000-2500" ]; then
    BASE_BUCKET_VAR="$BASE_BUCKET_3"
  else
    BASE_BUCKET_VAR="$BASE_BUCKET_4"
  fi

  BASE_SCORE=$(get_bucket_score "$BASE_BUCKET_VAR")
  BASE_ACC=$(get_bucket_accuracy "$BASE_BUCKET_VAR")
  BASE_ACC_PCT=$(awk "BEGIN {printf \"%.1f\", $BASE_ACC*100}")
  printf "%-15s | %-12s | %-10s\n" "Base ($BASE_MODEL)" "$BASE_SCORE" "${BASE_ACC_PCT}%"

  # Print checkpoint rows for this bucket
  for i in "${!CHECKPOINT_NAMES[@]}"; do
    if [ "$BUCKET_NAME" = "<1500" ]; then
      BUCKET_VAR="${CHECKPOINT_BUCKET_1[$i]}"
    elif [ "$BUCKET_NAME" = "1500-2000" ]; then
      BUCKET_VAR="${CHECKPOINT_BUCKET_2[$i]}"
    elif [ "$BUCKET_NAME" = "2000-2500" ]; then
      BUCKET_VAR="${CHECKPOINT_BUCKET_3[$i]}"
    else
      BUCKET_VAR="${CHECKPOINT_BUCKET_4[$i]}"
    fi

    SCORE=$(get_bucket_score "$BUCKET_VAR")
    ACC=$(get_bucket_accuracy "$BUCKET_VAR")
    ACC_PCT=$(awk "BEGIN {printf \"%.1f\", $ACC*100}")
    printf "%-15s | %-12s | %-10s\n" "${CHECKPOINT_NAMES[$i]}" "$SCORE" "${ACC_PCT}%"
  done
  echo ""
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Best checkpoint:"
# Find best performing checkpoint
BEST_IDX=0
BEST_PCT=0
for i in "${!CHECKPOINT_PCTS[@]}"; do
  CURR_PCT=${CHECKPOINT_PCTS[$i]}
  if (( $(awk "BEGIN {print ($CURR_PCT > $BEST_PCT)}") )); then
    BEST_PCT=$CURR_PCT
    BEST_IDX=$i
  fi
done

if [ ${#CHECKPOINT_NAMES[@]} -gt 0 ]; then
  echo "  ${CHECKPOINT_NAMES[$BEST_IDX]} with ${CHECKPOINT_PCTS[$BEST_IDX]}% (${CHECKPOINT_DIFFS[$BEST_IDX]}% vs base)"
fi
echo ""
