#!/bin/bash

# Compare Checkpoints Script
# Tests all Lichess DPO checkpoints against base model via head-to-head games

set -e  # Exit on error

# Configuration
BASE_MODEL="9M"
NUM_GAMES=50
CHECKPOINT_DIR="../checkpoints/${BASE_MODEL}_lichess"
RESULTS_DIR="../data/checkpoint_comparisons"

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
    --num_games=*)
      NUM_GAMES="${1#*=}"
      shift
      ;;
    --num_games)
      NUM_GAMES="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --base_model MODEL    Base model to compare against (9M, 136M, 270M) [default: 9M]"
      echo "  --num_games N         Number of games per matchup [default: 50]"
      echo "  --help                Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --base_model=9M --num_games=100"
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
echo "Checkpoint Comparison Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Games per matchup: $NUM_GAMES"
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

# Create results directory
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

# Arrays to store results
declare -a CHECKPOINT_NAMES
declare -a WIN_RATES
declare -a DRAWS
declare -a LOSSES
declare -a ELO_DIFFS

# Compare each checkpoint against base model
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

  echo "=========================================="
  echo "Comparing: $BASE_MODEL vs $AGENT"
  echo "=========================================="

  RESULT_FILE="$RESULTS_DIR/${BASE_MODEL}_vs_${AGENT}.txt"

  python compare_engines.py \
    --engine1=$BASE_MODEL \
    --engine2=$AGENT \
    --num_games=$NUM_GAMES > "$RESULT_FILE" 2>&1 || {
    echo "  Skipping (checkpoint not found or error)"
    continue
  }

  # Parse results - looking for lines like:
  # "9M: X wins"
  # "9M_lichess_200k: Y wins"
  # "Draws: Z"
  # "Approximate Elo difference: +/-N"

  # Extract win/loss/draw stats
  ENGINE1_WINS=$(grep "^${BASE_MODEL}: " "$RESULT_FILE" | grep -oP "\d+(?= wins)")
  ENGINE2_WINS=$(grep "^${AGENT}: " "$RESULT_FILE" | grep -oP "\d+(?= wins)")
  DRAW_COUNT=$(grep "^Draws: " "$RESULT_FILE" | grep -oP "\d+")

  if [ -z "$ENGINE1_WINS" ] || [ -z "$ENGINE2_WINS" ] || [ -z "$DRAW_COUNT" ]; then
    echo "  Skipping (could not parse results)"
    continue
  fi

  # Calculate win rate for engine2 (checkpoint)
  TOTAL_DECISIVE=$((ENGINE1_WINS + ENGINE2_WINS))
  if [ "$TOTAL_DECISIVE" -gt 0 ]; then
    WIN_RATE=$(awk "BEGIN {printf \"%.1f\", ($ENGINE2_WINS/($ENGINE1_WINS+$ENGINE2_WINS+$DRAW_COUNT))*100}")
  else
    WIN_RATE="0.0"
  fi

  # Extract Elo difference
  ELO_LINE=$(grep -E "Approximate Elo difference:" "$RESULT_FILE" | tail -1)
  if [ -n "$ELO_LINE" ]; then
    ELO_DIFF=$(echo "$ELO_LINE" | grep -oP "difference: \K[+-]?\d+")
  else
    # Check for special cases
    if grep -q "won all games" "$RESULT_FILE"; then
      if grep -q "${AGENT} won all games" "$RESULT_FILE"; then
        ELO_DIFF=">+400"
      else
        ELO_DIFF="<-400"
      fi
    else
      ELO_DIFF="N/A"
    fi
  fi

  echo "  Results: Engine2 ($AGENT) - Wins: $ENGINE2_WINS, Draws: $DRAW_COUNT, Losses: $ENGINE1_WINS"
  echo "  Win rate: $WIN_RATE% | Elo diff: $ELO_DIFF"
  echo ""

  CHECKPOINT_NAMES+=("$CHECKPOINT_LABEL")
  WIN_RATES+=("$WIN_RATE")
  DRAWS+=("$DRAW_COUNT")
  LOSSES+=("$ENGINE1_WINS")
  ELO_DIFFS+=("$ELO_DIFF")
done

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""

# Print table header
printf "%-15s | %-12s | %-10s | %-10s | %-12s\n" "Checkpoint" "W/D/L" "Win Rate" "Elo Diff" "vs Base"
printf "%-15s-+-%-12s-+-%-10s-+-%-10s-+-%-12s\n" "---------------" "------------" "----------" "----------" "------------"

# Print checkpoint rows
for i in "${!CHECKPOINT_NAMES[@]}"; do
  # Calculate wins for checkpoint
  WINS=$((NUM_GAMES - ${LOSSES[$i]} - ${DRAWS[$i]}))
  WDL="${WINS}/${DRAWS[$i]}/${LOSSES[$i]}"

  printf "%-15s | %-12s | %-10s | %-10s | %-12s\n" \
    "${CHECKPOINT_NAMES[$i]}" \
    "$WDL" \
    "${WIN_RATES[$i]}%" \
    "${ELO_DIFFS[$i]}" \
    "$BASE_MODEL"
done

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

# Find best checkpoint by Elo difference
echo "Best checkpoint by Elo:"
BEST_IDX=0
BEST_ELO=-9999
for i in "${!ELO_DIFFS[@]}"; do
  CURR_ELO_STR=${ELO_DIFFS[$i]}
  # Extract numeric value (handle >+400 and <-400)
  CURR_ELO=$(echo "$CURR_ELO_STR" | grep -oP "[+-]?\d+" | head -1)

  if [ -n "$CURR_ELO" ] && [ "$CURR_ELO" -gt "$BEST_ELO" ] 2>/dev/null; then
    BEST_ELO=$CURR_ELO
    BEST_IDX=$i
  fi
done

if [ ${#CHECKPOINT_NAMES[@]} -gt 0 ] && [ "$BEST_ELO" != "-9999" ]; then
  echo "  ${CHECKPOINT_NAMES[$BEST_IDX]} with Elo difference: ${ELO_DIFFS[$BEST_IDX]}"
fi
echo ""
