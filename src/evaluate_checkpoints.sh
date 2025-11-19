#!/bin/bash
# Evaluate multiple checkpoints on puzzles to track training progress

set -e

# Configuration
NUM_PUZZLES=100
RESULTS_DIR="data/checkpoint_evals"

# Create results directory
mkdir -p $RESULTS_DIR

# Setup environment
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

echo "=========================================="
echo "Checkpoint Evaluation on Puzzles"
echo "=========================================="
echo "Number of puzzles: $NUM_PUZZLES"
echo "Results directory: $RESULTS_DIR"
echo ""

# Define checkpoints to evaluate
CHECKPOINTS=(
    "9M"                # Baseline
    "9M_stream_1002"    # Stream training 1002 pairs
    "9M_stream_2039"    # Stream training 2039 pairs
    "9M_stream_3102"    # Stream training 3102 pairs
    "9M_stream_4259"    # Stream training 4259 pairs
    "9M_stream_5360"    # Stream training 5360 pairs
)

# Evaluate each checkpoint
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: $checkpoint"
    echo "=========================================="

    # Run evaluation
    output_file="../$RESULTS_DIR/${checkpoint}_puzzles_${NUM_PUZZLES}.txt"

    if [ -f "$output_file" ]; then
        echo "Results already exist at: $output_file"
        echo "Skipping (delete file to re-evaluate)"
    else
        echo "Running puzzle evaluation..."
        python puzzles.py --agent=$checkpoint --num_puzzles=$NUM_PUZZLES > $output_file 2>&1

        echo "Results saved to: $output_file"

        # Extract and display summary
        if grep -q "Puzzle solving accuracy" $output_file; then
            echo ""
            grep "Puzzle solving accuracy" $output_file
        fi
    fi
done

# Generate summary report
echo ""
echo "=========================================="
echo "Summary Report"
echo "=========================================="
echo ""

printf "%-20s %15s %15s\n" "Checkpoint" "Accuracy" "Pairs Trained"
printf "%-20s %15s %15s\n" "----------" "--------" "-------------"

for checkpoint in "${CHECKPOINTS[@]}"; do
    output_file="../$RESULTS_DIR/${checkpoint}_puzzles_${NUM_PUZZLES}.txt"

    if [ -f "$output_file" ]; then
        # Extract accuracy
        if grep -q "Puzzle solving accuracy" $output_file; then
            accuracy=$(grep "Puzzle solving accuracy" $output_file | grep -oP '\d+\.\d+%')

            # Extract pairs trained from checkpoint name
            if [[ $checkpoint == "9M" ]]; then
                pairs="0 (baseline)"
            elif [[ $checkpoint =~ _([0-9]+)k ]]; then
                k_value="${BASH_REMATCH[1]}"
                pairs="${k_value}k"
            else
                pairs="N/A"
            fi

            printf "%-20s %15s %15s\n" "$checkpoint" "$accuracy" "$pairs"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR/"
echo ""
