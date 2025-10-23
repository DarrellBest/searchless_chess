#!/bin/bash

# Complete HuggingFace Workflow Example
# This script demonstrates the full pipeline from checkpoint to HuggingFace Hub

set -e

echo "======================================"
echo "HuggingFace Model Packaging Workflow"
echo "======================================"
echo ""

# Configuration
CHECKPOINT_PATH="../checkpoints/9M_selfplay/4"
MODEL_NAME="9M"
OUTPUT_DIR="../hf_models/9M_selfplay"
REPO_NAME="searchless-chess-9M-selfplay"
HF_USERNAME=""  # Set your HuggingFace username here

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate searchless_chess
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects

cd src

# Step 1: Convert checkpoint
echo ""
echo "======================================"
echo "Step 1: Converting Checkpoint to HF Format"
echo "======================================"
python convert_to_hf.py \
  --checkpoint_path=$CHECKPOINT_PATH \
  --model_name=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --use_ema=True

# Step 2: Create README
echo ""
echo "======================================"
echo "Step 2: Creating Model Card (README.md)"
echo "======================================"

cat > $OUTPUT_DIR/README.md << 'EOF'
---
language: en
license: apache-2.0
tags:
- chess
- reinforcement-learning
- jax
- haiku
- self-play
library_name: jax
---

# Searchless Chess 9M (Self-Play Trained)

This is a 9 million parameter transformer-based chess engine trained using self-play with Stockfish-based rewards.

## Model Description

- **Architecture**: Transformer with 8 layers, 256 embedding dim, 8 attention heads
- **Training Method**: Self-play with Stockfish evaluation rewards
- **Framework**: JAX/Haiku
- **Parameters**: ~9 million
- **Base Model**: DeepMind's Searchless Chess
- **Training Iterations**: 4
- **Self-play Games**: ~80

## Quick Start

```python
from searchless_chess.src import hf_model
from huggingface_hub import snapshot_download

# Download model
model_path = snapshot_download(
    repo_id="YOUR_USERNAME/searchless-chess-9M-selfplay",
    local_dir="./chess_model"
)

# Load model
model = hf_model.SearchlessChessModel.from_pretrained("./chess_model")

# Predict move from starting position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
result = model.predict(fen)

print(f"Best move: {result['best_move']}")
print(f"Q-value: {result['q_value']:.4f}")
```

## Installation

Required dependencies:
```bash
pip install jax jaxlib dm-haiku orbax-checkpoint numpy huggingface-hub
```

## Training Details

- **Base Model**: 9M parameter action-value model
- **Training Algorithm**: Q-learning with Stockfish rewards
- **Reward Function**: Stockfish centipawn evaluation scaled by 0.01
- **Self-play Games**: 20 games per iteration
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Gradient Steps**: 100 per iteration
- **Optimizer**: Adam with gradient clipping (max norm 1.0)
- **EMA Decay**: 0.999

## Architecture

- **Input**: 77-token FEN representation
- **Embedding**: 256 dimensions
- **Layers**: 8 transformer blocks
- **Attention Heads**: 8 per layer
- **Output**: 128-bucket Q-value distribution
- **Positional Encoding**: Learned
- **Activation**: GELU in feed-forward layers

## Performance

Training improves over base model through self-play reinforcement learning.

## Limitations

- Model trained for limited iterations (early checkpoint)
- Performance may vary on tactical vs positional positions
- No explicit opening book or endgame tablebase
- Evaluation based on Q-values, not full game tree search

## Citation

Based on the Searchless Chess work by DeepMind Technologies Limited.

## License

Apache 2.0

---

For more information and training code, see the [original repository](https://github.com/google-deepmind/searchless_chess).
EOF

echo "README.md created at $OUTPUT_DIR/README.md"

# Step 3: Upload to HuggingFace (optional, commented out)
echo ""
echo "======================================"
echo "Step 3: Upload to HuggingFace Hub"
echo "======================================"
echo ""

if [ -z "$HF_USERNAME" ]; then
  echo "HF_USERNAME not set. Skipping upload."
  echo ""
  echo "To upload manually:"
  echo "  1. Set HF_USERNAME variable in this script"
  echo "  2. Run: huggingface-cli login"
  echo "  3. Run:"
  echo "     python upload_to_hf.py \\"
  echo "       --model_dir=$OUTPUT_DIR \\"
  echo "       --repo_name=$REPO_NAME \\"
  echo "       --username=YOUR_USERNAME"
else
  echo "Uploading to HuggingFace Hub..."
  python upload_to_hf.py \
    --model_dir=$OUTPUT_DIR \
    --repo_name=$REPO_NAME \
    --username=$HF_USERNAME \
    --private=False

  echo ""
  echo "Model uploaded successfully!"
  echo "View at: https://huggingface.co/$HF_USERNAME/$REPO_NAME"
fi

echo ""
echo "======================================"
echo "Workflow Complete!"
echo "======================================"
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the README.md and update as needed"
echo "  2. Login to HuggingFace: huggingface-cli login"
echo "  3. Upload: python upload_to_hf.py --model_dir=$OUTPUT_DIR --repo_name=$REPO_NAME --username=YOUR_USERNAME"
echo ""
echo "To download and use:"
echo "  python download_from_hf.py --repo_id=YOUR_USERNAME/$REPO_NAME"
echo ""
