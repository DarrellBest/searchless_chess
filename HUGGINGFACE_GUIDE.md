# HuggingFace Model Packaging Guide

This guide explains how to package and share your searchless chess models on HuggingFace Hub.

## Prerequisites

Install required packages:

```bash
pip install huggingface_hub
```

Create a HuggingFace account at https://huggingface.co and get your access token from https://huggingface.co/settings/tokens

Login to HuggingFace:

```bash
huggingface-cli login
```

## Step 1: Convert Checkpoint to HuggingFace Format

Convert your trained checkpoint to HuggingFace-compatible format:

```bash
cd src

# Convert the 9M selfplay model at iteration 4
python convert_to_hf.py \
  --checkpoint_path=../checkpoints/9M_selfplay/4 \
  --model_name=9M \
  --output_dir=../hf_models/9M_selfplay \
  --use_ema=True
```

This creates a directory with:
- `config.json` - Model configuration
- `params.npz` - Model weights as NumPy arrays
- `tree_structure.pkl` - JAX pytree structure
- `model_info.json` - Framework metadata

## Step 2: Create Model Card (README.md)

Create a README.md file in the model directory to document your model:

```bash
cat > ../hf_models/9M_selfplay/README.md << 'EOF'
---
language: en
license: apache-2.0
tags:
- chess
- reinforcement-learning
- jax
- haiku
library_name: jax
---

# Searchless Chess 9M (Self-Play Trained)

This is a 9 million parameter transformer-based chess engine trained using self-play with Stockfish-based rewards.

## Model Description

- **Architecture**: Transformer with 8 layers, 256 embedding dim, 8 attention heads
- **Training**: Self-play with Stockfish evaluation rewards
- **Framework**: JAX/Haiku
- **Parameters**: 9M
- **Base Model**: DeepMind's Searchless Chess

## Usage

```python
from searchless_chess.src import hf_model

# Load model
model = hf_model.SearchlessChessModel.from_pretrained("path/to/model")

# Predict move from FEN position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
result = model.predict(fen)

print(f"Best move: {result['best_move']}")
print(f"Q-value: {result['q_value']}")
```

## Training Details

- Training iterations: X
- Self-play games: Y
- Stockfish time per position: 0.01s
- Batch size: 32
- Learning rate: 1e-4

## Performance

- Puzzle accuracy: XX%
- Elo estimate: +XX relative to base model

## Citation

Based on the Searchless Chess work by DeepMind.

## License

Apache 2.0
EOF
```

## Step 3: Upload to HuggingFace Hub

Upload the model to HuggingFace Hub:

```bash
python upload_to_hf.py \
  --model_dir=../hf_models/9M_selfplay \
  --repo_name=searchless-chess-9M-selfplay \
  --username=YOUR_HF_USERNAME \
  --private=False
```

Set `--private=True` if you want to keep the model private.

## Step 4: Download and Use

Anyone can now download and use your model:

```bash
python download_from_hf.py \
  --repo_id=YOUR_USERNAME/searchless-chess-9M-selfplay \
  --output_dir=./downloaded_model
```

Or programmatically:

```python
from huggingface_hub import snapshot_download
from searchless_chess.src import hf_model

# Download model
model_path = snapshot_download(
    repo_id="YOUR_USERNAME/searchless-chess-9M-selfplay",
    local_dir="./my_model"
)

# Load and use
model = hf_model.SearchlessChessModel.from_pretrained("./my_model")
result = model.predict("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(result['best_move'])
```

## Alternative: Direct Python API

You can also use the HuggingFace Hub Python API directly:

```python
from huggingface_hub import HfApi, create_repo, upload_folder

# Create repository
repo_id = "username/searchless-chess-9M-selfplay"
create_repo(repo_id=repo_id, exist_ok=True)

# Upload model
upload_folder(
    folder_path="../hf_models/9M_selfplay",
    repo_id=repo_id,
    commit_message="Upload searchless chess model"
)
```

## Model Versioning

To upload different iterations:

```bash
# Convert iteration 10
python convert_to_hf.py \
  --checkpoint_path=../checkpoints/9M_selfplay/10 \
  --model_name=9M \
  --output_dir=../hf_models/9M_selfplay_iter10

# Upload with different tag
python upload_to_hf.py \
  --model_dir=../hf_models/9M_selfplay_iter10 \
  --repo_name=searchless-chess-9M-selfplay-iter10 \
  --username=YOUR_USERNAME
```

Or use git tags on the same repository:

```bash
cd ../hf_models/9M_selfplay
git tag v1.0  # Tag current version
# Update model files
git tag v2.0  # Tag new version
```

## Model Card Best Practices

Include in your README.md:
- Model description and architecture
- Training details (iterations, games, hyperparameters)
- Performance metrics (puzzle accuracy, Elo ratings)
- Usage examples
- Limitations and biases
- Citation information
- License

## Sharing with Specific Users

For private collaboration:

1. Create private repository: `--private=True`
2. Add collaborators on HuggingFace web interface
3. Share repository ID: `username/repo-name`

## Large Models

For 136M and 270M models, the upload may take longer:

```bash
# 136M model (~500MB)
python convert_to_hf.py \
  --checkpoint_path=../checkpoints/136M_selfplay/10 \
  --model_name=136M \
  --output_dir=../hf_models/136M_selfplay

# 270M model (~1GB)
python convert_to_hf.py \
  --checkpoint_path=../checkpoints/270M_selfplay/10 \
  --model_name=270M \
  --output_dir=../hf_models/270M_selfplay
```

HuggingFace supports large files via Git LFS automatically.

## Troubleshooting

### Authentication Error
```bash
# Re-login
huggingface-cli login
```

### Upload Failed
```bash
# Check network connection
# Try with smaller batch size or manual upload via web interface
```

### Model Loading Error
Ensure all dependencies are installed:
```bash
pip install jax jaxlib dm-haiku orbax-checkpoint numpy
```

## Resources

- HuggingFace Hub: https://huggingface.co
- Model Hub Docs: https://huggingface.co/docs/hub/models
- Transformers Docs: https://huggingface.co/docs/transformers
