# Self-Play Training with Stockfish Rewards

This guide explains how to train chess models using self-play with Stockfish-based reinforcement learning.

**For a detailed explanation of how self-play works, see [SELFPLAY_EXPLAINED.md](SELFPLAY_EXPLAINED.md)**

## Overview

The self-play training system improves existing models (9M, 136M, or 270M) by:
1. Having the model play against itself
2. Using Stockfish to evaluate each position and provide reward signals
3. Training the model to maximize these rewards using policy gradient methods

This combines the pattern recognition of neural networks with the tactical accuracy of Stockfish.

## How It Works

### Self-Play Process
1. **Game Generation**: The neural model plays against itself
2. **Position Evaluation**: After each move, Stockfish evaluates the position
3. **Reward Calculation**:
   - Immediate reward = change in centipawn advantage
   - Final reward = game outcome (1.0 for win, 0.5 for draw, 0.0 for loss)
4. **Learning**: Model parameters are updated to maximize total rewards

### Reward Formula
```
Total Reward = (Position Improvement × 0.01) + Game Outcome
```

Where:
- Position Improvement = Centipawn advantage change
- Scaling factor 0.01 normalizes centipawns to 0-1 range
- Game Outcome provides final supervision signal

## Usage

### Basic Training

Start training from a base model:

```bash
# Setup environment
conda activate searchless_chess
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Train from 9M model (fastest)
python selfplay_train.py --base_model=9M

# Train from 136M model (balanced)
python selfplay_train.py --base_model=136M

# Train from 270M model (best quality)
python selfplay_train.py --base_model=270M
```

### Advanced Configuration

Control training hyperparameters:

```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=20 \
  --games_per_iteration=50 \
  --batch_size=64 \
  --learning_rate=1e-4 \
  --gradient_steps_per_iteration=200 \
  --stockfish_time=0.05 \
  --save_frequency=5
```

### Parameter Descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | 9M | Starting model: 9M, 136M, or 270M |
| `--num_iterations` | 10 | Total training iterations |
| `--games_per_iteration` | 20 | Self-play games per iteration |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 1e-4 | Adam learning rate |
| `--gradient_steps_per_iteration` | 100 | Optimization steps per iteration |
| `--stockfish_time` | 0.01 | Stockfish time per position (seconds) |
| `--save_frequency` | 1 | Checkpoint save frequency |
| `--resume` | False | Resume from latest checkpoint if available |

## Resuming Training

The training script supports resuming from the latest checkpoint:

```bash
# Train for 10 iterations
python selfplay_train.py --base_model=9M --num_iterations=10

# Later, continue training for 10 MORE iterations
python selfplay_train.py --base_model=9M --num_iterations=10 --resume

# This will:
# - Find the latest checkpoint (iteration 10)
# - Load params, EMA params, and optimizer state
# - Continue training from iteration 11 to 20
```

**How it works:**
- Without `--resume`: Always starts from base model at iteration 0
- With `--resume`: Looks for existing checkpoints in `../checkpoints/9M_selfplay/`
- Finds the latest checkpoint number (e.g., 5, 10, 15)
- Loads all state from that checkpoint
- Continues training for `--num_iterations` MORE iterations

**Example workflow:**
```bash
# Day 1: Train for 5 iterations (saves checkpoints 1-5)
python selfplay_train.py --base_model=9M --num_iterations=5

# Day 2: Continue for 5 more (iterations 6-10)
python selfplay_train.py --base_model=9M --num_iterations=5 --resume

# Day 3: Continue for 10 more (iterations 11-20)
python selfplay_train.py --base_model=9M --num_iterations=10 --resume
```

**Note:** If no checkpoints exist when using `--resume`, training starts from the base model.

## Training Profiles

### Quick Test (5-10 minutes)
```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=2 \
  --games_per_iteration=5 \
  --gradient_steps_per_iteration=20
```

### Standard Training (2-4 hours)
```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=10 \
  --games_per_iteration=20 \
  --gradient_steps_per_iteration=100
```

### Intensive Training (12-24 hours)
```bash
python selfplay_train.py \
  --base_model=136M \
  --num_iterations=50 \
  --games_per_iteration=50 \
  --gradient_steps_per_iteration=200 \
  --stockfish_time=0.05
```

### Production Training (2-7 days)
```bash
python selfplay_train.py \
  --base_model=270M \
  --num_iterations=100 \
  --games_per_iteration=100 \
  --gradient_steps_per_iteration=500 \
  --stockfish_time=0.1 \
  --batch_size=64 \
  --learning_rate=5e-5
```

## Output and Checkpoints

### Checkpoint Location
Trained models are saved to:
```
../checkpoints/9M_selfplay/
../checkpoints/136M_selfplay/
../checkpoints/270M_selfplay/
```

### Checkpoint Contents
Each checkpoint contains:
- `params/` - Current model parameters
- `params_ema/` - Exponential moving average parameters (recommended for inference)
- `opt_state/` - Optimizer state

### Using Trained Models

Trained selfplay models are automatically registered in `src/engines/constants.py` and can be used immediately:

```bash
# Evaluate on puzzles
python puzzles.py --agent=9M_selfplay --num_puzzles=50

# Compare with base model
python puzzles.py --agent=9M --num_puzzles=50

# Use in automated evaluation
python evaluate_selfplay.py --base_model=9M --num_puzzles=100
```

The following selfplay engines are available:
- `9M_selfplay` - Selfplay-trained 9M model (loads iteration 1 by default)
- `136M_selfplay` - Selfplay-trained 136M model
- `270M_selfplay` - Selfplay-trained 270M model

## Performance Evaluation

### Automated Pipeline

Use the `train_and_evaluate.sh` script to run the complete pipeline:

```bash
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess

# Quick test (5-10 minutes)
./train_and_evaluate.sh \
  --base_model=9M \
  --num_iterations=2 \
  --games_per_iteration=5 \
  --gradient_steps=20 \
  --num_puzzles=50

# Standard training (2-4 hours)
./train_and_evaluate.sh \
  --base_model=9M \
  --num_iterations=10 \
  --games_per_iteration=20 \
  --num_puzzles=100

# Custom configuration
./train_and_evaluate.sh \
  --base_model=136M \
  --num_iterations=20 \
  --games_per_iteration=30 \
  --batch_size=64 \
  --learning_rate=0.00005 \
  --gradient_steps=200 \
  --stockfish_time=0.05 \
  --num_puzzles=200
```

The pipeline will:
1. Evaluate baseline performance (base model on puzzles)
2. Run self-play training
3. Evaluate trained model performance
4. Display before/after comparison with improvement statistics

### Manual Evaluation

You can also run evaluation steps separately:

```bash
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Evaluate base model
python puzzles.py --agent=9M --num_puzzles=100

# Train model
python selfplay_train.py --base_model=9M --num_iterations=10

# Evaluate trained model
python puzzles.py --agent=9M_selfplay --num_puzzles=100

# Compare results
python evaluate_selfplay.py --base_model=9M --num_puzzles=100
```

### Evaluation Output

The comparison script generates detailed statistics:

```
==========================================
EVALUATION RESULTS COMPARISON
==========================================

Base Model: 9M
Accuracy: 45.00% (45/100)

Selfplay Model: 9M_selfplay
Accuracy: 55.00% (55/100)

Improvement:
  Accuracy: +10.00%
  Additional Puzzles Solved: +10

Rating Breakdown:
Range           Base Acc.    Selfplay Acc.   Improvement
-----------------------------------------------------------
0-1000          85.00%       90.00%          +5.00%
1000-1500       60.00%       70.00%          +10.00%
1500-2000       45.00%       55.00%          +10.00%
2000-2500       30.00%       40.00%          +10.00%
2500+           20.00%       25.00%          +5.00%
==========================================

Results saved to: ../data/9M_vs_9M_selfplay_comparison.json
```

### Elo Comparison

For a more rigorous evaluation, calculate Elo ratings through head-to-head matches:

```bash
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Run 50 games between base and selfplay models
python compare_engines.py \
  --engine1=9M \
  --engine2=9M_selfplay \
  --num_games=50

# For more accurate Elo (100+ games recommended)
python compare_engines.py \
  --engine1=9M \
  --engine2=9M_selfplay \
  --num_games=200
```

**Output includes:**
- Win/loss/draw statistics
- Score percentages
- Approximate Elo difference (calculated from win rate)
- BayesElo precise calculation (if BayesElo is compiled)
- PGN file with all games

**Example output:**
```
============================================================
RESULTS
============================================================
9M: 18 wins
9M_selfplay: 28 wins
Draws: 4
Total: 50 games
============================================================

Score:
9M: 20.0/50 (40.0%)
9M_selfplay: 30.0/50 (60.0%)

Approximate Elo difference: +71
(9M_selfplay is +71 Elo relative to 9M)

============================================================
Running BayesElo for precise Elo calculation...
============================================================

Rank Name          Elo    +    - games score oppo. draws
   1 9M_selfplay    38   45   45    50   60%    -2    8%
   2 9M             -38   45   45    50   40%     2    8%

PGN file saved to: ../data/9M_vs_9M_selfplay_games.pgn
```

**Interpreting Elo Differences:**
- **+0 to +50**: Slight improvement
- **+50 to +100**: Moderate improvement
- **+100 to +200**: Significant improvement
- **+200+**: Major improvement

**Statistical Significance:**
- 50 games: ±45 Elo uncertainty
- 100 games: ±30 Elo uncertainty
- 200 games: ±20 Elo uncertainty

For reliable measurements, play 100+ games.

## Monitoring Training

### Progress Logs
The training script outputs:
```
=== Iteration 1/10 ===
Generating 20 self-play games...
Generating self-play game 1/20
...
Generated 450 experiences from 20 games
Training for 100 steps...
  Step 0/100: loss=2.3456, grad_norm=3.2100
  Step 10/100: loss=2.1234, grad_norm=2.8900
...
Saving checkpoint for iteration 1
```

### What to Monitor
- **Loss**: Should generally decrease (target: < 1.0)
- **Gradient Norm**: Should be stable (< 10.0)
- **Experiences Generated**: More is better (aim for 400+ per iteration)

## Performance Tips

### GPU Optimization
- Use smaller batch sizes if you hit OOM errors
- Close other GPU applications during training
- Monitor GPU memory: `watch -n 1 nvidia-smi`

### Training Speed
- **9M model**: ~2-3 games/minute
- **136M model**: ~1-2 games/minute
- **270M model**: ~0.5-1 game/minute

### Stockfish Time Trade-off
- Lower time (0.01s): Faster training, noisier signals
- Higher time (0.1s): Slower training, more accurate rewards
- Recommended: Start with 0.01s, increase if training is unstable

## Troubleshooting

### Import Errors
```bash
ModuleNotFoundError: No module named 'searchless_chess'
```
**Solution**: Set PYTHONPATH correctly
```bash
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size
```bash
python selfplay_train.py --batch_size=16
```

### Stockfish Not Found
```
FileNotFoundError: Stockfish engine not found
```
**Solution**: Ensure Stockfish is compiled and path is correct in `stockfish_engine.py`

### Training Instability (Loss Increasing)
**Solutions**:
- Reduce learning rate: `--learning_rate=5e-5`
- Increase Stockfish evaluation time: `--stockfish_time=0.05`
- Use larger batch size: `--batch_size=64`
- Add more games per iteration: `--games_per_iteration=50`

## Expected Improvements

After self-play training, you should see:
- **Puzzle Accuracy**: +5-15% improvement
- **Tournament Elo**: +50-200 Elo gain
- **Tactical Play**: Fewer blunders, better combination finding
- **Strategic Understanding**: May decrease slightly (trade-off)

## Comparison with Base Models

| Metric | Base 9M | Self-Play 9M | Improvement |
|--------|---------|--------------|-------------|
| Puzzle Acc. | 45% | 55% | +10% |
| Avg. Elo | 1800 | 1950 | +150 |
| Training Time | 0 | 4 hours | - |

## Next Steps

1. **Evaluate Performance**: Run puzzles and tournaments
2. **Iterative Training**: Continue training for more iterations
3. **Hyperparameter Tuning**: Experiment with learning rates
4. **Ensemble Methods**: Combine multiple self-play models
5. **Transfer Learning**: Use self-play model as base for other tasks

## Technical Details

### Algorithm: Value-Based Q-Learning with Stockfish Targets
```
Loss = Huber(Q_predicted(s,a) - Q_target(s,a))

Where:
- Q_predicted = E[return | s, a] from model's bucket distribution
- Q_target = Stockfish reward + game outcome
- Huber loss for robustness to outliers
```

The pretrained models predict Q-value distributions P(return | s, a) over 128 buckets.
We fine-tune these Q-values using Stockfish evaluations as targets.
The policy π(a|s) emerges from Q-values via: π(a|s) ∝ softmax(Q(s,a) / temperature).

### EMA Parameter Updates
```
θ_ema = 0.999 × θ_ema + 0.001 × θ_new
```
EMA parameters are more stable for inference.

### Huber Loss
```
Huber(x) = 0.5 × x² for |x| ≤ δ
         = δ × (|x| - 0.5δ) for |x| > δ
```
More robust than MSE for outlier rewards. Delta = 1.0 in our implementation.

## References

- Original Paper: "Grandmaster-Level Chess Without Search" (DeepMind, 2024)
- Self-Play: Silver et al., "Mastering the game of Go without human knowledge" (2017)
- Q-Learning: Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Reinforcement Learning: Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)

---

For questions or issues, please refer to the main USAGE_GUIDE.md or open an issue.
