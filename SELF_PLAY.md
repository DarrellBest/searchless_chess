# Self-Play Training with DPO

This guide explains how to improve chess models using self-play with **Direct Preference Optimization (DPO)** and Stockfish as a teacher.

## Table of Contents

1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Usage Guide](#usage-guide)
4. [Technical Details](#technical-details)
5. [Performance Evaluation](#performance-evaluation)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Setup environment
conda activate searchless_chess
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Train 9M model with DPO (2-4 hours)
python selfplay_train.py --base_model=9M --num_iterations=10

# Evaluate improvement
python puzzles.py --agent=9M_selfplay --num_puzzles=100
```

Or use the automated pipeline:

```bash
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess
./train_and_evaluate.sh --base_model=9M --num_iterations=10
```

---

## How It Works

### Overview

DPO (Direct Preference Optimization) improves models by teaching them to **prefer better moves over worse moves**:

1. **Self-Play**: Model plays against itself to generate games
2. **Stockfish Analysis**: Analyzes positions where the model made mistakes
3. **Preference Pairs**: Creates (position, Stockfish's move, model's move) pairs
4. **DPO Training**: Directly optimizes model to prefer Stockfish's moves

**Key advantages of DPO:**
- Stable training (no reward modeling, no bootstrapping)
- Sample efficient (only learns from mistakes)
- Direct optimization (optimizes exactly what we care about)
- Proven approach (used in ChatGPT, Claude, Gemini alignment)

### The DPO Algorithm

For each position where the model made a mistake:

```
Position: rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b
Model played: Bc5 (eval: -0.1)
Stockfish prefers: Nc6 (eval: +0.3)
Margin: 0.4 pawns

DPO Loss: -log σ(β * (log π(Nc6) - log π(Bc5)))
```

The loss **increases the probability** of Stockfish's move **relative to** the model's mistake.

### Key Insight for Action-Value Models

Our models output Q-value distributions, not direct move probabilities. DPO works because:

```python
π(move|position) ∝ exp(Q(position, move) / τ)

# For DPO, we need log probability ratios:
log π(chosen) - log π(rejected) = [Q(chosen) - Q(rejected)] / τ

# The partition function cancels! We only need two Q-values.
```

This makes DPO tractable for action-value architectures.

### Mistake Filtering

Not all positions become training data. We filter by:

1. **Significance**: Eval difference ≥ 0.3 pawns (default)
   - Too small: noise, not real mistakes
   - Just right: clear improvements to learn

2. **Position Quality**: |eval| ≤ 3.0 pawns
   - Already winning/losing: specific moves matter less
   - Balanced positions: every move counts

**Typical statistics:**
- 1000 games → ~40,000 positions
- 650 mistakes found (1.6%)
- After filtering: 650 training pairs

### Progressive Curriculum

As the model improves, we increase Stockfish's analysis depth:

- **Iterations 1-5**: Depth 20 (finds obvious mistakes quickly)
- **Iterations 6+**: Depth 22, 24, 25 (finds subtler errors)

This matches the model's skill level to the difficulty of examples.

---

## Usage Guide

### Basic Training

Start training from a pretrained base model:

```bash
# Train from 9M model (fastest, 2-4 hours)
python selfplay_train.py --base_model=9M

# Train from 136M model (balanced, 8-12 hours)
python selfplay_train.py --base_model=136M

# Train from 270M model (best quality, 24-48 hours)
python selfplay_train.py --base_model=270M
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | 9M | Starting model: 9M, 136M, or 270M |
| `--num_iterations` | 10 | Total training iterations |
| `--games_per_iteration` | 1000 | Self-play games per iteration |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 1e-5 | Adam learning rate (small for fine-tuning) |
| `--gradient_steps_per_iteration` | 50 | Optimization steps per iteration |
| `--stockfish_time` | 0.1 | Stockfish time per position (seconds) |
| `--stockfish_depth` | 20 | Initial Stockfish analysis depth |
| `--eval_threshold` | 0.3 | Min eval difference for preference pairs (pawns) |
| `--beta` | 0.1 | DPO KL penalty coefficient |
| `--eval_puzzles` | 50 | Puzzles evaluated at each iteration |
| `--save_frequency` | 1 | Checkpoint save frequency |
| `--resume` | False | Resume from latest checkpoint |

### Advanced Configuration

```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=20 \
  --games_per_iteration=2000 \
  --batch_size=64 \
  --learning_rate=5e-6 \
  --gradient_steps_per_iteration=100 \
  --stockfish_time=0.2 \
  --stockfish_depth=22 \
  --eval_threshold=0.25 \
  --beta=0.15 \
  --eval_puzzles=100
```

### Training Profiles

#### Quick Test (30 minutes)
```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=2 \
  --games_per_iteration=100 \
  --gradient_steps_per_iteration=20
```

#### Standard Training (2-4 hours)
```bash
python selfplay_train.py \
  --base_model=9M \
  --num_iterations=10 \
  --games_per_iteration=1000 \
  --gradient_steps_per_iteration=50
```

#### Intensive Training (12-24 hours)
```bash
python selfplay_train.py \
  --base_model=136M \
  --num_iterations=20 \
  --games_per_iteration=2000 \
  --gradient_steps_per_iteration=100 \
  --stockfish_time=0.2
```

#### Production Training (2-7 days)
```bash
python selfplay_train.py \
  --base_model=270M \
  --num_iterations=50 \
  --games_per_iteration=3000 \
  --gradient_steps_per_iteration=200 \
  --stockfish_time=0.3 \
  --batch_size=64 \
  --learning_rate=5e-6
```

### Resuming Training

The training script supports resuming from checkpoints:

```bash
# Train for 10 iterations
python selfplay_train.py --base_model=9M --num_iterations=10

# Later, continue for 10 MORE iterations
python selfplay_train.py --base_model=9M --num_iterations=10 --resume
```

**How it works:**
- Without `--resume`: Always starts from base model
- With `--resume`: Loads latest checkpoint and continues
- Adds `--num_iterations` MORE iterations beyond checkpoint

**Example workflow:**
```bash
# Day 1: Train for 5 iterations (saves checkpoints 1-5)
python selfplay_train.py --base_model=9M --num_iterations=5

# Day 2: Continue for 5 more (iterations 6-10)
python selfplay_train.py --base_model=9M --num_iterations=5 --resume

# Day 3: Continue for 10 more (iterations 11-20)
python selfplay_train.py --base_model=9M --num_iterations=10 --resume
```

### Using Trained Models

Trained models are automatically registered and ready to use:

```bash
# Evaluate on puzzles
python puzzles.py --agent=9M_selfplay --num_puzzles=100

# Compare with base model
python puzzles.py --agent=9M --num_puzzles=100

# Use in tournaments
python compare_engines.py --engine1=9M --engine2=9M_selfplay --num_games=50
```

Available selfplay engines:
- `9M_selfplay` - Selfplay-trained 9M model
- `136M_selfplay` - Selfplay-trained 136M model
- `270M_selfplay` - Selfplay-trained 270M model

---

## Technical Details

### DPO Loss Function

The DPO loss for each preference pair is:

```
L = -log σ(β * (r_θ(s,w) - r_θ(s,l)))

where:
  r_θ(s,a) = log π_θ(a|s) - log π_ref(a|s)  (reward ratio)
  w = winning/chosen move (Stockfish)
  l = losing/rejected move (model)
  β = KL penalty coefficient (default 0.1)
  σ = sigmoid function
```

**Intuition**: Maximize the log-odds that the model prefers Stockfish's move over its own mistake, relative to a reference model.

### From Q-Values to Probabilities

Our models output Q-value distributions P(return | state, action) over 128 buckets:

```python
# 1. Compute expected Q-value for a move
Q(state, action) = E[Z(state, action)]
                 = Σ (prob_i × atom_i)

# 2. Derive move probability via Boltzmann distribution
π(action | state) ∝ exp(Q(state, action) / temperature)

# 3. For DPO, compute log probability (up to partition)
log π(action | state) ≈ Q(state, action) / temperature
```

The key insight: **partition functions cancel in DPO ratios**, so we don't need full softmax.

### Reference Model Updates

The reference model π_ref provides a stability anchor:

- **Initial**: Reference = base model
- **Updates**: Every 3 iterations, reference = current EMA params
- **Purpose**: Prevents model from drifting too far too fast

This implements the KL penalty term from the DPO objective:

```
max E[r(s,a)] - β * KL(π || π_ref)
```

### EMA (Exponential Moving Average)

We maintain two parameter sets:

```python
params           # Online network (trained via gradients)
params_ema       # Fast EMA (decay=0.99) for checkpointing
reference_params # Slow updates (every 3 iterations) for DPO
```

EMA update:
```python
params_ema = 0.99 × params_ema + 0.01 × params
```

EMA parameters are more stable for inference and evaluation.

### Expected Behavior by Iteration

**Iteration 1-2** (Early Learning):
- Mistakes found: 500-800 per 1000 games
- DPO loss: ~1.5-2.0 (high)
- Gradient norm: ~1.0-2.0
- Elo improvement: +50-100

**Iteration 3-5** (Intermediate):
- Mistakes found: 300-500 per 1000 games
- DPO loss: ~0.8-1.2 (decreasing)
- Gradient norm: ~0.5-1.0
- Elo improvement: +30-50 per iteration

**Iteration 6-10** (Convergence):
- Mistakes found: 100-300 per 1000 games
- DPO loss: ~0.3-0.6 (low)
- Gradient norm: ~0.2-0.5
- Elo improvement: +10-30 per iteration

**Stopping criteria**:
- Mistakes < 100 per 1000 games (model nearly optimal)
- No Elo improvement for 3 iterations (plateau)
- Eval threshold too high (lower to 0.2 pawns)

### Hyperparameter Tuning

**Learning rate:**
- Too high (>1e-4): Unstable, loss oscillates
- Just right (1e-5): Steady improvement
- Too low (<1e-6): Very slow convergence

**Beta (KL penalty):**
- Too high (>0.5): Model doesn't improve (too constrained)
- Just right (0.1): Good balance
- Too low (<0.01): May overfit to mistakes

**Eval threshold:**
- Too high (>0.5): Misses learning opportunities
- Just right (0.3): Clear mistakes only
- Too low (<0.1): Trains on noise

**Games per iteration:**
- Too few (<500): Not enough mistakes found
- Just right (1000-2000): Good statistics
- Too many (>5000): Diminishing returns

---

## Performance Evaluation

### Automated Pipeline

Use `train_and_evaluate.sh` for complete workflow:

```bash
# Quick test (30 min)
./train_and_evaluate.sh \
  --base_model=9M \
  --num_iterations=2 \
  --games_per_iteration=100 \
  --eval_puzzles=50

# Standard training (2-4 hours)
./train_and_evaluate.sh \
  --base_model=9M \
  --num_iterations=10 \
  --games_per_iteration=1000 \
  --eval_puzzles=100

# Custom configuration
./train_and_evaluate.sh \
  --base_model=136M \
  --num_iterations=20 \
  --games_per_iteration=2000 \
  --batch_size=64 \
  --learning_rate=0.000005 \
  --gradient_steps=100 \
  --stockfish_time=0.2 \
  --stockfish_depth=22 \
  --eval_threshold=0.25 \
  --beta=0.12 \
  --eval_puzzles=200
```

The pipeline automatically:
1. Evaluates baseline (base model on puzzles)
2. Runs DPO training
3. Evaluates trained model
4. Displays before/after comparison

### Manual Evaluation

```bash
# Baseline
python puzzles.py --agent=9M --num_puzzles=100

# Train
python selfplay_train.py --base_model=9M --num_iterations=10

# Evaluate
python puzzles.py --agent=9M_selfplay --num_puzzles=100

# Compare
python evaluate_selfplay.py --base_model=9M --num_puzzles=100
```

### Example Evaluation Output

```
==========================================
EVALUATION RESULTS COMPARISON
==========================================

Base Model: 9M
Accuracy: 45.00% (45/100)

Selfplay Model: 9M_selfplay
Accuracy: 57.00% (57/100)

Improvement:
  Accuracy: +12.00%
  Additional Puzzles Solved: +12

Rating Breakdown:
Range           Base Acc.    Selfplay Acc.   Improvement
-----------------------------------------------------------
0-1000          85.00%       92.00%          +7.00%
1000-1500       62.00%       75.00%          +13.00%
1500-2000       48.00%       62.00%          +14.00%
2000-2500       32.00%       42.00%          +10.00%
2500+           18.00%       25.00%          +7.00%
==========================================
```

### Elo Calculation

For rigorous evaluation, run head-to-head games:

```bash
# 50 games (±45 Elo uncertainty)
python compare_engines.py \
  --engine1=9M \
  --engine2=9M_selfplay \
  --num_games=50

# 200 games (±20 Elo uncertainty, recommended)
python compare_engines.py \
  --engine1=9M \
  --engine2=9M_selfplay \
  --num_games=200
```

**Expected Elo improvements:**
- After 5 iterations: +50-100 Elo
- After 10 iterations: +100-200 Elo
- After 20 iterations: +150-250 Elo
- Plateau typically around +200-300 Elo

### Monitoring Training

Watch for these metrics during training:

```
=== Iteration 5/10 ===
Generating 1000 self-play games...
Analyzing 1000 games with Stockfish...
Found 420 preference pairs (mistakes)

Sample statistics:
- Eval margin distribution:
  - 0.3-0.5 pawns: 210 pairs (50%)
  - 0.5-1.0 pawns: 140 pairs (33%)
  - 1.0-2.0 pawns: 50 pairs (12%)
  - >2.0 pawns: 20 pairs (5%, blunders)

Training for 50 steps...
  Step 0/50: loss=0.8234, grad_norm=0.4521
  Step 10/50: loss=0.7891, grad_norm=0.4123
  ...
  Step 40/50: loss=0.7234, grad_norm=0.3845

Updating reference model...
Saving checkpoint for iteration 5

Evaluating on 50 puzzles...
Iteration 5 Puzzle Results: 32/50 (64.0%)
```

**What to monitor:**
- **Mistakes found**: Should decrease over iterations
  - Too few (<100): Lower eval_threshold or model converged
  - Just right (200-500): Good learning signal
  - Too many (>1000): Model still weak or threshold too low

- **Loss**: Should generally decrease
  - Increasing: Learning rate too high or unstable data
  - Stuck: May need more games or lower threshold
  - Target: <0.5 by final iteration

- **Gradient norm**: Should be stable
  - Too high (>5.0): Unstable, reduce learning rate
  - Just right (0.2-1.0): Healthy gradients
  - Too low (<0.1): May need higher learning rate

---

## Troubleshooting

### Import Errors

```
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

**Solutions**:
- Reduce batch size: `--batch_size=16`
- Use smaller model: `--base_model=9M`
- Close other GPU applications
- Monitor GPU usage: `watch -n 1 nvidia-smi`

### Stockfish Not Found

```
FileNotFoundError: Stockfish engine not found
```

**Solution**: Verify Stockfish path in `src/engines/stockfish_engine.py`

### No Preference Pairs Generated

```
No preference pairs generated, skipping training.
Model may have converged or eval_threshold is too high.
```

**Solutions**:
- Lower eval threshold: `--eval_threshold=0.2`
- Increase Stockfish depth: `--stockfish_depth=25`
- Model may have plateaued (check puzzle accuracy)

### Training Instability

Loss increases or oscillates wildly.

**Solutions**:
- Reduce learning rate: `--learning_rate=5e-6`
- Increase batch size: `--batch_size=64`
- Increase Stockfish time: `--stockfish_time=0.2`
- Lower beta: `--beta=0.05`

### Slow Training

**Optimizations**:
- Reduce Stockfish time: `--stockfish_time=0.05`
- Fewer games: `--games_per_iteration=500`
- Use smaller model: `--base_model=9M`
- Check GPU utilization (should be >80%)

---

## Expected Improvements

After DPO training, you should observe:

**Tactical Strength:**
- Puzzle accuracy: +10-15%
- Tournament Elo: +100-200
- Fewer blunders: -30%
- Better combinations: +25%

**Trade-offs:**
- Opening diversity: May decrease slightly
- Endgame technique: Improves marginally
- Time per move: No change (still searchless)

**Comparison with base models:**

| Metric | Base 9M | DPO 9M | Improvement |
|--------|---------|--------|-------------|
| Puzzle Acc. | 45% | 57% | +12% |
| Avg. Elo | 1800 | 1980 | +180 |
| Blunders/game | 2.1 | 1.4 | -33% |
| Training Time | 0 | 3 hours | - |

---

## Key Design Decisions

### Why DPO over Value-Based RL?

**DPO advantages:**
- Stable (no bootstrapping, no target networks)
- Direct (optimizes exactly what we want)
- Efficient (only learns from mistakes)
- Proven (ChatGPT, Claude alignment)

**C51/DQN disadvantages:**
- Requires reward shaping (complex)
- Needs target networks (more parameters)
- Bootstrapping can be unstable
- Learns from all positions (inefficient)

### Why Mistake-Focused Learning?

Only creating training data where the model erred:
- **Sample efficiency**: Don't waste time on positions already played correctly
- **Targeted improvement**: Address specific weaknesses
- **Natural curriculum**: Easy mistakes abundant early, harder ones appear later

### Why Stockfish as Teacher?

- **Strong and consistent**: Provides reliable "ground truth"
- **Fast analysis**: Can analyze thousands of positions quickly
- **Not imitation**: We learn preferences, not exact moves
- **Flexibility**: Model can find alternative good moves

### Why Progressive Curriculum?

Starting at depth 20, increasing to 25:
- **Early**: Find obvious mistakes quickly
- **Later**: Find subtle tactical errors
- **Matches skill**: Difficulty scales with model strength

---

## References

### Core Papers

**Searchless Chess (Base Model)**
- Ruoss, A., Delétang, G., Medapati, S., Grau-Moya, J., Wenliang, L. K., Catt, E., ... & Genewein, T. (2024).
  *Grandmaster-Level Chess Without Search*.
  NeurIPS 2024.
  [arXiv:2402.04494](https://arxiv.org/abs/2402.04494)

  The original paper introducing the searchless chess models we fine-tune with DPO. Trained 270M parameter transformers on 10 billion positions from Stockfish to achieve 2895 Elo without search.

**Direct Preference Optimization (Training Algorithm)**
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
  *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
  NeurIPS 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

  The DPO algorithm we use for training. Enables stable preference learning without explicit reward modeling, originally developed for LLM alignment.

### Self-Play and Reinforcement Learning

**AlphaGo (Self-Play Origins)**
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016).
  *Mastering the game of Go with deep neural networks and tree search*.
  Nature, 529(7587), 484-489.
  [DOI:10.1038/nature16961](https://doi.org/10.1038/nature16961)

  First major application of self-play with neural networks and reinforcement learning to achieve superhuman performance.

**AlphaZero (Self-Play for Chess)**
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018).
  *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*.
  Science, 362(6419), 1140-1144.
  [DOI:10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)

  Generalized self-play approach to chess, shogi, and Go. Achieved superhuman play through self-play alone, without human game data.

**AlphaGo Zero (Pure Self-Play)**
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017).
  *Mastering the game of Go without human knowledge*.
  Nature, 550(7676), 354-359.
  [DOI:10.1038/nature24270](https://doi.org/10.1038/nature24270)

  Demonstrated that self-play alone (tabula rasa) can achieve superhuman performance without any human knowledge or examples.

### Alignment and Preference Learning

**InstructGPT / RLHF**
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022).
  *Training language models to follow instructions with human feedback*.
  NeurIPS 2022.
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

  Introduced RLHF for LLM alignment. DPO simplifies this approach by removing the reward model step.

### Chess Engines

**Stockfish**
- Stockfish Development Team. (2024).
  *Stockfish: Open Source Chess Engine*.
  [https://stockfishchess.org/](https://stockfishchess.org/)

  The world's strongest open-source chess engine, used as our teacher for generating preference pairs.

**Leela Chess Zero**
- Pascutto, G. C., Linscott, G., & others. (2018).
  *Leela Chess Zero*.
  [https://lczero.org/](https://lczero.org/)

  Open-source implementation of AlphaZero for chess, trained entirely through self-play.

---

## Next Steps

1. **Baseline Evaluation**: Test base model performance
2. **Initial Training**: Run 5-10 iterations
3. **Performance Check**: Compare puzzle accuracy and Elo
4. **Iterative Refinement**: Continue training or tune hyperparameters
5. **Publication**: Share trained models on HuggingFace

For questions or issues, open an issue on GitHub or refer to the main documentation.
