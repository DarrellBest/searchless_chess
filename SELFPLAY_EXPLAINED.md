# Self-Play Training: How It Works

This document explains the self-play training system in detail, covering how games are generated, how moves are selected, and how the model learns from experience.

## Table of Contents

1. [Overview](#overview)
2. [Game Generation Process](#game-generation-process)
3. [Move Selection with Temperature](#move-selection-with-temperature)
4. [Reward Calculation](#reward-calculation)
5. [Learning Algorithm](#learning-algorithm)
6. [Complete Example](#complete-example)
7. [Key Design Decisions](#key-design-decisions)

---

## Overview

Self-play training improves chess models by:
1. Having the neural network play against itself
2. Using Stockfish to evaluate each position and provide reward signals
3. Training the model's Q-values to predict these rewards
4. Iteratively improving through multiple training cycles

This combines the pattern recognition capabilities of neural networks with the tactical accuracy of Stockfish.

---

## Game Generation Process

### Step-by-Step Breakdown

Each self-play game follows this loop (from `selfplay_generator.py`):

#### 1. Evaluate Position BEFORE Move

```python
score_before = self._get_stockfish_score(board)
```

- Stockfish evaluates the current position
- Returns centipawns from current player's perspective
- Example: `+150` means current player is up 1.5 pawns

#### 2. Neural Engine Plays a Move

```python
move = self.neural_engine.play(board)
```

- The neural model (9M/136M/270M) selects a move
- Uses temperature-based sampling (see next section)
- Playing against itself (same model for both White and Black)

#### 3. Apply the Move

```python
board.push(move)
```

The move is executed on the chess board.

#### 4. Evaluate Position AFTER Move

```python
score_after = self._get_stockfish_score(board)
```

- Stockfish evaluates the new position
- **Important:** Now from opponent's perspective (sides have flipped)

#### 5. Calculate Immediate Reward

```python
reward = (score_before + score_after) * self.reward_scaling
```

**Why `score_before + score_after`?**

- `score_before`: Your advantage before the move (from your perspective)
- `score_after`: Opponent's advantage after the move (from their perspective)
- If you improved the position, `score_after` (opponent's view) becomes negative
- Adding them captures the position change

**Example:**
- Before move: `+100` (you're up 1 pawn)
- After move: `-50` (opponent is down 0.5 pawns = you're up 0.5 pawns from their view)
- Reward: `(100 + (-50)) × 0.01 = 0.5`

The `0.01` scaling factor converts centipawns to a normalized 0-1 range.

#### 6. Store Experience

```python
experiences.append(
    SelfPlayExperience(
        tokenized_fen=tokenized_fen,  # Board state (77 tokens)
        action=action_idx,             # Move made (e.g., 1245)
        reward=reward,                 # Position improvement (e.g., 0.5)
        game_outcome=0.5,              # Placeholder (updated later)
    )
)
```

### Game Outcome Determination

After the game ends, determine the result:

```python
if board.is_checkmate():
    game_outcome = 0.0  # Side to move lost
elif board.is_stalemate() or board.can_claim_draw():
    game_outcome = 0.5  # Draw
else:
    # Game hit move limit or other termination
    final_score = self._get_stockfish_score(board)
    if abs(final_score) > 300:  # 3 pawns advantage
        game_outcome = 1.0 if final_score > 0 else 0.0
    else:
        game_outcome = 0.5  # Unclear position, treat as draw
```

### Outcome Propagation

**Critical step:** Assign the game outcome from each player's perspective

```python
for i in range(len(experiences)):
    if i % 2 == 0:
        experiences[i].game_outcome = game_outcome
    else:
        experiences[i].game_outcome = 1.0 - game_outcome
```

**Why alternate?**

- Move 0 (White's 1st move): If White won (1.0), assign 1.0
- Move 1 (Black's 1st move): If White won (1.0), Black gets 0.0
- Move 2 (White's 2nd move): White gets 1.0
- Move 3 (Black's 2nd move): Black gets 0.0
- And so on...

This ensures each experience has the correct win/loss signal from the player's perspective.

---

## Move Selection with Temperature

### How the Engine Selects Moves

From `neural_engines.py`, the `ActionValueEngine.play()` method:

```python
def play(self, board: chess.Board) -> chess.Move:
    # 1. Get Q-value distributions for all legal moves
    return_buckets_log_probs = self.analyse(board)['log_probs']
    return_buckets_probs = np.exp(return_buckets_log_probs)

    # 2. Compute expected Q-value (win probability) for each move
    win_probs = np.inner(return_buckets_probs, self._return_buckets_values)

    # 3. Adjust for repetition draws
    _update_scores_with_repetitions(board, win_probs)

    sorted_legal_moves = engine.get_ordered_legal_moves(board)

    # 4. Sample move based on temperature
    if self.temperature is not None:
        # Probabilistic: sample from softmax distribution
        probs = scipy.special.softmax(win_probs / self.temperature, axis=-1)
        return self._rng.choice(sorted_legal_moves, p=probs)
    else:
        # Deterministic: always pick best move
        best_index = np.argmax(win_probs)
        return sorted_legal_moves[best_index]
```

### Temperature Effects

Temperature controls exploration vs exploitation:

**temperature = None (greedy, deterministic):**
- Always picks the move with highest Q-value
- No randomness
- All games would be identical

**temperature = 0.1 (low, nearly greedy):**

Assume Q-values for starting position:
```
e4:   Q = 0.55
d4:   Q = 0.53
Nf3:  Q = 0.52
c4:   Q = 0.51
```

```python
probs = softmax([0.55, 0.53, 0.52, 0.51] / 0.1)
# Results: [98%, 1%, 0.5%, 0.5%]
# Almost always picks e4 (best move)
```

**temperature = 1.0 (balanced, default for self-play):**

```python
probs = softmax([0.55, 0.53, 0.52, 0.51] / 1.0)
# Results: [27%, 25%, 24%, 24%]
# Good exploration while still weighted by quality
```

**temperature = 10.0 (high, nearly uniform):**

```python
probs = softmax([0.55, 0.53, 0.52, 0.51] / 10.0)
# Results: [25.5%, 25%, 24.75%, 24.75%]
# Nearly random (ignores Q-values)
```

### Why Temperature = 1.0 for Self-Play?

In `selfplay_train.py`:

```python
neural_engine = neural_engines.ActionValueEngine(
    return_buckets_values=return_buckets_values,
    predict_fn=...,
    temperature=1.0,  # Enables diverse game generation
)
```

Benefits:
- **Exploration:** Model tries different openings and strategies
- **Diversity:** Each game is different, preventing overfitting
- **Quality:** Still weighted by Q-values (good moves more likely)
- **Coverage:** Generates training data across various positions

---

## Reward Calculation

### Two Components

Each experience receives a **combined reward**:

```python
total_reward = exp.reward + exp.game_outcome
```

#### Component 1: Immediate Reward (Position Change)

```python
reward = (score_before + score_after) * 0.01
```

- Measures tactical improvement of the move
- Based on Stockfish centipawn evaluation
- Scaled by 0.01 to normalize to ~0-1 range
- Can be positive (good move) or negative (bad move)

#### Component 2: Game Outcome (Final Result)

```python
game_outcome = 1.0  # Win
game_outcome = 0.5  # Draw
game_outcome = 0.0  # Loss
```

- Provides final strategic signal
- Did this move contribute to winning?
- Propagated backward with correct perspective

### Example Rewards

**Good tactical move in a won game:**
- Immediate: `+0.8` (improved position by 80 centipawns)
- Outcome: `+1.0` (won the game)
- Total: `1.8`

**Slight mistake in a drawn game:**
- Immediate: `-0.2` (worsened position by 20 centipawns)
- Outcome: `+0.5` (game drawn)
- Total: `0.3`

**Blunder in a lost game:**
- Immediate: `-3.5` (hung a piece, -350 centipawns)
- Outcome: `0.0` (lost the game)
- Total: `-3.5`

---

## Learning Algorithm

### Value-Based Q-Learning with Stockfish Targets

The model learns by minimizing the difference between predicted Q-values and Stockfish-based rewards.

### Loss Function

From `selfplay_train.py`:

```python
def loss_fn(params, sequences, rewards):
    # 1. Get Q-value distribution predictions
    bucket_log_probs = predictor.predict(params=params, targets=sequences, rng=None)

    # 2. Extract Q-value distribution for the action token (position -2)
    action_bucket_log_probs = bucket_log_probs[:, -2]
    action_bucket_probs = jnp.exp(action_bucket_log_probs)

    # 3. Compute predicted Q-value (expected return)
    predicted_q = jnp.dot(action_bucket_probs, return_buckets_values)

    # 4. Target Q-values come from Stockfish + game outcome
    target_q = rewards

    # 5. Compute TD error
    td_error = predicted_q - target_q

    # 6. Use Huber loss for robustness to outliers
    huber_delta = 1.0
    abs_error = jnp.abs(td_error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    huber_loss = 0.5 * quadratic ** 2 + huber_delta * linear

    return jnp.mean(huber_loss)
```

### Training Loop

Each iteration:

1. **Generate self-play games** (e.g., 20 games)
   - Neural model plays against itself
   - Stockfish evaluates each position
   - Collect experiences (state, action, reward)

2. **Shuffle experiences**
   - Randomize order to prevent sequential correlation
   - Batch into training examples

3. **Gradient descent** (e.g., 100 steps)
   - Update Q-values to match Stockfish targets
   - Use Adam optimizer with gradient clipping
   - Update EMA parameters for stable inference

4. **Save checkpoint**
   - Store updated parameters
   - Can be evaluated or used for next iteration

### Why Q-Learning (Not Policy Gradient)?

The pretrained models predict Q-value distributions, not action probabilities:

```
Model output: P(return | state, action)
  └─> 128 buckets representing return distribution
  └─> Q(s,a) = E[return]
  └─> Policy derived: π(a|s) ∝ softmax(Q(s,a) / temperature)
```

We keep this architecture and fine-tune Q-values using:
- **Targets:** Stockfish evaluations + game outcomes
- **Loss:** Huber loss (robust to outliers)
- **Benefit:** Uses ALL pretrained knowledge

Alternative (policy gradient) would require:
- Changing output layer (lose pretrained weights)
- Relearning move selection from scratch
- Less sample efficient

---

## Complete Example

### 3-Move Game Trace

**Starting position:** Standard chess opening

#### Move 1: White plays e4

**Before move:**
- Stockfish eval: `+20` centipawns (slight advantage)

**Move selected:**
- Q-values: e4 (0.55), d4 (0.53), Nf3 (0.52)
- Probs: [27%, 25%, 24%]
- Sampled: **e4**

**After move:**
- Stockfish eval: `-30` centipawns (from Black's perspective)

**Reward calculation:**
```python
reward = (20 + (-30)) * 0.01 = -0.1
```

**Game outcome:** White eventually wins (1.0)

**Final reward for this move:**
```python
total_reward = -0.1 + 1.0 = 0.9
```

**Interpretation:** Although the move slightly worsened position tactically, it was part of a winning strategy.

#### Move 2: Black plays e5

**Before move:**
- Stockfish eval: `+30` centipawns (Black's view)

**Move selected:**
- Q-values: e5 (0.52), Nc6 (0.51), d6 (0.48)
- Probs: [30%, 28%, 22%]
- Sampled: **e5**

**After move:**
- Stockfish eval: `-25` centipawns (from White's perspective)

**Reward calculation:**
```python
reward = (30 + (-25)) * 0.01 = 0.05
```

**Game outcome:** Black eventually loses (0.0)

**Final reward for this move:**
```python
total_reward = 0.05 + 0.0 = 0.05
```

**Interpretation:** Decent tactical move, but in a losing game.

#### Move 3: White plays Nf3

**Before move:**
- Stockfish eval: `+25` centipawns

**Move selected:**
- Q-values: Nf3 (0.56), Bc4 (0.54), d4 (0.52)
- Probs: [32%, 27%, 24%]
- Sampled: **Nf3**

**After move:**
- Stockfish eval: `-20` centipawns (Black's view)

**Reward calculation:**
```python
reward = (25 + (-20)) * 0.01 = 0.05
```

**Game outcome:** White wins (1.0)

**Final reward:**
```python
total_reward = 0.05 + 1.0 = 1.05
```

### Training on This Game

The model receives three training examples:

| Move | State    | Action | Target Q-value |
|------|----------|--------|----------------|
| 1    | [start]  | e4     | 0.9            |
| 2    | [e4]     | e5     | 0.05           |
| 3    | [e4 e5]  | Nf3    | 1.05           |

Model updates:
- If current Q(start, e4) = 0.5, loss pulls it toward 0.9
- If current Q([e4], e5) = 0.3, loss pulls it toward 0.05
- If current Q([e4 e5], Nf3) = 0.7, loss pulls it toward 1.05

Over many games, Q-values converge to Stockfish-informed predictions.

---

## Key Design Decisions

### 1. Self-Play (Not Fixed Opponent)

**Why:** Model plays both sides
- Explores own weaknesses
- No ceiling from opponent's strength
- Generates unlimited training data

**Alternative:** Play against Stockfish
- Would learn to mimic Stockfish
- Lose "searchless" property

### 2. Stockfish Rewards (Not Just Game Outcome)

**Why:** Immediate feedback on every move
- Faster learning (don't wait until game end)
- Tactical accuracy (catches blunders)
- Credit assignment (which moves were good?)

**Alternative:** Only use win/loss
- Sparse signal (only at game end)
- Slower convergence

### 3. Reward Scaling (0.01)

**Why:** Normalize centipawns to 0-1 range
- 100 centipawns (1 pawn) → 1.0 reward
- Compatible with [0.0, 1.0] game outcomes
- Stable gradient magnitudes

### 4. Combined Reward (Immediate + Outcome)

**Why:** Captures both tactics and strategy
- Immediate: Is this move tactically sound?
- Outcome: Did it contribute to winning?
- Balances short-term and long-term thinking

### 5. Temperature = 1.0

**Why:** Balanced exploration
- Not too greedy (avoids repetitive games)
- Not too random (still prioritizes good moves)
- Diverse training data

### 6. Q-Learning (Not Policy Gradient)

**Why:** Leverages pretrained Q-value architecture
- Uses ALL pretrained weights
- No architectural changes needed
- Sample efficient

**Alternative:** Policy gradient
- Would lose pretrained knowledge
- Requires more training data

### 7. Huber Loss (Not MSE)

**Why:** Robust to outlier rewards
- Blunders create huge negative rewards
- MSE would overweight these outliers
- Huber clips large errors (linear beyond threshold)

### 8. EMA Parameters

**Why:** Stable inference
- Training params can be noisy
- EMA smooths updates (0.999 decay)
- Better for evaluation

---

## Summary

Self-play training works by:

1. **Game Generation:** Neural model plays itself with temperature-based move sampling
2. **Reward Calculation:** Stockfish evaluates each move (immediate) + final game outcome
3. **Learning:** Q-values updated to match Stockfish targets using Huber loss
4. **Iteration:** Process repeats, gradually improving the model

The result is a model that combines:
- **Neural pattern recognition** (what moves to consider)
- **Stockfish tactical accuracy** (which moves are objectively good)
- **Strategic understanding** (which moves lead to winning positions)

All while remaining **searchless** (no tree search at inference time).
