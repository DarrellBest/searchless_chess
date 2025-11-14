# DPO Constrained Greedy Fix

## 1. Context

This project implements **Direct Preference Optimization (DPO)** fine-tuning for Searchless Chess models. DPO trains neural chess engines to prefer better moves (identified by Stockfish) over worse moves (the model's own mistakes) without requiring a separate reward model.

During DPO training, we observed a critical failure mode: **Q-value drift** and **"intruder moves"** - moves that were ranked very low by the base model but jumped to the top rank in the trained model, beating even Stockfish's chosen moves under naive greedy decoding.

## 2. Original Problem Example

### FEN Position
```
1R6/3q1ppk/6n1/b2p2Pp/2pP2b1/p1P5/P1B1rPPB/2Q2NK1 b - - 0 20
```

### BASE MODEL Ranking (Before Training)
```
1. d7e7 (rejected) Q=0.5831  ← Model's original top choice
2. d7e6           Q=0.5816
3. g4f5           Q=0.5739
4. a5c7 (chosen)  Q=0.5675  ← Stockfish's best move
5. h5h4           Q=0.5528
...
14. e2c2           Q=0.3755  ← Random rank-14 move
```

### TRAINED MODEL Ranking (After DPO Training)
```
1. e2c2           Q=0.4622  ← NOW RANK 1! (was rank 14)
2. a5c3           Q=0.4450
3. a5c7 (chosen)  Q=0.4426  ← Dropped to rank 3
4. e2e8           Q=0.4395
5. a5d8           Q=0.4362
```

### Q-Value Changes
- **Chosen (a5c7)**: 0.5675 → 0.4426 (Δ -0.1249) ↓
- **Rejected (d7e7)**: 0.5831 → 0.4296 (Δ -0.1535) ↓↓
- **Intruder (e2c2)**: 0.3755 → 0.4622 (Δ +0.0868) ↑

### Why This Breaks Greedy Decoding

Under standard greedy decoding (argmax over Q-values), the trained model would select **e2c2** because it has the highest Q-value (0.4622), even though:
1. The base model ranked it 14th (Q=0.3755)
2. Stockfish's chosen move **a5c7** is ranked 3rd (Q=0.4426)
3. This represents a catastrophic failure where a clearly inferior move beats the expert's choice

The root cause is **pessimistic Q-value drift**: DPO training can satisfy the preference constraint (chosen > rejected) by simply making all Q-values smaller, which causes relative rankings to shift unpredictably for moves not in the training pairs.

## 3. Old Behavior (Before Fix)

The original `ActionValueEngine.play()` method used pure greedy decoding:

```python
# Old behavior (simplified)
def play(self, board: chess.Board) -> chess.Move:
    return_buckets_log_probs = self.analyse(board)['log_probs']
    return_buckets_probs = np.exp(return_buckets_log_probs)
    win_probs = np.inner(return_buckets_probs, self._return_buckets_values)
    _update_scores_with_repetitions(board, win_probs)
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    
    # Pure greedy: pick move with highest Q-value
    best_index = np.argmax(win_probs)
    return sorted_legal_moves[best_index]
```

**Problem**: This allowed any move with the highest `Q_online` (including e2c2) to be selected, even if the base model thought it was very bad. There was no mechanism to prevent intruder moves that drifted upward during training.

## 4. New Behavior: Constrained Greedy Decoding

The fix implements **constrained greedy decoding** that uses the base model's top-K moves as a "trust region":

```python
# New behavior: Constrained Greedy Algorithm
# From src/engines/neural_engines.py:115-145

if self.reference_predict_fn is not None:
    # Step 1: Compute Q_base from reference model for all legal moves
    legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
    legal_actions = np.array(legal_actions, dtype=np.int32)
    legal_actions = np.expand_dims(legal_actions, axis=-1)
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    sequences_base = np.stack([tokenized_fen] * len(legal_actions))
    dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
    sequences_base = np.concatenate(
        [sequences_base, legal_actions, dummy_return_buckets],
        axis=1,
    )
    
    # Get reference model Q-values
    ref_return_buckets_log_probs = self.reference_predict_fn(sequences_base)[:, -1]
    ref_return_buckets_probs = np.exp(ref_return_buckets_log_probs)
    q_base = np.inner(ref_return_buckets_probs, self._return_buckets_values)
    
    # Step 2: Q_online is already computed as win_probs
    q_online = win_probs
    
    # Step 3-4: Identify top-K moves according to Q_base
    k = min(self.constrained_greedy_k, len(sorted_legal_moves))
    topk_indices = np.argsort(q_base)[-k:]  # Top-K indices
    
    # Step 5-6: Among top-K candidates, pick the one with highest Q_online
    candidate_q_online = q_online[topk_indices]
    best_candidate_idx = np.argmax(candidate_q_online)
    best_index = topk_indices[best_candidate_idx]
    
    return sorted_legal_moves[best_index]
```

**Key Changes**:
1. **Compute Q_base**: Get Q-values from the reference (base) model for all legal moves
2. **Select top-K**: Identify the K highest-ranked moves according to Q_base (default K=5)
3. **Re-rank within top-K**: Among these candidates, select the move with highest Q_online
4. **Exclude intruders**: Moves outside the base model's top-K cannot be selected, even if they have high Q_online

## 5. Q-Value Preservation (DPO Loss Fix)

To address the root cause (pessimistic Q-value drift), we added **Q-value preservation regularization** to the DPO loss:

```python
# From src/dpo_loss.py:247-288

# CRITICAL FIX: Add Q-value preservation regularization to prevent pessimistic drift
# The problem: Without normalization, the model can satisfy DPO by making all Q-values
# smaller (as long as chosen > rejected). This causes:
# 1. Pessimistic drift (all Q-values trend downward)
# 2. Wrong rankings (moves not in training pairs can rank above chosen moves)

# Recover raw Q-values from temperature-scaled log probabilities
# log_pi = Q / temperature, so Q = log_pi * temperature
online_chosen_q = log_pi_chosen * temperature
ref_chosen_q = log_ref_chosen * temperature

# Q-value preservation loss: penalize deviation from reference Q-values
# This prevents the model from making all Q-values smaller
# We use L2 loss on the chosen move's Q-value to keep it close to reference
q_preservation_loss_chosen = jnp.mean((online_chosen_q - ref_chosen_q) ** 2)

# ENHANCEMENT: Also constrain rejected move to prevent overall scale drift
# If rejected drifts too far, it can cause other moves to jump above chosen
online_rejected_q = log_pi_rejected * temperature
ref_rejected_q = log_ref_rejected * temperature
q_preservation_loss_rejected = jnp.mean((online_rejected_q - ref_rejected_q) ** 2)

# Combined preservation loss (weight chosen more heavily since it's the target)
q_preservation_loss = 0.7 * q_preservation_loss_chosen + 0.3 * q_preservation_loss_rejected

# Total loss with Q-value preservation and optional KL anchor
total_loss = dpo_loss_value + q_preservation_weight * q_preservation_loss
```

**How This Reduces Pessimistic Drift**:

The Q-value preservation loss penalizes the model when the chosen move's Q-value deviates too far from the reference model's Q-value. This prevents the model from satisfying the DPO constraint (chosen > rejected) by simply making all Q-values smaller. By constraining both chosen and rejected moves (70% weight on chosen, 30% on rejected), we stabilize the overall Q-value scale while still allowing the model to learn preferences.

## 6. How the Fix Resolves the Example

Returning to our concrete FEN example:

```
1R6/3q1ppk/6n1/b2p2Pp/2pP2b1/p1P5/P1B1rPPB/2Q2NK1 b - - 0 20
```

### Standard Greedy (Without Fix)
- Would select **e2c2** because it has the highest Q_online (0.4622)
- ❌ **Problem**: This is a catastrophic failure - a rank-14 move beats Stockfish's choice

### Constrained Greedy (With Fix)
1. **Compute Q_base** from reference model:
   - Top-5 moves: `d7e7 (0.5831), d7e6 (0.5816), g4f5 (0.5739), a5c7 (0.5675), h5h4 (0.5528)`
   - e2c2 is rank 14 (Q_base = 0.3755) - **not in top-K**

2. **Candidate set**: Only the top-5 moves from Q_base are considered:
   - `{d7e7, d7e6, g4f5, a5c7, h5h4}`

3. **Re-rank within candidates** using Q_online:
   - Among these 5, **a5c7** has the highest Q_online (0.4426)
   - ✅ **Selected**: a5c7 (Stockfish's chosen move)

4. **e2c2 is excluded**: Since it's not in the base model's top-K, it never enters the candidate set, regardless of its Q_online value.

### Summary of Fix Benefits

- ✅ **Intruder moves outside base top-K cannot be played anymore**: e2c2 (rank 14) is excluded from consideration
- ✅ **Q-preservation keeps chosen/rejected values close to reference**: Prevents overall pessimistic drift
- ✅ **Together, they fix the original failure mode**: The engine now reliably prefers Stockfish-chosen moves (a5c7) in problematic positions

## 7. Summary

### Problem
DPO training caused **Q-value drift** and **intruder moves** beating Stockfish's choice under greedy decoding. In the example, e2c2 (rank 14 in base model) jumped to rank 1 in the trained model, beating Stockfish's chosen move a5c7.

### Fix
Two complementary mechanisms:

1. **Q-value preservation regularization** in DPO loss: Penalizes deviation of chosen/rejected Q-values from reference, preventing pessimistic drift
2. **Constrained greedy decoding**: Uses base model's top-K moves as a trust region, only re-ranking within that set using Q_online

### Result
The engine now reliably prefers Stockfish-chosen moves in problematic positions like the example FEN. Intruder moves like e2c2 can no longer be selected because they are excluded from the candidate set (not in base top-K). The fix ensures that only moves the base model considered reasonable can be selected, while still allowing the trained model to learn preferences within that trusted set.

