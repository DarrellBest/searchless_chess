"""Dynamic DPO pair generation with hard negative mining.

This module generates DPO pairs on-the-fly during training by finding "intruder"
moves that have higher Q-values than the preferred move. This ensures the
preferred move becomes the global maximum, not just relatively better.

The algorithm (called per batch of positions):
1. For each position in batch:
   - Parse FEN to get all legal moves
   - Skip if too many legal moves (>80, too expensive)
   - Batch-compute Q-values for ALL legal moves using current model (EMA params)
   - Get Q-value of preferred move: Q_preferred
   - Find ALL intruders:
     * Intruder = legal move where Q(move) > Q_preferred AND move NOT IN good_moves
   - Sort intruders by Q-value (highest first)
   - Create one DPO pair for EVERY intruder (no limit if max_pairs_per_position <= 0):
     * If 15 intruders found → create 15 pairs from this position
   - Each pair: (chosen=preferred_move, rejected=intruder_i)
2. Return flat list of all pairs from all positions in batch

This self-corrects as training progresses, targeting current mistakes.
"""

import chess
import jax.numpy as jnp
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import utils


def decode_position_to_fen(position_tokens: np.ndarray) -> str:
  """Converts tokenized position back to FEN string.

  Args:
    position_tokens: [seq_len] Tokenized position (without action/return).

  Returns:
    FEN string for the position.
  """
  # Decode tokens back to FEN
  # The tokenizer encodes FEN strings character by character
  fen_chars = [tokenizer.VOCAB[token] for token in position_tokens if token < len(tokenizer.VOCAB)]
  fen = ''.join(fen_chars).strip()
  return fen


def get_legal_move_actions(fen: str) -> list[int]:
  """Gets action indices for all legal moves in a position.

  Args:
    fen: FEN string for the position.

  Returns:
    List of action indices (as defined in utils.MOVE_TO_ACTION).
  """
  board = chess.Board(fen)
  sorted_legal_moves = sorted(board.legal_moves, key=lambda m: m.uci())
  legal_actions = [utils.MOVE_TO_ACTION[m.uci()] for m in sorted_legal_moves]
  return legal_actions


def compute_q_values_batch(
    params,
    predictor,
    positions: np.ndarray,
    actions: np.ndarray,
    return_buckets_values: np.ndarray,
) -> np.ndarray:
  """Computes Q-values for a batch of (position, action) pairs.

  Args:
    params: Model parameters.
    predictor: Transformer predictor.
    positions: [batch_size, seq_len] Tokenized positions.
    actions: [batch_size] Action indices.
    return_buckets_values: [n_buckets] Return bucket values for Q computation.

  Returns:
    [batch_size] Q-values for each (position, action) pair.
  """
  batch_size = len(actions)

  # Create sequences: [position, action, dummy_return]
  dummy_returns = np.zeros((batch_size, 1), dtype=np.int32)
  action_tokens = actions[:, None]
  sequences = np.concatenate([positions, action_tokens, dummy_returns], axis=1)

  # Get return distributions
  return_logprobs = predictor.predict(
      params=params,
      targets=jnp.array(sequences),
      rng=None
  )[:, -2]  # [batch_size, n_buckets]

  # Convert to Q-values
  return_probs = jnp.exp(return_logprobs)
  q_values = jnp.sum(return_probs * return_buckets_values, axis=-1)

  return np.array(q_values)


def generate_dynamic_pairs(
    params,
    predictor,
    positions: np.ndarray,
    fen_strings: list[str],
    preferred_moves: np.ndarray,
    return_buckets_values: np.ndarray,
    good_moves_list: list[list[int]] = None,
    max_pairs_per_position: int = 5,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[int], list[int]]:
  """Generates dynamic DPO pairs by finding intruder moves.

  For each position:
  1. Compute Q-values for all legal moves
  2. Find intruders: moves with Q > Q(preferred) AND NOT in good_moves list
  3. Create pairs: (chosen=preferred_move, rejected=intruder)

  Args:
    params: Model parameters.
    predictor: Transformer predictor.
    positions: [batch_size, seq_len] Tokenized positions.
    fen_strings: [batch_size] FEN strings for positions.
    preferred_moves: [batch_size] Preferred move indices (from Stockfish/Lichess).
    return_buckets_values: [n_buckets] Return bucket values.
    good_moves_list: [batch_size] List of lists containing good move indices.
                     Intruders will NOT be selected from these moves.
    max_pairs_per_position: Maximum pairs to generate per position (0 or negative = unlimited).
    verbose: Whether to print diagnostics.

  Returns:
    Tuple of (positions_list, chosen_moves_list, rejected_moves_list) where
    each list contains the generated DPO pairs.
  """
  new_positions = []
  new_chosen = []
  new_rejected = []

  stats = {
      'total_positions': 0,
      'positions_with_intruders': 0,
      'total_intruders': 0,
      'total_pairs_generated': 0,
  }

  for i in range(len(positions)):
    stats['total_positions'] += 1

    # Get FEN string and position tokens
    fen = fen_strings[i]
    pos_tokens = positions[i]

    # Remove padding zeros at the end
    pos_tokens_clean = pos_tokens[pos_tokens > 0]

    # Get legal moves from FEN
    try:
      legal_actions = get_legal_move_actions(fen)
    except Exception as e:
      if verbose:
        print(f"Warning: Could not get legal moves for position {i}: {e}")
        print(f"  FEN: {fen}")
      continue

    # Skip if too many legal moves (expensive to evaluate)
    if len(legal_actions) > 80:
      if verbose:
        print(f"Skipping position {i}: {len(legal_actions)} legal moves (too many)")
      continue

    # Compute Q-values for all legal moves
    # Create batch of (position, action) pairs
    # Need to pad back to consistent length for batching
    # Max position length = SEQUENCE_LENGTH (77) since we add action (1) + return (1) = 79 total
    max_pos_len = tokenizer.SEQUENCE_LENGTH  # 77 tokens for position
    padded_pos = np.zeros(max_pos_len, dtype=np.int32)
    padded_pos[:min(len(pos_tokens_clean), max_pos_len)] = pos_tokens_clean[:max_pos_len]

    batch_positions = np.tile(padded_pos[None, :], (len(legal_actions), 1))
    batch_actions = np.array(legal_actions, dtype=np.int32)

    q_values = compute_q_values_batch(
        params,
        predictor,
        batch_positions,
        batch_actions,
        return_buckets_values,
    )

    # Get Q-value of preferred move
    preferred_action = preferred_moves[i]
    if preferred_action not in legal_actions:
      if verbose:
        print(f"Warning: Preferred move {preferred_action} not in legal moves for position {i}")
      continue

    preferred_idx = legal_actions.index(preferred_action)
    q_preferred = q_values[preferred_idx]

    # Get list of good moves for this position (if provided)
    good_moves = set(good_moves_list[i]) if good_moves_list else set()

    # Find intruders: legal moves with Q > Q(preferred) AND NOT in good_moves
    intruders = []
    for j, (action, q) in enumerate(zip(legal_actions, q_values)):
      # Skip if it's the preferred move itself
      if action == preferred_action:
        continue

      # Skip if it's in the good moves list
      if action in good_moves:
        if verbose:
          print(f"    Skipping move {action}: Q={q:.4f} (in good moves list)")
        continue

      # This is a bad intruder!
      if q > q_preferred:
        intruders.append((action, q))

    if len(intruders) > 0:
      stats['positions_with_intruders'] += 1
      stats['total_intruders'] += len(intruders)

      # Sort intruders by Q-value (highest first)
      intruders.sort(key=lambda x: x[1], reverse=True)

      # Apply limit only if max_pairs_per_position > 0
      if max_pairs_per_position > 0:
        intruders = intruders[:max_pairs_per_position]

      # Create pairs for each intruder
      for intruder_action, intruder_q in intruders:
        new_positions.append(pos_tokens_clean)
        new_chosen.append(preferred_action)
        new_rejected.append(intruder_action)
        stats['total_pairs_generated'] += 1

        if verbose:
          print(f"Position {i}: Created pair - "
                f"Q(preferred={preferred_action})={q_preferred:.4f} > "
                f"Q(intruder={intruder_action})={intruder_q:.4f}")

  # Always print stats if we processed any positions
  if stats['total_positions'] > 0:
    print(f"\nDynamic pair generation stats:")
    print(f"  Positions processed: {stats['total_positions']}")
    print(f"  Positions with intruders: {stats['positions_with_intruders']}")
    print(f"  Total intruders found: {stats['total_intruders']}")
    print(f"  Pairs generated: {stats['total_pairs_generated']}")
    if stats['positions_with_intruders'] > 0:
      avg_intruders = stats['total_intruders'] / stats['positions_with_intruders']
      print(f"  Avg intruders per position: {avg_intruders:.2f}")

    # Debug info if no pairs were generated
    if verbose and stats['total_pairs_generated'] == 0:
      print(f"\n  DEBUG: No pairs generated")
      print(f"  This could mean:")
      print(f"    1. Position decoding is failing (check errors above)")
      print(f"    2. Preferred moves already have highest Q-values (good!)")
      print(f"    3. All positions were skipped (too many legal moves)")

  return new_positions, new_chosen, new_rejected


def augment_batch_with_dynamic_pairs(
    params,
    predictor,
    positions: np.ndarray,
    chosen_moves: np.ndarray,
    rejected_moves: np.ndarray,
    return_buckets_values: np.ndarray,
    good_moves_list: list[list[int]] = None,
    augmentation_ratio: float = 0.5,
    max_pairs_per_position: int = 3,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Augments a training batch with dynamically generated pairs.

  Args:
    params: Model parameters.
    predictor: Transformer predictor.
    positions: [batch_size, seq_len] Original positions.
    chosen_moves: [batch_size] Original chosen moves.
    rejected_moves: [batch_size] Original rejected moves.
    return_buckets_values: [n_buckets] Return bucket values.
    augmentation_ratio: Fraction of batch to use for dynamic pair generation.
    max_pairs_per_position: Maximum pairs per position.
    verbose: Whether to print diagnostics.

  Returns:
    Augmented (positions, chosen_moves, rejected_moves) arrays.
  """
  batch_size = len(positions)
  n_dynamic = int(batch_size * augmentation_ratio)

  if n_dynamic == 0:
    return positions, chosen_moves, rejected_moves

  # Select random subset for dynamic pair generation
  indices = np.random.choice(batch_size, size=n_dynamic, replace=False)
  subset_positions = positions[indices]
  subset_preferred = chosen_moves[indices]
  subset_good_moves = [good_moves_list[i] for i in indices] if good_moves_list else None

  # Generate dynamic pairs
  new_positions, new_chosen, new_rejected = generate_dynamic_pairs(
      params=params,
      predictor=predictor,
      positions=subset_positions,
      preferred_moves=subset_preferred,
      return_buckets_values=return_buckets_values,
      good_moves_list=subset_good_moves,
      max_pairs_per_position=max_pairs_per_position,
      verbose=verbose,
  )

  if len(new_positions) == 0:
    return positions, chosen_moves, rejected_moves

  # Pad new positions to same length as original
  max_len = positions.shape[1]
  padded_positions = []
  for pos in new_positions:
    if len(pos) < max_len:
      padded = np.zeros(max_len, dtype=positions.dtype)
      padded[:len(pos)] = pos
      padded_positions.append(padded)
    else:
      padded_positions.append(pos[:max_len])

  # Concatenate with original batch
  all_positions = np.vstack([positions, np.array(padded_positions)])
  all_chosen = np.concatenate([chosen_moves, np.array(new_chosen)])
  all_rejected = np.concatenate([rejected_moves, np.array(new_rejected)])

  return all_positions, all_chosen, all_rejected
