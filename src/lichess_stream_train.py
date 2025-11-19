"""Streaming DPO training with dynamic pair generation.

Goal: Train model by streaming positions from Lichess, generating dynamic preference pairs
on-the-fly based on current model mistakes.

Step-by-Step Execution:
1. Stream positions from Lichess database (compressed zstd file)
2. For each position - Parse and batch (FEN-4 to FEN-6, tokenize, extract preferred/good moves)
3. When batch reaches size - Generate pairs dynamically based on current model intruders
4. Accumulate generated pairs until reaching pairs_per_update threshold (default: 1000)
5. When threshold reached - Train with DPO loss + Q-value anchoring, then clear accumulator
6. Always clear position batch (even on error) - positions never reused (ephemeral training data)
7. Checkpoint saving every N pairs
8. Safety checks (KL divergence threshold, max positions)

Key Properties:
- Ephemeral data: Each position used exactly once, never cached or reused
- Dynamic pairs: Pairs depend on current model state (what it's currently wrong about)
- Multiple pairs per position: Unlimited - one pair for each intruder (could be 0, could be 20+)
- Selective training: Only train on positions where model makes mistakes (has intruders)
- Smart filtering: Preserves Stockfish's acceptable alternatives (doesn't penalize good moves)
- Q-value anchoring: Prevents absolute Q-value drift while learning relative preferences
- Adaptive curriculum: As model improves, fewer intruders found, pairs become harder
"""

import logging
import os
import shutil
import time

from absl import app
from absl import flags
from jax import random as jrandom
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import dynamic_dpo_generator
from searchless_chess.src import lichess_position_cache
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils

# Training hyperparameters
_BASE_MODEL = flags.DEFINE_string('base_model', '9M', 'Base model to fine-tune.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'Number of positions per batch for pair generation.')
_PAIRS_PER_UPDATE = flags.DEFINE_integer('pairs_per_update', 1000, 'Number of pairs to accumulate before training update.')
_TRAIN_MINI_BATCH = flags.DEFINE_integer('train_mini_batch', 128, 'Mini-batch size for gradient accumulation during training.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 2e-6, 'Learning rate.')
_BETA = flags.DEFINE_float('beta', 10.0, 'DPO KL penalty coefficient.')
_TEMPERATURE = flags.DEFINE_float('temperature', 1.0, 'Temperature for action selection.')
_MAX_GRAD_NORM = flags.DEFINE_float('max_grad_norm', 1.0, 'Maximum gradient norm.')
_MAX_KL_DIVERGENCE = flags.DEFINE_float('max_kl_divergence', 0.5, 'Stop if KL exceeds this.')
_ANCHOR_WEIGHT = flags.DEFINE_float('anchor_weight', 1.0, 'Q-value anchoring weight.')
_DYNAMIC_PAIRS_MAX_PER_POS = flags.DEFINE_integer('dynamic_pairs_max_per_pos', -1, 'Max pairs per position (-1=unlimited).')
_CHECKPOINT_EVERY = flags.DEFINE_integer('checkpoint_every', 1000, 'Checkpoints saved after each training update.')
_MAX_POSITIONS = flags.DEFINE_integer('max_positions', -1, 'Max positions to process (-1=unlimited).')
_LICHESS_DB_PATH = flags.DEFINE_string('lichess_db_path', '../data/lichess_db_eval.jsonl.zst', 'Path to Lichess DB.')

_EMA_DECAY = 0.999


def dpo_loss_fn(
    params,
    reference_params,
    positions,
    chosen_moves,
    rejected_moves,
    predictor,
    beta,
    temperature,
    anchor_weight=1.0,
):
  """DPO loss with Q-value anchoring."""
  batch_size = len(chosen_moves)

  # Create sequences
  dummy_returns = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  chosen_actions = chosen_moves[:, None]
  rejected_actions = rejected_moves[:, None]

  chosen_sequences = jnp.concatenate([positions, chosen_actions, dummy_returns], axis=1)
  rejected_sequences = jnp.concatenate([positions, rejected_actions, dummy_returns], axis=1)

  # Get return distributions
  chosen_return_logprobs = predictor.predict(params=params, targets=chosen_sequences, rng=None)[:, -2]
  rejected_return_logprobs = predictor.predict(params=params, targets=rejected_sequences, rng=None)[:, -2]
  ref_chosen_return_logprobs = predictor.predict(params=reference_params, targets=chosen_sequences, rng=None)[:, -2]
  ref_rejected_return_logprobs = predictor.predict(params=reference_params, targets=rejected_sequences, rng=None)[:, -2]

  # Convert to Q-values
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
  return_values = jnp.array(return_buckets_values, dtype=jnp.float32)

  chosen_q = jnp.sum(jnp.exp(chosen_return_logprobs) * return_values, axis=-1)
  rejected_q = jnp.sum(jnp.exp(rejected_return_logprobs) * return_values, axis=-1)
  ref_chosen_q = jnp.sum(jnp.exp(ref_chosen_return_logprobs) * return_values, axis=-1)
  ref_rejected_q = jnp.sum(jnp.exp(ref_rejected_return_logprobs) * return_values, axis=-1)

  # DPO loss
  pi_logratios = (chosen_q - rejected_q) / temperature
  ref_logratios = (ref_chosen_q - ref_rejected_q) / temperature
  logits = beta * (pi_logratios - ref_logratios)
  dpo_loss = -jnp.mean(jax.nn.log_sigmoid(logits))

  # Q-value anchoring
  anchor_loss = jnp.square(chosen_q - ref_chosen_q).mean() + jnp.square(rejected_q - ref_rejected_q).mean()
  total_loss = dpo_loss + anchor_weight * anchor_loss

  # Metrics
  accuracy = jnp.mean(pi_logratios > 0)
  q_all = jnp.concatenate([chosen_q, rejected_q])
  q_ref_all = jnp.concatenate([ref_chosen_q, ref_rejected_q])

  metrics = {
      'loss': total_loss,
      'dpo_loss': dpo_loss,
      'anchor_loss': anchor_loss,
      'accuracy': accuracy,
      'kl_divergence_mean': jnp.mean(jnp.abs(chosen_q - ref_chosen_q) + jnp.abs(rejected_q - ref_rejected_q)),
      'q_chosen_mean': jnp.mean(chosen_q),
      'q_rejected_mean': jnp.mean(rejected_q),
      'q_diff_mean': jnp.mean(chosen_q - rejected_q),
      'q_all_mean': jnp.mean(q_all),
      'q_all_std': jnp.std(q_all),
      'q_ref_all_mean': jnp.mean(q_ref_all),
      'q_ref_all_std': jnp.std(q_ref_all),
      'logits_mean': jnp.mean(logits),
  }

  return total_loss, metrics


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('='*80)
  logging.info('Streaming DPO Training with Dynamic Pairs')
  logging.info('='*80)
  logging.info(f'Base model: {_BASE_MODEL.value}')
  logging.info(f'Batch size (positions): {_BATCH_SIZE.value}')
  logging.info(f'Pairs per update: {_PAIRS_PER_UPDATE.value}')
  logging.info(f'Learning rate: {_LEARNING_RATE.value}')
  logging.info(f'Beta: {_BETA.value}')
  logging.info(f'Anchor weight: {_ANCHOR_WEIGHT.value}')
  logging.info(f'Max pairs per position: {_DYNAMIC_PAIRS_MAX_PER_POS.value}')
  logging.info(f'Streaming from: {_LICHESS_DB_PATH.value}')
  logging.info('='*80)

  # Build model
  logging.info('\nBuilding model...')
  match _BASE_MODEL.value:
    case '9M':
      num_layers, embedding_dim, num_heads = 8, 256, 8
    case '136M':
      num_layers, embedding_dim, num_heads = 8, 1024, 8
    case '270M':
      num_layers, embedding_dim, num_heads = 16, 1024, 8
    case _:
      raise ValueError(f'Unknown model: {_BASE_MODEL.value}')

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=128,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)

  # Initialize
  rng = jrandom.PRNGKey(42)
  dummy_targets = np.ones((1, 1), dtype=np.uint32)
  initial_params = predictor.initial_params(rng=rng, targets=dummy_targets)

  # Load base model
  logging.info(f'Loading base model {_BASE_MODEL.value}...')
  base_checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{_BASE_MODEL.value}')
  params = training_utils.load_parameters(checkpoint_dir=base_checkpoint_dir, params=initial_params, step=-1)
  params_ema = params

  # Setup
  checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{_BASE_MODEL.value}_stream')
  os.makedirs(checkpoint_dir, exist_ok=True)

  optimizer = optax.adamw(learning_rate=_LEARNING_RATE.value)
  opt_state = optimizer.init(params)
  reference_params = params

  # Get return bucket values
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
  return_values_array = np.array(return_buckets_values, dtype=np.float32)

  # Training loop
  logging.info('\n=== Starting Streaming Training ===')
  logging.info('Process: Stream positions -> Parse/batch -> Generate dynamic pairs -> Train -> Clear batch')
  logging.info('Pairs generated: One pair per intruder (unlimited by default)')
  logging.info('='*80 + '\n')

  @jax.jit
  def compute_grads(params, ref_params, pos, chosen, rejected):
    """Compute gradients for a mini-batch."""
    def loss_fn(p):
      return dpo_loss_fn(p, ref_params, pos, chosen, rejected, predictor,
                        _BETA.value, _TEMPERATURE.value, _ANCHOR_WEIGHT.value)
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return grads, loss_val, metrics

  @jax.jit
  def apply_grads(params, params_ema, opt_state, grads):
    """Apply accumulated gradients."""
    grad_norm = optax.global_norm(grads)
    grads, _ = optax.clip_by_global_norm(_MAX_GRAD_NORM.value).update(grads, opt_state)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params_ema = jax.tree.map(
        lambda ema, new: _EMA_DECAY * ema + (1 - _EMA_DECAY) * new,
        params_ema,
        new_params,
    )
    return new_params, new_params_ema, new_opt_state, grad_norm

  batch_count = 0
  total_pairs = 0
  total_positions_processed = 0
  total_loss = 0.0
  total_kl = 0.0
  last_checkpoint_pairs = 0

  # Timing and progress tracking
  training_start_time = time.time()
  last_progress_time = training_start_time
  pairs_at_last_progress = 0

  # STEP 1: Stream positions and generate pairs on-the-fly
  # Initialize batch accumulators (cleared after each batch)
  position_batch = []
  fen_batch = []
  preferred_batch = []
  good_moves_batch = []

  # Accumulate pairs until reaching pairs_per_update threshold
  accumulated_positions = []
  accumulated_chosen = []
  accumulated_rejected = []

  for position_idx, position_data in enumerate(lichess_position_cache.stream_lichess_positions(
      _LICHESS_DB_PATH.value,
      max_positions=_MAX_POSITIONS.value if _MAX_POSITIONS.value > 0 else None
  )):
    # STEP 2: For each position - Parse and batch
    # Convert FEN-4 to FEN-6, tokenize, extract moves
    try:
      # Parse position: FEN-4 -> FEN-6 (append 0 20 for halfmove/fullmove)
      fen4 = position_data['fen']
      fen6 = lichess_position_cache.fen4_to_fen6(fen4)

      # Extract moves:
      # - preferred_move: first move of first PV (Stockfish's top choice)
      # - good_moves: first move of ALL PVs (acceptable alternatives)
      preferred_move, good_moves = lichess_position_cache.extract_moves_from_evals(position_data['evals'])

      # Tokenize FEN string to numpy array (length 77)
      tokenized_fen = tokenizer.tokenize(fen6).astype(np.int32)

      # Convert moves to action indices via MOVE_TO_ACTION dictionary
      preferred_action = utils.MOVE_TO_ACTION[preferred_move]
      good_actions = [utils.MOVE_TO_ACTION[m] for m in good_moves]

      # Append to batch
      position_batch.append(tokenized_fen)
      fen_batch.append(fen6)
      preferred_batch.append(preferred_action)
      good_moves_batch.append(good_actions)
      total_positions_processed += 1
    except Exception as e:
      if position_idx < 5:  # Debug first few
        logging.error(f'ERROR parsing position {position_idx}: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
      else:
        logging.warning(f'Skipping position {position_idx} (parse error: {type(e).__name__})')
      continue

    # STEP 3: When batch reaches size - Generate pairs dynamically
    if len(position_batch) >= _BATCH_SIZE.value:
      try:
        logging.info(f'Processing batch with {len(position_batch)} positions...')

        # Pad all tokenized positions to same length
        max_len = max(len(pos) for pos in position_batch)
        positions = np.zeros((len(position_batch), max_len), dtype=np.uint32)
        for i, pos in enumerate(position_batch):
          positions[i, :len(pos)] = pos

        preferred_moves = np.array(preferred_batch, dtype=np.int32)

        logging.info(f'Generating dynamic pairs...')
        # Call generate_dynamic_pairs():
        # For each position in batch:
        #   - Parse FEN to get all legal moves
        #   - Skip if too many legal moves (>80, too expensive)
        #   - Batch-compute Q-values for ALL legal moves using current model (EMA params)
        #   - Get Q-value of preferred move: Q_preferred
        #   - Find ALL intruders: Q(move) > Q_preferred AND move NOT IN good_moves
        #   - Sort intruders by Q-value (highest first)
        #   - Create one DPO pair for EVERY intruder (no limit if max_pairs_per_position=-1)
        # Returns flat list of all pairs from all positions in batch
        is_verbose = (batch_count < 3) or (batch_count % 100 == 0)

        new_positions, new_chosen, new_rejected = dynamic_dpo_generator.generate_dynamic_pairs(
            params=params_ema,
            predictor=predictor,
            positions=positions,
            fen_strings=fen_batch,
            preferred_moves=preferred_moves,
            return_buckets_values=return_values_array,
            good_moves_list=good_moves_batch,
            max_pairs_per_position=_DYNAMIC_PAIRS_MAX_PER_POS.value,
            verbose=is_verbose,
        )

        # STEP 4: If pairs were generated - Accumulate them
        if len(new_positions) > 0:
          # Add new pairs to accumulator
          accumulated_positions.extend(new_positions)
          accumulated_chosen.extend(new_chosen)
          accumulated_rejected.extend(new_rejected)

          # Calculate pairs/second
          current_time = time.time()
          elapsed_since_progress = current_time - last_progress_time
          pairs_since_progress = len(accumulated_chosen) - pairs_at_last_progress
          pairs_per_second = pairs_since_progress / elapsed_since_progress if elapsed_since_progress > 0 else 0

          # Progress update
          logging.info(
              f'Generated {len(new_positions)} pairs | '
              f'Accumulated: {len(accumulated_chosen)}/{_PAIRS_PER_UPDATE.value} | '
              f'Rate: {pairs_per_second:.1f} pairs/sec'
          )

          # Update progress tracking
          last_progress_time = current_time
          pairs_at_last_progress = len(accumulated_chosen)

        else:
          if is_verbose:
            logging.info(f'Batch {total_positions_processed//32}: No intruders, skipping')

      except Exception as e:
        logging.error(f'ERROR processing batch: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
      finally:
        # STEP 5: Always clear batch (even on error)
        # Discard all positions, FENs, moves from batch
        # Reset batch arrays to empty
        # Positions never reused (ephemeral training data)
        position_batch = []
        fen_batch = []
        preferred_batch = []
        good_moves_batch = []

      # STEP 4b: Train when we've accumulated enough pairs (check after clearing batch)
      if len(accumulated_chosen) >= _PAIRS_PER_UPDATE.value:
        try:
          num_pairs = len(accumulated_chosen)
          logging.info(f'\nTraining on {num_pairs} pairs using gradient accumulation (mini-batch={_TRAIN_MINI_BATCH.value})...')

          # Pad all pair positions to same length
          max_len = max(len(seq) for seq in accumulated_positions)
          all_positions = np.zeros((num_pairs, max_len), dtype=np.uint32)
          for i, seq in enumerate(accumulated_positions):
            all_positions[i, :len(seq)] = seq

          all_chosen = np.array(accumulated_chosen, dtype=np.int32)
          all_rejected = np.array(accumulated_rejected, dtype=np.int32)

          # Train in mini-batches with gradient accumulation
          mini_batch_size = _TRAIN_MINI_BATCH.value
          num_mini_batches = (num_pairs + mini_batch_size - 1) // mini_batch_size

          # Initialize accumulated gradients
          accumulated_grads = None
          total_loss_val = 0.0
          final_metrics = None

          for mb_idx in range(num_mini_batches):
            start_idx = mb_idx * mini_batch_size
            end_idx = min(start_idx + mini_batch_size, num_pairs)

            mb_positions = jnp.array(all_positions[start_idx:end_idx])
            mb_chosen = jnp.array(all_chosen[start_idx:end_idx])
            mb_rejected = jnp.array(all_rejected[start_idx:end_idx])

            # Compute gradients for this mini-batch
            grads, loss_val, metrics = compute_grads(
                params, reference_params,
                mb_positions, mb_chosen, mb_rejected
            )

            # Accumulate gradients
            if accumulated_grads is None:
              accumulated_grads = grads
            else:
              accumulated_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, grads)

            total_loss_val += float(loss_val)
            final_metrics = metrics

            if (mb_idx + 1) % 5 == 0 or mb_idx == num_mini_batches - 1:
              logging.info(f'  Mini-batch {mb_idx + 1}/{num_mini_batches} processed')

          # Average accumulated gradients
          accumulated_grads = jax.tree.map(lambda g: g / num_mini_batches, accumulated_grads)
          avg_loss = total_loss_val / num_mini_batches

          # Apply accumulated gradients
          params, params_ema, opt_state, grad_norm = apply_grads(
              params, params_ema, opt_state, accumulated_grads
          )

          total_loss += avg_loss
          total_kl += float(final_metrics['kl_divergence_mean'])
          batch_count += 1
          total_pairs += num_pairs

          # Clear accumulator after training
          accumulated_positions = []
          accumulated_chosen = []
          accumulated_rejected = []

          # Reset progress tracking for next accumulation cycle
          last_progress_time = time.time()
          pairs_at_last_progress = 0

          # STEP 7: Safety checks - If KL_divergence > threshold: stop training (model diverging)
          if float(final_metrics['kl_divergence_mean']) > _MAX_KL_DIVERGENCE.value:
            logging.warning(f'\nKL divergence ({final_metrics["kl_divergence_mean"]:.4f}) exceeded threshold!')
            logging.warning('Stopping training.')
            break

          # Progress logging
          cumulative_avg_loss = total_loss / batch_count
          cumulative_avg_kl = total_kl / batch_count
          logging.info(
              f'Positions: {total_positions_processed:,} | Pairs: {total_pairs:,} | Updates: {batch_count} | '
              f'Loss: {cumulative_avg_loss:.4f} (DPO: {final_metrics["dpo_loss"]:.4f}, Anchor: {final_metrics["anchor_loss"]:.4f}) | '
              f'KL: {cumulative_avg_kl:.4f} | Acc: {final_metrics["accuracy"]:.2%} | '
              f'Q_mean: {final_metrics["q_all_mean"]:.4f} (std: {final_metrics["q_all_std"]:.4f})'
          )

          # STEP 6: Checkpoint saving after every training update
          # Save both regular params and EMA params
          # Checkpoint directory named by total pair count (e.g., 1000, 2000, 3000)
          checkpoint_name = f'{total_pairs}'
          logging.info(f'Saving checkpoint: {checkpoint_name}')

          step_dir = os.path.join(checkpoint_dir, checkpoint_name)
          os.makedirs(step_dir, exist_ok=True)

          training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params,
                                        step=checkpoint_name, use_ema_params=False)
          training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params_ema,
                                        step=checkpoint_name, use_ema_params=True)

          last_checkpoint_pairs = total_pairs
          logging.info('Checkpoint saved!\n')

        except Exception as e:
          logging.error(f'ERROR during training: {type(e).__name__}: {e}')
          import traceback
          traceback.print_exc()
          # Clear accumulator on error to prevent repeated OOM
          accumulated_positions = []
          accumulated_chosen = []
          accumulated_rejected = []

  # Final checkpoint
  if total_pairs > last_checkpoint_pairs:
    checkpoint_name = f'{total_pairs}'
    logging.info(f'\nSaving final checkpoint: {checkpoint_name}')
    step_dir = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(step_dir, exist_ok=True)

    training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params,
                                   step=checkpoint_name, use_ema_params=False)
    training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params_ema,
                                   step=checkpoint_name, use_ema_params=True)

  # Summary with timing
  total_training_time = time.time() - training_start_time
  hours = int(total_training_time // 3600)
  minutes = int((total_training_time % 3600) // 60)
  seconds = int(total_training_time % 60)

  logging.info('\n' + '='*80)
  logging.info('Training Complete!')
  logging.info('='*80)
  logging.info(f'Total time: {hours}h {minutes}m {seconds}s ({total_training_time:.1f}s)')
  logging.info(f'Total positions processed: {total_positions_processed:,}')
  logging.info(f'Total pairs trained: {total_pairs:,}')
  logging.info(f'Total training updates: {batch_count:,}')
  if batch_count > 0:
    logging.info(f'Average loss: {total_loss/batch_count:.4f}')
    logging.info(f'Average KL: {total_kl/batch_count:.4f}')
  if total_pairs > 0 and total_training_time > 0:
    avg_pairs_per_sec = total_pairs / total_training_time
    logging.info(f'Average generation rate: {avg_pairs_per_sec:.2f} pairs/sec')
  logging.info(f'Checkpoints saved to: {checkpoint_dir}')
  logging.info('='*80)
  logging.info('\nTraining Progression Example:')
  logging.info('Position 1: 5 intruders → 5 DPO pairs')
  logging.info('Position 2: 0 intruders → 0 pairs (model already correct)')
  logging.info('Position 3: 12 intruders → 12 DPO pairs')
  logging.info('Position 4: 1 intruder → 1 pair')
  logging.info('Batch total: 18 pairs → train on all 18, update model once')
  logging.info('='*80)


if __name__ == '__main__':
  app.run(main)
