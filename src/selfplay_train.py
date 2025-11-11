"""Self-play training script with DPO (Direct Preference Optimization)."""

from collections.abc import Sequence
import copy
import functools
import logging as python_logging
import os
import warnings

from absl import app
from absl import flags
from absl import logging
import chess
import haiku as hk
import jax

# Suppress harmless warnings and verbose logging
warnings.filterwarnings('ignore', message='.*sharding.*')
warnings.filterwarnings('ignore', message='.*Conversion for.*PositionalSharding.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/XLA warnings
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Suppress TPU warnings

# Suppress verbose library logging (keep training script messages)
python_logging.getLogger('orbax.checkpoint').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.sharding').setLevel(python_logging.CRITICAL)
python_logging.getLogger('jax._src.xla_bridge').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.array_metadata_store').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.sharding_impls').setLevel(python_logging.CRITICAL)

from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import constants
from searchless_chess.src import dpo_generator
from searchless_chess.src import lichess_dpo_generator
from searchless_chess.src import dpo_loss
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import constants as engine_constants
from searchless_chess.src.engines import neural_engines


_BASE_MODEL = flags.DEFINE_enum(
    'base_model',
    '9M',
    ['9M', '136M', '270M'],
    'The base model to start from for self-play training.',
)

_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations',
    5,
    'Number of self-play training iterations (start small to test, then scale up).',
)

_GAMES_PER_ITERATION = flags.DEFINE_integer(
    'games_per_iteration',
    100,
    'Number of self-play games per iteration (100 games = ~600-1000 preference pairs).',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    8,
    'Batch size for training (small batch = more gradient steps per epoch).',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    1e-5,
    'Learning rate for Adam optimizer (CRITICAL: very small to avoid destroying base model knowledge).',
)

_GRADIENT_STEPS_PER_ITERATION = flags.DEFINE_integer(
    'gradient_steps_per_iteration',
    -1,
    'Number of gradient steps per self-play iteration. Set to -1 to use all available batches (1 full epoch).',
)

_STOCKFISH_DEPTH = flags.DEFINE_integer(
    'stockfish_depth',
    22,
    'Depth for Stockfish analysis (starts here, increases with curriculum). Depth 22 balances quality vs speed.',
)

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    16,
    'Number of parallel workers for Stockfish analysis.',
)

_EVAL_THRESHOLD = flags.DEFINE_float(
    'eval_threshold',
    1.0,
    'Minimum evaluation difference (in pawns) to create a preference pair. 1.0 = only real blunders (hanging pieces, major mistakes).',
)

_BETA = flags.DEFINE_float(
    'beta',
    0.1,
    'KL penalty coefficient for DPO loss. Recommended range: 0.1-0.5. Higher = stay closer to base model.',
)

_UPDATE_REF_EVERY = flags.DEFINE_integer(
    'update_ref_every',
    10,
    'Update reference model every N iterations (higher = more stable, keeps policy from drifting too far).',
)

_DPO_TEMPERATURE = flags.DEFINE_float(
    'dpo_temperature',
    1.0,
    'Temperature for DPO probability computation.',
)

_SAVE_FREQUENCY = flags.DEFINE_integer(
    'save_frequency',
    1,
    'How often to save checkpoints (in iterations).',
)

_RESUME = flags.DEFINE_boolean(
    'resume',
    False,
    'Resume training from latest checkpoint if available.',
)

def _load_base_model(model_name: str) -> tuple[hk.Params, transformer.TransformerConfig]:
  """Loads a pretrained base model.

  Args:
    model_name: Name of the model ('9M', '136M', or '270M').

  Returns:
    Tuple of (parameters, config).
  """
  logging.info(f'Loading base model: {model_name}')

  # Determine config based on model size
  num_return_buckets = 128

  if model_name == '9M':
    num_layers = 8
    embedding_dim = 256
    num_heads = 8
  elif model_name == '136M':
    num_layers = 8
    embedding_dim = 1024
    num_heads = 8
  else:  # 270M
    num_layers = 16
    embedding_dim = 1024
    num_heads = 8

  # Keep the original action-value architecture (predicting Q-value distributions)
  # We'll derive action probabilities from Q-values via softmax
  config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=num_return_buckets,  # Keep original: predict Q-value distributions
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  # Build predictor to get param structure
  predictor = transformer.build_transformer_predictor(config)
  dummy_params = predictor.initial_params(
      rng=jrandom.PRNGKey(0),
      targets=np.zeros((1, 1), dtype=np.uint32),
  )

  # Load checkpoint
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{model_name}',
  )

  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=dummy_params,
      step=6_400_000,
      use_ema_params=False,
  )

  logging.info(f'Successfully loaded {model_name} model from {checkpoint_dir}')
  return params, config


def _make_dpo_loss_fn(predictor, z_atoms, beta=0.1, temperature=1.0, kl_penalty=0.0):
  """Creates a DPO loss function with optional KL anchor.

  DPO directly optimizes the model to prefer better moves (from Stockfish)
  over worse moves (the model's own mistakes) without needing a separate
  reward model or value function.

  Args:
    predictor: The transformer predictor.
    z_atoms: Array of return bucket atoms (support values).
    beta: KL penalty coefficient (default 0.1).
    temperature: Temperature for action selection (default 1.0).
    kl_penalty: Weight for explicit KL anchor (default 0.0).

  Returns:
    Loss function that takes (online_params, reference_params, batch_data).
  """
  def loss_fn(online_params, reference_params, positions, chosen_moves, rejected_moves):
    """Computes DPO loss for preference pairs.

    Args:
      online_params: Current model parameters.
      reference_params: Reference model parameters (frozen).
      positions: [batch_size, seq_len] Tokenized positions.
      chosen_moves: [batch_size] Better move indices (Stockfish).
      rejected_moves: [batch_size] Worse move indices (model's moves).

    Returns:
      Tuple of (loss, metrics_dict).
    """
    return dpo_loss.dpo_loss(
        online_params=online_params,
        reference_params=reference_params,
        predictor=predictor,
        positions=positions,
        chosen_moves=chosen_moves,
        rejected_moves=rejected_moves,
        z_atoms=z_atoms,
        beta=beta,
        temperature=temperature,
        kl_penalty=kl_penalty,
    )

  return loss_fn


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting DPO self-play training')
  logging.info(f'Base model: {_BASE_MODEL.value}')
  logging.info(f'Iterations: {_NUM_ITERATIONS.value}')
  logging.info(f'Games per iteration: {_GAMES_PER_ITERATION.value}')
  logging.info(f'Stockfish depth: {_STOCKFISH_DEPTH.value}')
  logging.info(f'Eval threshold: {_EVAL_THRESHOLD.value} pawns')
  logging.info(f'DPO beta: {_BETA.value}')

  # Load base model
  params, config = _load_base_model(_BASE_MODEL.value)
  params_ema = copy.deepcopy(params)  # For checkpointing (fast EMA)
  reference_params = copy.deepcopy(params)  # For DPO reference (updated periodically)

  # Build predictor
  predictor = transformer.build_transformer_predictor(config)

  # Create optimizer
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(_LEARNING_RATE.value),
  )
  opt_state = optimizer.init(params)

  # Setup checkpoint directory
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{_BASE_MODEL.value}_selfplay',
  )
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Check for existing checkpoints to resume from (BEFORE sharding)
  start_iteration = 0
  if _RESUME.value and os.path.exists(checkpoint_dir):
    # Find latest checkpoint
    checkpoint_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()
    ]
    if checkpoint_dirs:
      latest_iteration = max(int(d) for d in checkpoint_dirs)
      logging.info(f'Found checkpoint at iteration {latest_iteration}')
      logging.info('Resuming training from checkpoint...')

      # Load params using training_utils (handles sharding correctly)
      params = training_utils.load_parameters(
          checkpoint_dir=checkpoint_dir,
          params=params,
          step=latest_iteration,
          use_ema_params=False,
      )

      # Load EMA params
      params_ema = training_utils.load_parameters(
          checkpoint_dir=checkpoint_dir,
          params=params_ema,
          step=latest_iteration,
          use_ema_params=True,
      )

      # Initialize reference params from EMA (for DPO)
      reference_params = copy.deepcopy(params_ema)

      # Load optimizer state (use raw checkpointer with restore_args)
      latest_checkpoint = os.path.join(checkpoint_dir, str(latest_iteration))
      checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

      # Create restore args to handle sharded checkpoints
      restore_args = ocp.checkpoint_utils.construct_restore_args(opt_state)
      opt_state = checkpointer.restore(
          os.path.join(latest_checkpoint, 'opt_state'),
          item=opt_state,
          restore_args=restore_args,
      )

      start_iteration = latest_iteration
      logging.info(f'Resumed from iteration {latest_iteration}')
      logging.info(f'Continuing training for {_NUM_ITERATIONS.value} more iterations')
    else:
      logging.info('No checkpoints found, starting from base model')
  elif _RESUME.value:
    logging.info('Resume flag set but no checkpoint directory found, starting from base model')

  # Setup sharding for distributed training (AFTER loading checkpoints)
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  sharding = jax.sharding.PositionalSharding(devices)
  sharding = sharding.reshape((jax.device_count(), 1))

  params = training_utils.replicate(params, sharding)
  params_ema = training_utils.replicate(params_ema, sharding)
  reference_params = training_utils.replicate(reference_params, sharding)
  opt_state = training_utils.replicate(opt_state, sharding)

  # Get return bucket values (support atoms) for Q-value computation
  num_return_buckets = 128
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
  # Convert to JAX array
  z_atoms = jnp.array(return_buckets_values, dtype=jnp.float32)

  # Create DPO loss and gradient functions
  loss_fn = _make_dpo_loss_fn(
      predictor,
      z_atoms,
      beta=_BETA.value,
      temperature=_DPO_TEMPERATURE.value,
      kl_penalty=0.0  # Start at 0.0, increase if drift occurs
  )

  # Wrapper to extract loss for gradient computation
  def loss_for_grad(params, reference_params, positions, chosen_moves, rejected_moves):
    loss, _ = loss_fn(params, reference_params, positions, chosen_moves, rejected_moves)
    return loss

  # Gradient wrt first argument (online_params)
  grad_fn = jax.value_and_grad(loss_for_grad, argnums=0)

  @jax.jit
  def update_step(params, params_ema, reference_params, opt_state, positions, chosen_moves, rejected_moves):
    """Single gradient update step with DPO."""
    # Compute loss and gradients using online params and reference params
    loss_val, grads = grad_fn(params, reference_params, positions, chosen_moves, rejected_moves)

    # Get full metrics (without gradient computation)
    _, metrics = loss_fn(params, reference_params, positions, chosen_moves, rejected_moves)

    # Apply gradients to online params
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Update fast EMA params (for checkpointing, decay=0.99)
    fast_ema_decay = 0.99
    params_ema = jax.tree.map(
        lambda ema, new: fast_ema_decay * ema + (1 - fast_ema_decay) * new,
        params_ema,
        params,
    )

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)

    return params, params_ema, opt_state, loss_val, grad_norm, metrics

  # Setup checkpoint manager
  checkpoint_manager = training_utils.get_checkpoint_manager(
      ckpt_frequency=_SAVE_FREQUENCY.value,
      max_to_keep=5,
      save_frequency=_SAVE_FREQUENCY.value,
      checkpoint_dir=checkpoint_dir,
  )

  # Warm-up phase: Train on cached preferences if starting fresh (not resuming)
  cache_dir = os.path.join(os.getcwd(), f'../checkpoints/{_BASE_MODEL.value}_selfplay')
  os.makedirs(cache_dir, exist_ok=True)
  preference_cache_file = os.path.join(cache_dir, 'preference_cache.json')

  if not _RESUME.value and os.path.exists(preference_cache_file):
    import json
    logging.info('\n=== Warm-up Phase: Training on Cached Preferences ===')

    # Load cached preferences
    with open(preference_cache_file, 'r') as f:
      cached_prefs = json.load(f)

    if cached_prefs:
      logging.info(f'Found {len(cached_prefs)} cached preference pairs from previous runs')
      logging.info('Training on cached data to bootstrap model before generating new games...')

      # Convert cached preferences to training format
      import numpy as np
      warm_up_positions = []
      warm_up_chosen = []
      warm_up_rejected = []

      for pref_tuple in cached_prefs:
        fen, chosen_move, rejected_move = pref_tuple
        # Tokenize position
        tokenized_pos = tokenizer.tokenize(fen)
        dummy_action = np.array([0], dtype=np.int32)
        dummy_return = np.array([0], dtype=np.int32)
        pos_seq = np.concatenate([tokenized_pos, dummy_action, dummy_return])

        # Convert moves to action indices using the global dictionary
        chosen_action = utils.MOVE_TO_ACTION[chosen_move]
        rejected_action = utils.MOVE_TO_ACTION[rejected_move]

        warm_up_positions.append(pos_seq)
        warm_up_chosen.append(chosen_action)
        warm_up_rejected.append(rejected_action)

      # Convert to numpy arrays
      warm_up_positions = np.array(warm_up_positions)
      warm_up_chosen = np.array(warm_up_chosen, dtype=np.int32)
      warm_up_rejected = np.array(warm_up_rejected, dtype=np.int32)

      num_warm_up = len(warm_up_positions)
      total_batches = (num_warm_up + _BATCH_SIZE.value - 1) // _BATCH_SIZE.value

      logging.info(f'Warm-up training: {total_batches} batches ({num_warm_up} examples)')

      # Train on cached data (1 epoch)
      batch_idx = 0
      for step in range(total_batches):
        start_idx = (batch_idx % total_batches) * _BATCH_SIZE.value
        end_idx = min(start_idx + _BATCH_SIZE.value, num_warm_up)

        # Create batch and move to GPU
        positions = jnp.array(warm_up_positions[start_idx:end_idx])
        chosen_moves = jnp.array(warm_up_chosen[start_idx:end_idx])
        rejected_moves = jnp.array(warm_up_rejected[start_idx:end_idx])
        batch_idx += 1

        # Shard data
        positions = jax.lax.with_sharding_constraint(positions, sharding)
        chosen_moves = jax.lax.with_sharding_constraint(chosen_moves, sharding)
        rejected_moves = jax.lax.with_sharding_constraint(rejected_moves, sharding)

        # Update parameters
        params, params_ema, opt_state, loss_val, grad_norm, metrics = update_step(
            params, params_ema, reference_params, opt_state, positions, chosen_moves, rejected_moves
        )

        if step % 50 == 0:
          logging.info(
              f'  Warm-up step {step}/{total_batches}: '
              f'loss={float(loss_val):.4f}, grad_norm={float(grad_norm):.4f}, '
              f'reward_acc={float(metrics["reward_accuracy"]):.3f}, '
              f'kl={float(metrics["kl_loss"]):.4f}'
          )

      logging.info(f'Warm-up complete! Bootstrapped model from {num_warm_up} cached preferences')

      # Save checkpoint after warm-up (iteration 0)
      logging.info('Saving warm-up checkpoint (iteration 0)...')
      checkpoint_manager.save(
          step=0,
          items=dict(
              params=params,
              params_ema=params_ema,
              opt_state=opt_state,
          ),
      )
      logging.info('Warm-up checkpoint saved')
    else:
      logging.info('No cached preferences found, starting fresh')
  elif _RESUME.value:
    logging.info('Resuming training - skipping warm-up phase')

  # Main DPO self-play training loop
  total_iterations = start_iteration + _NUM_ITERATIONS.value
  current_sf_depth = _STOCKFISH_DEPTH.value

  for iteration in range(start_iteration, total_iterations):
    logging.info(f'\n=== Iteration {iteration + 1}/{total_iterations} ===')

    # Create neural engine with current parameters
    # Un-shard params for inference
    local_params = jax.device_get(params_ema)

    # Use ActionValueEngine with greedy play (temperature=None) for deterministic games
    neural_engine = neural_engines.ActionValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=local_params,
            batch_size=1,
        ),
        temperature=None,  # Greedy play - always pick best move
    )

    # Generate self-play games and create preference pairs using DPO
    generator = dpo_generator.DPOSelfPlayGenerator(
        neural_engine=neural_engine,
        stockfish_depth=current_sf_depth,
        stockfish_time_limit=None,  # Only use depth, not time
        max_moves_per_game=200,
        eval_threshold=_EVAL_THRESHOLD.value,
        max_position_eval=3.0,
        temperature=1.0,
        num_workers=_NUM_WORKERS.value,
    )

    logging.info(f'Generating {_GAMES_PER_ITERATION.value} self-play games...')
    logging.info(f'Using Stockfish depth: {current_sf_depth}')

    # Path to openings directory
    openings_dir = os.path.join(os.getcwd(), '../data/chess-openings-master')

    # Cache file paths (defined earlier, reuse here)
    used_openings_file = os.path.join(cache_dir, 'used_openings.json')
    # preference_cache_file already defined before warm-up phase

    # Generate preference pairs with deduplication
    batch_iterator, sampled_opening_indices, new_preferences = generator.generate_batch(
        num_games=_GAMES_PER_ITERATION.value,
        batch_size=_BATCH_SIZE.value,
        openings_dir=openings_dir,
        used_openings_file=used_openings_file,
        preference_cache_file=preference_cache_file,
    )

    # Collect raw preference pairs (not pre-batched to save memory)
    all_positions_raw = []
    all_chosen_moves_raw = []
    all_rejected_moves_raw = []

    for positions, chosen_moves, rejected_moves in batch_iterator:
      # Unbatch and collect as individual examples
      for i in range(positions.shape[0]):
        all_positions_raw.append(positions[i])
        all_chosen_moves_raw.append(chosen_moves[i])
        all_rejected_moves_raw.append(rejected_moves[i])

    generator.close()

    if not all_positions_raw:
      logging.warning('No preference pairs generated, skipping training.')
      logging.warning('Model may have converged or eval_threshold is too high.')
      continue

    # Convert to numpy arrays (keep on CPU to save GPU memory)
    import numpy as np
    all_positions_raw = np.array(all_positions_raw)
    all_chosen_moves_raw = np.array(all_chosen_moves_raw)
    all_rejected_moves_raw = np.array(all_rejected_moves_raw)

    num_examples = len(all_positions_raw)
    total_batches = (num_examples + _BATCH_SIZE.value - 1) // _BATCH_SIZE.value

    logging.info(f'Collected {num_examples} preference pairs from {_GAMES_PER_ITERATION.value} games')

    # If gradient_steps is -1, use all available batches (1 full epoch)
    if _GRADIENT_STEPS_PER_ITERATION.value == -1:
      num_steps = total_batches
      logging.info(f'Training for {num_steps} steps (all available batches = 1 epoch)...')
    else:
      num_steps = _GRADIENT_STEPS_PER_ITERATION.value
      logging.info(f'Training for {num_steps} steps ({num_steps/total_batches:.2f} epochs)...')

    batch_idx = 0

    for step in range(num_steps):
      # Create batch on-the-fly and move to GPU
      start_idx = (batch_idx % total_batches) * _BATCH_SIZE.value
      end_idx = min(start_idx + _BATCH_SIZE.value, num_examples)

      # Convert numpy slice to JAX array (CPU -> GPU)
      positions = jnp.array(all_positions_raw[start_idx:end_idx])
      chosen_moves = jnp.array(all_chosen_moves_raw[start_idx:end_idx])
      rejected_moves = jnp.array(all_rejected_moves_raw[start_idx:end_idx])
      batch_idx += 1

      # Shard data
      positions = jax.lax.with_sharding_constraint(positions, sharding)
      chosen_moves = jax.lax.with_sharding_constraint(chosen_moves, sharding)
      rejected_moves = jax.lax.with_sharding_constraint(rejected_moves, sharding)

      # Update parameters with DPO
      params, params_ema, opt_state, loss_val, grad_norm, metrics = update_step(
          params, params_ema, reference_params, opt_state, positions, chosen_moves, rejected_moves
      )

      if step % 10 == 0:
        logging.info(
            f'  Step {step}/{num_steps}: '
            f'loss={float(loss_val):.4f}, grad_norm={float(grad_norm):.4f}, '
            f'reward_acc={float(metrics["reward_accuracy"]):.3f}, '
            f'reward_margin={float(metrics["reward_margin"]):.3f}, '
            f'kl={float(metrics["kl_loss"]):.4f}'
        )

    # Update cache files after successful training
    # 1. Update used openings cache
    if sampled_opening_indices:
      import json
      existing_used_openings = []
      if os.path.exists(used_openings_file):
        with open(used_openings_file, 'r') as f:
          existing_used_openings = json.load(f)
      existing_used_openings.extend(sampled_opening_indices)
      with open(used_openings_file, 'w') as f:
        json.dump(existing_used_openings, f)
      logging.info(f'Updated used openings cache: {len(existing_used_openings)} total openings used')

    # 2. Update preference cache
    if new_preferences:
      existing_preferences = []
      if os.path.exists(preference_cache_file):
        with open(preference_cache_file, 'r') as f:
          existing_preferences = json.load(f)
      # Convert set of tuples to list for JSON serialization
      new_prefs_list = [list(pref) for pref in new_preferences]
      existing_preferences.extend(new_prefs_list)
      with open(preference_cache_file, 'w') as f:
        json.dump(existing_preferences, f)
      logging.info(f'Updated preference cache: {len(existing_preferences)} total preference pairs cached')

    # Update reference model periodically
    if (iteration + 1) % _UPDATE_REF_EVERY.value == 0:
      logging.info('Updating reference model...')
      reference_params = jax.tree.map(lambda x: x, params_ema)

    # Progressive curriculum: increase Stockfish depth after iteration 5
    if iteration >= 5:
      current_sf_depth = min(current_sf_depth + 2, 30)

    # Save checkpoint
    if (iteration + 1) % _SAVE_FREQUENCY.value == 0:
      logging.info(f'Saving checkpoint for iteration {iteration + 1}')
      checkpoint_manager.save(
          step=iteration + 1,
          items=dict(
              params=params,
              params_ema=params_ema,
              opt_state=opt_state,
          ),
      )


  # Wait for all checkpoints to finish saving
  logging.info('Waiting for checkpoint finalization...')
  checkpoint_manager.wait_until_finished()

  logging.info('Self-play training complete!')
  logging.info(f'Final model saved to: {checkpoint_dir}')


if __name__ == '__main__':
  app.run(main)
