"""Self-play training script with Stockfish-based rewards."""

from collections.abc import Sequence
import copy
import functools
import logging as python_logging
import os
import warnings

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax

# Suppress harmless warnings and verbose logging
warnings.filterwarnings('ignore', message='.*sharding.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/XLA warnings
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Suppress TPU warnings

# Suppress verbose library logging (keep training script messages)
python_logging.getLogger('orbax.checkpoint').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.sharding').setLevel(python_logging.ERROR)
python_logging.getLogger('jax._src.xla_bridge').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.array_metadata_store').setLevel(python_logging.WARNING)

from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import constants
from searchless_chess.src import selfplay_generator
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
    10,
    'Number of self-play training iterations.',
)

_GAMES_PER_ITERATION = flags.DEFINE_integer(
    'games_per_iteration',
    20,
    'Number of self-play games per iteration.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size for training.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    1e-4,
    'Learning rate for Adam optimizer.',
)

_GRADIENT_STEPS_PER_ITERATION = flags.DEFINE_integer(
    'gradient_steps_per_iteration',
    100,
    'Number of gradient steps per self-play iteration.',
)

_STOCKFISH_TIME = flags.DEFINE_float(
    'stockfish_time',
    0.01,
    'Time limit for Stockfish evaluation (seconds).',
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


def _make_reward_loss_fn(predictor, return_buckets_values):
  """Creates a value-based loss function for Q-learning with rewards.

  Args:
    predictor: The transformer predictor.
    return_buckets_values: Array of return bucket values for computing Q-values.

  Returns:
    Loss function that takes params, sequences, and rewards.
  """
  def loss_fn(params, sequences, rewards):
    """Computes value-based loss: trains Q-values to match rewards.

    This is similar to Deep Q-Learning (DQN) where we train Q(s,a) to match
    observed returns. We weight the loss by reward magnitude to focus on
    important transitions.

    Args:
      params: Model parameters.
      sequences: [batch_size, seq_len] input sequences.
      rewards: [batch_size] target Q-values (rewards from Stockfish + outcomes).

    Returns:
      Scalar loss value.
    """
    # Get Q-value distribution predictions (log probabilities over return buckets)
    # Shape: [batch_size, seq_len, num_return_buckets]
    bucket_log_probs = predictor.predict(params=params, targets=sequences, rng=None)

    # Extract Q-value distribution for the action token (second to last position)
    # Shape: [batch_size, num_return_buckets]
    action_bucket_log_probs = bucket_log_probs[:, -2]
    action_bucket_probs = jnp.exp(action_bucket_log_probs)

    # Compute predicted Q-value (expected return)
    # Shape: [batch_size]
    predicted_q = jnp.dot(action_bucket_probs, return_buckets_values)

    # Target Q-values come from rewards (Stockfish eval + game outcome)
    target_q = rewards

    # Compute TD error
    td_error = predicted_q - target_q

    # Use Huber loss for robustness to outliers
    huber_delta = 1.0
    abs_error = jnp.abs(td_error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    huber_loss = 0.5 * quadratic ** 2 + huber_delta * linear

    # Mean loss
    loss = jnp.mean(huber_loss)

    return loss

  return loss_fn


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting self-play training with Stockfish rewards')
  logging.info(f'Base model: {_BASE_MODEL.value}')
  logging.info(f'Iterations: {_NUM_ITERATIONS.value}')
  logging.info(f'Games per iteration: {_GAMES_PER_ITERATION.value}')

  # Load base model
  params, config = _load_base_model(_BASE_MODEL.value)
  params_ema = copy.deepcopy(params)

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
  opt_state = training_utils.replicate(opt_state, sharding)

  # Get return bucket values for Q-value computation
  num_return_buckets = 128
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )

  # Create loss and gradient functions
  loss_fn = _make_reward_loss_fn(predictor, return_buckets_values)
  grad_fn = jax.value_and_grad(loss_fn)

  @jax.jit
  def update_step(params, params_ema, opt_state, sequences, rewards):
    """Single gradient update step."""
    loss_val, grads = grad_fn(params, sequences, rewards)

    # Apply gradients
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Update EMA params
    ema_decay = 0.999
    params_ema = jax.tree.map(
        lambda ema, new: ema_decay * ema + (1 - ema_decay) * new,
        params_ema,
        params,
    )

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)

    return params, params_ema, opt_state, loss_val, grad_norm

  # Setup checkpoint manager
  checkpoint_manager = training_utils.get_checkpoint_manager(
      ckpt_frequency=_SAVE_FREQUENCY.value,
      max_to_keep=5,
      save_frequency=_SAVE_FREQUENCY.value,
      checkpoint_dir=checkpoint_dir,
  )

  # Main self-play training loop
  total_iterations = start_iteration + _NUM_ITERATIONS.value
  for iteration in range(start_iteration, total_iterations):
    logging.info(f'\n=== Iteration {iteration + 1}/{total_iterations} ===')

    # Create neural engine with current parameters
    # Un-shard params for inference
    local_params = jax.device_get(params_ema)

    # Use ActionValueEngine since we're keeping the Q-value architecture
    neural_engine = neural_engines.ActionValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=local_params,
            batch_size=1,
        ),
        temperature=1.0,
    )

    # Generate self-play games
    generator = selfplay_generator.SelfPlayGenerator(
        neural_engine=neural_engine,
        stockfish_time_limit=_STOCKFISH_TIME.value,
        max_moves_per_game=200,
        temperature=1.0,
        reward_scaling=0.01,
    )

    logging.info(f'Generating {_GAMES_PER_ITERATION.value} self-play games...')

    # Collect experiences
    all_sequences = []
    all_rewards = []

    for sequences_batch, rewards_batch in generator.generate_batch(
        num_games=_GAMES_PER_ITERATION.value,
        batch_size=_BATCH_SIZE.value,
    ):
      all_sequences.append(sequences_batch)
      all_rewards.append(rewards_batch)

    generator.close()

    if not all_sequences:
      logging.warning('No experiences generated, skipping training.')
      continue

    # Train on collected experiences
    logging.info(f'Training for {_GRADIENT_STEPS_PER_ITERATION.value} steps...')

    total_batches = len(all_sequences)
    batch_idx = 0

    for step in range(_GRADIENT_STEPS_PER_ITERATION.value):
      # Cycle through batches
      sequences = all_sequences[batch_idx % total_batches]
      rewards = all_rewards[batch_idx % total_batches]
      batch_idx += 1

      # Shard data
      sequences = jax.lax.with_sharding_constraint(sequences, sharding)
      rewards = jax.lax.with_sharding_constraint(rewards, sharding)

      # Update parameters
      params, params_ema, opt_state, loss_val, grad_norm = update_step(
          params, params_ema, opt_state, sequences, rewards
      )

      if step % 10 == 0:
        logging.info(
            f'  Step {step}/{_GRADIENT_STEPS_PER_ITERATION.value}: '
            f'loss={float(loss_val):.4f}, grad_norm={float(grad_norm):.4f}'
        )

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
