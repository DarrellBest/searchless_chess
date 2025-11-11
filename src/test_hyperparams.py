"""Quick hyperparameter testing for DPO training.

Runs small-scale experiments (500 pairs, 100 batches) to validate hyperparameters
before committing to full training runs.
"""

import logging
import os

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import optax

from searchless_chess.src import lichess_dpo_generator
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import neural_engines

# Quick test parameters
_BASE_MODEL = flags.DEFINE_string('base_model', '9M', 'Base model to test')
_LICHESS_DB_PATH = flags.DEFINE_string(
    'lichess_db_path',
    '../data/lichess_db_eval.jsonl.zst',
    'Path to Lichess database'
)

# Hyperparameters to test
_LEARNING_RATES = flags.DEFINE_list(
    'learning_rates',
    ['1e-7', '5e-7', '1e-6', '5e-6', '1e-5'],
    'Learning rates to test (comma-separated)'
)
_BETAS = flags.DEFINE_list(
    'betas',
    ['0.05', '0.1', '0.2'],
    'Beta values to test'
)
_TEMPERATURES = flags.DEFINE_list(
    'temperatures',
    ['0.5', '1.0', '2.0'],
    'Temperature values to test'
)

# Test configuration
_NUM_PAIRS = flags.DEFINE_integer('num_pairs', 500, 'Pairs for quick test')
_NUM_BATCHES = flags.DEFINE_integer('num_batches', 100, 'Batches to test')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'Batch size')


def dpo_loss_fn(params, reference_params, positions, chosen_moves, rejected_moves,
                predictor, beta, temperature):
  """Simplified DPO loss for testing."""
  policy_values = predictor.apply(params, positions)
  reference_values = predictor.apply(reference_params, positions)

  policy_logits = policy_values / temperature
  ref_logits = reference_values / temperature

  policy_logprobs = jax.nn.log_softmax(policy_logits, axis=-1)
  ref_logprobs = jax.nn.log_softmax(ref_logits, axis=-1)

  batch_indices = jnp.arange(len(chosen_moves))
  policy_chosen = policy_logprobs[batch_indices, chosen_moves]
  policy_rejected = policy_logprobs[batch_indices, rejected_moves]
  ref_chosen = ref_logprobs[batch_indices, chosen_moves]
  ref_rejected = ref_logprobs[batch_indices, rejected_moves]

  pi_logratios = policy_chosen - policy_rejected
  ref_logratios = ref_chosen - ref_rejected

  logits = beta * (pi_logratios - ref_logratios)
  loss = -jnp.mean(jax.nn.log_sigmoid(logits))

  accuracy = jnp.mean(pi_logratios > 0)
  policy_probs = jnp.exp(policy_logprobs)
  kl_div = jnp.sum(policy_probs * (policy_logprobs - ref_logprobs), axis=-1).mean()

  return loss, {'accuracy': accuracy, 'kl': kl_div}


def test_hyperparams(base_model, lr, beta, temperature):
  """Run quick test with given hyperparameters."""
  logging.info(f'\nTesting: LR={lr}, beta={beta}, temp={temperature}')

  # Build model
  match base_model:
    case '9M':
      num_layers, embedding_dim, num_heads = 8, 256, 8
    case '136M':
      num_layers, embedding_dim, num_heads = 8, 1024, 8
    case '270M':
      num_layers, embedding_dim, num_heads = 16, 1024, 8

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
  rng = jrandom.PRNGKey(42)
  initial_params = predictor.initial_params(
      rng=rng, targets=np.ones((1, 1), dtype=np.uint32)
  )

  # Load base model
  base_checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{base_model}')
  params = training_utils.load_parameters(
      checkpoint_dir=base_checkpoint_dir, params=initial_params, step=-1
  )
  reference_params = params

  # Generate test data
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
  predict_fn = neural_engines.wrap_predict_fn(
      predictor=predictor, params=params, batch_size=1
  )

  generator = lichess_dpo_generator.LichessDPOGenerator(
      predict_fn=predict_fn,
      database_path=_LICHESS_DB_PATH.value,
  )

  logging.info(f'Generating {_NUM_PAIRS.value} preference pairs...')
  batch_iterator, _ = generator.generate_batch(
      num_pairs_target=_NUM_PAIRS.value,
      batch_size=_BATCH_SIZE.value,
  )

  # Convert to arrays
  all_positions = []
  all_chosen = []
  all_rejected = []

  for positions_batch, chosen_batch, rejected_batch in batch_iterator():
    max_len = max(len(seq) for seq in positions_batch)
    positions = np.zeros((len(positions_batch), max_len), dtype=np.uint32)
    for i, seq in enumerate(positions_batch):
      positions[i, :len(seq)] = seq

    all_positions.append(positions)
    all_chosen.extend(chosen_batch)
    all_rejected.extend(rejected_batch)

  all_positions = np.vstack(all_positions)
  all_chosen = np.array(all_chosen, dtype=np.int32)
  all_rejected = np.array(all_rejected, dtype=np.int32)

  # Setup optimizer
  optimizer = optax.adamw(learning_rate=lr)
  opt_state = optimizer.init(params)

  # Training loop
  @jax.jit
  def update_step(params, ref_params, opt_state, pos, chosen, rejected):
    def loss_fn(p):
      return dpo_loss_fn(p, ref_params, pos, chosen, rejected, predictor, beta, temperature)

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad_norm = optax.global_norm(grads)
    grads, _ = optax.clip_by_global_norm(1.0).update(grads, opt_state)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val, grad_norm, metrics

  # Test for N batches
  num_samples = min(len(all_positions), _NUM_BATCHES.value * _BATCH_SIZE.value)
  indices = np.arange(num_samples)
  np.random.shuffle(indices)

  losses = []
  accuracies = []
  kls = []
  grad_norms = []

  for batch_idx in range(_NUM_BATCHES.value):
    start_idx = batch_idx * _BATCH_SIZE.value
    end_idx = start_idx + _BATCH_SIZE.value
    if end_idx > num_samples:
      break

    batch_indices = indices[start_idx:end_idx]
    positions = jnp.array(all_positions[batch_indices])
    chosen = jnp.array(all_chosen[batch_indices])
    rejected = jnp.array(all_rejected[batch_indices])

    params, opt_state, loss_val, grad_norm, metrics = update_step(
        params, reference_params, opt_state, positions, chosen, rejected
    )

    losses.append(float(loss_val))
    accuracies.append(float(metrics['accuracy']))
    kls.append(float(metrics['kl']))
    grad_norms.append(float(grad_norm))

  # Analyze results
  avg_loss = np.mean(losses)
  final_loss = losses[-1]
  loss_improvement = losses[0] - losses[-1]

  avg_acc = np.mean(accuracies)
  final_acc = accuracies[-1]

  avg_kl = np.mean(kls)
  max_kl = np.max(kls)
  final_kl = kls[-1]

  avg_grad_norm = np.mean(grad_norms)

  # Verdict
  verdict = []
  if final_kl > 0.5:
    verdict.append('FAIL: KL too high')
  elif final_kl < 0.01:
    verdict.append('WARN: KL too low (not learning)')
  else:
    verdict.append('PASS: KL in range')

  if loss_improvement < 0:
    verdict.append('FAIL: Loss increasing')
  elif loss_improvement < 0.05:
    verdict.append('WARN: Loss barely decreasing')
  else:
    verdict.append('PASS: Loss decreasing')

  if final_acc < 0.55:
    verdict.append('WARN: Low accuracy')
  elif final_acc > 0.85:
    verdict.append('PASS: Good accuracy')

  if avg_grad_norm > 2.0:
    verdict.append('WARN: Unstable gradients')

  overall = 'GOOD' if 'FAIL' not in ' '.join(verdict) else 'BAD'

  result = {
      'lr': lr,
      'beta': beta,
      'temperature': temperature,
      'avg_loss': avg_loss,
      'final_loss': final_loss,
      'loss_improvement': loss_improvement,
      'avg_accuracy': avg_acc,
      'final_accuracy': final_acc,
      'avg_kl': avg_kl,
      'max_kl': max_kl,
      'final_kl': final_kl,
      'avg_grad_norm': avg_grad_norm,
      'verdict': ', '.join(verdict),
      'overall': overall,
  }

  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many arguments')

  logging.info('='*80)
  logging.info('Quick Hyperparameter Test')
  logging.info('='*80)
  logging.info(f'Model: {_BASE_MODEL.value}')
  logging.info(f'Test pairs: {_NUM_PAIRS.value}')
  logging.info(f'Test batches: {_NUM_BATCHES.value}')
  logging.info('='*80)

  results = []

  # Test all combinations
  for lr_str in _LEARNING_RATES.value:
    for beta_str in _BETAS.value:
      for temp_str in _TEMPERATURES.value:
        lr = float(lr_str)
        beta = float(beta_str)
        temp = float(temp_str)

        try:
          result = test_hyperparams(_BASE_MODEL.value, lr, beta, temp)
          results.append(result)

          logging.info(f'\nResults for LR={lr}, beta={beta}, temp={temp}:')
          logging.info(f'  Loss: {result["final_loss"]:.4f} (Δ={result["loss_improvement"]:+.4f})')
          logging.info(f'  Accuracy: {result["final_accuracy"]:.2%}')
          logging.info(f'  KL divergence: {result["final_kl"]:.4f} (max={result["max_kl"]:.4f})')
          logging.info(f'  Grad norm: {result["avg_grad_norm"]:.4f}')
          logging.info(f'  Verdict: {result["verdict"]}')
          logging.info(f'  Overall: {result["overall"]}')

        except Exception as e:
          logging.error(f'Failed for LR={lr}, beta={beta}, temp={temp}: {e}')

  # Summary
  logging.info('\n' + '='*80)
  logging.info('SUMMARY')
  logging.info('='*80)

  good_results = [r for r in results if r['overall'] == 'GOOD']
  if good_results:
    # Sort by loss improvement
    good_results.sort(key=lambda x: x['loss_improvement'], reverse=True)

    logging.info(f'\nFound {len(good_results)} good configurations:')
    for i, r in enumerate(good_results[:3], 1):
      logging.info(f'\n#{i}: LR={r["lr"]}, beta={r["beta"]}, temp={r["temperature"]}')
      logging.info(f'    Loss improvement: {r["loss_improvement"]:.4f}')
      logging.info(f'    Final accuracy: {r["final_accuracy"]:.2%}')
      logging.info(f'    Final KL: {r["final_kl"]:.4f}')
      logging.info(f'    Verdict: {r["verdict"]}')

    best = good_results[0]
    logging.info('\n' + '='*80)
    logging.info('RECOMMENDED HYPERPARAMETERS:')
    logging.info('='*80)
    logging.info(f'  --learning_rate={best["lr"]}')
    logging.info(f'  --beta={best["beta"]}')
    logging.info(f'  --temperature={best["temperature"]}')
  else:
    logging.warning('No good configurations found!')
    logging.warning('Try reducing learning rates or adjusting beta range')


if __name__ == '__main__':
  app.run(main)
