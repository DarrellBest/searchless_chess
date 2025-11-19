"""DPO (Direct Preference Optimization) loss for chess move learning.

This implements Direct Preference Optimization from Rafailov et al. 2023
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model".

DPO directly optimizes a model to prefer better moves (from Stockfish) over
worse moves (the model's own mistakes) without needing a separate reward model.

The loss function is:
  L_DPO = -E[log σ(β * (r_θ(s,w) - r_θ(s,l)))]

where:
  r_θ(s,a) = log π_θ(a|s) - log π_ref(a|s)
  w = winning/better move (Stockfish)
  l = losing/worse move (LLM)
  β = KL penalty coefficient
  σ = sigmoid function

References:
  Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
  Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
  https://arxiv.org/abs/2305.18290
"""

import jax
import jax.numpy as jnp


def compute_move_log_probs(
    params,
    predictor,
    positions: jnp.ndarray,
    moves: jnp.ndarray,
    z_atoms: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
  """Computes log probabilities of moves given positions.

  For action-value models, we derive move probabilities from Q-values:
    π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)

  where Q(s,a) = E[Z(s,a)] (expected return).

  Args:
    params: Model parameters.
    predictor: Transformer predictor function.
    positions: [batch_size, seq_len] Tokenized positions (FENs).
    moves: [batch_size] Move indices to compute log probs for.
    z_atoms: [n_atoms] Support atoms for Q-value distribution.
    temperature: Temperature for softmax over Q-values.

  Returns:
    [batch_size] Log probabilities of the specified moves.
  """
  batch_size = positions.shape[0]
  n_atoms = z_atoms.shape[0]

  # To get Q-values for all actions, we need to query the model for each action
  # The model takes [position_tokens, action, dummy_return] as input
  # and outputs a distribution over returns for that action

  # We'll compute Q-values for all possible actions (inefficient but correct)
  # In practice, we only need Q-values for legal moves, but for simplicity:

  # For each position, we need to:
  # 1. Get Q-value distributions for all actions
  # 2. Compute expected Q-values
  # 3. Apply softmax to get move probabilities
  # 4. Extract log prob for the specified move

  # This is computationally expensive - we need to do num_actions forward passes
  # per position. For now, let's compute it for the moves we care about.

  # Actually, looking at the model architecture more carefully:
  # The model outputs [batch_size, seq_len, n_atoms]
  # where seq_len includes position tokens + action token + return token
  # The output at position -2 (action token) gives the Q-value distribution

  # But we need Q-values for ALL actions to compute the probability.
  # This is the key challenge: the action-value model requires evaluating
  # all actions to get a policy distribution.

  # For DPO, we need both the chosen and rejected move probabilities.
  # Let's implement a helper that computes Q-values for specific actions.

  def get_q_value(pos_tokens, action_idx):
    """Get Q-value for a specific action."""
    # Create input: [pos_tokens, action_idx, 0]
    dummy_return = jnp.array([0], dtype=jnp.int32)
    action_token = jnp.array([action_idx], dtype=jnp.int32)
    input_seq = jnp.concatenate([pos_tokens, action_token, dummy_return])

    # Get distribution over returns
    log_probs = predictor.predict(
        params=params,
        targets=input_seq[None, :],
        rng=None
    )[0, -2]  # Distribution at action position

    probs = jnp.exp(log_probs)

    # Expected Q-value
    q_value = jnp.sum(probs * z_atoms)
    return q_value

  # For DPO, we actually need to compute probabilities over a set of legal moves
  # This is problematic because we don't have legal move masks here.
  #
  # Alternative approach: Use a simplified version where we approximate
  # the probability using only the Q-values of chosen vs rejected:
  #   π(a|s) ≈ exp(Q(s,a)/τ) / [exp(Q(s,chosen)/τ) + exp(Q(s,rejected)/τ)]
  #
  # This is an approximation but makes the computation tractable.
  #
  # Actually, for DPO we don't need the exact probabilities, we need
  # log π(a|s). If we assume the partition function is approximately constant
  # (or cancels out in the ratio), we can use:
  #   log π(a|s) ≈ Q(s,a) / τ - log Z(s)
  #
  # where Z(s) is the partition function. In the DPO loss, we compute:
  #   log π(chosen) - log π(rejected) = [Q(chosen) - Q(rejected)] / τ
  #
  # The partition function cancels! This makes DPO tractable with action-value models.

  # So we just need to compute Q-values for the specified moves
  log_probs_list = []

  for i in range(batch_size):
    pos_tokens = positions[i, :-2]  # Remove action and return tokens
    action_idx = moves[i]

    # Create input sequence
    dummy_return = jnp.array([0], dtype=jnp.int32)
    action_token = jnp.array([action_idx], dtype=jnp.int32)
    input_seq = jnp.concatenate([pos_tokens, action_token, dummy_return])

    # Get Q-value distribution
    log_dist = predictor.predict(
        params=params,
        targets=input_seq[None, :],
        rng=None
    )[0, -2]  # [n_atoms]

    probs = jnp.exp(log_dist)
    q_value = jnp.sum(probs * z_atoms)

    # Log probability (up to partition function)
    log_prob = q_value / temperature
    log_probs_list.append(log_prob)

  return jnp.array(log_probs_list)


def dpo_loss(
    online_params,
    reference_params,
    predictor,
    positions: jnp.ndarray,
    chosen_moves: jnp.ndarray,
    rejected_moves: jnp.ndarray,
    z_atoms: jnp.ndarray,
    beta: float = 10.0,
    temperature: float = 1.0,
    anchor_weight: float = 0.0,
) -> tuple[jnp.ndarray, dict]:
  """Computes DPO loss for preference pairs with Q-value anchoring.

  The loss encourages the model to assign higher probability to chosen moves
  (from Stockfish) compared to rejected moves (model's mistakes), relative
  to a reference model. Q-value anchoring prevents absolute Q-value drift by
  penalizing squared deviation from reference Q-values.

  IMPORTANT - Beta Scaling for Chess Q-Values:
    Chess Q-values are in range [-0.3, 0.3], much smaller than language model
    log probabilities (typically -5 to -15). With typical Q-value differences
    of ~0.1 and beta=0.1:
      - logits = 0.1 * 0.1 = 0.01  (too small!)
      - sigmoid(0.01) ≈ 0.5025     (only 50% preference)

    For strong preference signals, use beta = 5.0-20.0:
      - logits = 10.0 * 0.1 = 1.0  (strong signal)
      - sigmoid(1.0) ≈ 0.73        (73% preference)

  IMPORTANT - Q-Value Anchoring:
    DPO loss is invariant to shifting all Q-values by a constant, which can
    cause absolute Q-value drift (both chosen/rejected drop while random moves
    drift upward). Anchoring prevents this by penalizing squared Q-value
    deviation from the reference model. Recommended: anchor_weight = 0.5-2.0.

  Args:
    online_params: Current model parameters.
    reference_params: Reference model parameters (frozen).
    predictor: Transformer predictor function.
    positions: [batch_size, seq_len] Tokenized positions.
    chosen_moves: [batch_size] Better move indices (Stockfish).
    rejected_moves: [batch_size] Worse move indices (model's moves).
    z_atoms: [n_atoms] Support atoms for Q-value distributions.
    beta: KL penalty coefficient (default 10.0 for chess Q-values).
    temperature: Temperature for action selection (default 1.0).
    anchor_weight: Weight for Q-value anchoring to reference (default 0.0).

  Returns:
    Tuple of (loss, metrics_dict) where metrics contains:
      - dpo_loss: Base DPO loss
      - anchor_loss: Q-value anchoring loss
      - total_loss: Combined loss
      - reward_accuracy: % where chosen > rejected
      - reward_margin: Average log prob margin
      - q_chosen_mean/q_rejected_mean/q_diff_mean: Q-value diagnostics
      - q_all_mean/q_all_std: Overall Q-value statistics
      - q_ref_all_mean/q_ref_all_std: Reference Q-value statistics
      - logits_mean: Average DPO logits (beta * Q-diff)
  """
  # Compute log probabilities from online model
  log_pi_chosen = compute_move_log_probs(
      online_params, predictor, positions, chosen_moves, z_atoms, temperature
  )
  log_pi_rejected = compute_move_log_probs(
      online_params, predictor, positions, rejected_moves, z_atoms, temperature
  )

  # Compute log probabilities from reference model (stop gradient)
  log_ref_chosen = compute_move_log_probs(
      jax.lax.stop_gradient(reference_params),
      predictor,
      positions,
      chosen_moves,
      z_atoms,
      temperature
  )
  log_ref_rejected = compute_move_log_probs(
      jax.lax.stop_gradient(reference_params),
      predictor,
      positions,
      rejected_moves,
      z_atoms,
      temperature
  )

  # Compute reward ratios: r_θ(s,a) = log π_θ(a|s) - log π_ref(a|s)
  r_chosen = log_pi_chosen - log_ref_chosen
  r_rejected = log_pi_rejected - log_ref_rejected

  # DPO loss: -E[log σ(β * (r_chosen - r_rejected))]
  logits = beta * (r_chosen - r_rejected)
  dpo_loss_value = -jax.nn.log_sigmoid(logits).mean()

  # Q-value anchoring: Penalize squared deviation from reference Q-values
  # This prevents absolute Q-value drift while learning relative preferences
  # Extract Q-values from log-probs (log_prob = Q/temperature)
  q_chosen = log_pi_chosen * temperature
  q_rejected = log_pi_rejected * temperature
  q_ref_chosen = log_ref_chosen * temperature
  q_ref_rejected = log_ref_rejected * temperature

  anchor_loss_chosen = jnp.square(q_chosen - q_ref_chosen).mean()
  anchor_loss_rejected = jnp.square(q_rejected - q_ref_rejected).mean()
  anchor_loss = anchor_loss_chosen + anchor_loss_rejected

  # Total loss with optional anchoring
  total_loss = dpo_loss_value
  if anchor_weight > 0.0:
    total_loss = total_loss + anchor_weight * anchor_loss

  # Compute metrics for monitoring
  reward_accuracy = (r_chosen > r_rejected).astype(jnp.float32).mean()
  reward_margin = (r_chosen - r_rejected).mean()

  # Q-value statistics
  q_diff = q_chosen - q_rejected
  q_all = jnp.concatenate([q_chosen, q_rejected])
  q_ref_all = jnp.concatenate([q_ref_chosen, q_ref_rejected])

  metrics = {
      'dpo_loss': dpo_loss_value,
      'anchor_loss': anchor_loss,
      'total_loss': total_loss,
      'reward_accuracy': reward_accuracy,
      'reward_margin': reward_margin,
      'q_chosen_mean': q_chosen.mean(),
      'q_rejected_mean': q_rejected.mean(),
      'q_diff_mean': q_diff.mean(),
      'q_all_mean': q_all.mean(),
      'q_all_std': q_all.std(),
      'q_ref_all_mean': q_ref_all.mean(),
      'q_ref_all_std': q_ref_all.std(),
      'logits_mean': logits.mean(),
  }

  return total_loss, metrics


def make_dpo_loss_fn(predictor, z_atoms, beta=10.0, temperature=1.0, anchor_weight=0.0):
  """Creates a DPO loss function.

  Args:
    predictor: Transformer predictor function.
    z_atoms: Support atoms for Q-value distributions.
    beta: KL penalty coefficient (default 10.0 for chess Q-values).
    temperature: Temperature for action selection.
    anchor_weight: Weight for Q-value anchoring to reference (default 0.0).

  Returns:
    Loss function that takes (online_params, reference_params, batch).
  """
  def loss_fn(online_params, reference_params, positions, chosen_moves, rejected_moves):
    return dpo_loss(
        online_params=online_params,
        reference_params=reference_params,
        predictor=predictor,
        positions=positions,
        chosen_moves=chosen_moves,
        rejected_moves=rejected_moves,
        z_atoms=z_atoms,
        beta=beta,
        temperature=temperature,
        anchor_weight=anchor_weight,
    )

  return loss_fn
