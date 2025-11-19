"""Inspect DPO loss gradients to check if they point in the right direction."""

import jax
import jax.numpy as jnp
import numpy as np
import chess
import os
from jax import random as jrandom
from lichess_dpo_generator import LichessDPOGenerator
from engines.constants import ENGINE_BUILDERS
from searchless_chess.src import utils, training_utils, transformer
from searchless_chess.src import tokenizer as token_module

def main():
    print("=" * 80)
    print("Inspecting DPO Gradients")
    print("=" * 80)

    # Load model (similar to lichess_train.py)
    print("\nLoading 9M model...")

    # Model config for 9M
    num_heads = 8
    num_layers = 8
    embedding_dim = 256

    predictor_config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=128,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=token_module.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = transformer.build_transformer_predictor(config=predictor_config)

    rng = jrandom.PRNGKey(42)
    dummy_targets = np.ones((1, 1), dtype=np.uint32)
    initial_params = predictor.initial_params(rng=rng, targets=dummy_targets)

    base_checkpoint_dir = os.path.join(os.getcwd(), '../checkpoints/9M')
    params = training_utils.load_parameters(
        checkpoint_dir=base_checkpoint_dir,
        params=initial_params,
        step=-1,
    )

    # Create a simple predict function for the generator
    def simple_predict_fn(sequences):
        """Wrapper to use with generator."""
        return predictor.predict(params=params, targets=sequences, rng=None)

    # Get a single training pair
    generator = LichessDPOGenerator(
        predict_fn=simple_predict_fn,
        database_path='../data/lichess_db_eval.jsonl.zst'
    )

    # Find first pair where model disagrees
    for position_data in generator.stream_positions(max_positions=100):
        try:
            best_line = generator.get_best_line(position_data['evals'])
            acceptable_moves = generator.get_all_pv_first_moves(position_data['evals'])

            fen4 = position_data['fen']
            fen6 = generator.fen4_to_fen6(fen4)
            board = chess.Board(fen6)

            if board.is_game_over():
                continue

            model_move = generator.predict_model_move(board)

            if model_move.uci() not in acceptable_moves:
                chosen_move = best_line[0]
                rejected_move = model_move.uci()

                print(f"\nTest position:")
                print(f"  FEN: {fen6}")
                print(f"  Chosen (Stockfish best): {chosen_move}")
                print(f"  Rejected (model choice): {rejected_move}")
                break
        except:
            continue

    # Get Q-values for this pair
    from searchless_chess.src import tokenizer

    sorted_legal_moves = list(board.legal_moves)
    chosen_move_obj = chess.Move.from_uci(chosen_move)
    rejected_move_obj = chess.Move.from_uci(rejected_move)

    chosen_action = utils.MOVE_TO_ACTION[chosen_move]
    rejected_action = utils.MOVE_TO_ACTION[rejected_move]

    # Tokenize position
    tokenized_fen = token_module.tokenize(board.fen()).astype(np.int32)
    position = jnp.array(tokenized_fen, dtype=jnp.uint32)

    # Create sequences
    dummy_return = jnp.zeros((1,), dtype=jnp.int32)
    chosen_seq = jnp.concatenate([position, jnp.array([chosen_action], dtype=jnp.int32), dummy_return])
    rejected_seq = jnp.concatenate([position, jnp.array([rejected_action], dtype=jnp.int32), dummy_return])

    # Get Q-values
    def get_q(p, seq):
        logprobs = predictor.predict(params=p, targets=seq[None, :], rng=None)[:, -1]
        probs = jnp.exp(logprobs)
        _, return_values = utils.get_uniform_buckets_edges_values(128)
        return_values = jnp.array(return_values, dtype=jnp.float32)
        q = jnp.sum(probs * return_values, axis=-1)[0]
        return q

    q_chosen = get_q(params, chosen_seq)
    q_rejected = get_q(params, rejected_seq)

    print(f"\nInitial Q-values:")
    print(f"  Q(chosen)  = {q_chosen:.4f}")
    print(f"  Q(rejected) = {q_rejected:.4f}")
    print(f"  Q-diff = {q_chosen - q_rejected:.4f}")

    # Define simple DPO loss for this single pair
    def loss_fn(p):
        q_c = get_q(p, chosen_seq)
        q_r = get_q(p, rejected_seq)

        # Reference is initial params (frozen)
        ref_q_c = get_q(params, chosen_seq)
        ref_q_r = get_q(params, rejected_seq)

        # DPO loss
        beta = 10.0
        temp = 1.0

        pi_logratio = (q_c - q_r) / temp
        ref_logratio = (ref_q_c - ref_q_r) / temp
        logits = beta * (pi_logratio - ref_logratio)
        loss = -jax.nn.log_sigmoid(logits)

        return loss, (q_c, q_r, pi_logratio, logits)

    # Compute loss and gradients
    (loss_val, (q_c, q_r, pi_logratio, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    print(f"\nLoss computation:")
    print(f"  pi_logratio = {pi_logratio:.4f}")
    print(f"  logits = {logits:.4f}")
    print(f"  loss = {loss_val:.4f}")

    # Check gradient direction
    # If we apply a small negative step (gradient descent), what happens to Q-values?
    learning_rate = 0.01
    test_params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

    q_chosen_after = get_q(test_params, chosen_seq)
    q_rejected_after = get_q(test_params, rejected_seq)

    print(f"\nAfter one gradient step (lr={learning_rate}):")
    print(f"  Q(chosen)  = {q_chosen_after:.4f}  (delta: {q_chosen_after - q_chosen:+.4f})")
    print(f"  Q(rejected) = {q_rejected_after:.4f}  (delta: {q_rejected_after - q_rejected:+.4f})")
    print(f"  Q-diff = {q_chosen_after - q_rejected_after:.4f}  (delta: {(q_chosen_after - q_rejected_after) - (q_chosen - q_rejected):+.4f})")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    delta_chosen = float(q_chosen_after - q_chosen)
    delta_rejected = float(q_rejected_after - q_rejected)
    delta_diff = float((q_chosen_after - q_rejected_after) - (q_chosen - q_rejected))

    print(f"\nExpected behavior:")
    print(f"  - Q(chosen) should INCREASE (delta > 0)")
    print(f"  - Q(rejected) should DECREASE (delta < 0)")
    print(f"  - Q-diff should INCREASE (delta > 0)")

    print(f"\nActual behavior:")
    print(f"  - Q(chosen):   delta = {delta_chosen:+.4f}  {'✓ CORRECT' if delta_chosen > 0 else '✗ WRONG'}")
    print(f"  - Q(rejected): delta = {delta_rejected:+.4f}  {'✓ CORRECT' if delta_rejected < 0 else '✗ WRONG'}")
    print(f"  - Q-diff:      delta = {delta_diff:+.4f}  {'✓ CORRECT' if delta_diff > 0 else '✗ WRONG'}")

    if delta_chosen < 0 and delta_rejected < 0:
        print("\n❌ CRITICAL BUG: Both Q-values are DECREASING!")
        print("   This confirms the gradient has the WRONG SIGN somewhere.")
        print("   The model is learning to be pessimistic rather than preferential.")
    elif delta_chosen < 0:
        print("\n❌ BUG: Q(chosen) is decreasing when it should increase!")
        print("   Gradient direction is wrong.")
    elif delta_rejected > 0:
        print("\n❌ BUG: Q(rejected) is increasing when it should decrease!")
        print("   Gradient direction is wrong.")
    elif delta_diff <= 0:
        print("\n❌ BUG: Q-diff is not increasing!")
        print("   DPO is not working as intended.")
    else:
        print("\n✓ Gradients point in the CORRECT direction!")
        print("  The bug must be elsewhere (e.g., data loading, model drift, etc.)")

    print("=" * 80)

if __name__ == '__main__':
    main()
