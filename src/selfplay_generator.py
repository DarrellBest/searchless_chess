"""Generates self-play games with Stockfish-based rewards."""

from collections.abc import Iterator
import dataclasses

import chess
import chess.engine
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import stockfish_engine


@dataclasses.dataclass
class SelfPlayExperience:
  """A single experience from self-play.

  Attributes:
    tokenized_fen: Tokenized board state.
    action: The action taken (move index).
    reward: Reward from Stockfish evaluation (centipawn advantage change).
    game_outcome: Final game outcome (1.0 for win, 0.0 for loss, 0.5 for draw).
  """
  tokenized_fen: np.ndarray
  action: int
  reward: float
  game_outcome: float


class SelfPlayGenerator:
  """Generates self-play games for reinforcement learning.

  This generator creates games where a neural engine plays against itself,
  with Stockfish providing reward signals based on position evaluations.
  """

  def __init__(
      self,
      neural_engine: engine_lib.Engine,
      stockfish_time_limit: float = 0.01,
      max_moves_per_game: int = 200,
      temperature: float = 1.0,
      reward_scaling: float = 0.01,
      game_outcome_weight: float = 0.1,
  ):
    """Initializes the self-play generator.

    Args:
      neural_engine: The neural engine that plays against itself.
      stockfish_time_limit: Time limit for Stockfish evaluation per position.
      max_moves_per_game: Maximum moves before declaring a draw.
      temperature: Temperature for move sampling (0.0 = greedy).
      reward_scaling: Scaling factor for centipawn rewards (default 0.01).
      game_outcome_weight: Weight for final game outcome (default 0.1).
    """
    self.neural_engine = neural_engine
    self.stockfish_engine = stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=stockfish_time_limit)
    )
    self.max_moves_per_game = max_moves_per_game
    self.temperature = temperature
    self.reward_scaling = reward_scaling
    self.game_outcome_weight = game_outcome_weight

  def _get_stockfish_score(self, board: chess.Board) -> float:
    """Gets the Stockfish score for a position (in centipawns).

    Args:
      board: The chess board to evaluate.

    Returns:
      Score in centipawns from the current player's perspective.
      Returns 0.0 if mate is detected.
    """
    info = self.stockfish_engine.analyse(board)
    score = info['score'].relative

    if score.is_mate():
      # Mate scores: large bonus/penalty
      mate_in = score.mate()
      if mate_in > 0:
        return 10000.0  # Winning mate
      else:
        return -10000.0  # Losing mate
    else:
      return float(score.score())

  def generate_game(self) -> list[SelfPlayExperience]:
    """Generates a single self-play game with reward signals.

    Returns:
      List of experiences from the game.
    """
    board = chess.Board()
    experiences = []
    move_count = 0

    while not board.is_game_over() and move_count < self.max_moves_per_game:
      # Skip if game should end due to repetition or fifty-move rule
      if board.can_claim_draw():
        break

      # Get current position evaluation
      score_before = self._get_stockfish_score(board)

      # Store the current state
      tokenized_fen = tokenizer.tokenize(board.fen())

      # Let the neural engine play a move
      move = self.neural_engine.play(board)
      action_idx = utils.MOVE_TO_ACTION[move.uci()]

      # Make the move
      board.push(move)
      move_count += 1

      # Get evaluation after the move (from opponent's perspective)
      score_after = self._get_stockfish_score(board)

      # Reward is the improvement in position
      # score_before is from our perspective, score_after is from opponent's perspective
      # So improvement = -score_after - score_before
      # Example: before=+100 (up 1 pawn), after=-200 (opp down 2 pawns) -> improvement = 200-100 = +100
      reward = (-score_after - score_before) * self.reward_scaling

      # Store experience (will update game_outcome at end)
      experiences.append(
          SelfPlayExperience(
              tokenized_fen=tokenized_fen,
              action=action_idx,
              reward=reward,
              game_outcome=0.5,  # Will be updated
          )
      )

    # Determine final game outcome from White's and Black's perspectives
    if board.is_checkmate():
      # The side to move lost
      if board.turn == chess.WHITE:
        # White is checkmated, Black won
        white_outcome = 0.0
        black_outcome = 1.0
      else:
        # Black is checkmated, White won
        white_outcome = 1.0
        black_outcome = 0.0
    elif board.is_stalemate() or board.can_claim_draw():
      white_outcome = 0.5
      black_outcome = 0.5
    else:
      # Use final evaluation to determine outcome
      final_score = self._get_stockfish_score(board)
      if abs(final_score) > 300:  # Clear advantage
        if final_score > 0:
          # Side to move has advantage
          if board.turn == chess.WHITE:
            white_outcome = 1.0
            black_outcome = 0.0
          else:
            white_outcome = 0.0
            black_outcome = 1.0
        else:
          # Side to move is losing
          if board.turn == chess.WHITE:
            white_outcome = 0.0
            black_outcome = 1.0
          else:
            white_outcome = 1.0
            black_outcome = 0.0
      else:
        white_outcome = 0.5
        black_outcome = 0.5

    # Propagate game outcome backwards based on which side made each move
    for i in range(len(experiences)):
      # i=0,2,4,... are White's moves (even indices)
      # i=1,3,5,... are Black's moves (odd indices)
      if i % 2 == 0:
        experiences[i].game_outcome = white_outcome
      else:
        experiences[i].game_outcome = black_outcome

    return experiences

  def generate_batch(
      self, num_games: int, batch_size: int
  ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generates batches of self-play experiences.

    Args:
      num_games: Number of games to generate.
      batch_size: Size of each training batch.

    Yields:
      Tuple of (sequences, rewards) where:
        - sequences: [batch_size, seq_len] tokenized states + actions
        - rewards: [batch_size] reward values
    """
    all_experiences = []

    # Generate games
    for game_idx in range(num_games):
      print(f'Generating self-play game {game_idx + 1}/{num_games}')
      experiences = self.generate_game()
      all_experiences.extend(experiences)

    print(f'Generated {len(all_experiences)} experiences from {num_games} games')

    # Shuffle experiences
    rng = np.random.default_rng()
    rng.shuffle(all_experiences)

    # Create batches
    num_batches = len(all_experiences) // batch_size
    for batch_idx in range(num_batches):
      start_idx = batch_idx * batch_size
      end_idx = start_idx + batch_size
      batch_experiences = all_experiences[start_idx:end_idx]

      # Create sequences: [tokenized_fen, action, dummy_return]
      sequences = []
      rewards = []

      for exp in batch_experiences:
        action = np.array([exp.action], dtype=np.int32)
        dummy_return = np.array([0], dtype=np.int32)
        sequence = np.concatenate([exp.tokenized_fen, action, dummy_return])
        sequences.append(sequence)

        # Combined reward: immediate centipawn change + weighted final outcome
        # This balances learning from position evaluation vs game result
        total_reward = exp.reward + self.game_outcome_weight * exp.game_outcome
        rewards.append(total_reward)

      sequences = np.stack(sequences).astype(np.int32)
      rewards = np.array(rewards, dtype=np.float32)

      yield sequences, rewards

  def close(self):
    """Closes the Stockfish engine."""
    # Stockfish engine cleanup is handled automatically
    pass
