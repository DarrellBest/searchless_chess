"""Generates preference pairs for DPO training using self-play and Stockfish analysis."""

import dataclasses
from collections.abc import Iterator

import chess
import chess.engine
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import stockfish_engine


@dataclasses.dataclass
class GameTrajectory:
  """A single game trajectory for later analysis.

  Stores the raw game moves without evaluation, allowing for batch
  analysis by Stockfish later.
  """
  positions: list[str]
  moves: list[str]
  move_numbers: list[int]


@dataclasses.dataclass
class PreferencePair:
  """A preference pair for DPO training.

  Attributes:
    position: FEN string of the position.
    tokenized_position: Tokenized FEN for model input.
    chosen_move: Better move (from Stockfish).
    rejected_move: Worse move (from model).
    chosen_action: Action index for chosen move.
    rejected_action: Action index for rejected move.
    eval_margin: Difference in evaluation (Stockfish - LLM) in pawns.
  """
  position: str
  tokenized_position: np.ndarray
  chosen_move: str
  rejected_move: str
  chosen_action: int
  rejected_action: int
  eval_margin: float


class DPOSelfPlayGenerator:
  """Generates self-play games and creates preference pairs using Stockfish.

  This generator creates games where a neural engine plays against itself,
  then analyzes positions where the model made mistakes (compared to Stockfish)
  to create preference pairs for DPO training.
  """

  def __init__(
      self,
      neural_engine: engine_lib.Engine,
      stockfish_depth: int = 20,
      stockfish_time_limit: float = 0.1,
      max_moves_per_game: int = 200,
      eval_threshold: float = 0.3,
      max_position_eval: float = 3.0,
      temperature: float = 1.0,
  ):
    """Initializes the DPO self-play generator.

    Args:
      neural_engine: The neural engine that plays against itself.
      stockfish_depth: Depth for Stockfish analysis (default 20).
      stockfish_time_limit: Time limit for Stockfish evaluation per position.
      max_moves_per_game: Maximum moves before declaring a draw.
      eval_threshold: Minimum eval difference to create preference pair (pawns).
      max_position_eval: Maximum absolute eval to include position (pawns).
      temperature: Temperature for move sampling during self-play.
    """
    self.neural_engine = neural_engine
    self.stockfish_engine = stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=stockfish_time_limit, depth=stockfish_depth)
    )
    self.max_moves_per_game = max_moves_per_game
    self.eval_threshold = eval_threshold
    self.max_position_eval = max_position_eval
    self.temperature = temperature

  def generate_game(self) -> GameTrajectory:
    """Generates a single self-play game.

    Returns:
      GameTrajectory with positions and moves from the game.
    """
    board = chess.Board()
    positions = []
    moves = []
    move_numbers = []

    move_count = 0
    while not board.is_game_over() and move_count < self.max_moves_per_game:
      fen = board.fen()
      move = self.neural_engine.play(board)

      positions.append(fen)
      moves.append(move.uci())
      move_numbers.append(board.fullmove_number)

      board.push(move)
      move_count += 1

    return GameTrajectory(
        positions=positions,
        moves=moves,
        move_numbers=move_numbers,
    )

  def create_preferences(
      self, trajectories: list[GameTrajectory]
  ) -> list[PreferencePair]:
    """Analyzes game trajectories and creates preference pairs.

    For each position in the trajectories:
    1. Get Stockfish's best move and evaluation
    2. Evaluate the model's move
    3. If the model made a mistake (eval_diff > threshold), create a preference pair

    Args:
      trajectories: List of game trajectories to analyze.

    Returns:
      List of preference pairs where the model made mistakes.
    """
    preferences = []

    for traj_idx, trajectory in enumerate(trajectories):
      print(f'Analyzing trajectory {traj_idx + 1}/{len(trajectories)}...')

      for pos_idx, (fen, llm_move) in enumerate(zip(trajectory.positions, trajectory.moves)):
        board = chess.Board(fen)

        try:
          llm_move_obj = chess.Move.from_uci(llm_move)
        except ValueError:
          continue

        if llm_move_obj not in board.legal_moves:
          continue

        # Get Stockfish's preferred move and evaluation
        sf_result = self.stockfish_engine.analyse(board)
        sf_score = sf_result['score'].relative

        # Handle mate scores
        if sf_score.is_mate():
          mate_in = sf_score.mate()
          sf_eval = 100.0 if mate_in > 0 else -100.0
        else:
          sf_eval = sf_score.score() / 100.0

        # Skip if position is already decided (too winning or losing)
        if abs(sf_eval) > self.max_position_eval:
          continue

        # Get Stockfish's best move
        if 'pv' in sf_result and len(sf_result['pv']) > 0:
          sf_move = sf_result['pv'][0]
        else:
          continue

        # Skip if model already played Stockfish's move
        if llm_move_obj == sf_move:
          continue

        # Evaluate the model's move
        board.push(llm_move_obj)
        llm_result = self.stockfish_engine.analyse(board)
        llm_score = llm_result['score'].relative

        if llm_score.is_mate():
          mate_in = llm_score.mate()
          # Note: This is from opponent's perspective, so flip sign
          llm_eval = -100.0 if mate_in > 0 else 100.0
        else:
          # Flip sign because it's from opponent's perspective
          llm_eval = -llm_score.score() / 100.0

        board.pop()

        # Calculate evaluation difference
        eval_diff = sf_eval - llm_eval

        # Only create preference pair if significant mistake
        if eval_diff < self.eval_threshold:
          continue

        # Create preference pair
        tokenized_pos = tokenizer.tokenize(fen)

        try:
          sf_action = utils.MOVE_TO_ACTION[sf_move.uci()]
          llm_action = utils.MOVE_TO_ACTION[llm_move]
        except KeyError:
          continue

        preferences.append(
            PreferencePair(
                position=fen,
                tokenized_position=tokenized_pos,
                chosen_move=sf_move.uci(),
                rejected_move=llm_move,
                chosen_action=sf_action,
                rejected_action=llm_action,
                eval_margin=eval_diff,
            )
        )

    return preferences

  def generate_batch(
      self, num_games: int, batch_size: int
  ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generates batches of preference pairs for DPO training.

    Args:
      num_games: Number of self-play games to generate.
      batch_size: Size of each training batch.

    Yields:
      Tuple of (positions, chosen_moves, rejected_moves) where:
        - positions: [batch_size, seq_len] tokenized positions
        - chosen_moves: [batch_size] action indices for Stockfish moves
        - rejected_moves: [batch_size] action indices for model moves
    """
    # Generate games
    trajectories = []
    for game_idx in range(num_games):
      print(f'Generating self-play game {game_idx + 1}/{num_games}')
      trajectory = self.generate_game()
      trajectories.append(trajectory)

    # Analyze and create preference pairs
    print(f'\nAnalyzing {len(trajectories)} games with Stockfish...')
    preferences = self.create_preferences(trajectories)

    print(f'Found {len(preferences)} preference pairs (mistakes)')

    if len(preferences) == 0:
      print('No preferences found, model may have converged or threshold too high')
      return

    # Shuffle preferences
    rng = np.random.default_rng()
    rng.shuffle(preferences)

    # Create batches
    num_batches = len(preferences) // batch_size

    for batch_idx in range(num_batches):
      start_idx = batch_idx * batch_size
      end_idx = start_idx + batch_size
      batch_prefs = preferences[start_idx:end_idx]

      # Prepare batch data
      positions = []
      chosen_moves = []
      rejected_moves = []

      for pref in batch_prefs:
        # Position sequence: [tokenized_fen, dummy_action, dummy_return]
        # We'll add the action later in the loss computation
        dummy_action = np.array([0], dtype=np.int32)
        dummy_return = np.array([0], dtype=np.int32)
        pos_seq = np.concatenate([pref.tokenized_position, dummy_action, dummy_return])
        positions.append(pos_seq)

        chosen_moves.append(pref.chosen_action)
        rejected_moves.append(pref.rejected_action)

      positions = np.stack(positions).astype(np.int32)
      chosen_moves = np.array(chosen_moves, dtype=np.int32)
      rejected_moves = np.array(rejected_moves, dtype=np.int32)

      yield positions, chosen_moves, rejected_moves

  def close(self):
    """Closes the Stockfish engine."""
    pass
