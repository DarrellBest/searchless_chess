"""Head-to-head engine comparison with Elo calculation."""

from collections.abc import Sequence
import copy
import os
import subprocess
import tempfile

from absl import app
from absl import flags
import chess
import chess.pgn
import numpy as np

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import stockfish_engine


_ENGINE_1 = flags.DEFINE_string(
    'engine1',
    '9M',
    'First engine to compare.',
)

_ENGINE_2 = flags.DEFINE_string(
    'engine2',
    '9M_selfplay',
    'Second engine to compare.',
)

_NUM_GAMES = flags.DEFINE_integer(
    'num_games',
    50,
    'Number of games to play (must be even for balanced colors).',
)

_OUTPUT_PGN = flags.DEFINE_string(
    'output_pgn',
    None,
    'Path to save PGN file (optional).',
)


# Evaluation engine for early termination
_EVAL_STOCKFISH_ENGINE = stockfish_engine.StockfishEngine(
    limit=chess.engine.Limit(time=0.01)
)
_MIN_SCORE_TO_STOP = 1300


def _play_game(
    engines: tuple[engine.Engine, engine.Engine],
    engines_names: tuple[str, str],
    white_name: str,
    initial_board: chess.Board | None = None,
) -> chess.pgn.Game:
  """Plays a game of chess between two engines."""
  if initial_board is None:
    initial_board = chess.Board()
  white_player = engines_names.index(white_name)
  current_player = white_player if initial_board.turn else 1 - white_player
  board = initial_board
  result = None

  while not (
      board.is_game_over()
      or board.can_claim_fifty_moves()
      or board.is_repetition()
  ):
    # Check if one side has overwhelming advantage
    info = _EVAL_STOCKFISH_ENGINE.analyse(board)
    score = info['score'].relative

    # Handle mate positions
    if score.is_mate():
      mate_value = score.mate()
      if mate_value > 0:
        result = '1-0' if board.turn else '0-1'
      else:
        result = '0-1' if board.turn else '1-0'
      break

    # Handle large material advantage
    if abs(score.score()) > _MIN_SCORE_TO_STOP:
      if score.score() > 0:
        result = '1-0' if board.turn else '0-1'
      else:
        result = '0-1' if board.turn else '1-0'
      break

    # Play move
    move = engines[current_player].play(board)
    board.push(move)
    current_player = 1 - current_player

  # Determine result if not already set
  if result is None:
    if board.is_checkmate():
      result = '0-1' if board.turn else '1-0'
    else:
      result = '1/2-1/2'

  # Create PGN game
  game = chess.pgn.Game()
  game.headers['White'] = engines_names[white_player]
  game.headers['Black'] = engines_names[1 - white_player]
  game.headers['Result'] = result

  # Add moves
  node = game
  for move in initial_board.move_stack + board.move_stack[len(initial_board.move_stack):]:
    node = node.add_variation(move)

  return game


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _NUM_GAMES.value % 2 != 0:
    raise ValueError('num_games must be even to balance colors')

  engine1_name = _ENGINE_1.value
  engine2_name = _ENGINE_2.value

  print(f'\n{"="*60}')
  print(f'Head-to-Head Engine Comparison')
  print(f'{"="*60}')
  print(f'Engine 1: {engine1_name}')
  print(f'Engine 2: {engine2_name}')
  print(f'Games: {_NUM_GAMES.value}')
  print(f'{"="*60}\n')

  # Load engines
  print(f'Loading engines...')
  engine1 = constants.ENGINE_BUILDERS[engine1_name]()
  engine2 = constants.ENGINE_BUILDERS[engine2_name]()
  print(f'Engines loaded.\n')

  # Load openings
  openings_path = os.path.join(os.getcwd(), '../data/eco_openings.pgn')
  opening_boards = []

  with open(openings_path, 'r') as file:
    while (game := chess.pgn.read_game(file)) is not None:
      opening_boards.append(game.end().board())

  # Sample openings
  rng = np.random.default_rng(seed=42)
  opening_indices = rng.choice(
      np.arange(len(opening_boards)),
      size=_NUM_GAMES.value // 2,
      replace=False,
  )
  opening_boards = [opening_boards[idx] for idx in opening_indices]

  # Play games
  games = []
  results = {engine1_name: 0, engine2_name: 0, 'draws': 0}

  for i, opening_board in enumerate(opening_boards):
    # Game 1: engine1 as white
    print(f'Game {i*2 + 1}/{_NUM_GAMES.value}: {engine1_name} (White) vs {engine2_name} (Black)')
    game1 = _play_game(
        engines=(engine1, engine2),
        engines_names=(engine1_name, engine2_name),
        white_name=engine1_name,
        initial_board=copy.deepcopy(opening_board),
    )
    games.append(game1)

    result1 = game1.headers['Result']
    if result1 == '1-0':
      results[engine1_name] += 1
    elif result1 == '0-1':
      results[engine2_name] += 1
    else:
      results['draws'] += 1
    print(f'  Result: {result1}')

    # Game 2: engine2 as white
    print(f'Game {i*2 + 2}/{_NUM_GAMES.value}: {engine2_name} (White) vs {engine1_name} (Black)')
    game2 = _play_game(
        engines=(engine2, engine1),
        engines_names=(engine2_name, engine1_name),
        white_name=engine2_name,
        initial_board=copy.deepcopy(opening_board),
    )
    games.append(game2)

    result2 = game2.headers['Result']
    if result2 == '1-0':
      results[engine2_name] += 1
    elif result2 == '0-1':
      results[engine1_name] += 1
    else:
      results['draws'] += 1
    print(f'  Result: {result2}\n')

  # Save PGN
  pgn_path = _OUTPUT_PGN.value
  if pgn_path is None:
    pgn_path = f'../data/{engine1_name}_vs_{engine2_name}_games.pgn'

  print(f'Saving games to {pgn_path}...')
  with open(pgn_path, 'w') as file:
    for game in games:
      file.write(str(game))
      file.write('\n\n')

  # Display results
  print(f'\n{"="*60}')
  print('RESULTS')
  print(f'{"="*60}')
  print(f'{engine1_name}: {results[engine1_name]} wins')
  print(f'{engine2_name}: {results[engine2_name]} wins')
  print(f'Draws: {results["draws"]}')
  print(f'Total: {_NUM_GAMES.value} games')
  print(f'{"="*60}')

  # Calculate win percentage
  engine1_score = results[engine1_name] + 0.5 * results['draws']
  engine2_score = results[engine2_name] + 0.5 * results['draws']
  engine1_pct = (engine1_score / _NUM_GAMES.value) * 100
  engine2_pct = (engine2_score / _NUM_GAMES.value) * 100

  print(f'\nScore:')
  print(f'{engine1_name}: {engine1_score}/{_NUM_GAMES.value} ({engine1_pct:.1f}%)')
  print(f'{engine2_name}: {engine2_score}/{_NUM_GAMES.value} ({engine2_pct:.1f}%)')

  # Calculate approximate Elo difference
  # Using formula: Elo_diff ≈ -400 * log10(1/win_rate - 1)
  if engine1_score > 0 and engine1_score < _NUM_GAMES.value:
    win_rate = engine1_score / _NUM_GAMES.value
    elo_diff = -400 * np.log10(1 / win_rate - 1)
    print(f'\nApproximate Elo difference: {elo_diff:+.0f}')
    print(f'({engine2_name} is {elo_diff:+.0f} Elo relative to {engine1_name})')
  elif engine1_score == _NUM_GAMES.value:
    print(f'\n{engine1_name} won all games (Elo diff > +400)')
  else:
    print(f'\n{engine2_name} won all games (Elo diff < -400)')

  # Run BayesElo if available
  bayeselo_path = os.path.join(os.getcwd(), '../BayesElo/bayeselo')
  if os.path.exists(bayeselo_path):
    print(f'\n{"="*60}')
    print('Running BayesElo for precise Elo calculation...')
    print(f'{"="*60}\n')

    # Create BayesElo commands file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cmd_file:
      cmd_file.write(f'readpgn {os.path.abspath(pgn_path)}\n')
      cmd_file.write('elo\n')
      cmd_file.write('mm\n')
      cmd_file.write('ratings\n')
      cmd_file.write('x\n')
      cmd_file.write('x\n')
      cmd_filename = cmd_file.name

    # Run BayesElo
    result = subprocess.run(
        [bayeselo_path],
        stdin=open(cmd_filename, 'r'),
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    os.unlink(cmd_filename)
  else:
    print(f'\nBayesElo not found at {bayeselo_path}')
    print('For precise Elo calculation, compile BayesElo.')

  print(f'\nPGN file saved to: {pgn_path}')


if __name__ == '__main__':
  app.run(main)
